"""Build tables (CSV + Markdown) and figures (PNG) from every dataset in results/raw/.

Scenario A = hyperbolic generator, scenario B = exponential + logarithmic creep.
Means are taken over replicas (generator seeds) and model seeds; the reported
"± sd" is the standard deviation between replicas of the replica-level metric.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from settlement.metrics import HORIZON_BINS, HORIZON_LABELS, paired_tests_blocked  # noqa: E402
from settlement.protocol import temporal_split  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW, TAB, FIG = (os.path.join(ROOT, "results", d) for d in ("raw", "tables", "figures"))
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

ORDER = ["B0", "B1", "B2", "M1", "M1-H", "M2", "M3"]
SCEN = {"hyperbolic": "A (hiperbólico)", "exponential": "B (exponencial + fluencia)"}
NAMES = {"B0": "B0 Persistencia", "B1": "B1 Hiperbólico", "B2": "B2 Asaoka", "M1": "M1 Proceso gaussiano",
         "M1-H": "M1-H Híbrido", "M2": "M2 Gradient boosting", "M3": "M3 LSTM"}
COLORS = {"B0": "#52514e", "B1": "#6250d6", "B2": "#e87ba4", "M1": "#1baf7a", "M1-H": "#eda100",
          "M2": "#eb6834", "M3": "#2a78d6"}
STYLE = {m: ("--" if m.startswith("B") else "-") for m in ORDER}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2a28", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "legend.frameon": False, "savefig.dpi": 200})


def save(df, name, floatfmt=".2f"):
    df.to_csv(os.path.join(TAB, f"{name}.csv"))
    with open(os.path.join(TAB, f"{name}.md"), "w") as f:
        f.write(df.to_markdown(floatfmt=floatfmt))


def load(name):
    parts = []
    for d in sorted(glob.glob(os.path.join(RAW, "*_s*"))):
        f = os.path.join(d, name)
        if not os.path.exists(f):
            continue
        x = pd.read_csv(f) if f.endswith((".csv", ".gz")) else pd.read_json(f)
        tag = os.path.basename(d)
        x["dataset"], x["curve"], x["gen_seed"] = tag, tag.split("_s")[0], int(tag.split("_s")[1])
        parts.append(x)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def mae(e):
    return float(np.mean(np.abs(e)))


def replica_metrics(p, keys):
    """Metrics per replica (dataset), averaging model seeds inside each replica."""
    g = p.groupby(["curve", "dataset", *keys])
    m = pd.DataFrame({"MAE": g["err"].apply(mae),
                      "RMSE": g["err"].apply(lambda e: float(np.sqrt(np.mean(e ** 2)))),
                      "MASE": g["abs_scaled"].mean(),
                      "MAE_true": g["err_true"].apply(mae)})
    sid = ["seed"] if "seed" in p else []
    if sid:
        p = p.assign(seed=p["seed"].fillna(-1))
    last = p.loc[p.groupby(["curve", "dataset", *keys, *sid, "point_id"])["day"].idxmax()]
    m["MAE_last"] = last.groupby(["curve", "dataset", *keys])["err"].apply(mae)
    return m


def pm(mu, sd):
    return f"{mu:.2f}" if not np.isfinite(sd) else f"{mu:.2f} ± {sd:.2f}"


def main():
    e1 = load("e1_predictions.csv.gz")
    fails = load("e1_fit_status.csv")
    hampel = load("e1_hampel_flags.csv")
    info = load("e1_model_info.json")
    e2, e3, e4 = load("e2_predictions.csv.gz"), load("e3_predictions.csv.gz"), load("e4_predictions.csv.gz")
    datasets = {os.path.basename(f)[:-4]: pd.read_csv(f) for f in sorted(glob.glob(os.path.join(ROOT, "data", "synthetic", "*_s*.csv")))}
    S = {"datasets": sorted(datasets)}

    # ---------------- Table 1: datasets ----------------
    rows = []
    for tag, df in datasets.items():
        last = df.sort_values("day").groupby("point_id").tail(1)
        for emb, g in df.groupby("embankment"):
            l = last[last["embankment"] == emb]
            rows.append(dict(dataset=tag, curve=tag.split("_s")[0], embankment=emb, points=g["point_id"].nunique(),
                             stations=g["station"].nunique(), lines=g["edge_dist_m"].nunique(),
                             campaigns=g["day"].nunique(), duration=g["day"].max(), obs=len(g),
                             final_min=l["true_mm"].min(), final_max=l["true_mm"].max(),
                             heave_pct=100 * (l["true_mm"] > 0).mean(), outliers=int(g["outlier_injected"].sum())))
    t1 = pd.DataFrame(rows)
    save(t1.set_index(["dataset", "embankment"]), "t1_datasets", ".1f")
    t1a = t1[t1["dataset"] == "hyperbolic_s2026"].set_index("embankment")
    save(t1a, "t1a_dataset_A_s2026", ".1f")
    S["dataset_totals"] = t1.groupby("dataset")[["points", "obs", "outliers"]].sum().to_dict(orient="index")
    fr = t1.groupby(["curve", "embankment"]).agg(fmin=("final_min", "min"), fmax=("final_max", "max"),
                                                 heave_min=("heave_pct", "min"), heave_max=("heave_pct", "max")).round(1)
    S["final_range_by_curve"] = {f"{c}|{e}": v for (c, e), v in fr.to_dict(orient="index").items()}

    # ---------------- Table 2: E1 main ----------------
    rm = replica_metrics(e1, ["cut", "model"])
    agg = rm.groupby(["curve", "cut", "model"]).agg(["mean", "std"])
    ref = agg[("MAE", "mean")].xs("B0", level="model")
    rel = agg[("MAE", "mean")] / ref.reindex(pd.MultiIndex.from_arrays(
        [agg.index.get_level_values("curve"), agg.index.get_level_values("cut")])).to_numpy()
    t2 = pd.DataFrame(index=agg.index)
    for c in ["MAE", "RMSE", "MASE", "MAE_last", "MAE_true"]:
        t2[c] = [pm(a, b) for a, b in zip(agg[(c, "mean")], agg[(c, "std")])]
    t2.insert(3, "relMAE", rel.round(3))
    t2 = t2.reindex(pd.MultiIndex.from_product([["hyperbolic", "exponential"], [30, 50, 70], ORDER],
                                               names=["curve", "cut", "model"]))
    save(t2, "t2_e1_main")
    S["e1_mae"] = {f"{c}|{k}|{m}": round(v, 3) for (c, k, m), v in agg[("MAE", "mean")].items()}
    S["e1_mae_sd_replicas"] = {f"{c}|{k}|{m}": round(v, 3) for (c, k, m), v in agg[("MAE", "std")].items()}
    S["e1_relmae"] = {f"{c}|{k}|{m}": round(v, 3) for (c, k, m), v in rel.items()}
    S["e1_mae_last"] = {f"{c}|{k}|{m}": round(v, 3) for (c, k, m), v in agg[("MAE_last", "mean")].items()}
    S["e1_mae_true"] = {f"{c}|{k}|{m}": round(v, 3) for (c, k, m), v in agg[("MAE_true", "mean")].items()}
    # seed spread of M2/M3 inside a replica
    sp = e1[e1["model"].isin(["M2", "M3"])].groupby(["curve", "dataset", "cut", "model", "seed"])["err"].apply(mae)
    S["seed_range"] = sp.groupby(["curve", "cut", "model"]).agg(["min", "max"]).round(2).reset_index().to_dict(orient="records")

    # ---------------- Table 3: MAE by horizon ----------------
    cut_day = e1["cut"] / 100 * e1.groupby(["dataset", "embankment"])["day"].transform("max")
    e1["h"] = pd.cut(e1["day"] - cut_day, HORIZON_BINS, labels=HORIZON_LABELS)
    t3 = e1.groupby(["curve", "cut", "h", "model"], observed=True)["err"].apply(mae).unstack("model")[ORDER]
    t3["n_obs_per_model"] = e1[e1["model"] == "B0"].groupby(["curve", "cut", "h"], observed=True).size()
    save(t3, "t3_e1_by_horizon")
    S["e1_by_horizon"] = {f"{c}|{k}|{h}": v for (c, k, h), v in t3.round(2).to_dict(orient="index").items()}

    # ---------------- Table 4: geotechnical fit failures ----------------
    f = fails[fails["model"].isin(["B1", "B2"])].copy()
    t4 = (100 * (1 - f.groupby(["curve", "cut", "model"])["fit_ok"].mean())).unstack("model").round(1)
    conv = []
    for (curve, cut, mod), g in f.groupby(["curve", "cut", "model"]):
        ok = set(zip(g.loc[g["fit_ok"], "dataset"], g.loc[g["fit_ok"], "point_id"]))
        sub = e1[(e1["curve"] == curve) & (e1["cut"] == cut)]
        sub = sub[[k in ok for k in zip(sub["dataset"], sub["point_id"])]]
        m = sub.groupby("model")["err"].apply(mae)
        conv.append(dict(curve=curve, cut=cut, model=mod, n_conv=len(ok), MAE_conv=m[mod], MAE_B0_conv=m["B0"]))
    t4b = pd.DataFrame(conv).set_index(["curve", "cut", "model"])
    lastt = pd.concat([d.sort_values("day").groupby("point_id").tail(1).assign(dataset=t)
                       for t, d in datasets.items()]).set_index(["dataset", "point_id"])["true_mm"]
    f["heave"] = [lastt.get((d, p), np.nan) > 0 for d, p in zip(f["dataset"], f["point_id"])]
    t4c = (100 * (1 - f.groupby(["curve", "model", "heave"])["fit_ok"].mean())).round(1)
    save(t4, "t4_fit_failure", ".1f")
    save(t4b, "t4b_converged_subset")
    save(t4c.to_frame("fail_pct"), "t4c_failure_by_sign", ".1f")
    S["fit_failure"] = {f"{c}|{k}": v for (c, k), v in t4.to_dict(orient="index").items()}
    S["converged"] = t4b.round(3).reset_index().to_dict(orient="records")
    S["failure_by_sign"] = {"|".join(map(str, k)): v for k, v in t4c.to_dict().items()}

    # ---------------- Table 5: uncertainty ----------------
    rows = []
    for (curve, cut, mod), g in e1[e1["model"].isin(["M1", "M1-H"])].groupby(["curve", "cut", "model"]):
        lo, hi = g["pred"] - 1.96 * g["std"], g["pred"] + 1.96 * g["std"]
        rows.append(dict(curve=curve, cut=cut, model=mod,
                         coverage_95=100 * ((g["settlement_mm"] >= lo) & (g["settlement_mm"] <= hi)).mean(),
                         width_mm=(hi - lo).mean()))
    t5 = pd.DataFrame(rows).set_index(["curve", "cut", "model"])
    save(t5, "t5_gp_coverage", ".1f")
    S["coverage"] = t5.round(1).reset_index().to_dict(orient="records")
    sig = {"T1": 1.0, "T2": 1.2, "T3": 0.9, "T4": 1.0, "T5": 1.5, "T6": 1.1}
    nz = info[info["model"] == "M1"][["curve", "dataset", "cut", "embankment", "noise_mm"]].copy()
    nz["noise_true_mm"] = nz["embankment"].map(lambda e: float(np.hypot(sig[e], 0.5)))
    nz["abs_err"] = (nz["noise_mm"] - nz["noise_true_mm"]).abs()
    save(nz.set_index(["dataset", "cut", "embankment"]), "t5b_gp_noise")
    S["gp_noise_abs_err"] = dict(mean=round(nz["abs_err"].mean(), 3), max=round(nz["abs_err"].max(), 3),
                                 est_range=[round(nz["noise_mm"].min(), 2), round(nz["noise_mm"].max(), 2)],
                                 true_range=[round(nz["noise_true_mm"].min(), 2), round(nz["noise_true_mm"].max(), 2)])

    # ---------------- Table 6: station-block Wilcoxon ----------------
    tests = []
    for curve, g in e1.groupby("curve"):
        avg = g.groupby(["cut", "model", "dataset", "embankment", "station", "point_id", "day"], as_index=False)["err"].mean()
        r = paired_tests_blocked(avg)
        r.insert(0, "curve", curve)
        tests.append(r)
    t6 = pd.concat(tests).set_index(["curve", "cut", "model"])
    save(t6, "t6_wilcoxon_blocks", ".4f")
    S["wilcoxon"] = t6.round(4).reset_index().to_dict(orient="records")

    # ---------------- Table 7: ablation (seed 0 in both arms) ----------------
    base = e1[e1["model"].isin(["M2", "M3"]) & (e1["seed"] == 0)]
    ab = pd.concat([base, e3])
    t7r = replica_metrics(ab, ["cut", "model"])["MAE"].groupby(["curve", "cut", "model"]).agg(["mean", "std"])
    t7 = pd.Series([pm(a, b) for a, b in zip(t7r["mean"], t7r["std"])], index=t7r.index).unstack("model")
    save(t7, "t7_ablation")
    S["ablation"] = {f"{c}|{k}|{m}": round(v, 3) for (c, k, m), v in t7r["mean"].items()}

    # ---------------- Table 8: E2 transfer (LOEO) ----------------
    e130 = e1[e1["cut"] == 30]
    rows = []
    for curve in ["hyperbolic", "exponential"]:
        for mod in ["M2", "M3"]:
            a = e2[(e2["curve"] == curve) & (e2["model"] == mod)]
            b = e130[(e130["curve"] == curve) & (e130["model"] == mod) & (e130["seed"] == 0)]
            c = e130[e130["curve"] == curve]
            per = []
            for (ds, emb), g in a.groupby(["dataset", "embankment"]):
                cc = c[(c["dataset"] == ds) & (c["embankment"] == emb)]
                bb = b[(b["dataset"] == ds) & (b["embankment"] == emb)]
                per.append(dict(e2=mae(g["err"]), e1=mae(bb["err"]),
                                b1=mae(cc[cc["model"] == "B1"]["err"]), m1h=mae(cc[cc["model"] == "M1-H"]["err"])))
            per = pd.DataFrame(per)
            rows.append(dict(curve=curve, model=mod, MAE_E2=mae(a["err"]), MAE_same_model_E1_30=mae(b["err"]),
                             MAE_B1_30=mae(c[c["model"] == "B1"]["err"]), MAE_M1H_30=mae(c[c["model"] == "M1-H"]["err"]),
                             MAE_B0_30=mae(c[c["model"] == "B0"]["err"]),
                             sites_better_than_B1=f"{int((per['e2'] < per['b1']).sum())}/{len(per)}",
                             sites_better_than_M1H=f"{int((per['e2'] < per['m1h']).sum())}/{len(per)}"))
    t8 = pd.DataFrame(rows).set_index(["curve", "model"])
    save(t8, "t8_e2_transfer")
    S["e2"] = t8.round(3).reset_index().to_dict(orient="records")

    # ---------------- Table 9: E4 data volume ----------------
    t9 = e4.groupby(["curve", "k_other"]).agg(MAE=("err", mae), train_points=("train_points", "mean"))
    refs = e130.groupby(["curve", "model"])["err"].apply(mae).unstack("model")[["B0", "B1", "M1-H"]]
    t9 = t9.join(refs, on="curve")
    save(t9, "t9_e4_data_volume")
    S["e4"] = t9.round(3).reset_index().to_dict(orient="records")

    # ---------------- Table 10: Hampel ----------------
    rows = []
    for tag, df in datasets.items():
        for frac in (0.3, 0.5, 0.7):
            tr, _ = temporal_split(df, frac)
            h = hampel[(hampel["dataset"] == tag) & (hampel["cut"] == int(frac * 100))]
            rows.append(dict(dataset=tag, cut=int(frac * 100), injected=int(tr["outlier_injected"].sum()),
                             flagged=len(h), true_pos=int(h["outlier_injected"].sum())))
    t10 = pd.DataFrame(rows).groupby("cut")[["injected", "flagged", "true_pos"]].sum()
    t10["precision"] = t10["true_pos"] / t10["flagged"]
    t10["recall"] = t10["true_pos"] / t10["injected"]
    save(t10, "t10_hampel")
    S["hampel"] = t10.round(3).reset_index().to_dict(orient="records")

    m3 = info[info["model"] == "M3"]
    S["m3_epoch1_share"] = m3.assign(e1=m3["best_epoch"] <= 1).groupby(["curve", "cut"])["e1"].mean().round(2).to_dict()
    S["m3_epoch1_share"] = {f"{k[0]}|{k[1]}": v for k, v in S["m3_epoch1_share"].items()}
    S["runtime_min"] = {os.path.basename(d): json.load(open(os.path.join(d, "run_meta.json")))["minutes"]
                        for d in sorted(glob.glob(os.path.join(RAW, "*_s*"))) if os.path.exists(os.path.join(d, "run_meta.json"))}
    json.dump(S, open(os.path.join(TAB, "summary.json"), "w"), indent=1, default=str)

    # =============== FIGURES ===============
    dfA = datasets["hyperbolic_s2026"]
    dfB = datasets.get("exponential_s2026")
    # Fig 1: example series (A, B)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), sharex=True)
    picks = [(dfA, "T1-E08-L1", "A"), (dfA, "T1-E01-L1", "A"), (dfA, "T1-E05-L1", "A"), (dfB, "T4-E10-L1", "B")]
    for ax, (d, pid, sc) in zip(axes.flat, picks):
        g = d[d["point_id"] == pid].sort_values("day")
        ax.plot(g["day"], g["true_mm"], color="#2a78d6", lw=2, label="Señal real (sin ruido)")
        ax.plot(g["day"], g["settlement_mm"], "o", ms=4, color="#52514e", mfc="white", mew=1.2, label="Observación")
        out = g[g["outlier_injected"]]
        ax.plot(out["day"], out["settlement_mm"], "x", ms=7, color="#e34948", mew=1.6, label="Error grueso")
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_title(f"Escenario {sc} · {pid}", fontsize=9, color="#2b2a28", loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Días desde la primera campaña")
    for ax in axes[:, 0]:
        ax.set_ylabel("Movimiento (mm)")
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_series_ejemplo.png"))
    plt.close(fig)

    # Fig 2: MAE by horizon, scenarios in rows
    fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.2), sharey=True)
    for r, curve in enumerate(["hyperbolic", "exponential"]):
        for c, cut in enumerate([30, 50, 70]):
            ax = axes[r, c]
            sub = t3.loc[(curve, cut)]
            x = np.arange(len(sub))
            for mod in ORDER:
                ax.plot(x, sub[mod], STYLE[mod], marker="o", ms=3.5, lw=1.8, color=COLORS[mod], label=mod)
            ax.set_xticks(x)
            ax.set_xticklabels(sub.index, rotation=30, ha="right", fontsize=7.5)
            ax.set_title(f"Escenario {'A' if r == 0 else 'B'} · corte {cut} %", fontsize=8.5, loc="left")
        axes[r, 0].set_ylabel("MAE (mm)")
    axes[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_mae_por_horizonte.png"))
    plt.close(fig)

    # Fig 3: example forecast (scenario A, replica 1)
    pid, cut = "T1-E08-L1", 50
    g = dfA[dfA["point_id"] == pid].sort_values("day")
    cd = cut / 100 * dfA[dfA["embankment"] == "T1"]["day"].max()
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.plot(g["day"], g["true_mm"], color="#c3c2b7", lw=3, label="Señal real")
    ax.plot(g["day"], g["settlement_mm"], "o", ms=4, color="#52514e", mfc="white", mew=1.2, label="Observación")
    sub = e1[(e1["dataset"] == "hyperbolic_s2026") & (e1["point_id"] == pid) & (e1["cut"] == cut)]
    for mod in ["B0", "B1", "B2", "M1-H", "M2", "M3"]:
        s = sub[sub["model"] == mod].groupby("day")[["pred", "std"]].mean().reset_index()
        ax.plot(s["day"], s["pred"], STYLE[mod], lw=2, color=COLORS[mod], label=NAMES[mod])
        if mod == "M1-H":
            ax.fill_between(s["day"], s["pred"] - 1.96 * s["std"], s["pred"] + 1.96 * s["std"],
                            color=COLORS[mod], alpha=0.15, lw=0)
    ax.axvline(cd, color="#8a8984", lw=1)
    ax.text(cd + 5, 0.3, "corte 50 %", fontsize=7.5, color="#52514e")
    ax.set_xlabel("Días desde la primera campaña")
    ax.set_ylabel("Movimiento (mm)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_pronostico_ejemplo.png"))
    plt.close(fig)

    # Fig 4: data-volume curve (E4)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, curve in zip(axes, ["hyperbolic", "exponential"]):
        s = t9.loc[curve]
        ax.plot(s["train_points"], s["MAE"], "-o", color=COLORS["M2"], lw=2, ms=5, label="M2 con k terraplenes completos")
        ax.set_xticks(s["train_points"])
        ax.set_xticklabels([f"k={k}\n{p:.0f}" for k, p in zip(s.index, s["train_points"])], fontsize=7.5)
        for ref, ls in [("B1", "--"), ("M1-H", "-")]:
            ax.axhline(s[ref].iloc[0], ls=ls, lw=1.4, color=COLORS[ref], label=f"{NAMES[ref]} (corte 30 %)")
        ax.set_title(f"Escenario {'A' if curve == 'hyperbolic' else 'B'}", fontsize=9, loc="left")
    fig.supxlabel("Terraplenes completos añadidos (k) y puntos de control en el entrenamiento", fontsize=9, color="#2b2a28")
    axes[0].set_ylabel("MAE (mm)")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_volumen_datos.png"))
    plt.close(fig)

    # Fig 5: LSTM learning curves (scenario A, replica 1, seed 0)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), sharey=True)
    for ax, cut in zip(axes, [30, 50, 70]):
        r = info[(info["dataset"] == "hyperbolic_s2026") & (info["model"] == "M3") & (info["cut"] == cut) & (info["seed"] == 0)].iloc[0]
        ax.plot(r["curve_train"], color="#2a78d6", lw=2, label="Entrenamiento")
        ax.plot(r["curve_valid"], color="#eb6834", lw=2, label="Validación interna")
        ax.axvline(r["best_epoch"] - 1, color="#8a8984", lw=1, ls=":")
        ax.set_title(f"Corte al {cut} %", fontsize=9, loc="left")
        ax.set_xlabel("Época")
    axes[0].set_ylabel("MSE (mm²)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_curvas_lstm.png"))
    plt.close(fig)
    print("ok")


if __name__ == "__main__":
    main()
