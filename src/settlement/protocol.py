"""Experimental protocols.

E1 - temporal origin: each embankment is cut at 30, 50 and 70 % of its duration;
     models learn from data before the cut and forecast every later campaign.
E2 - transfer between sites: leave-one-embankment-out (LOEO) for the
     transferable models M2/M3 (complete series of the other embankments plus
     the first 30 % of the held-out one).
E3 - ablation of the spatial features of M2/M3.
E4 - data-volume curve: error of M2 as the number of complete embankments
     available for training grows (0, 1, 2, 3, 5).
Random splits of individual observations are never used.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .baselines import BASELINES
from .metrics import mase_scale
from .models import GBMModel, GPModel, HybridGPModel, LSTMModel
from .preprocess import clean_training


def temporal_split(df: pd.DataFrame, frac: float):
    cut_day = df.groupby("embankment")["day"].transform(lambda s: frac * s.max())
    return df[df["day"] <= cut_day].copy(), df[df["day"] > cut_day].copy()


def _rows(model, test, pred, scale, cut, extra=None):
    out = test[["embankment", "point_id", "station", "edge_dist_m", "day", "settlement_mm", "true_mm"]].copy()
    out["model"], out["cut"], out["pred"] = model, cut, pred
    out["err"] = out["pred"] - out["settlement_mm"]
    out["err_true"] = out["pred"] - out["true_mm"]
    out["abs_scaled"] = np.abs(out["err"]) / out["point_id"].map(scale)
    for k, v in (extra or {}).items():
        out[k] = v
    return out


def run_baselines(train_c, test, scale, cut):
    res, fails = [], []
    for name, fn in BASELINES.items():
        pred = np.full(len(test), np.nan)
        for pid, idx in test.groupby("point_id").groups.items():
            g = train_c[train_c["point_id"] == pid].sort_values("day")
            p, ok = fn(g["day"].to_numpy(float), g["y"].to_numpy(float), test.loc[idx, "day"].to_numpy(float))
            pred[test.index.get_indexer(idx)] = p
            fails.append(dict(model=name, cut=cut, point_id=pid, fit_ok=ok))
        res.append(_rows(name, test, pred, scale, cut))
    return res, pd.DataFrame(fails)


def run_e1(df: pd.DataFrame, cuts=(0.3, 0.5, 0.7), seeds=(0, 1, 2, 3, 4), log=print):
    preds, fails, hampel_logs, info = [], [], [], []
    for frac in cuts:
        cut = int(frac * 100)
        train, test = temporal_split(df, frac)
        test = test[test["point_id"].isin(train["point_id"])]
        train_c, hlog = clean_training(train)
        hlog["cut"] = cut
        hampel_logs.append(hlog)
        scale = mase_scale(train)
        t0 = time.time()
        b, f = run_baselines(train_c, test, scale, cut)
        preds += b
        fails.append(f)
        gp = GPModel(seed=0).fit(train_c)
        m, s = gp.predict(test)
        preds.append(_rows("M1", test, m, scale, cut, {"std": s}))
        for emb in gp.gps:
            info.append(dict(cut=cut, model="M1", embankment=emb, noise_mm=gp.noise_mm(emb),
                             kernel=str(gp.gps[emb].kernel_)))
        hy = HybridGPModel(seed=0).fit(train_c)
        m, s = hy.predict(test)
        preds.append(_rows("M1-H", test, m, scale, cut, {"std": s}))
        log(f"[E1 {cut}%] baselines + GP done in {time.time() - t0:.0f}s")
        for seed in seeds:
            gbm = GBMModel(seed=seed).fit(train_c)
            preds.append(_rows("M2", test, gbm.predict(train_c, test), scale, cut, {"seed": seed}))
            info.append(dict(cut=cut, model="M2", seed=seed, params=str(gbm.best_params_), iters=gbm.best_iter_))
            lstm = LSTMModel(seed=seed).fit(train_c)
            preds.append(_rows("M3", test, lstm.predict(train_c, test), scale, cut, {"seed": seed}))
            info.append(dict(cut=cut, model="M3", seed=seed, best_epoch=lstm.best_epoch_,
                             curve_train=lstm.valid_history["train"], curve_valid=lstm.valid_history["valid"]))
        log(f"[E1 {cut}%] all models done in {time.time() - t0:.0f}s")
    return (pd.concat(preds, ignore_index=True), pd.concat(fails, ignore_index=True),
            pd.concat(hampel_logs, ignore_index=True), pd.DataFrame(info))


def run_e3_ablation(df, cuts=(0.3, 0.5, 0.7), seeds=(0, 1, 2), log=print):
    """Ablation (preliminary): M2 and M3 without spatial features."""
    preds = []
    for frac in cuts:
        cut = int(frac * 100)
        train, test = temporal_split(df, frac)
        test = test[test["point_id"].isin(train["point_id"])]
        train_c, _ = clean_training(train)
        scale = mase_scale(train)
        for seed in seeds:
            g = GBMModel(seed=seed, use_spatial=False).fit(train_c)
            preds.append(_rows("M2-noSpatial", test, g.predict(train_c, test), scale, cut, {"seed": seed}))
            l = LSTMModel(seed=seed, use_spatial=False).fit(train_c)
            preds.append(_rows("M3-noSpatial", test, l.predict(train_c, test), scale, cut, {"seed": seed}))
        log(f"[E3 {cut}%] done")
    return pd.concat(preds, ignore_index=True)


def run_e2(df: pd.DataFrame, seeds=(0,), frac: float = 0.3, log=print):
    """E2 - leave-one-embankment-out (LOEO): M2 and M3 learn from the complete
    series of every other embankment plus the first 30 % of the held-out one."""
    preds = []
    for emb in sorted(df["embankment"].unique()):
        target = df[df["embankment"] == emb]
        others = df[df["embankment"] != emb]
        tr_t, te_t = temporal_split(target, frac)
        te_t = te_t[te_t["point_id"].isin(tr_t["point_id"])]
        train_c, _ = clean_training(pd.concat([others, tr_t]))
        scale = mase_scale(tr_t)
        tr_target_c = train_c[train_c["embankment"] == emb]
        for seed in seeds:
            g = GBMModel(seed=seed).fit(train_c)
            preds.append(_rows("M2", te_t, g.predict(tr_target_c, te_t), scale, 30, {"seed": seed}))
            l = LSTMModel(seed=seed).fit(train_c)
            preds.append(_rows("M3", te_t, l.predict(tr_target_c, te_t), scale, 30, {"seed": seed}))
        log(f"[E2 LOEO] {emb} done")
    return pd.concat(preds, ignore_index=True)


def run_e4(df: pd.DataFrame, ks=(0, 1, 2, 3, 5), seed: int = 0, frac: float = 0.3, log=print):
    """E4 - data-volume curve: M2 learns from the first 30 % of the target
    embankment plus k complete embankments (the next k in a fixed rotation)."""
    embs = sorted(df["embankment"].unique())
    preds = []
    for i, emb in enumerate(embs):
        target = df[df["embankment"] == emb]
        tr_t, te_t = temporal_split(target, frac)
        te_t = te_t[te_t["point_id"].isin(tr_t["point_id"])]
        scale = mase_scale(tr_t)
        rotation = [embs[(i + j) % len(embs)] for j in range(1, len(embs))]
        for k in ks:
            others = df[df["embankment"].isin(rotation[:k])]
            train_c, _ = clean_training(pd.concat([others, tr_t]))
            g = GBMModel(seed=seed).fit(train_c)
            pts = int(train_c["point_id"].nunique())
            preds.append(_rows("M2", te_t, g.predict(train_c[train_c["embankment"] == emb], te_t), scale, 30,
                               {"k_other": k, "train_points": pts, "train_obs": len(train_c)}))
        log(f"[E4] {emb} done")
    return pd.concat(preds, ignore_index=True)
