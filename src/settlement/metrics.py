"""Metrics and paired statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HORIZON_BINS = [0, 90, 180, 365, 10_000]
HORIZON_LABELS = ["0-90 d", "91-180 d", "181-365 d", ">365 d"]


def mase_scale(train: pd.DataFrame) -> pd.Series:
    """Hyndman & Koehler (2006) scale per point: in-sample MAE of the one-step
    naive forecast over the training window (raw observations)."""
    return (train.sort_values("day").groupby("point_id")["settlement_mm"]
            .apply(lambda s: np.mean(np.abs(np.diff(s.to_numpy()))) if len(s) > 1 else np.nan))


def summarise(pred: pd.DataFrame, group_cols=("model",)) -> pd.DataFrame:
    """pred has one row per (model, test observation) with columns err (pred - obs),
    err_true (pred - noiseless signal, synthetic only), abs_scaled (|err| / MASE scale)."""
    g = pred.groupby(list(group_cols))
    out = pd.DataFrame({
        "MAE_mm": g["err"].apply(lambda e: np.mean(np.abs(e))),
        "RMSE_mm": g["err"].apply(lambda e: np.sqrt(np.mean(e ** 2))),
        "MASE": g["abs_scaled"].mean(),
        "MAE_true_mm": g["err_true"].apply(lambda e: np.mean(np.abs(e))),
        "n_obs": g.size(),
    })
    return out


def relative_mae(table: pd.DataFrame, by: str = "cut") -> pd.Series:
    """MAE of each model divided by the MAE of persistence (B0) in the same group."""
    ref = table.xs("B0", level="model")["MAE_mm"]
    return table["MAE_mm"] / ref.reindex(table.index.get_level_values(by)).to_numpy()


def last_campaign_error(pred: pd.DataFrame) -> pd.DataFrame:
    last = pred.loc[pred.groupby(["model", "cut", "point_id"])["day"].idxmax()]
    return last.groupby(["model", "cut"])["err"].apply(lambda e: np.mean(np.abs(e))).rename("MAE_last_mm")


def holm(pvals: pd.Series) -> pd.Series:
    order = pvals.sort_values()
    m = len(order)
    adj, running = [], 0.0
    for i, p in enumerate(order.to_numpy()):
        running = max(running, min(1.0, (m - i) * p))
        adj.append(running)
    return pd.Series(adj, index=order.index).reindex(pvals.index)


def paired_tests(pred: pd.DataFrame, models=("M1", "M2", "M3"), baselines=("B0", "B1", "B2")):
    """Wilcoxon signed-rank test, paired by point, of each model against the best
    baseline (lowest MAE) in each cut; Holm correction over all comparisons."""
    pm = pred.groupby(["cut", "model", "point_id"])["err"].apply(lambda e: np.mean(np.abs(e))).unstack("model")
    rows = []
    for cut, g in pm.groupby(level="cut"):
        best = g[list(baselines)].mean().idxmin()
        for m in models:
            if m not in g:
                continue
            d = (g[m] - g[best]).dropna()
            stat, p = wilcoxon(d) if (d != 0).any() else (np.nan, 1.0)
            rows.append(dict(cut=cut, model=m, best_baseline=best, median_diff_mm=d.median(),
                             mean_diff_mm=d.mean(), share_better=(d < 0).mean(), n_points=len(d), p=p))
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p"])
    return res


def paired_tests_blocked(pred: pd.DataFrame, block_cols=("dataset", "embankment", "station"),
                         models=("M1", "M1-H", "M2", "M3"), baselines=("B0", "B1", "B2")):
    """Wilcoxon signed-rank test paired by BLOCK (default: station of each dataset)
    rather than by point: points of a station share their profile and campaign
    errors, so treating them as independent would overstate the evidence.
    The best baseline (lowest mean block MAE) is chosen per cut; Holm over all tests."""
    keys = ["cut", "model", *block_cols]
    pb = pred.groupby(keys)["err"].apply(lambda e: np.mean(np.abs(e))).unstack("model")
    rows = []
    for cut, g in pb.groupby(level="cut"):
        best = g[list(baselines)].mean().idxmin()
        for m in models:
            if m not in g:
                continue
            d = (g[m] - g[best]).dropna()
            stat, p = wilcoxon(d) if (d != 0).any() else (np.nan, 1.0)
            rows.append(dict(cut=cut, model=m, best_baseline=best, median_diff_mm=d.median(),
                             mean_diff_mm=d.mean(), share_better=(d < 0).mean(), n_blocks=len(d), p=p))
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p"])
    return res
