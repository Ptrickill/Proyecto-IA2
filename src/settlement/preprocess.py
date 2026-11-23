"""Pre-processing: outlier screening (Hampel), constant-step interpolation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def hampel_flags(y: np.ndarray, half_window: int = 2, n_sigma: float = 3.0,
                 floor_mm: float = 1.5) -> np.ndarray:
    """Hampel filter: flag y[i] if |y[i] - median| > n_sigma * 1.4826 * MAD of a
    centred window. A floor (in mm) avoids flagging ordinary survey noise when
    the window happens to be almost flat (MAD close to 0). The first reading
    (reference, always 0) is never flagged."""
    y = np.asarray(y, float)
    n = len(y)
    flags = np.zeros(n, bool)
    for i in range(1, n):
        lo, hi = max(0, i - half_window), min(n, i + half_window + 1)
        w = y[lo:hi]
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        thr = max(n_sigma * mad, floor_mm * n_sigma)
        flags[i] = abs(y[i] - med) > thr
    return flags


def clean_training(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag outliers in the training window of each point and replace them by
    the window median for model fitting. Flags are returned for analysis:
    nothing is removed silently. Test targets are never modified."""
    out, log = [], []
    for pid, g in train.groupby("point_id", sort=False):
        g = g.sort_values("day").copy()
        y = g["settlement_mm"].to_numpy()
        f = hampel_flags(y)
        y_clean = y.copy()
        for i in np.where(f)[0]:
            lo, hi = max(0, i - 2), min(len(y), i + 3)
            y_clean[i] = np.median(np.delete(y[lo:hi], i - lo))
        g["y"] = y_clean
        g["hampel_flag"] = f
        out.append(g)
        log.append(g[f][["point_id", "day", "settlement_mm", "y", "outlier_injected"]])
    return pd.concat(out), (pd.concat(log) if log else pd.DataFrame())


def interp_grid(t: np.ndarray, y: np.ndarray, t_end: float, step: float, n_steps: int) -> np.ndarray:
    """Linear interpolation on a constant grid that ends at t_end and goes back
    n_steps * step days. Times before the first reading take the reference
    value (0 mm by construction)."""
    grid = t_end - step * np.arange(n_steps, -1, -1)
    grid = np.clip(grid, t[0], None)
    return np.interp(grid, t, y)
