"""Point-wise baselines: persistence (B0), hyperbolic (B1), Asaoka (B2).

Each function receives the cleaned training series of ONE point (t, y) and
the query times, and returns (prediction, fit_ok). When a geotechnical fit
fails (no convergence or a non-physical asymptote), the prediction falls back
to persistence and fit_ok=False, so that the failure rate can be reported and
the metrics can be computed both on all points and on converged points.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit


def _nonphysical(s_final: float, y: np.ndarray) -> bool:
    return (not np.isfinite(s_final)) or abs(s_final) > 3 * np.max(np.abs(y)) + 5.0


def persistence(t, y, t_query):
    return np.full(len(t_query), y[-1], float), True


def hyperbolic(t, y, t_query):
    """Hyperbolic method s(t') = t' / (a + b t'), parameterised as
    s = S_inf * t' / (T + t') so that both settlement and heave are allowed.
    S_inf = 1/b is the final movement and T = a/b the time to reach half of it."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    tp, tq = t - t[0], np.asarray(t_query, float) - t[0]
    if len(t) < 4:
        return persistence(t, y, t_query)[0], False

    def f(x, s_inf, tau):
        return s_inf * x / (tau + x)

    s0 = y[-1] * 1.3 if abs(y[-1]) > 0.1 else 0.5
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (s_inf, tau), _ = curve_fit(f, tp, y, p0=[s0, max(30.0, np.median(tp))],
                                        bounds=([-200, 1.0], [200, 5000.0]), maxfev=5000)
    except Exception:
        return persistence(t, y, t_query)[0], False
    if _nonphysical(s_inf, y) or tau >= 4999 or tau <= 1.01:
        return persistence(t, y, t_query)[0], False
    return f(tq, s_inf, tau), True


def asaoka(t, y, t_query, step: float = 30.0):
    """Asaoka (1978): on a constant step, s_j = b0 + b1 s_{j-1}; the final
    movement is b0 / (1 - b1) and the forecast decays geometrically from the
    last reading towards it. Requires 0 < b1 < 1."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    n_steps = int((t[-1] - t[0]) // step)
    if n_steps < 4:
        return persistence(t, y, t_query)[0], False
    grid = t[-1] - step * np.arange(n_steps, -1, -1)
    s = np.interp(grid, t, y)
    X = np.column_stack([np.ones(n_steps), s[:-1]])
    b0, b1 = np.linalg.lstsq(X, s[1:], rcond=None)[0]
    if not (0.0 < b1 < 1.0):
        return persistence(t, y, t_query)[0], False
    s_final = b0 / (1 - b1)
    if _nonphysical(s_final, y):
        return persistence(t, y, t_query)[0], False
    k = (np.asarray(t_query, float) - t[-1]) / step
    return s_final + (y[-1] - s_final) * b1 ** k, True


BASELINES = {"B0": persistence, "B1": hyperbolic, "B2": asaoka}
