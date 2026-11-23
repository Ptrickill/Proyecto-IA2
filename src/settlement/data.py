"""Data generation and loading.

The synthetic generator reproduces the regime described in Deliverable 1:
millimetre-scale settlements, irregular survey campaigns shared by all the
points of an embankment, points that heave instead of settling, seasonal
swelling, slow creep, survey noise, a common-mode (benchmark) error per
campaign and occasional gross errors (outliers).

Sign convention: negative = settlement, positive = heave, in millimetres,
relative to the first reading of each point.

The real data must follow the same long-format schema (see SCHEMA) so that
the rest of the pipeline runs unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SCHEMA = [
    "embankment",     # anonymous code: T1, T2, ...
    "point_id",       # unique id, e.g. T1-E03-L2
    "station",        # station index along the axis (0, 1, 2, ...)
    "abscissa_m",     # chainage relative to the start of the embankment (m)
    "edge_dist_m",    # distance from the embankment edge (m)
    "day",            # days since the first campaign of the embankment
    "settlement_mm",  # observed cumulative movement (mm), relative to first reading
]
SYNTHETIC_EXTRA = ["true_mm", "outlier_injected"]


@dataclass
class EmbankmentSpec:
    code: str
    n_stations: int
    station_spacing_m: float
    lines_m: list
    duration_days: int
    profile_mm: list | None = None      # final movement per station (None -> random smooth)
    profile_scale_mm: float = 10.0      # used when profile_mm is None
    seasonal_amp_mm: float = 0.8        # mean amplitude of seasonal swelling
    noise_mm: float = 1.0               # point survey noise (1 sigma)
    common_mode_mm: float = 0.5         # campaign benchmark error (1 sigma)
    outlier_rate: float = 0.02
    missing_rate: float = 0.04
    creep_fraction: float = 0.2
    tau_days: float = 90.0              # characteristic consolidation time
    extra: dict = field(default_factory=dict)


# Six embankments per scenario (292 control points). T1 reproduces the geometry
# and order of magnitude of the reference case (10 stations every 20 m, 5 lines
# from 1.0 to 10.0 m, ~557 days, final movements between +2 and -12.6 mm).
# The others vary magnitude, geometry, campaign density, noise and the share of
# heaving points, so that transfer between sites and the amount of training
# data can be studied.
DEFAULT_SPECS = [
    EmbankmentSpec("T1", 10, 20.0, [1.0, 3.25, 5.5, 7.75, 10.0], 557,
                   profile_mm=[-1.6, 2.0, 2.0, 0.2, -2.6, -12.0, -8.4, -9.6, -11.4, -12.6]),
    EmbankmentSpec("T2", 12, 20.0, [1.0, 3.25, 5.5, 7.75, 10.0], 480,
                   profile_scale_mm=14.0, seasonal_amp_mm=0.6, noise_mm=1.2),
    EmbankmentSpec("T3", 8, 25.0, [1.5, 4.0, 6.5, 9.0], 620,
                   profile_scale_mm=5.0, seasonal_amp_mm=1.4, noise_mm=0.9),
    EmbankmentSpec("T4", 15, 20.0, [1.0, 4.0, 7.0, 10.0], 700,
                   profile_scale_mm=16.0, seasonal_amp_mm=0.5, noise_mm=1.0, tau_days=140.0),
    EmbankmentSpec("T5", 10, 25.0, [2.0, 6.0, 10.0], 520,
                   profile_scale_mm=9.0, seasonal_amp_mm=0.9, noise_mm=1.5, tau_days=70.0),
    EmbankmentSpec("T6", 12, 20.0, [1.0, 3.25, 5.5, 7.75, 10.0], 650,
                   profile_scale_mm=11.0, seasonal_amp_mm=1.0, noise_mm=1.1, creep_fraction=0.35),
]

CURVES = ("hyperbolic", "exponential")


def _smooth_profile(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random profile along the axis (squared-exponential GP sample),
    biased towards settlement but allowing heave at some stations."""
    x = np.arange(n)[:, None]
    k = np.exp(-0.5 * (x - x.T) ** 2 / 2.0 ** 2) + 1e-6 * np.eye(n)
    z = rng.multivariate_normal(np.zeros(n), k)
    return scale * (0.5 * z - 0.7)


def _campaign_days(duration: int, rng: np.random.Generator) -> np.ndarray:
    """Irregular campaigns: dense at the beginning, sparser later."""
    days, t = [0.0], 0.0
    while True:
        frac = t / duration
        gap = max(7.0, rng.normal(14 + 34 * frac, 5))
        t += gap
        if t > duration:
            break
        days.append(round(t))
    if duration - days[-1] > 10:
        days.append(duration)
    return np.array(days, dtype=float)


def generate_embankment(spec: EmbankmentSpec, rng: np.random.Generator, curve: str = "hyperbolic") -> pd.DataFrame:
    """curve = "hyperbolic" (scenario A) or "exponential" (scenario B: Terzaghi-type
    exponential primary consolidation plus logarithmic secondary compression).
    Scenario B breaks the circularity between generator and hyperbolic models."""
    days = _campaign_days(spec.duration_days, rng)
    n_c = len(days)
    common = rng.normal(0, spec.common_mode_mm, n_c)
    common[0] = 0.0
    phase = rng.uniform(0, 365)
    if spec.profile_mm is not None:
        profile = np.asarray(spec.profile_mm, float)
    else:
        profile = _smooth_profile(spec.n_stations, spec.profile_scale_mm, rng)
    # consolidation time scale varies smoothly along the axis
    tau_station = np.exp(np.log(spec.tau_days) + 0.35 * np.cumsum(rng.normal(0, 0.3, spec.n_stations)))
    lines = np.asarray(spec.lines_m, float)
    dmax = lines.max()

    rows = []
    for s in range(spec.n_stations):
        for li, d in enumerate(lines):
            edge_factor = 1.25 - 0.5 * d / dmax          # edge (new fill) moves more
            s_final = profile[s] * edge_factor + rng.normal(0, 0.8)
            tau = tau_station[s] * np.exp(rng.normal(0, 0.15))
            amp = spec.seasonal_amp_mm * (1.4 - 0.8 * d / dmax) * np.exp(rng.normal(0, 0.2))
            creep = rng.normal(0, 0.5) / 100 if rng.random() < spec.creep_fraction else 0.0
            if curve == "hyperbolic":
                cons = s_final * days / (tau + days)
            else:   # 90 % of the movement as primary (exponential), 10 % as secondary (log) at ~600 d
                cons = s_final * (0.9 * (1 - np.exp(-days / tau)) + 0.1 * np.log1p(days / tau) / np.log1p(600 / tau))
            seas = amp * (np.sin(2 * np.pi * (days - phase) / 365) - np.sin(2 * np.pi * (-phase) / 365))
            true = cons + seas + creep * days
            obs = true + common + rng.normal(0, spec.noise_mm, n_c)
            out = rng.random(n_c) < spec.outlier_rate
            out[0] = False
            obs = obs + out * rng.choice([-1, 1], n_c) * rng.uniform(4, 8, n_c)
            obs = obs - obs[0]                         # relative to the first reading
            miss = rng.random(n_c) < spec.missing_rate
            miss[0] = False
            pid = f"{spec.code}-E{s:02d}-L{li + 1}"
            for k in range(n_c):
                if miss[k]:
                    continue
                rows.append((spec.code, pid, s, s * spec.station_spacing_m, d, days[k],
                             round(obs[k], 2), round(true[k], 3), bool(out[k])))
    return pd.DataFrame(rows, columns=SCHEMA + SYNTHETIC_EXTRA)


def generate_dataset(seed: int = 2026, specs=None, curve: str = "hyperbolic") -> pd.DataFrame:
    if curve not in CURVES:
        raise ValueError(f"curve must be one of {CURVES}")
    rng = np.random.default_rng(seed)
    specs = specs or DEFAULT_SPECS
    return pd.concat([generate_embankment(sp, rng, curve) for sp in specs], ignore_index=True)


def load_dataset(path: str) -> pd.DataFrame:
    """Load a long-format CSV with at least the SCHEMA columns."""
    df = pd.read_csv(path)
    missing = [c for c in SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "true_mm" not in df.columns:
        df["true_mm"] = np.nan
    if "outlier_injected" not in df.columns:
        df["outlier_injected"] = False
    return df.sort_values(["point_id", "day"]).reset_index(drop=True)
