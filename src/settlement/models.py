"""Global machine-learning models: GP (M1), gradient boosting (M2), LSTM (M3).

Common interface: fit(train) and predict(query) where
  * train: cleaned training observations (columns of SCHEMA + 'y'),
  * query: rows (point_id, day, ...) to forecast after the cut.
M2 and M3 are "direct" multi-horizon models: from an anchor reading (the last
one available for the point) they predict the change y(t_anchor + h) - y(t_anchor)
for any horizon h. Predicting the change anchors every model to persistence,
which is the natural reference in a noisy, millimetre-scale regime.
"""
from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from .preprocess import interp_grid

# ----------------------------------------------------------------------------
# M1 - Gaussian process regression over (time, abscissa, edge distance)
# ----------------------------------------------------------------------------


class GPModel:
    """One GP per embankment (spatial coordinates are site specific).
    Kernel: C * Matern(nu=3/2, anisotropic over t, x, d) + white noise.
    Hyper-parameters are fitted by maximising the marginal likelihood on the
    training window only. Predictive std includes the white-noise term, so the
    95 % interval refers to a new *observation*, which is what is evaluated."""

    def __init__(self, n_restarts: int = 2, seed: int = 0):
        self.n_restarts, self.seed = n_restarts, seed
        self.gps, self.scalers = {}, {}

    @staticmethod
    def _X(df):
        return df[["day", "abscissa_m", "edge_dist_m"]].to_numpy(float)

    def fit(self, train: pd.DataFrame):
        for emb, g in train.groupby("embankment"):
            X = self._X(g)
            mu, sd = X.mean(0), X.std(0) + 1e-9
            kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                      * Matern(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=1.5)
                      + WhiteKernel(0.1, (1e-3, 1e1)))
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=self.n_restarts, random_state=self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit((X - mu) / sd, g["y"].to_numpy())
            self.gps[emb], self.scalers[emb] = gp, (mu, sd)
        return self

    def noise_mm(self, emb):
        """Estimated observation-noise std (mm) for an embankment."""
        gp = self.gps[emb]
        return float(np.sqrt(gp.kernel_.k2.noise_level) * gp._y_train_std)

    def predict(self, query: pd.DataFrame):
        mean = np.full(len(query), np.nan)
        std = np.full(len(query), np.nan)
        for emb, idx in query.groupby("embankment").groups.items():
            gp = self.gps[emb]
            mu, sd = self.scalers[emb]
            q = query.loc[idx]
            m, s = gp.predict((self._X(q) - mu) / sd, return_std=True)
            pos = query.index.get_indexer(idx)
            mean[pos], std[pos] = m, s     # sklearn's std already contains the white kernel
        return mean, std


# ----------------------------------------------------------------------------
# Feature construction shared by M2 and M3
# ----------------------------------------------------------------------------

GRID_STEP, GRID_STEPS = 30.0, 3   # LSTM input: anchor and 3 previous 30-day steps


def _neighbour_table(train: pd.DataFrame) -> dict:
    """Sum and count of cleaned values per (embankment, station, day), used to
    compute the 'neighbour mean': points at the same and adjacent stations,
    excluding the point itself."""
    agg = train.groupby(["embankment", "station", "day"])["y"].agg(["sum", "count"])
    return {k: (v["sum"], v["count"]) for k, v in agg.iterrows()}


def _neighbour_mean(tab, emb, station, day, own_value=None):
    tot, cnt = 0.0, 0
    for s in (station - 1, station, station + 1):
        v = tab.get((emb, s, day))
        if v is not None:
            tot, cnt = tot + v[0], cnt + v[1]
    if own_value is not None:
        tot, cnt = tot - own_value, cnt - 1
    return tot / cnt if cnt > 0 else np.nan


MAX_TARGETS_PER_ANCHOR = 6


def build_pairs(train: pd.DataFrame, query: pd.DataFrame | None = None, min_anchor_idx: int = 1,
                max_targets: int = MAX_TARGETS_PER_ANCHOR):
    """Build (anchor, target) samples.
    Training mode (query is None): every anchor i >= min_anchor_idx and every
    later target j inside the training window of the same point.
    To keep the cost linear in the number of campaigns, each anchor keeps at most
    `max_targets` later targets, evenly spaced so that short and long horizons
    remain represented (deterministic).
    Forecast mode: anchor = last training reading of the point, targets = query rows.
    Returns a feature DataFrame, the LSTM sequences, the target change (NaN in
    forecast mode), the anchor value and the row index of the query."""
    tab = _neighbour_table(train)
    feats, seqs, ys, anchors, qidx = [], [], [], [], []
    series = {pid: g.sort_values("day") for pid, g in train.groupby("point_id")}
    targets = None if query is None else query.groupby("point_id")
    for pid, g in series.items():
        t, y = g["day"].to_numpy(float), g["y"].to_numpy(float)
        emb, st, d = g["embankment"].iat[0], int(g["station"].iat[0]), float(g["edge_dist_m"].iat[0])
        if query is None:
            anchor_ids = range(min_anchor_idx, len(t) - 1)
        else:
            if pid not in targets.groups:
                continue
            anchor_ids = [len(t) - 1]
        for i in anchor_ids:
            ti, yi = t[i], y[i]
            d1 = yi - y[i - 1] if i >= 1 else np.nan
            d2 = yi - y[i - 2] if i >= 2 else np.nan
            if i >= 2:
                slope = np.polyfit(t[max(0, i - 3):i + 1], y[max(0, i - 3):i + 1], 1)[0] * 100
            else:
                slope = np.nan
            nb = _neighbour_mean(tab, emb, st, ti, yi)
            nb_prev = _neighbour_mean(tab, emb, st, t[i - 1], y[i - 1]) if i >= 1 else np.nan
            seq = interp_grid(t[:i + 1], y[:i + 1], ti, GRID_STEP, GRID_STEPS) - yi
            if query is None:
                later = np.arange(i + 1, len(t))
                if len(later) > max_targets:
                    later = later[np.unique(np.round(np.linspace(0, len(later) - 1, max_targets)).astype(int))]
                tgt = [(t[j], y[j] - yi, None) for j in later]
            else:
                q = query.loc[targets.groups[pid]]
                tgt = [(tq, np.nan, ix) for tq, ix in zip(q["day"].to_numpy(float), q.index)]
            for tj, dy, ix in tgt:
                feats.append((yi, d1, d2, slope, ti, tj - ti, d, nb - yi if np.isfinite(nb) else np.nan,
                              nb - nb_prev if np.isfinite(nb) and np.isfinite(nb_prev) else np.nan))
                seqs.append(seq)
                ys.append(dy)
                anchors.append(yi)
                qidx.append(ix)
    cols = ["y_anchor", "diff1", "diff2", "slope_100d", "t_anchor", "horizon", "edge_dist",
            "nb_minus_own", "nb_diff1"]
    return (pd.DataFrame(feats, columns=cols), np.array(seqs, float), np.array(ys, float),
            np.array(anchors, float), np.array(qidx, dtype=object))


# ----------------------------------------------------------------------------
# M2 - gradient boosting (LightGBM)
# ----------------------------------------------------------------------------


class GBMModel:
    GRID = [dict(num_leaves=7, min_child_samples=10), dict(num_leaves=7, min_child_samples=30),
            dict(num_leaves=15, min_child_samples=10), dict(num_leaves=15, min_child_samples=30)]

    def __init__(self, seed: int = 0, use_spatial: bool = True):
        self.seed, self.use_spatial = seed, use_spatial

    def _cols(self, X):
        drop = [] if self.use_spatial else ["edge_dist", "nb_minus_own", "nb_diff1"]
        return X.drop(columns=drop)

    def _params(self, extra):
        return dict(objective="l2", learning_rate=0.05, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, random_state=self.seed, verbose=-1,
                    n_jobs=1, **extra)

    def fit(self, train: pd.DataFrame, valid_frac: float = 0.75):
        fit_tr, val_tr = split_train_valid(train, valid_frac)
        Xf, _, yf, _, _ = build_pairs(fit_tr)
        Xv, yv = validation_pairs(fit_tr, val_tr)
        best = (np.inf, None, None)
        for g in self.GRID:
            m = lgb.LGBMRegressor(n_estimators=600, **self._params(g))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(self._cols(Xf), yf, eval_set=[(self._cols(Xv), yv)],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
            score = m.best_score_["valid_0"]["l2"]
            if score < best[0]:
                best = (score, g, max(20, m.best_iteration_ or 600))
        self.best_params_, self.best_iter_ = best[1], best[2]
        X, _, y, _, _ = build_pairs(train)
        self.model = lgb.LGBMRegressor(n_estimators=self.best_iter_, **self._params(self.best_params_))
        self.model.fit(self._cols(X), y)
        return self

    def predict(self, train: pd.DataFrame, query: pd.DataFrame):
        X, _, _, anchor, qidx = build_pairs(train, query)
        pred = anchor + self.model.predict(self._cols(X))
        out = pd.Series(pred, index=list(qidx))
        return out.reindex(query.index).to_numpy()


def split_train_valid(train: pd.DataFrame, frac: float = 0.75):
    """Temporal split inside the training window of each embankment."""
    cut = train.groupby("embankment")["day"].transform(lambda s: s.min() + frac * (s.max() - s.min()))
    return train[train["day"] <= cut], train[train["day"] > cut]


def validation_pairs(fit_tr: pd.DataFrame, val_tr: pd.DataFrame, return_seq: bool = False):
    """Forecast-mode pairs from the end of the inner window to validation targets."""
    X, S, _, anchor, qidx = build_pairs(fit_tr, val_tr)
    ytrue = val_tr.loc[list(qidx), "y"].to_numpy() - anchor
    return (X, S, ytrue) if return_seq else (X, ytrue)


# ----------------------------------------------------------------------------
# M3 - small LSTM
# ----------------------------------------------------------------------------


class _LSTMNet(nn.Module):
    def __init__(self, n_static: int, hidden: int = 16, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + n_static, 16), nn.ReLU(),
                                  nn.Linear(16, 1))

    def forward(self, seq, static):
        _, (h, _) = self.lstm(seq)
        return self.head(torch.cat([h[-1], static], 1)).squeeze(1)


class LSTMModel:
    STATIC = ["horizon", "t_anchor", "edge_dist", "nb_minus_own", "nb_diff1"]
    Y_SCALE = 5.0   # mm

    def __init__(self, seed: int = 0, hidden: int = 16, max_epochs: int = 200, patience: int = 25,
                 use_spatial: bool = True):
        self.seed, self.hidden, self.max_epochs, self.patience = seed, hidden, max_epochs, patience
        self.use_spatial = use_spatial
        self.history = {"train": [], "valid": []}

    def _static(self, X):
        cols = self.STATIC if self.use_spatial else ["horizon", "t_anchor"]
        Z = X[cols].fillna(0.0).to_numpy(float)
        return (Z - self.mu) / self.sd

    def _tensors(self, X, S, y=None):
        seq = torch.tensor(S[:, :, None] / self.Y_SCALE, dtype=torch.float32)
        st = torch.tensor(self._static(X), dtype=torch.float32)
        yy = None if y is None else torch.tensor(y / self.Y_SCALE, dtype=torch.float32)
        return seq, st, yy

    def _train(self, X, S, y, Xv=None, Sv=None, yv=None, epochs=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        cols = self.STATIC if self.use_spatial else ["horizon", "t_anchor"]
        Z = X[cols].fillna(0.0).to_numpy(float)
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        net = _LSTMNet(len(cols), self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
        seq, st, yy = self._tensors(X, S, y)
        if Xv is not None:
            sv, stv, yvv = self._tensors(Xv, Sv, yv)
        g = torch.Generator().manual_seed(self.seed)
        best, best_state, wait, best_ep = np.inf, None, 0, 0
        n_ep = epochs or self.max_epochs
        for ep in range(n_ep):
            net.train()
            perm = torch.randperm(len(yy), generator=g)
            tot = 0.0
            for b in range(0, len(yy), 256):
                ix = perm[b:b + 256]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(seq[ix], st[ix]), yy[ix])
                loss.backward()
                opt.step()
                tot += loss.item() * len(ix)
            self.history["train"].append(tot / len(yy) * self.Y_SCALE ** 2)
            if Xv is not None:
                net.eval()
                with torch.no_grad():
                    vl = nn.functional.mse_loss(net(sv, stv), yvv).item() * self.Y_SCALE ** 2
                self.history["valid"].append(vl)
                if vl < best - 1e-4:
                    best, wait, best_ep = vl, 0, ep + 1
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_ep

    def fit(self, train: pd.DataFrame, valid_frac: float = 0.75):
        fit_tr, val_tr = split_train_valid(train, valid_frac)
        Xf, Sf, yf, _, _ = build_pairs(fit_tr)
        Xv, Sv, yv = validation_pairs(fit_tr, val_tr, return_seq=True)
        _, self.best_epoch_ = self._train(Xf, Sf, yf, Xv, Sv, yv)
        self.valid_history = dict(self.history)
        self.history = {"train": [], "valid": []}
        X, S, y, _, _ = build_pairs(train)
        self.net, _ = self._train(X, S, y, epochs=max(10, self.best_epoch_))
        return self

    def predict(self, train: pd.DataFrame, query: pd.DataFrame):
        X, S, _, anchor, qidx = build_pairs(train, query)
        seq, st, _ = self._tensors(X, S)
        self.net.eval()
        with torch.no_grad():
            d = self.net(seq, st).numpy() * self.Y_SCALE
        return pd.Series(anchor + d, index=list(qidx)).reindex(query.index).to_numpy()


# ----------------------------------------------------------------------------
# M1-H (exploratory) - hyperbolic trend per point + GP on the residuals
# ----------------------------------------------------------------------------


class HybridGPModel:
    """Exploratory hybrid announced in Deliverable 1: the hyperbolic method
    (persistence when it fails) supplies the trend of each point and a GP over
    (t, x, d) models the residual. Far from the data the GP residual returns to
    zero, so the forecast degrades gracefully towards the geotechnical trend
    instead of towards the training mean."""

    def __init__(self, seed: int = 0, n_restarts: int = 2):
        self.gp = GPModel(n_restarts=n_restarts, seed=seed)

    def fit(self, train: pd.DataFrame):
        from .baselines import hyperbolic
        self.trend_ = {}
        parts = []
        for pid, g in train.groupby("point_id"):
            g = g.sort_values("day").copy()
            t, y = g["day"].to_numpy(float), g["y"].to_numpy(float)
            self.trend_[pid] = (t, y)
            fit, _ = hyperbolic(t, y, t)
            g["y"] = y - fit
            parts.append(g)
        self.gp.fit(pd.concat(parts))
        return self

    def predict(self, query: pd.DataFrame):
        from .baselines import hyperbolic
        trend = np.full(len(query), np.nan)
        for pid, idx in query.groupby("point_id").groups.items():
            t, y = self.trend_[pid]
            trend[query.index.get_indexer(idx)] = hyperbolic(t, y, query.loc[idx, "day"].to_numpy(float))[0]
        m, s = self.gp.predict(query)
        return trend + m, s
