"""Forecasting models. Every model maps a feature row for day D to 24 hourly EE prices.

Two families:
  * Direct models (naive, LEAR, LightGBM) fit on the features as they are.
  * Feature-map + ridge models (linear, ESN, quantum reservoir, random features): a fixed,
    untrained map phi(x) followed by a ridge readout. Because the readout and inputs are
    identical inside each comparison, any difference comes from the map alone.
All price targets are modelled in a variance-stabilised space,
z = asinh((y - median) / MAD), which tames spikes (Uniejewski et al., 2018).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoLarsIC, Ridge
from sklearn.preprocessing import StandardScaler

from .features import HOURS, hour_block
from .qrc import QuantumReservoir

PRICE_PREFIXES = ("ee_d1_h", "ee_d2_h", "ee_d7_h", "fi_d1_h", "ee_d1_mean", "ee_d1_min", "ee_d1_max")


class AsinhTarget:
    def fit(self, y: np.ndarray):
        self.med = float(np.median(y))
        self.mad = float(np.median(np.abs(y - self.med))) or 1.0
        return self

    def fwd(self, y):
        return np.arcsinh((y - self.med) / self.mad)

    def inv(self, z):
        return self.med + self.mad * np.sinh(z)


# ---------- naive ----------
class Naive:
    def __init__(self, lag: int):
        self.lag = lag
        self.name = f"naive_d{lag}"

    def fit(self, X, Y):
        return self

    def predict(self, X):
        return hour_block(X, f"ee_d{self.lag}")


# ---------- LEAR (LASSO-estimated autoregressive, Lago et al. 2021) ----------
class LEAR:
    name = "lear"

    def __init__(self, windows=(364, 728)):
        self.windows = windows

    def _design(self, X, tt):
        cols = []
        for c in X.columns:
            v = X[c].to_numpy(dtype=float)
            if c.startswith(PRICE_PREFIXES):
                v = tt.fwd(v)
            cols.append(v)
        return np.column_stack(cols)

    def fit(self, X, Y):
        self.members = []
        for w in self.windows:
            Xw, Yw = X.iloc[-w:], Y.iloc[-w:].to_numpy()
            tt = AsinhTarget().fit(Yw.ravel())
            D = self._design(Xw, tt)
            sc = StandardScaler().fit(D)
            Ds = sc.transform(D)
            Z = tt.fwd(Yw)
            regs = [LassoLarsIC(criterion="aic", max_iter=500).fit(Ds, Z[:, h]) for h in HOURS]
            self.members.append((tt, sc, regs))
        return self

    def predict(self, X):
        preds = []
        for tt, sc, regs in self.members:
            Ds = sc.transform(self._design(X, tt))
            Z = np.column_stack([r.predict(Ds) for r in regs])
            preds.append(tt.inv(Z))
        return np.mean(preds, axis=0)


# ---------- LightGBM on a (day, hour) long table ----------
def long_table(X: pd.DataFrame) -> pd.DataFrame:
    day_cols = [c for c in X.columns if "_h" not in c or not c.split("_h")[-1].isdigit()]
    weather_vars = sorted({c.rsplit("_h", 1)[0] for c in X.columns if c.startswith("w_") and c.split("_h")[-1].isdigit()})
    base = X[day_cols].to_numpy()
    rows = []
    for h in HOURS:
        hb = np.column_stack(
            [
                X[f"ee_d1_h{h:02d}"],
                X[f"ee_d2_h{h:02d}"],
                X[f"ee_d7_h{h:02d}"],
                X[f"fi_d1_h{h:02d}"],
                X[f"ee_d1_h{(h - 1) % 24:02d}"],
                X[f"ee_d1_h{(h + 1) % 24:02d}"],
                np.full(len(X), h),
            ]
            + [X[f"{w}_h{h:02d}"] for w in weather_vars]
        )
        rows.append(np.column_stack([base, hb]))
    arr = np.stack(rows, axis=1).reshape(len(X) * 24, -1)  # day-major, then hour
    names = day_cols + ["lag1", "lag2", "lag7", "fi_lag1", "lag1_prev_h", "lag1_next_h", "hour"] + weather_vars
    return pd.DataFrame(arr, columns=names)


class GBM:
    def __init__(self, alpha: float | None = None, window: int = 728, n_estimators: int = 600):
        self.alpha = alpha
        self.window = window
        self.n_estimators = n_estimators
        self.name = "lgbm" if alpha is None else f"lgbm_q{int(alpha * 100):02d}"

    def fit(self, X, Y):
        import lightgbm as lgb

        Xw, Yw = X.iloc[-self.window:], Y.iloc[-self.window:].to_numpy()
        params = dict(
            n_estimators=self.n_estimators, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, random_state=0,
        )
        if self.alpha is None:
            params["objective"] = "l1"
        else:
            params.update(objective="quantile", alpha=self.alpha)
        self.m = lgb.LGBMRegressor(**params).fit(long_table(Xw), Yw.reshape(-1))
        return self

    def predict(self, X):
        return self.m.predict(long_table(X)).reshape(len(X), 24)


# ---------- feature maps + ridge readout ----------
class MapRidge:
    """phi is fitted once on a 'map fit' period (before the evaluation window) and then frozen;
    only the ridge readout is refitted as the rolling origin moves."""

    def __init__(self, name: str, featmap, alpha: float = 1.0, window: int = 728):
        self.name, self.featmap, self.alpha, self.window = name, featmap, alpha, window

    def fit(self, X, Y):
        Phi = self.featmap.transform(X)[-self.window:]
        Yw = Y.to_numpy()[-self.window:]
        self.tt = AsinhTarget().fit(Yw.ravel())
        self.reg = Ridge(alpha=self.alpha).fit(Phi, self.tt.fwd(Yw))
        return self

    def predict(self, X):
        return self.tt.inv(self.reg.predict(self.featmap.transform(X)))


class StandardMap:
    """Standardised raw features: the plain linear model."""

    def fit(self, X):
        self.sc = StandardScaler().fit(X.to_numpy())
        return self

    def transform(self, X):
        return self.sc.transform(X.to_numpy())


class PCAMap:
    """Standardise then project to k principal components (the input to the small reservoirs)."""

    def __init__(self, k: int = 8):
        self.k = k

    def fit(self, X):
        self.sc = StandardScaler().fit(X.to_numpy())
        self.pca = PCA(n_components=self.k, random_state=0).fit(self.sc.transform(X.to_numpy()))
        Z = self.pca.transform(self.sc.transform(X.to_numpy()))
        self.lo, self.hi = np.percentile(Z, 1, axis=0), np.percentile(Z, 99, axis=0)
        return self

    def transform(self, X):
        Z = self.pca.transform(self.sc.transform(X.to_numpy()))
        return np.clip((Z - self.lo) / (self.hi - self.lo), 0.0, 1.0)  # in [0, 1]


class ESNMap:
    """Leaky echo-state network run over the sequence of days. State at day D only sees
    inputs up to D, and those only hold information known on D-1."""

    def __init__(self, n: int = 400, rho: float = 0.9, leak: float = 0.3, in_scale: float = 0.5, seed: int = 0):
        self.n, self.rho, self.leak, self.in_scale, self.seed = n, rho, leak, in_scale, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        self.sc = StandardScaler().fit(X.to_numpy())
        d = X.shape[1]
        W = rng.normal(size=(self.n, self.n)) * (rng.random((self.n, self.n)) < 0.05)
        W *= self.rho / max(abs(np.linalg.eigvals(W)))
        self.W = W
        self.Win = rng.uniform(-1, 1, size=(self.n, d)) * self.in_scale / np.sqrt(d)
        return self

    def transform(self, X):
        U = np.clip(self.sc.transform(X.to_numpy()), -5, 5)
        x = np.zeros(self.n)
        S = np.empty((len(U), self.n))
        for i, u in enumerate(U):
            x = (1 - self.leak) * x + self.leak * np.tanh(self.W @ x + self.Win @ u)
            S[i] = x
        return np.hstack([S, U])


class QRCMap:
    """Simulated quantum reservoir on PCA-compressed inputs (see qrc.py)."""

    def __init__(self, k: int = 8, tau: float = 1.0, h: float = 1.0, steps: int = 3, seed: int = 0):
        self.pca = PCAMap(k)
        self.qr = QuantumReservoir(n_qubits=k, tau=tau, h=h, steps=steps, seed=seed)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.qr.features(self.pca.transform(X))


class RFFMap:
    """Null control for the quantum reservoir: random cosine features of the same size,
    on the same PCA inputs. If the quantum map can't beat this, its 'quantumness' adds nothing."""

    def __init__(self, k: int = 8, n_features: int = 108, scale: float = 3.0, seed: int = 0):
        self.pca, self.n_features, self.scale, self.seed = PCAMap(k), n_features, scale, seed

    def fit(self, X):
        self.pca.fit(X)
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(scale=self.scale, size=(self.pca.k, self.n_features))
        self.b = rng.uniform(0, 2 * np.pi, self.n_features)
        return self

    def transform(self, X):
        return np.cos(self.pca.transform(X) @ self.W + self.b)


class PCALinearMap:
    """Linear baseline on the same 8 PCA inputs the reservoirs see."""

    def __init__(self, k: int = 8):
        self.pca = PCAMap(k)

    def fit(self, X):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)
