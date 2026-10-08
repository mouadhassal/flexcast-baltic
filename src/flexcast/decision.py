"""Train the forecast for the battery's profit, not for price error (v3).

Method: SPO+ ("smart predict, then optimize", Elmachtoub & Grigas, 2022). For a decision
problem whose cost is linear in the price vector p, the SPO+ loss is a convex surrogate for
the regret "profit of the plan made on the forecast vs. the plan made on the truth".
Its subgradient w.r.t. the forecast p_hat is

    2 * ( x*(p) - x*(2 p_hat - p) )

where x*(q) is the optimal net-charge schedule for price vector q. So each training step
needs the battery schedule for every training day. The exact LP (battery.py) is too slow
for that inner loop, so here a vectorised dynamic program on a 0.1 MWh state-of-charge
grid solves all days at once. Evaluation still uses the exact LP.

Model: a linear map from standardised features to the 24 hourly prices, warm-started at
the ridge (squared-error) solution, then fine-tuned with Adam on SPO+ plus a small
squared-error term `mu` that keeps the price level sensible.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .battery import Battery


class BatchDP:
    """Exact optimum on a discretised SoC grid, for a batch of days at once."""

    def __init__(self, b: Battery = Battery(), step: float = 0.1):
        self.b = b
        eta = np.sqrt(b.round_trip)
        self.levels = np.round(np.arange(0, b.energy_mwh + 1e-9, step), 6)
        n = len(self.levels)
        self.start = int(np.argmin(np.abs(self.levels - b.soc0_frac * b.energy_mwh)))
        ds = self.levels[None, :] - self.levels[:, None]                 # (from, to) SoC change
        charge = np.where(ds > 0, ds / eta, 0.0)                         # MW drawn from grid
        discharge = np.where(ds < 0, -ds * eta, 0.0)                     # MW delivered to grid
        feasible = (charge <= b.power_mw + 1e-9) & (discharge <= b.power_mw + 1e-9)
        self.net = charge - discharge                                    # grid-side net charge x
        self.wear = b.cost_eur_per_mwh / 2 * (charge + discharge)
        self.pen = np.where(feasible, 0.0, np.inf)
        self.n = n

    def solve(self, prices: np.ndarray) -> np.ndarray:
        """prices (B, 24) -> optimal net-charge schedule x (B, 24) minimising p.x + wear."""
        B, T = prices.shape
        cost = np.full((B, self.n), np.inf)
        cost[:, self.start] = 0.0
        arg = np.empty((T, B, self.n), dtype=np.int16)
        for t in range(T):
            step = prices[:, t, None, None] * self.net[None] + self.wear[None] + self.pen[None]  # (B, from, to)
            tot = cost[:, :, None] + step
            arg[t] = np.argmin(tot, axis=1)
            cost = np.take_along_axis(tot, arg[t][:, None, :], axis=1)[:, 0, :]
        x = np.empty((B, T))
        s = np.full(B, self.start)
        for t in range(T - 1, -1, -1):
            prev = arg[t][np.arange(B), s]
            x[:, t] = self.net[prev, s]
            s = prev
        return x


class SPOLinear:
    name = "spo_linear"

    def __init__(self, alpha: float = 100.0, lr: float = 0.5, epochs: int = 30, mu: float = 0.05,
                 window: int = 728, batch: int = 64, seed: int = 0):
        self.alpha, self.lr, self.epochs, self.mu = alpha, lr, epochs, mu
        self.window, self.batch, self.seed = window, batch, seed
        self.dp = BatchDP()

    def fit(self, X, Y):
        Xw, Yw = X.to_numpy()[-self.window:], Y.to_numpy()[-self.window:]
        self.sc = StandardScaler().fit(Xw)
        F = self.sc.transform(Xw)
        warm = Ridge(alpha=self.alpha).fit(F, Yw)                      # squared-error warm start
        W, b = warm.coef_.T.copy(), warm.intercept_.copy()             # (d, 24), (24,)
        x_true = self.dp.solve(Yw)                                     # fixed for the whole fit
        rng = np.random.default_rng(self.seed)
        mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
        b1, b2, eps, k = 0.9, 0.999, 1e-8, 0
        scale = np.std(Yw)                                             # keeps the MSE term comparable
        for _ in range(self.epochs):
            order = rng.permutation(len(F))
            for i in range(0, len(F), self.batch):
                idx = order[i:i + self.batch]
                f, y = F[idx], Yw[idx]
                p_hat = f @ W + b
                g = 2 * (x_true[idx] - self.dp.solve(2 * p_hat - y))   # SPO+ subgradient wrt p_hat
                g = g + self.mu * 2 * (p_hat - y) / scale
                gW = f.T @ g / len(idx) + self.alpha / len(F) * (W - warm.coef_.T)
                gb = g.mean(axis=0)
                k += 1
                for P, G, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                    m *= b1; m += (1 - b1) * G
                    v *= b2; v += (1 - b2) * G * G
                    P -= self.lr * (m / (1 - b1 ** k)) / (np.sqrt(v / (1 - b2 ** k)) + eps)
        self.W, self.b = W, b
        return self

    def predict(self, X):
        return self.sc.transform(X.to_numpy()) @ self.W + self.b


class RidgeRaw:
    """The warm start alone: ridge on the same features, squared error, raw prices.
    Comparing SPOLinear with this isolates the effect of training for profit."""
    name = "ridge_raw"

    def __init__(self, alpha: float = 100.0, window: int = 728):
        self.alpha, self.window = alpha, window

    def fit(self, X, Y):
        Xw, Yw = X.to_numpy()[-self.window:], Y.to_numpy()[-self.window:]
        self.sc = StandardScaler().fit(Xw)
        self.m = Ridge(alpha=self.alpha).fit(self.sc.transform(Xw), Yw)
        return self

    def predict(self, X):
        return self.m.predict(self.sc.transform(X.to_numpy()))
