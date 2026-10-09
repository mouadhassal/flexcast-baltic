"""mFRR bid assembly as a QUBO, and four ways to solve it (see PREREGISTRATION_quantum.md).

f(x) = sum_i c_i P_i x_i + lam * (sum_i P_i x_i - T)^2,   x in {0,1}^n
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp, minimize

ASSET_TYPES = {  # power range (MW), activation cost range (EUR/MWh)
    "heat_pump": ((0.02, 0.10), (5, 30)),
    "commercial_hvac": ((0.05, 0.30), (10, 50)),
    "industrial_load": ((0.20, 0.80), (40, 120)),
    "small_battery": ((0.10, 0.50), (8, 25)),
}


@dataclass
class Instance:
    P: np.ndarray        # MW
    c: np.ndarray        # EUR/MWh
    T: float
    lam: float
    kinds: list

    @property
    def n(self):
        return len(self.P)

    def linear(self):
        return self.c * self.P


def make_instance(n: int, seed: int, T: float = 1.0) -> Instance:
    rng = np.random.default_rng(seed)
    kinds = list(rng.choice(list(ASSET_TYPES), size=n, p=[0.4, 0.25, 0.15, 0.2]))
    P = np.array([rng.uniform(*ASSET_TYPES[k][0]) for k in kinds])
    c = np.array([rng.uniform(*ASSET_TYPES[k][1]) for k in kinds])
    a = c * P
    lam = 2.0 * a.max() / P.min() ** 2       # missing by the smallest asset always costs more than any activation
    return Instance(P=P, c=c, T=T, lam=lam, kinds=kinds)


def energy(inst: Instance, X: np.ndarray) -> np.ndarray:
    """f for a batch of bitstrings X (m, n)."""
    X = np.atleast_2d(X)
    return X @ inst.linear() + inst.lam * (X @ inst.P - inst.T) ** 2


def all_bitstrings(n: int) -> np.ndarray:
    return ((np.arange(2 ** n)[:, None] >> np.arange(n)[::-1]) & 1).astype(np.int8)  # qubit 0 = leftmost bit


def exhaustive(inst: Instance):
    t = time.perf_counter()
    X = all_bitstrings(inst.n)
    E = energy(inst, X)
    i = int(np.argmin(E))
    return X[i], float(E[i]), time.perf_counter() - t, E


def simulated_annealing(inst: Instance, sweeps: int = 1000, restarts: int = 20, seed: int = 0):
    t = time.perf_counter()
    rng = np.random.default_rng(seed)
    a, P, lam, T = inst.linear(), inst.P, inst.lam, inst.T
    best_x, best_e = None, np.inf
    scale = np.abs(energy(inst, rng.integers(0, 2, (200, inst.n)))).std() + 1e-9
    temps = np.geomspace(2.0 * scale, 1e-4 * scale, sweeps)
    for _ in range(restarts):
        x = rng.integers(0, 2, inst.n)
        s = float(x @ P)
        e = float(x @ a + lam * (s - T) ** 2)
        for temp in temps:
            for i in rng.permutation(inst.n):
                d = 1 - 2 * x[i]                       # +1 switch on, -1 switch off
                s_new = s + d * P[i]
                de = d * a[i] + lam * ((s_new - T) ** 2 - (s - T) ** 2)
                if de <= 0 or rng.random() < np.exp(-de / temp):
                    x[i] += d; s = s_new; e += de
            if e < best_e:
                best_e, best_x = e, x.copy()
    return best_x, float(best_e), time.perf_counter() - t


class QAOA:
    """Exact statevector QAOA. Cost Hamiltonian is diagonal: H_C|x> = f(x)|x> (rescaled);
    mixer is sum_i X_i. Angles found with COBYLA, warm-started layer by layer (INTERP)."""

    def __init__(self, inst: Instance, E_all: np.ndarray):
        self.n = inst.n
        self.E = E_all
        self.Es = (E_all - E_all.mean()) / (E_all.std() + 1e-12)    # rescale so angles are O(1)

    def state(self, gammas, betas):
        n = self.n
        psi = np.full(2 ** n, 2 ** (-n / 2), dtype=complex)
        for g, b in zip(gammas, betas):
            psi = psi * np.exp(-1j * g * self.Es)
            c, s = np.cos(b), -1j * np.sin(b)
            psi = psi.reshape((2,) * n)
            for q in range(n):                                       # exp(-i b X) on each qubit
                psi = np.moveaxis(psi, q, 0)
                p0, p1 = psi[0].copy(), psi[1].copy()
                psi[0], psi[1] = c * p0 + s * p1, s * p0 + c * p1
                psi = np.moveaxis(psi, 0, q)
            psi = psi.reshape(-1)
        return psi

    def expectation(self, theta):
        p = len(theta) // 2
        probs = np.abs(self.state(theta[:p], theta[p:])) ** 2
        return float(probs @ self.Es)

    def optimise(self, p_max: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        results = {}
        gam, bet = np.array([]), np.array([])
        for p in range(1, p_max + 1):
            if p == 1:
                starts = [np.array([g, b]) for g, b in rng.uniform([0.05, 0.05], [1.0, 0.8], size=(4, 2))]
            else:                                                    # INTERP initialisation
                xs = np.linspace(0, 1, p - 1)
                xn = np.linspace(0, 1, p)
                gi = np.interp(xn, xs, gam) if p > 2 else np.full(p, gam[0])
                bi = np.interp(xn, xs, bet) if p > 2 else np.full(p, bet[0])
                starts = [np.concatenate([gi, bi])]
            best = None
            for s0 in starts:
                r = minimize(self.expectation, s0, method="COBYLA", options={"maxiter": 300, "rhobeg": 0.2})
                if best is None or r.fun < best.fun:
                    best = r
            gam, bet = best.x[:p], best.x[p:]
            probs = np.abs(self.state(gam, bet)) ** 2
            results[p] = (gam.copy(), bet.copy(), probs / probs.sum())
        return results


def sample_best(probs: np.ndarray, E: np.ndarray, shots: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(probs), size=shots, p=probs)
    j = idx[np.argmin(E[idx])]
    return int(j), float(E[j])


def milp_constrained(inst: Instance, tol: float = 0.05):
    """min sum c_i P_i x_i  s.t.  T <= sum P_i x_i <= T + tol,  x binary."""
    t = time.perf_counter()
    res = milp(c=inst.linear(), constraints=LinearConstraint(inst.P[None, :], inst.T, inst.T + tol),
               integrality=np.ones(inst.n), bounds=Bounds(0, 1), options={"time_limit": 60})
    return res, time.perf_counter() - t
