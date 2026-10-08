"""A small simulated quantum reservoir (exact statevector, numpy only).

Same family as the QRCx model (transverse-field Ising dynamics), cut down to 8 qubits so
it runs in seconds on a laptop:

    H = sum_{i<j} J_ij Z_i Z_j + h * sum_i X_i,    J_ij ~ U(-1, 1), fixed by seed.

Each day's input x in [0,1]^k is loaded as a product state prod_i RY(pi * x_i)|0>, evolved
under U = exp(-i H tau) for `steps` steps, and read out after every step as <Z_i> and
<Z_i Z_j>. With k = 8 and 3 steps that is 3 * (8 + 28) = 108 features per day.

Memory across days is not carried in the quantum state here; the day features already
contain the D-1, D-2 and D-7 lags. This keeps the comparison with the classical maps clean.
`shots` adds the sampling noise you would get on hardware (Gaussian approximation).
"""
from __future__ import annotations

import itertools

import numpy as np


class QuantumReservoir:
    def __init__(self, n_qubits: int = 8, tau: float = 1.0, h: float = 1.0, steps: int = 3,
                 seed: int = 0, shots: int | None = None):
        self.k, self.tau, self.h, self.steps, self.shots = n_qubits, tau, h, steps, shots
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(seed + 1)
        dim = 2 ** n_qubits
        bits = (np.arange(dim)[:, None] >> np.arange(n_qubits)[::-1]) & 1   # qubit 0 is the leftmost
        self.zsign = 1.0 - 2.0 * bits                                         # (dim, k), Z eigenvalues
        self.pairs = list(itertools.combinations(range(n_qubits), 2))

        diag = np.zeros(dim)
        for i, j in self.pairs:
            diag += rng.uniform(-1, 1) * self.zsign[:, i] * self.zsign[:, j]
        H = np.diag(diag).astype(complex)
        for q in range(n_qubits):  # transverse field: X on qubit q flips bit q
            flip = np.arange(dim) ^ (1 << (n_qubits - 1 - q))
            H[np.arange(dim), flip] += h
        w, V = np.linalg.eigh(H)
        self.U = (V * np.exp(-1j * w * tau)) @ V.conj().T

    def _encode(self, X: np.ndarray) -> np.ndarray:
        theta = np.pi * X                                         # (n, k)
        amp0, amp1 = np.cos(theta / 2), np.sin(theta / 2)
        psi = np.ones((len(X), 1))
        for q in range(self.k):                                   # Kronecker product, qubit 0 first
            psi = np.einsum("na,nb->nab", psi, np.stack([amp0[:, q], amp1[:, q]], axis=1)).reshape(len(X), -1)
        return psi.astype(complex)

    def features(self, X: np.ndarray) -> np.ndarray:
        psi = self._encode(np.asarray(X, dtype=float))
        feats = []
        zz = np.stack([self.zsign[:, i] * self.zsign[:, j] for i, j in self.pairs], axis=1)
        for _ in range(self.steps):
            psi = psi @ self.U.T
            p = np.abs(psi) ** 2
            obs = np.hstack([p @ self.zsign, p @ zz])
            if self.shots:
                obs = obs + self.rng.normal(size=obs.shape) * np.sqrt(np.clip(1 - obs**2, 0, 1) / self.shots)
            feats.append(obs)
        return np.hstack(feats)
