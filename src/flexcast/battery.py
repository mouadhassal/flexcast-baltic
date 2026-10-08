"""Turn a price forecast into a battery schedule, then settle it at real prices.

Asset (default): 1 MW / 2 MWh battery, 90% round-trip efficiency, starts and ends each
day at 50% state of charge, and a 5 EUR/MWh throughput cost standing in for degradation.
The schedule is the exact optimum (linear program) for the *forecast* prices; profit is
then counted at the *actual* prices. Perfect foresight (schedule on actual prices) is the
ceiling, so 'capture' = realised profit / perfect-foresight profit.

This is a day-ahead arbitrage proxy. It ignores intraday re-trading, balancing markets,
grid fees and bid-curve mechanics; those are listed as next steps in the report.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass(frozen=True)
class Battery:
    power_mw: float = 1.0
    energy_mwh: float = 2.0
    round_trip: float = 0.90
    soc0_frac: float = 0.5
    cost_eur_per_mwh: float = 5.0


def schedule(prices: np.ndarray, b: Battery = Battery()) -> tuple[np.ndarray, np.ndarray]:
    """Optimal hourly charge c_t and discharge d_t (MW, 1-hour steps) for a price vector."""
    T = len(prices)
    eta = np.sqrt(b.round_trip)
    # variables: c (T), d (T), s (T)  -> state of charge at end of hour t
    n = 3 * T
    cost = np.concatenate([prices + b.cost_eur_per_mwh / 2, -prices + b.cost_eur_per_mwh / 2, np.zeros(T)])
    A_eq = np.zeros((T + 1, n))
    b_eq = np.zeros(T + 1)
    s0 = b.soc0_frac * b.energy_mwh
    for t in range(T):
        A_eq[t, 2 * T + t] = 1.0                 # s_t
        if t > 0:
            A_eq[t, 2 * T + t - 1] = -1.0        # - s_{t-1}
        A_eq[t, t] = -eta                        # - eta * c_t
        A_eq[t, T + t] = 1.0 / eta               # + d_t / eta
        b_eq[t] = s0 if t == 0 else 0.0
    A_eq[T, 2 * T + T - 1] = 1.0                 # end where we started
    b_eq[T] = s0
    bounds = [(0, b.power_mw)] * (2 * T) + [(0, b.energy_mwh)] * T
    res = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    return res.x[:T], res.x[T:2 * T]


def settle(c: np.ndarray, d: np.ndarray, prices: np.ndarray, b: Battery = Battery()) -> float:
    """Profit in EUR of executing schedule (c, d) at the given prices."""
    return float(np.sum(prices * (d - c)) - b.cost_eur_per_mwh / 2 * np.sum(c + d))


def backtest(forecast: np.ndarray, actual: np.ndarray, b: Battery = Battery(), trade_mask=None) -> np.ndarray:
    """Daily profit (EUR) of scheduling on `forecast` and settling at `actual`.
    `trade_mask[i] = False` means stay idle that day (profit 0)."""
    out = np.zeros(len(actual))
    for i in range(len(actual)):
        if trade_mask is not None and not trade_mask[i]:
            continue
        c, d = schedule(forecast[i], b)
        out[i] = settle(c, d, actual[i], b)
    return out
