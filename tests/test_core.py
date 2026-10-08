import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flexcast import battery, features  # noqa: E402
from flexcast.qrc import QuantumReservoir  # noqa: E402


def test_battery_flat_prices_does_nothing():
    c, d = battery.schedule(np.full(24, 50.0))
    assert np.allclose(c, 0, atol=1e-7) and np.allclose(d, 0, atol=1e-7)


def test_battery_buys_low_sells_high_within_limits():
    p = np.array([10.0] * 12 + [200.0] * 12)
    b = battery.Battery()
    c, d = battery.schedule(p, b)
    assert c[:12].sum() > 0.9 and d[12:].sum() > 0.9
    assert c.max() <= b.power_mw + 1e-9 and d.max() <= b.power_mw + 1e-9
    assert battery.settle(c, d, p, b) > 0


def test_battery_soc_returns_to_start():
    rng = np.random.default_rng(1)
    p = rng.normal(80, 40, 24)
    b = battery.Battery()
    c, d = battery.schedule(p, b)
    eta = np.sqrt(b.round_trip)
    assert abs(np.sum(eta * c - d / eta)) < 1e-6


def test_perfect_foresight_is_an_upper_bound():
    rng = np.random.default_rng(2)
    actual = rng.normal(80, 50, (20, 24))
    noisy = actual + rng.normal(0, 30, actual.shape)
    assert battery.backtest(actual, actual).sum() >= battery.backtest(noisy, actual).sum() - 1e-6


def test_batch_dp_is_close_to_exact_lp():
    from flexcast.decision import BatchDP

    rng = np.random.default_rng(4)
    p = rng.normal(80, 50, (30, 24))
    x = BatchDP().solve(p)
    b = battery.Battery()
    c, d = np.clip(x, 0, None), np.clip(-x, 0, None)
    dp_profit = np.sum(p * (d - c)) - b.cost_eur_per_mwh / 2 * np.sum(c + d)
    lp_profit = battery.backtest(p, p).sum()
    assert 0.95 * lp_profit <= dp_profit <= lp_profit + 1e-6
    assert np.all(np.abs(x) <= b.power_mw / np.sqrt(b.round_trip) + 1e-9)


def test_quantum_reservoir_is_unitary_and_bounded():
    qr = QuantumReservoir(n_qubits=4, steps=2)
    assert np.allclose(qr.U @ qr.U.conj().T, np.eye(16), atol=1e-10)
    f = qr.features(np.random.default_rng(0).random((5, 4)))
    assert f.shape == (5, 2 * (4 + 6)) and np.all(np.abs(f) <= 1 + 1e-9)


def test_quantum_reservoir_encoding_matches_single_qubit_expectation():
    qr = QuantumReservoir(n_qubits=3, tau=1e-12, steps=1)  # U ~ identity
    x = np.array([[0.0, 0.5, 1.0]])
    z = qr.features(x)[0, :3]
    assert np.allclose(z, np.cos(np.pi * x[0]), atol=1e-6)  # <Z> of RY(theta)|0> is cos(theta)


def test_features_do_not_see_the_target_day():
    idx = pd.date_range("2025-01-01", "2025-03-31 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(3)
    prices = pd.DataFrame({a: rng.normal(80, 20, len(idx)) for a in ["ee", "fi", "lv", "lt"]}, index=idx)
    system = pd.DataFrame(
        {c: rng.normal(500, 50, len(idx)) for c in
         ["production", "consumption", "production_renewable", "solar_energy_production", "ac_balance"]},
        index=idx,
    )
    X1, _, _ = features.build(prices, system)
    day = pd.Timestamp("2025-03-01")
    local = prices.index.tz_convert("Europe/Berlin")
    mask = local.normalize().tz_localize(None) == day
    prices2 = prices.copy()
    prices2.loc[mask] += 1000.0                    # change everything about day D itself
    system2 = system.copy()
    system2.loc[mask] += 1000.0
    X2, _, _ = features.build(prices2, system2)
    pd.testing.assert_series_equal(X1.loc[day], X2.loc[day])
