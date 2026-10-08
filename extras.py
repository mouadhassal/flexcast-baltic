"""Analyses run after the locked test, from the saved test forecasts only (no refitting).

  * H3 (pre-registered): MAE difference between the quantum reservoir and the random-feature
    control, with the same block-bootstrap CI as the main table.
  * EXPLORATORY (not pre-registered, labelled as such in the report):
      - the most any trust gate could have saved on this test window;
      - monthly MAE and capture for the ensemble vs naive_d1;
      - how often the forecast picks the right cheapest / dearest hours.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from flexcast import battery, evaluate  # noqa: E402

T = ROOT / "results" / "test"
y = pd.read_csv(T / "actual.csv", index_col=0, parse_dates=True)
Y = y.to_numpy()
fc = lambda k: pd.read_csv(T / f"forecast_{k}.csv", index_col=0, parse_dates=True).to_numpy()
prof = lambda k: pd.read_csv(T / f"profit_{k}.csv", index_col=0, parse_dates=True).iloc[:, 0]

out = {}

# H3
d = evaluate.daily_abs_err(fc("pca8_qrc"), Y) - evaluate.daily_abs_err(fc("pca8_rff"), Y)   # >0: QRC worse
m, lo, hi = evaluate.bootstrap_mean_ci(d)
d2 = evaluate.daily_abs_err(fc("pca8_qrc"), Y) - evaluate.daily_abs_err(fc("pca8_linear"), Y)
m2, lo2, hi2 = evaluate.bootstrap_mean_ci(d2)
out["H3_qrc_minus_rff_mae"] = {"mean": m, "ci_lo": lo, "ci_hi": hi}
out["H3_qrc_minus_pca8linear_mae"] = {"mean": m2, "ci_lo": lo2, "ci_hi": hi2}

# Exploratory: ceiling on what a gate could save = all losses on losing days
summary = {}
for k in ("ensemble", "esn", "lear", "lgbm", "naive_d1"):
    p = prof(k)
    summary[k] = {"total_eur": float(p.sum()), "losing_days": int((p < 0).sum()),
                  "loss_on_losing_days_eur": float(p[p < 0].sum()),
                  "max_gain_from_perfect_gate_pct": float(-p[p < 0].sum() / p.sum() * 100)}
out["exploratory_gate_ceiling"] = summary

# Exploratory: monthly view
pf = pd.Series(battery.backtest(Y, Y), index=y.index)
rows = []
for mth, idx in y.groupby(y.index.to_period("M")).groups.items():
    sel = y.index.isin(idx)
    rows.append({
        "month": str(mth), "days": int(sel.sum()),
        "mae_naive_d1": evaluate.mae(fc("naive_d1")[sel], Y[sel]),
        "mae_ensemble": evaluate.mae(fc("ensemble")[sel], Y[sel]),
        "capture_naive_d1_pct": 100 * prof("naive_d1")[sel].sum() / pf[sel].sum(),
        "capture_ensemble_pct": 100 * prof("ensemble")[sel].sum() / pf[sel].sum(),
        "perfect_foresight_eur": float(pf[sel].sum()),
    })
monthly = pd.DataFrame(rows)
monthly.to_csv(ROOT / "results" / "exploratory_monthly.csv", index=False)

# Exploratory: does the forecast find the cheapest 2 and dearest 2 hours?
def hit_rate(F, k=2):
    lo_hit = np.mean([len(set(np.argsort(f)[:k]) & set(np.argsort(a)[:k])) / k for f, a in zip(F, Y)])
    hi_hit = np.mean([len(set(np.argsort(f)[-k:]) & set(np.argsort(a)[-k:])) / k for f, a in zip(F, Y)])
    return float(lo_hit), float(hi_hit)

out["exploratory_hour_hit_rate_k2"] = {k: dict(zip(("cheapest", "dearest"), hit_rate(fc(k))))
                                       for k in ("ensemble", "lear", "lgbm", "naive_d1")}
# Exploratory, added after reading Prokhorov-style "price shape" benchmarks in the literature:
# does a dumb hourly profile do as well in euros? Trailing 28-day mean by hour is known at D-1.
# The month-hour average uses the whole month (including future days), so it is only a yardstick.
from scipy.stats import kendalltau  # noqa: E402
sys.path.insert(0, str(ROOT))
from run import load  # noqa: E402

X, Yall, _ = load()
trail = np.array([Yall.loc[d - pd.Timedelta(days=28): d - pd.Timedelta(days=1)].to_numpy().mean(0) for d in y.index])
oracle_mh = y.groupby(y.index.to_period("M")).transform("mean").to_numpy()
pf_total = battery.backtest(Y, Y).sum()
bench = {}
for name, F in [("naive_d1", fc("naive_d1")), ("trailing_28d_profile", trail),
                ("oracle_month_hour_avg_NOT_LIVE", oracle_mh), ("ensemble", fc("ensemble")), ("esn", fc("esn"))]:
    pr = battery.backtest(F, Y)
    bench[name] = {"mae": evaluate.mae(F, Y), "capture_pct": float(100 * pr.sum() / pf_total),
                   "profit_eur": float(pr.sum()), "eur_per_mw_year": float(pr.sum() / len(Y) * 365),
                   "mean_daily_kendall_tau": float(np.mean([kendalltau(f, a)[0] for f, a in zip(F, Y)])),
                   "losing_days": int((pr < 0).sum())}
dd = battery.backtest(fc("ensemble"), Y) - battery.backtest(trail, Y)
bench["ensemble_minus_trailing_profile_eur_per_day"] = dict(zip(("mean", "ci_lo", "ci_hi"), evaluate.bootstrap_mean_ci(dd)))
bench["perfect_foresight_eur_per_mw_year"] = float(pf_total / len(Y) * 365)
out["exploratory_profile_benchmark"] = bench
(ROOT / "results" / "extras.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
print(monthly.round(1).to_string(index=False))
