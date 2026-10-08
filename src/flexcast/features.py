"""Build one feature row per delivery day D, using only what is known at the
SDAC gate closure (12:00 CET on D-1).

Information set, and why each item is allowed:
  * Day-ahead prices for D-1 (all zones): published around 12:45 CET on D-2. Known.
  * Prices for D-2 and D-7: older, known.
  * Estonian system actuals (consumption, production, renewables, solar, AC balance):
    we only use full day D-2, so there is no doubt about publication delay.
  * Calendar features of D (weekday, month, Estonian public holidays): known in advance.
Nothing about day D itself (no weather forecasts, no TSO plans) is used. That is a
deliberate handicap; real weather forecasts are the obvious next feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter

from .data import to_daily_hourly

HOURS = list(range(24))


def estonian_holidays(years) -> set:
    days = set()
    for y in years:
        e = pd.Timestamp(easter(y))
        for m, d in [(1, 1), (2, 24), (5, 1), (6, 23), (6, 24), (8, 20), (12, 24), (12, 25), (12, 26)]:
            days.add(pd.Timestamp(y, m, d))
        days.update({e - pd.Timedelta(days=2), e, e + pd.Timedelta(days=49)})  # Good Friday, Easter, Pentecost
    return days


def price_tables(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {area: to_daily_hourly(prices[area].dropna()) for area in ["ee", "fi", "lv", "lt"]}


def system_daily(system: pd.DataFrame) -> pd.DataFrame:
    local = system.tz_convert("Europe/Berlin")
    daily = local.groupby(local.index.date).mean()
    daily.index = pd.to_datetime(daily.index)
    return daily


def build(prices: pd.DataFrame, system: pd.DataFrame, weather: pd.DataFrame | None = None
          ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (X, Y, tables). X has one row per day D with features; Y has the 24 hourly
    EE prices of D. Rows lacking any input are dropped. `weather` (from weather.build) is
    indexed by day D and holds forecasts issued before gate closure, so it joins as-is."""
    t = price_tables(prices)
    ee = t["ee"]
    days = ee.index
    sysd = system_daily(system).reindex(days)

    def lag(df: pd.DataFrame, k: int, prefix: str) -> pd.DataFrame:
        out = df.reindex(days - pd.Timedelta(days=k))
        out.index = days
        out.columns = [f"{prefix}_h{h:02d}" for h in HOURS]
        return out

    parts = [
        lag(ee, 1, "ee_d1"),
        lag(ee, 2, "ee_d2"),
        lag(ee, 7, "ee_d7"),
        lag(t["fi"], 1, "fi_d1"),
    ]
    d1 = ee.reindex(days - pd.Timedelta(days=1))
    d1.index = days
    lt1 = t["lt"].reindex(days - pd.Timedelta(days=1))
    lt1.index = days
    fi1 = t["fi"].reindex(days - pd.Timedelta(days=1))
    fi1.index = days
    stats = pd.DataFrame(
        {
            "ee_d1_mean": d1.mean(axis=1),
            "ee_d1_std": d1.std(axis=1),
            "ee_d1_min": d1.min(axis=1),
            "ee_d1_max": d1.max(axis=1),
            "ee_d1_spread": d1.max(axis=1) - d1.min(axis=1),
            "lt_minus_ee_d1": (lt1 - d1).mean(axis=1),
            "fi_minus_ee_d1": (fi1 - d1).mean(axis=1),
        },
        index=days,
    )
    s2 = sysd.reindex(days - pd.Timedelta(days=2))
    s2.index = days
    s2.columns = [f"sys_d2_{c}" for c in s2.columns]

    hol = estonian_holidays(range(days.min().year, days.max().year + 1))
    dow = days.dayofweek
    cal = pd.DataFrame(index=days)
    for k in range(7):
        cal[f"dow_{k}"] = (dow == k).astype(float)
    cal["holiday"] = days.isin(list(hol)).astype(float)
    doy = days.dayofyear.values
    cal["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    cal["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # Baltic grid synchronised with Continental Europe on 9 Feb 2025; regime flag.
    cal["post_sync"] = (days >= pd.Timestamp("2025-02-09")).astype(float)

    X = pd.concat(parts + [stats, s2, cal], axis=1)
    if weather is not None:
        X = X.join(weather, how="left")
    Y = ee.copy()
    Y.columns = [f"y_h{h:02d}" for h in HOURS]
    ok = X.notna().all(axis=1) & Y.notna().all(axis=1)
    return X[ok], Y[ok], t


def hour_block(X: pd.DataFrame, prefix: str) -> np.ndarray:
    return X[[f"{prefix}_h{h:02d}" for h in HOURS]].to_numpy()
