"""Archived numerical weather *forecasts* from Open-Meteo's Previous Runs API (no key).

We use the `_previous_day2` fields: the value predicted 48 hours before each valid hour.
The latest valid hour of market day D is D 23:00 CET, so its forecast was issued no later
than D-2 23:00 CET, well before the 12:00 CET gate closure on D-1. The earliest hour is
issued even earlier. So no information after gate closure is used. (The 24-hour fields
would leak for late hours of D, so they are deliberately not used.)

Coverage: the archive is complete from about March 2024 (most models are archived from
January 2024), so weather-based models train on a shorter history.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests

from .data import RAW, to_daily_hourly

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
VARS = ["wind_speed_100m", "temperature_2m", "shortwave_radiation"]

# Points chosen for what moves Baltic prices: Finnish and Baltic wind fleets, demand
# centres (temperature), and the solar-heavy south.
SITES = {
    "fi_ostrobothnia": (63.1, 21.6),   # west-coast wind cluster
    "fi_oulu_coast": (64.8, 24.9),     # northern wind cluster
    "ee_west": (58.8, 23.0),           # West Estonia / islands wind
    "lt_coast": (55.8, 21.3),          # Lithuanian coastal wind
    "tallinn": (59.44, 24.75),
    "helsinki": (60.17, 24.94),
    "vilnius": (54.69, 25.28),
}


def fetch_site(name: str, lat: float, lon: float, start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    out = RAW / f"weather_{name}_{start}_{end}.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, index_col=0, parse_dates=True)
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end, "timezone": "GMT",
              "hourly": ",".join(f"{v}_previous_day2" for v in VARS)}
    for attempt in range(4):
        try:
            r = requests.get(API, params=params, timeout=180)
            r.raise_for_status()
            h = r.json()["hourly"]
            break
        except (requests.RequestException, KeyError, ValueError):
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    df = pd.DataFrame(h)
    df.index = pd.to_datetime(df.pop("time")).dt.tz_localize("UTC")
    df.columns = [c.replace("_previous_day2", "") for c in df.columns]
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    return df


def wind_power_proxy(speed_kmh: pd.Series) -> pd.Series:
    """Rough turbine curve: 0 below 3 m/s, cubic ramp to rated at 12 m/s, flat above,
    cut-out at 25 m/s. Makes wind roughly linear in output for the linear models."""
    v = speed_kmh / 3.6
    p = ((v - 3) / 9).clip(0, 1) ** 3
    return p.where(v < 25, 0.0)


def build(start: str, end: str) -> pd.DataFrame:
    """One row per CET market day D, 24-hour blocks plus daily means for:
    wind proxy (FI mean of two clusters, EE, LT), temperature (mean of 3 cities),
    solar radiation (mean of 3 cities)."""
    raw = {n: fetch_site(n, *ll, start, end) for n, ll in SITES.items()}
    series = {
        "w_wind_fi": (wind_power_proxy(raw["fi_ostrobothnia"]["wind_speed_100m"])
                      + wind_power_proxy(raw["fi_oulu_coast"]["wind_speed_100m"])) / 2,
        "w_wind_ee": wind_power_proxy(raw["ee_west"]["wind_speed_100m"]),
        "w_wind_lt": wind_power_proxy(raw["lt_coast"]["wind_speed_100m"]),
        "w_temp": sum(raw[c]["temperature_2m"] for c in ("tallinn", "helsinki", "vilnius")) / 3,
        "w_solar": sum(raw[c]["shortwave_radiation"] for c in ("tallinn", "helsinki", "vilnius")) / 3,
    }
    blocks = []
    for name, s in series.items():
        wide = to_daily_hourly(s.dropna())
        wide.columns = [f"{name}_h{h:02d}" for h in range(24)]
        blocks.append(wide)
        blocks.append(wide.mean(axis=1).rename(f"{name}_mean").to_frame())
    W = pd.concat(blocks, axis=1)
    return W.replace([np.inf, -np.inf], np.nan)
