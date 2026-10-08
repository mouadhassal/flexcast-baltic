"""Download and cache public Elering data (Nord Pool day-ahead prices and Estonian system data).

Everything here comes from https://dashboard.elering.ee/api, which needs no key.
Raw responses are cached as CSV under data/raw so a run is reproducible even if
Elering later revises a value.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

API = "https://dashboard.elering.ee/api"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MARKET_TZ = "Europe/Berlin"  # SDAC delivery days and gate closure (12:00) are in CET/CEST


def _get(path: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    params = {
        "start": start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%S.999Z"),
    }
    for attempt in range(4):
        try:
            r = requests.get(f"{API}/{path}", params=params, timeout=120)
            r.raise_for_status()
            body = r.json()
            if not body.get("success", False):
                raise RuntimeError(f"Elering returned success=false for {path} {params}")
            return body["data"]
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def _year_chunks(start: str, end: str):
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC")
    cur = s
    while cur < e:
        nxt = min(cur + pd.DateOffset(months=6), e)
        yield cur, nxt - pd.Timedelta(milliseconds=1)
        cur = nxt


def fetch_prices(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Day-ahead prices (EUR/MWh) for ee, fi, lv, lt at native resolution, UTC index."""
    out = RAW / f"prices_{start}_{end}.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, index_col=0, parse_dates=True)
    frames = []
    for a, b in _year_chunks(start, end):
        data = _get("nps/price", a, b)
        cols = {}
        for area, rows in data.items():
            s = pd.Series({r["timestamp"]: r["price"] for r in rows}, name=area, dtype=float)
            cols[area] = s
        df = pd.DataFrame(cols)
        df.index = pd.to_datetime(df.index, unit="s", utc=True)
        frames.append(df)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    return df


def fetch_system(start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Estonian power-system actuals (MW). Elering reports these every 5 min with
    values filled in at its own reporting cadence; we keep non-null rows only."""
    out = RAW / f"system_{start}_{end}.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, index_col=0, parse_dates=True)
    frames = []
    for a, b in _year_chunks(start, end):
        rows = _get("system", a, b)
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df.index = pd.to_datetime(df.pop("timestamp"), unit="s", utc=True)
        frames.append(df)
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    keep = ["production", "consumption", "production_renewable", "solar_energy_production", "ac_balance"]
    df = df[keep].dropna(how="all")
    RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    return df


def to_daily_hourly(series: pd.Series) -> pd.DataFrame:
    """Turn a UTC series into one row per CET/CEST market day and 24 columns (hour 0..23).

    Quarter-hour prices (SDAC went to 15-minute products on 1 Oct 2025) are averaged to the
    hour. On the 25-hour DST day the repeated hour is averaged; on the 23-hour day the
    missing hour is filled from its neighbours.
    """
    local = series.tz_convert(MARKET_TZ)
    frame = pd.DataFrame({"v": local.values}, index=local.index)
    frame["day"] = frame.index.date
    frame["hour"] = frame.index.hour
    wide = frame.groupby(["day", "hour"])["v"].mean().unstack("hour")
    wide = wide.reindex(columns=range(24))
    # Keep only days we actually observed (>= 23 hours); this drops a partial last day.
    wide = wide[wide.notna().sum(axis=1) >= 23]
    wide = wide.interpolate(axis=1, limit_direction="both")
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "day"
    return wide
