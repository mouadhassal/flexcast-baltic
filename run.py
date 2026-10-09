"""Entry point.

    python run.py validate   # tune on the validation window, write results/chosen_config.json
    python run.py test       # needs PREREGISTRATION.md; runs the frozen configs once on the test window
    python run.py figures    # charts for the report

Windows (all CET/CEST delivery days):
    validation  2025-10-01 .. 2026-03-31
    test        2026-04-01 .. 2026-10-08   (locked; touched once)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from flexcast import battery, data, evaluate, features  # noqa: E402
from flexcast.backtest import rolling  # noqa: E402
from flexcast.models import (GBM, LEAR, ESNMap, MapRidge, Naive, PCALinearMap, QRCMap,  # noqa: E402
                             RFFMap, StandardMap)

RESULTS = ROOT / "results"
VAL = ("2025-10-01", "2026-03-31")
TEST = ("2026-04-01", "2026-10-08")
DATA_RANGE = ("2023-01-01", "2026-10-09")
ENSEMBLE = ("lear", "lgbm", "esn")  # equal-weight average, fixed in advance (no weights fitted)


def load():
    p = data.fetch_prices(*DATA_RANGE)
    s = data.fetch_system(*DATA_RANGE)
    return features.build(p, s)


# ---- the search space, deliberately small ----
def grid():
    g = {"naive_d1": [{}], "naive_d7": [{}], "lear": [{}]}
    g["lgbm"] = [{"n_estimators": n} for n in (300, 600)]
    g["linear"] = [{"alpha": a} for a in (1, 10, 100, 1000, 3000, 10000)]
    g["esn"] = [{"alpha": a, "rho": r, "leak": lk} for a in (1, 10, 100, 300, 1000) for r in (0.5, 0.9) for lk in (0.3, 1.0)]
    alphas = (0.1, 1, 10, 30, 100)
    g["pca8_linear"] = [{"alpha": a} for a in alphas]
    g["pca8_rff"] = [{"alpha": a, "scale": s} for a in alphas for s in (1.0, 3.0)]
    g["pca8_qrc"] = [{"alpha": a, "tau": t, "h": h} for a in alphas for t in (0.5, 1.0, 2.0) for h in (0.5, 1.0)]
    return g


def factory(name: str, cfg: dict):
    if name == "naive_d1":
        return lambda: Naive(1)
    if name == "naive_d7":
        return lambda: Naive(7)
    if name == "lear":
        return lambda: LEAR()
    if name == "lgbm":
        return lambda: GBM(n_estimators=cfg["n_estimators"])
    if name == "linear":
        return lambda: MapRidge(name, StandardMap(), alpha=cfg["alpha"])
    if name == "esn":
        return lambda: MapRidge(name, ESNMap(rho=cfg["rho"], leak=cfg["leak"]), alpha=cfg["alpha"])
    if name == "pca8_linear":
        return lambda: MapRidge(name, PCALinearMap(8), alpha=cfg["alpha"])
    if name == "pca8_rff":
        return lambda: MapRidge(name, RFFMap(8, n_features=108, scale=cfg["scale"]), alpha=cfg["alpha"])
    if name == "pca8_qrc":
        return lambda: MapRidge(name, QRCMap(8, tau=cfg["tau"], h=cfg["h"], steps=3), alpha=cfg["alpha"])
    raise KeyError(name)


def validate():
    X, Y, _ = load()
    y = Y.loc[VAL[0]:VAL[1]].to_numpy()
    rows, chosen, best_fc = [], {}, {}
    for name, cfgs in grid().items():
        best = None
        for cfg in cfgs:
            t0 = time.time()
            f = rolling(factory(name, cfg), X, Y, *VAL).to_numpy()
            m = evaluate.mae(f, y)
            rows.append({"model": name, "config": json.dumps(cfg), "val_mae": m, "val_rmse": evaluate.rmse(f, y),
                         "seconds": round(time.time() - t0, 1)})
            print(f"{name:12s} {json.dumps(cfg):48s} MAE {m:7.2f}  ({time.time() - t0:.0f}s)", flush=True)
            if best is None or m < best[0]:
                best = (m, cfg, f)
        chosen[name] = best[1]
        best_fc[name] = best[2]
    ens = np.mean([best_fc[k] for k in ENSEMBLE], axis=0)
    rows.append({"model": "ensemble", "config": json.dumps(ENSEMBLE), "val_mae": evaluate.mae(ens, y),
                 "val_rmse": evaluate.rmse(ens, y), "seconds": 0})
    print(f"ensemble     {json.dumps(ENSEMBLE):48s} MAE {evaluate.mae(ens, y):7.2f}")
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "validation_grid.csv", index=False)
    (RESULTS / "chosen_config.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps(chosen, indent=2))


def guard_mask(point, q10, q90, b=battery.Battery()):
    """Trust gate (pre-registered): schedule on the point forecast, then re-price that schedule
    pessimistically - charging hours at q90, discharging hours at q10. Trade only if the
    schedule still makes money in that pessimistic world; otherwise stay idle and flag the
    day for a human."""
    mask = np.zeros(len(point), dtype=bool)
    for i in range(len(point)):
        c, d = battery.schedule(point[i], b)
        pess = np.where(d > c, q10[i], q90[i])
        mask[i] = battery.settle(c, d, pess, b) > 0
    return mask


def test():
    prereg = ROOT / "PREREGISTRATION.md"
    if not prereg.exists():
        raise SystemExit("Write and freeze PREREGISTRATION.md before running the test window.")
    sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    chosen = json.loads((RESULTS / "chosen_config.json").read_text())
    X, Y, _ = load()
    yt = Y.loc[TEST[0]:TEST[1]]
    y = yt.to_numpy()
    fc = {}
    for name, cfg in chosen.items():
        t0 = time.time()
        fc[name] = rolling(factory(name, cfg), X, Y, *TEST)
        print(f"{name:12s} done in {time.time() - t0:.0f}s", flush=True)
    fc["ensemble"] = pd.DataFrame(np.mean([fc[k].to_numpy() for k in ENSEMBLE], axis=0),
                                  index=fc["lear"].index, columns=Y.columns)
    for a in (0.1, 0.9):
        fc[f"lgbm_q{int(a * 100):02d}"] = rolling(lambda a=a: GBM(alpha=a, n_estimators=chosen["lgbm"]["n_estimators"]),
                                                   X, Y, *TEST)
    out = RESULTS / "test"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in fc.items():
        v.to_csv(out / f"forecast_{k}.csv")
    yt.to_csv(out / "actual.csv")

    point_models = list(chosen) + ["ensemble"]
    base = evaluate.daily_abs_err(fc["naive_d1"].to_numpy(), y)
    pf = battery.backtest(y, y)
    naive_profit = battery.backtest(fc["naive_d1"].to_numpy(), y)
    rows = []
    for k in point_models:
        f = fc[k].to_numpy()
        de = evaluate.daily_abs_err(f, y)
        diff = evaluate.bootstrap_mean_ci(base - de)       # > 0 means better than naive_d1
        prof = battery.backtest(f, y)
        dprof = evaluate.bootstrap_mean_ci(prof - naive_profit)
        rows.append({
            "model": k, "mae": evaluate.mae(f, y), "rmse": evaluate.rmse(f, y),
            "mae_skill_vs_naive_d1_pct": 100 * (1 - evaluate.mae(f, y) / evaluate.mae(fc["naive_d1"].to_numpy(), y)),
            "mae_gain_vs_naive_d1": diff[0], "mae_gain_ci_lo": diff[1], "mae_gain_ci_hi": diff[2],
            "profit_eur": prof.sum(), "capture_pct": 100 * prof.sum() / pf.sum(),
            "profit_gain_vs_naive_d1_eur_per_day": dprof[0], "profit_gain_ci_lo": dprof[1], "profit_gain_ci_hi": dprof[2],
            "losing_days": int((prof < 0).sum()),
        })
        pd.Series(prof, index=yt.index).to_csv(out / f"profit_{k}.csv")
    table = pd.DataFrame(rows).sort_values("mae")
    table.to_csv(RESULTS / "test_metrics.csv", index=False)

    # uncertainty: calibration of the 80% interval and the trust gate
    q10, q90, med = (fc[k].to_numpy() for k in ("lgbm_q10", "lgbm_q90", "lgbm"))
    lo, hi = np.minimum(q10, q90), np.maximum(q10, q90)
    mask = guard_mask(med, lo, hi)
    raw = battery.backtest(med, y)
    gated = np.where(mask, raw, 0.0)
    flagged = ~mask
    guard = {
        "interval_80_coverage": evaluate.coverage(lo, hi, y),
        "pinball_q10": evaluate.pinball(lo, y, 0.1), "pinball_q90": evaluate.pinball(hi, y, 0.9),
        "days": int(len(y)), "days_flagged_idle": int(flagged.sum()),
        "profit_ungated_eur": float(raw.sum()), "profit_gated_eur": float(gated.sum()),
        "profit_on_flagged_days_if_traded_eur": float(raw[flagged].sum()),
        "losing_days_ungated": int((raw < 0).sum()), "losing_days_gated": int((gated < 0).sum()),
        "loss_eur_ungated": float(raw[raw < 0].sum()), "loss_eur_gated": float(gated[gated < 0].sum()),
        "perfect_foresight_eur": float(pf.sum()),
    }
    pd.Series(flagged, index=yt.index, name="flagged").to_csv(out / "guard_flags.csv")
    meta = {"preregistration_sha256": sha, "test_window": TEST, "run_utc": pd.Timestamp.utcnow().isoformat(),
            "chosen_config": chosen, "guard": guard}
    (RESULTS / "test_summary.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 220)
    print(table.round(2).to_string(index=False))
    print(json.dumps(guard, indent=2))


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = ROOT / "figures"
    figs.mkdir(exist_ok=True)
    out = RESULTS / "test"
    tm = pd.read_csv(RESULTS / "test_metrics.csv")
    y = pd.read_csv(out / "actual.csv", index_col=0, parse_dates=True)

    # 1. skill with CI
    t = tm.sort_values("mae_gain_vs_naive_d1")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(t["mae_gain_vs_naive_d1"], t["model"],
                xerr=[t["mae_gain_vs_naive_d1"] - t["mae_gain_ci_lo"], t["mae_gain_ci_hi"] - t["mae_gain_vs_naive_d1"]],
                fmt="o", capsize=3)
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("MAE reduction vs 'same as yesterday' (EUR/MWh), 95% block-bootstrap CI")
    ax.set_title("Forecast error, test window Apr-Oct 2026")
    fig.tight_layout(); fig.savefig(figs / "skill_ci.png", dpi=150); plt.close(fig)

    # 2. capture
    t = tm.sort_values("capture_pct")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(t["model"], t["capture_pct"])
    ax.set_xlabel("Share of perfect-foresight battery profit captured (%)")
    ax.set_title("What the forecast is worth: 1 MW / 2 MWh battery, EE day-ahead")
    fig.tight_layout(); fig.savefig(figs / "capture.png", dpi=150); plt.close(fig)

    # 3. a sample week with the 80% band
    med = pd.read_csv(out / "forecast_lgbm.csv", index_col=0, parse_dates=True)
    q10 = pd.read_csv(out / "forecast_lgbm_q10.csv", index_col=0, parse_dates=True)
    q90 = pd.read_csv(out / "forecast_lgbm_q90.csv", index_col=0, parse_dates=True)
    wk = slice("2026-09-14", "2026-09-20")
    flat = lambda d: d.loc[wk].to_numpy().ravel()
    hrs = np.arange(len(flat(y)))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.fill_between(hrs, np.minimum(flat(q10), flat(q90)), np.maximum(flat(q10), flat(q90)), alpha=0.25, label="80% band (LightGBM quantiles)")
    ax.plot(hrs, flat(y), lw=1.4, label="actual EE price")
    ax.plot(hrs, flat(med), lw=1.2, label="forecast (made before 12:00 CET on D-1)")
    ax.set_xlabel("hour, 14-20 Sep 2026"); ax.set_ylabel("EUR/MWh"); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(figs / "sample_week.png", dpi=150); plt.close(fig)

    # 4. cumulative profit
    fig, ax = plt.subplots(figsize=(8, 3.8))
    pf = pd.Series(battery.backtest(y.to_numpy(), y.to_numpy()), index=y.index)
    ax.plot(pf.cumsum(), label="perfect foresight", color="black", lw=1)
    for k in ("ensemble", "lear", "lgbm", "naive_d1"):
        p = pd.read_csv(out / f"profit_{k}.csv", index_col=0, parse_dates=True).iloc[:, 0]
        ax.plot(p.cumsum(), label=k)
    ax.set_ylabel("cumulative EUR"); ax.legend(fontsize=8); ax.set_title("Battery arbitrage on forecasts, settled at real prices")
    fig.tight_layout(); fig.savefig(figs / "cumulative_profit.png", dpi=150); plt.close(fig)

    # 5. the whole study on one chart (test window, capture of perfect-foresight profit)
    if (V3 / "test_metrics.csv").exists():
        v1m = pd.read_csv(RESULTS / "test_metrics.csv").set_index("model")["capture_pct"]
        v3m = pd.read_csv(V3 / "test_metrics.csv").set_index("model")["capture_pct"]
        items = [("same as yesterday", v1m["naive_d1"], "grey"),
                 ("28-day hourly profile\n(no model)", v3m["trailing_28d_profile"], "black"),
                 ("v1 ensemble\n(prices + grid data)", v1m["ensemble"], "tab:blue"),
                 ("v2 ensemble\n(+ weather forecasts)", v3m["ensemble_w"], "tab:blue"),
                 ("v2 linear\n(+ weather forecasts)", v3m["linear_w"], "tab:blue"),
                 ("v3 ridge, trained on error", v3m["ridge_raw_w"], "tab:orange"),
                 ("v3 same model,\ntrained on profit (SPO+)", v3m["spo_w"], "tab:green")]
        if (V4 / "test_metrics.csv").exists():
            v4m = pd.read_csv(V4 / "test_metrics.csv").set_index("model")["capture_pct"]
            items += [("v4 asinh ridge, trained on error", v4m["asinh_ridge_w"], "tab:orange"),
                      ("v4 same model, SPO+\n(unstable, see paper)", v4m["spo_asinh_w"], "tab:red")]
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.barh([i[0] for i in items][::-1], [i[1] for i in items][::-1], color=[i[2] for i in items][::-1])
        ax.axvline(v3m["trailing_28d_profile"], color="black", ls="--", lw=1)
        for k, (_, v, _) in enumerate(items[::-1]):
            ax.text(v + 0.5, k, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_xlim(0, 80)
        ax.set_xlabel("share of perfect-foresight battery profit, test window Apr-Oct 2026 (%)")
        ax.set_title("What each step was worth (1 MW / 2 MWh, EE day-ahead)")
        fig.tight_layout(); fig.savefig(figs / "study_summary.png", dpi=150); plt.close(fig)
    print("figures written to", figs)


# ---------------- v2: weather forecasts (see PREREGISTRATION_v2.md) ----------------
V2 = RESULTS / "v2"
V2_MODELS = ("lear", "lgbm", "linear", "esn")


def load_v2():
    from flexcast import weather
    p = data.fetch_prices(*DATA_RANGE)
    s = data.fetch_system(*DATA_RANGE)
    W = weather.build(*DATA_RANGE)
    X, Y, _ = features.build(p, s, W)
    ok = X.notna().all(axis=1)
    Xw, Yw = X[ok], Y[ok]
    Xnw = Xw[[c for c in Xw.columns if not c.startswith("w_")]]
    return {"w": Xw, "nw": Xnw}, Yw, Y


def validate_w():
    Xs, Y, _ = load_v2()
    y = Y.loc[VAL[0]:VAL[1]].to_numpy()
    rows, chosen = [], {}
    g = grid()
    for tag, X in Xs.items():
        best_fc = {}
        for name in V2_MODELS:
            best = None
            for cfg in g[name]:
                f = rolling(factory(name, cfg), X, Y, *VAL).to_numpy()
                m = evaluate.mae(f, y)
                rows.append({"variant": tag, "model": name, "config": json.dumps(cfg), "val_mae": m})
                print(f"{tag:3s} {name:8s} {json.dumps(cfg):48s} MAE {m:7.2f}", flush=True)
                if best is None or m < best[0]:
                    best = (m, cfg, f)
            chosen[f"{name}_{tag}"] = best[1]
            best_fc[name] = best[2]
        ens = np.mean([best_fc[k] for k in ENSEMBLE], axis=0)
        rows.append({"variant": tag, "model": "ensemble", "config": json.dumps(ENSEMBLE), "val_mae": evaluate.mae(ens, y)})
        print(f"{tag:3s} ensemble MAE {evaluate.mae(ens, y):7.2f}", flush=True)
    V2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(V2 / "validation_grid.csv", index=False)
    (V2 / "chosen_config.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps(chosen, indent=2))


def test_w():
    prereg = ROOT / "PREREGISTRATION_v2.md"
    sha = hashlib.sha256(prereg.read_bytes()).hexdigest()
    chosen = json.loads((V2 / "chosen_config.json").read_text())
    Xs, Y, Yall = load_v2()
    yt = Y.loc[TEST[0]:TEST[1]]
    y = yt.to_numpy()
    fc = {}
    for key, cfg in chosen.items():
        name, tag = key.rsplit("_", 1)
        fc[key] = rolling(factory(name, cfg), Xs[tag], Y, *TEST).to_numpy()
        print(f"{key:10s} done", flush=True)
    for tag in ("w", "nw"):
        fc[f"ensemble_{tag}"] = np.mean([fc[f"{k}_{tag}"] for k in ENSEMBLE], axis=0)
    # reference points from v1 and the trailing profile
    v1 = lambda k: pd.read_csv(RESULTS / "test" / f"forecast_{k}.csv", index_col=0, parse_dates=True).loc[yt.index].to_numpy()
    fc["v1_ensemble"] = v1("ensemble")
    fc["naive_d1"] = v1("naive_d1")
    fc["trailing_28d_profile"] = np.array([Yall.loc[d - pd.Timedelta(days=28): d - pd.Timedelta(days=1)].to_numpy().mean(0)
                                           for d in yt.index])
    out = V2 / "test"
    out.mkdir(parents=True, exist_ok=True)
    pf = battery.backtest(y, y)
    prof = {k: battery.backtest(f, y) for k, f in fc.items()}
    rows = []
    for k, f in fc.items():
        pd.DataFrame(f, index=yt.index, columns=yt.columns).to_csv(out / f"forecast_{k}.csv")
        rows.append({"model": k, "mae": evaluate.mae(f, y), "rmse": evaluate.rmse(f, y),
                     "profit_eur": prof[k].sum(), "capture_pct": 100 * prof[k].sum() / pf.sum(),
                     "losing_days": int((prof[k] < 0).sum())})
    table = pd.DataFrame(rows).sort_values("mae")
    table.to_csv(V2 / "test_metrics.csv", index=False)

    def ci(a):
        return dict(zip(("mean", "ci_lo", "ci_hi"), evaluate.bootstrap_mean_ci(a)))

    dae = lambda k: evaluate.daily_abs_err(fc[k], y)
    hyp = {
        "W1_mae_gain_weather_vs_same_rows_no_weather": ci(dae("ensemble_nw") - dae("ensemble_w")),
        "W2_eur_per_day_ensemble_w_minus_profile": ci(prof["ensemble_w"] - prof["trailing_28d_profile"]),
        "W3_capture_pct": {k: 100 * prof[k].sum() / pf.sum() for k in ("ensemble_w", "v1_ensemble", "trailing_28d_profile")},
        "extra_mae_gain_vs_v1_ensemble": ci(dae("v1_ensemble") - dae("ensemble_w")),
        "extra_eur_per_day_ensemble_w_minus_v1_ensemble": ci(prof["ensemble_w"] - prof["v1_ensemble"]),
        "extra_eur_per_day_best_single_w_minus_profile": {k: ci(prof[k] - prof["trailing_28d_profile"])
                                                          for k in ("lear_w", "lgbm_w", "esn_w", "linear_w")},
    }
    meta = {"preregistration_v2_sha256": sha, "test_window": TEST, "chosen_config": chosen, "hypotheses": hyp}
    (V2 / "test_summary.json").write_text(json.dumps(meta, indent=2))
    pd.set_option("display.width", 200)
    print(table.round(2).to_string(index=False))
    print(json.dumps(hyp, indent=2))


# ---------------- v3: train for profit (see PREREGISTRATION_v3.md) ----------------
V3 = RESULTS / "v3"


def grid_v3():
    return {
        "ridge_raw_w": [{"alpha": a} for a in (100, 1000)],
        "spo_w": [{"alpha": a, "lr": lr, "epochs": ep, "mu": mu}
                  for a in (100, 1000) for lr in (0.003, 0.01, 0.03) for ep in (10, 30) for mu in (0.05, 0.3)],
    }


def factory_v3(name, cfg):
    from flexcast.decision import RidgeRaw, SPOLinear
    if name == "ridge_raw_w":
        return lambda: RidgeRaw(alpha=cfg["alpha"])
    return lambda: SPOLinear(alpha=cfg["alpha"], lr=cfg["lr"], epochs=cfg["epochs"], mu=cfg["mu"])


def validate_p():
    Xs, Y, _ = load_v2()
    X = Xs["w"]
    y = Y.loc[VAL[0]:VAL[1]].to_numpy()
    pf = battery.backtest(y, y).sum()
    rows, chosen = [], {}
    for name, cfgs in grid_v3().items():
        best = None
        for cfg in cfgs:
            t0 = time.time()
            f = rolling(factory_v3(name, cfg), X, Y, *VAL).to_numpy()
            cap = 100 * battery.backtest(f, y).sum() / pf
            rows.append({"model": name, "config": json.dumps(cfg), "val_mae": evaluate.mae(f, y), "val_capture_pct": cap})
            print(f"{name:12s} {json.dumps(cfg):62s} MAE {evaluate.mae(f, y):6.2f} capture {cap:5.1f}% ({time.time() - t0:.0f}s)", flush=True)
            if best is None or cap > best[0]:
                best = (cap, cfg)
        chosen[name] = best[1]
    V3.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(V3 / "validation_grid.csv", index=False)
    (V3 / "chosen_config.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps(chosen, indent=2))


def test_p():
    sha = hashlib.sha256((ROOT / "PREREGISTRATION_v3.md").read_bytes()).hexdigest()
    chosen = json.loads((V3 / "chosen_config.json").read_text())
    Xs, Y, Yall = load_v2()
    yt = Y.loc[TEST[0]:TEST[1]]
    y = yt.to_numpy()
    fc = {k: rolling(factory_v3(k, cfg), Xs["w"], Y, *TEST).to_numpy() for k, cfg in chosen.items()}
    v2 = lambda k: pd.read_csv(V2 / "test" / f"forecast_{k}.csv", index_col=0, parse_dates=True).loc[yt.index].to_numpy()
    for k in ("ensemble_w", "linear_w", "trailing_28d_profile", "naive_d1"):
        fc[k] = v2(k)
    out = V3 / "test"
    out.mkdir(parents=True, exist_ok=True)
    pf = battery.backtest(y, y)
    prof = {k: battery.backtest(f, y) for k, f in fc.items()}
    rows = []
    for k, f in fc.items():
        pd.DataFrame(f, index=yt.index, columns=yt.columns).to_csv(out / f"forecast_{k}.csv")
        rows.append({"model": k, "mae": evaluate.mae(f, y), "capture_pct": 100 * prof[k].sum() / pf.sum(),
                     "profit_eur": prof[k].sum(), "losing_days": int((prof[k] < 0).sum())})
    table = pd.DataFrame(rows).sort_values("capture_pct", ascending=False)
    table.to_csv(V3 / "test_metrics.csv", index=False)

    def ci(a):
        return dict(zip(("mean", "ci_lo", "ci_hi"), evaluate.bootstrap_mean_ci(a)))

    dae = lambda k: evaluate.daily_abs_err(fc[k], y)
    hyp = {
        "P1_eur_per_day_spo_minus_ridge_raw": ci(prof["spo_w"] - prof["ridge_raw_w"]),
        "P2_eur_per_day_spo_minus_profile": ci(prof["spo_w"] - prof["trailing_28d_profile"]),
        "P3_mae_spo_minus_ridge_raw": ci(dae("spo_w") - dae("ridge_raw_w")),
        "extra_eur_per_day_spo_minus_ensemble_w": ci(prof["spo_w"] - prof["ensemble_w"]),
    }
    (V3 / "test_summary.json").write_text(json.dumps({"preregistration_v3_sha256": sha, "test_window": TEST,
                                                      "chosen_config": chosen, "hypotheses": hyp}, indent=2))
    print(table.round(2).to_string(index=False))
    print(json.dumps(hyp, indent=2))


# ---------------- v4: SPO+ on the stronger (asinh) model (see PREREGISTRATION_v4.md) ----------------
V4 = RESULTS / "v4"


def grid_v4():
    return {
        "asinh_ridge_w": [{"alpha": a, "epochs": 0} for a in (300, 1000, 3000)],
        "spo_asinh_w": [{"alpha": a, "lr": lr, "epochs": ep, "mu": 0.05}
                        for a in (300, 1000, 3000) for lr in (0.003, 0.01, 0.03) for ep in (10, 30)],
    }


def factory_v4(name, cfg):
    from flexcast.decision import SPOAsinh
    return lambda: SPOAsinh(alpha=cfg["alpha"], lr=cfg.get("lr", 0.0), epochs=cfg["epochs"], mu=cfg.get("mu", 0.05))


def validate_v4():
    Xs, Y, _ = load_v2()
    X = Xs["w"]
    y = Y.loc[VAL[0]:VAL[1]].to_numpy()
    pf = battery.backtest(y, y).sum()
    rows, chosen = [], {}
    for name, cfgs in grid_v4().items():
        best = None
        for cfg in cfgs:
            t0 = time.time()
            f = rolling(factory_v4(name, cfg), X, Y, *VAL).to_numpy()
            try:  # a diverged model (non-finite or absurd prices) is a failed config, never selectable
                if not np.all(np.isfinite(f)) or np.abs(f).max() > 1e5:
                    raise RuntimeError("diverged")
                cap = 100 * battery.backtest(f, y).sum() / pf
                mae_v = evaluate.mae(f, y)
            except RuntimeError:
                cap, mae_v = float("nan"), float("nan")
            rows.append({"model": name, "config": json.dumps(cfg), "val_mae": mae_v, "val_capture_pct": cap,
                         "failed": bool(np.isnan(cap))})
            print(f"{name:14s} {json.dumps(cfg):62s} MAE {mae_v:6.2f} capture {cap:5.1f}% ({time.time() - t0:.0f}s)", flush=True)
            if not np.isnan(cap) and (best is None or cap > best[0]):
                best = (cap, cfg)
        chosen[name] = best[1]
    V4.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(V4 / "validation_grid.csv", index=False)
    (V4 / "chosen_config.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps(chosen, indent=2))


def test_v4():
    sha = hashlib.sha256((ROOT / "PREREGISTRATION_v4.md").read_bytes()).hexdigest()
    chosen = json.loads((V4 / "chosen_config.json").read_text())
    Xs, Y, _ = load_v2()
    yt = Y.loc[TEST[0]:TEST[1]]
    y = yt.to_numpy()
    fc = {k: rolling(factory_v4(k, cfg), Xs["w"], Y, *TEST).to_numpy() for k, cfg in chosen.items()}
    v3 = lambda k: pd.read_csv(V3 / "test" / f"forecast_{k}.csv", index_col=0, parse_dates=True).loc[yt.index].to_numpy()
    for k in ("linear_w", "ensemble_w", "spo_w", "ridge_raw_w", "trailing_28d_profile", "naive_d1"):
        fc[k] = v3(k)
    out = V4 / "test"
    out.mkdir(parents=True, exist_ok=True)
    pf = battery.backtest(y, y)
    prof = {k: battery.backtest(f, y) for k, f in fc.items()}
    rows = []
    for k, f in fc.items():
        pd.DataFrame(f, index=yt.index, columns=yt.columns).to_csv(out / f"forecast_{k}.csv")
        pd.Series(prof[k], index=yt.index).to_csv(out / f"profit_{k}.csv")
        rows.append({"model": k, "mae": evaluate.mae(f, y), "capture_pct": 100 * prof[k].sum() / pf.sum(),
                     "profit_eur": prof[k].sum(), "losing_days": int((prof[k] < 0).sum())})
    table = pd.DataFrame(rows).sort_values("capture_pct", ascending=False)
    table.to_csv(V4 / "test_metrics.csv", index=False)

    def ci(a):
        return dict(zip(("mean", "ci_lo", "ci_hi"), evaluate.bootstrap_mean_ci(a)))

    hyp = {
        "Q1_eur_per_day_spo_asinh_minus_asinh_ridge": ci(prof["spo_asinh_w"] - prof["asinh_ridge_w"]),
        "Q2_eur_per_day_spo_asinh_minus_profile": ci(prof["spo_asinh_w"] - prof["trailing_28d_profile"]),
        "extra_eur_per_day_spo_asinh_minus_naive_d1": ci(prof["spo_asinh_w"] - prof["naive_d1"]),
        "extra_mae_spo_asinh_minus_asinh_ridge": ci(evaluate.daily_abs_err(fc["spo_asinh_w"], y)
                                                    - evaluate.daily_abs_err(fc["asinh_ridge_w"], y)),
    }
    (V4 / "test_summary.json").write_text(json.dumps({"preregistration_v4_sha256": sha, "test_window": TEST,
                                                      "chosen_config": chosen, "hypotheses": hyp}, indent=2))
    print(table.round(2).to_string(index=False))
    print(json.dumps(hyp, indent=2))


# ---------------- live: the clean out-of-sample test that grows every day ----------------
def live():
    """Score the frozen v1/v2/v3 configs on delivery days after the test window (9 Oct 2026 on),
    which no model or decision in this repo has seen. Rerun any day to extend it."""
    from flexcast import weather
    today = pd.Timestamp.now(tz="Europe/Tallinn").normalize().tz_localize(None)
    end = (today + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    start_live = "2026-10-09"
    p = data.fetch_prices(DATA_RANGE[0], end, refresh=True)
    s = data.fetch_system(DATA_RANGE[0], end, refresh=True)
    W = weather.build(DATA_RANGE[0], end)
    X, Y, _ = features.build(p, s)
    XW, YW, _ = features.build(p, s, W)
    okw = XW.notna().all(axis=1)
    XW, YW = XW[okw], YW[okw]
    days = Y.loc[start_live:].index.intersection(YW.index)
    if len(days) == 0:
        raise SystemExit("No complete delivery days after 8 Oct 2026 yet.")
    a, b = str(days[0].date()), str(days[-1].date())
    y = Y.loc[a:b].to_numpy()
    fc = {}
    c1 = json.loads((RESULTS / "chosen_config.json").read_text())
    for k in ("lear", "lgbm", "esn", "naive_d1"):
        fc[k] = rolling(factory(k, c1[k]), X, Y, a, b).to_numpy()
    fc["ensemble"] = np.mean([fc[k] for k in ENSEMBLE], axis=0)
    c2 = json.loads((V2 / "chosen_config.json").read_text())
    Xnw = XW[[c for c in XW.columns if not c.startswith("w_")]]
    for k in ENSEMBLE:
        fc[f"{k}_w"] = rolling(factory(k, c2[f"{k}_w"]), XW, YW, a, b).to_numpy()
    fc["ensemble_w"] = np.mean([fc[f"{k}_w"] for k in ENSEMBLE], axis=0)
    c3 = json.loads((V3 / "chosen_config.json").read_text())
    for k, cfg in c3.items():
        fc[k] = rolling(factory_v3(k, cfg), XW, YW, a, b).to_numpy()
    fc["trailing_28d_profile"] = np.array([Y.loc[d - pd.Timedelta(days=28): d - pd.Timedelta(days=1)].to_numpy().mean(0)
                                           for d in Y.loc[a:b].index])
    pf = battery.backtest(y, y)
    rows = []
    for k, f in fc.items():
        pr = battery.backtest(f, y)
        rows.append({"model": k, "days": len(y), "mae": evaluate.mae(f, y),
                     "capture_pct": 100 * pr.sum() / pf.sum(), "profit_eur": pr.sum()})
    out = RESULTS / "live"
    out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows).sort_values("capture_pct", ascending=False)
    t.to_csv(out / "live_metrics.csv", index=False)
    print(f"live window {a} .. {b} ({len(y)} days)")
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    {"validate": validate, "test": test, "figures": figures,
     "validate_w": validate_w, "test_w": test_w,
     "validate_p": validate_p, "test_p": test_p,
     "validate_v4": validate_v4, "test_v4": test_v4, "live": live}[sys.argv[1]]()
