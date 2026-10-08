# Pre-registration v2: does adding weather forecasts change the answer?

Written **before** any weather model was fitted or evaluated. `run.py test_w` records the SHA-256 of this file.

## Honest status of this test

The test window (1 Apr – 8 Oct 2026) was already used once, in v1. I have seen how prices behaved in it and how the v1 models did. So v2 is a **second look at a seen window, not a clean out-of-sample test**. I keep the window anyway, so v1 and v2 are comparable day for day.

All choices are made on the validation window only:
- which sites and variables
- the feature design
- the hyperparameters

The sites and variables below were picked before any model run, from domain reasoning.

The next truly clean test is the period after 8 Oct 2026, which I will run once enough days exist.

## What changes from v1

- **New inputs.** Weather forecasts from the Open-Meteo Previous Runs API, using `_previous_day2` fields only (issued 48 h before the valid hour). So for every hour of day D the forecast was issued no later than D-2 23:00 CET, before the 12:00 CET gate closure on D-1.
- **Sites:**
  - wind: Finnish west coast (63.1, 21.6), Finnish north coast (64.8, 24.9), West Estonia (58.8, 23.0), Lithuanian coast (55.8, 21.3)
  - temperature and solar radiation: Tallinn, Helsinki, Vilnius
- **Features:** a wind power proxy (0 below 3 m/s, cubic to rated at 12 m/s, cut-out at 25 m/s) for FI (mean of 2 sites), EE and LT, plus mean temperature and mean solar radiation. Each comes as 24 hourly values for day D plus a daily mean.
- **History:** the weather archive is complete only from about March 2024, so weather models train on rows from 2024-03-08 onward.

## Models

| name | inputs | rows |
|---|---|---|
| `lear_w`, `lgbm_w`, `esn_w`, `linear_w` | v1 features + weather | from 2024-03-08 |
| `ensemble_w` | equal-weight mean of `lear_w`, `lgbm_w`, `esn_w` | – |
| `lear_nw`, `lgbm_nw`, `esn_nw`, `ensemble_nw` | v1 features only | **same rows** as the `_w` models (ablation) |

Hyperparameter grids are the same as v1 for LightGBM, linear and ESN, selected by validation MAE. LEAR has none.

## Hypotheses

- **W1 (accuracy):** `ensemble_w` has lower test MAE than `ensemble_nw`, with the CI of the daily MAE difference excluding 0.
  - *Prediction:* holds.
- **W2 (euros, the one that matters):** `ensemble_w` earns more than the trailing 28-day hourly profile, with the CI of the daily € difference excluding 0.
  - *Prediction:* uncertain. I expect a small positive point estimate, possibly with a CI touching 0.
- **W3 (capture):** report `ensemble_w` capture next to v1 (65.0%) and the profile (65.2%). No threshold.

All other v2 numbers are descriptive. Everything is reported whether it helps or not.

## Frozen v2 config

`results/v2/chosen_config.json` SHA-256: `6cc4b31498a796e35090ff724382d76d30cc420ea01556a26d3e15a9b42d0022`

Validation MAE (EUR/MWh, 1 Oct 2025 – 31 Mar 2026, same rows): ensemble_w 32.6 vs ensemble_nw 37.5.
