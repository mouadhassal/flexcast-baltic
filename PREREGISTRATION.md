# Pre-registration: test window for flexcast

Written after the validation run and **before** the test window (1 Apr – 8 Oct 2026) was
forecast by any model. `run.py test` records the SHA-256 of this file in
`results/test_summary.json`, so any later edit is detectable.

## Frozen setup

- **Target:** the 24 hourly Estonian (EE) day-ahead prices of delivery day D. Quarter-hour prices are averaged to the hour.
- **Information set:** what is known at 12:00 CET on D-1. This is prices up to D-1, Estonian system actuals up to D-2, and the calendar. No weather forecasts are used.
- **Test window:** delivery days 2026-04-01 to 2026-10-08 (191 days). It is used once.
- **Protocol:** rolling origin. Every model is refitted every 7 days on all history before the block. Feature maps (scaler, PCA, reservoirs) are fitted once on data before 2026-04-01 and then frozen.
- **Models and hyperparameters:** exactly as in `results/chosen_config.json`, selected by validation MAE (1 Oct 2025 – 31 Mar 2026). The SHA-256 of that file is given below.
- **Battery:** 1 MW / 2 MWh, 90% round-trip efficiency, 50% state of charge at the start and end of each day, and a 5 EUR/MWh throughput cost. The schedule is the exact linear-programming optimum on the forecast. Profit is settled at actual prices.
- **Uncertainty:** day-level moving-block bootstrap (block of 7 days, 5,000 resamples), 95% intervals.

## Metrics

- **Primary:** MAE in EUR/MWh over all test hours.
- **Secondary:** RMSE.
- **Decision metric:**
  - realised battery profit (EUR)
  - capture, meaning profit as a share of the perfect-foresight profit
  - number of losing days

## Hypotheses and predictions

- **H1 (forecasting):** LightGBM and LEAR both have lower MAE than `naive_d1` ("same as yesterday"). The 95% CI of the daily MAE reduction must exclude 0.
- **H2 (value):** the lowest-MAE model earns more battery profit than scheduling on `naive_d1`. The 95% CI of the daily profit gain must exclude 0.
  - *Prediction:* this is uncertain. Arbitrage profit depends on getting the *order* of hours right, not the level, so a large MAE gain may give a small profit gain.
- **H3 (quantum reservoir, expected null):** at equal input (8 PCA components), equal feature count (108) and equal ridge readout, the simulated quantum reservoir (`pca8_qrc`) does **not** beat random cosine features (`pca8_rff`).
  - *Prediction:* the CI of the MAE difference between them includes 0, or favours RFF.
  - *Reporting:* if QRC wins with a CI excluding 0, it is reported as a result. Either way it is reported.
- **H4 (trust gate):** the guard in `run.py::guard_mask` reduces losing days compared with ungated trading on the LightGBM median.
  - *Prediction:* losing days fall, and total profit changes by less than 5%.
- **Calibration check:** coverage of the LightGBM [q10, q90] interval is between 70% and 90%. The nominal coverage is 80%.

## Rules

- No model, feature, hyperparameter or window is changed after the test forecasts exist.
- Anything computed after the test run that is not listed here is labelled **exploratory** in the report.
- All test numbers are reported, including the unfavourable ones.

## Frozen config hash

`results/chosen_config.json` SHA-256: `60550e9f335635f38ab6baeb6b60e89484ac9bceb783f6afc9ac6cb318a4cda2`

The test also adds an `ensemble` model: an equal-weight average of `lear`, `lgbm` and `esn`. No weights are fitted. The members were fixed before the test run.

## Validation results these choices came from (1 Oct 2025 – 31 Mar 2026, MAE EUR/MWh)

| Model | MAE |
|---|---|
| lear | 37.1 |
| ensemble | 37.8 |
| esn | 40.2 (approx.) |
| lgbm | 41.5 |
| pca8_rff | 42.8 |
| pca8_qrc | 43.1 |
| pca8_linear | 44.1 |
| linear | 44.9 |
| naive_d1 | 53.7 |
| naive_d7 | 58.8 |

Full grid: `results/validation_grid.csv`.
