# Pre-registration v4: profit training (SPO+) on the stronger model

Written before any v4 model was run on any window. `run.py test_v4` records the SHA-256 of this file.

## Honest status

This is the fourth look at the 1 Apr – 8 Oct 2026 window, so it is **not a clean test**. The clean evidence comes from `run.py live` on days after 8 Oct 2026, and v4 will be added to it.

## What changes from v3

v3 put SPO+ on a raw-price ridge, which was the weakest base (61.7% capture). v4 puts it on the stronger model family: the asinh-space ridge that captured 67.1% in v2 (`linear_w`).

- **Model:** a linear model in z = asinh((p − median)/MAD), with prices p = median + MAD·sinh(z). The SPO+ subgradient with respect to p goes through the chain rule (·MAD·cosh z), plus `mu` times squared error in z, plus an L2 pull to the warm start.
- **Control:** `asinh_ridge_w`, the same class with 0 epochs, i.e. the warm start alone.
- **Grid (validation, picked by capture):** alpha ∈ {300, 1000, 3000}; for `spo_asinh_w` also lr ∈ {0.003, 0.01, 0.03}, epochs ∈ {10, 30}, mu = 0.05. Weekly refits; the same rows, features and battery as v3.

## Hypotheses (test window)

- **Q1:** `spo_asinh_w` earns more than `asinh_ridge_w`, with the CI of the daily € difference excluding 0.
  - *Prediction:* a smaller gain than v3's +€6.6/day, because the base is already better. CI probably touching 0.
- **Q2:** `spo_asinh_w` earns more than the trailing 28-day profile, with the CI excluding 0.
  - *Prediction:* uncertain. v2's `linear_w` was +€4.1/day (CI −0.15 – 8.4) against the profile, so this is plausible but not expected to be clearly shown.

Everything is reported whatever the outcome.

## Amendment (before any test run)

The first validation run crashed. The config {alpha 300, lr 0.01, epochs 30} diverged (validation MAE 467,834 €/MWh): `sinh` amplifies the SPO+ steps on price spikes, and the next diverged forecast made the battery LP fail.

**Rule added:** a config whose validation forecasts are non-finite or exceed 1e5 €/MWh in absolute value, or make the LP fail, is recorded as **failed** and cannot be selected. The grid, hypotheses and test are otherwise unchanged. The divergence is itself a result and is reported.

## Frozen v4 config

- **Chosen:** `asinh_ridge_w` = {alpha 300}, validation capture 65.2%. `spo_asinh_w` = {alpha 300, lr 0.003, epochs 30}, validation capture 64.1%.
- **Divergence:** 9 of 18 SPO+ configs diverged on validation.
- **Grid edge:** alpha = 300 is the lower edge of the pre-registered grid. It is reported as-is, not extended.
- `results/v4/chosen_config.json` SHA-256: `f5e19070c1f2d62f283f17d39dc85410093fc99916d5a3cc101f6ad1473da39a`
