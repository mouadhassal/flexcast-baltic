# Pre-registration v3: train the forecast for battery profit (SPO+)

Written before the v3 validation grid and before any v3 model touched the test window. `run.py test_p` records the SHA-256 of this file.

## Honest status

- The test window (1 Apr – 8 Oct 2026) was used in v1 and looked at again in v2. v3 is a **third look at a seen window, not a clean test**.
- Hyperparameters are chosen on the validation window only, by validation **profit**, since profit is the target.
- One setup check was run before this file: on the training data before the validation window, SPO+ fine-tuning raised in-sample capture from 83.1% to 84.6% across step sizes 0.003–0.1. A rough four-setting probe on the validation window showed step sizes 0.3–3 *hurt*, which is why the grid below uses smaller ones. I report both.

## Models (same inputs as v2: v1 features + weather forecasts, rows from 2024-03-08)

- **`ridge_raw_w`:** ridge regression in raw price space, squared error. This is the warm start, and the ablation.
- **`spo_w`:** the same linear model, initialised at `ridge_raw_w`, then fine-tuned with Adam on the SPO+ loss plus `mu` times a scaled squared-error term, with an L2 pull back to the warm start. The battery schedule inside training comes from a batched dynamic program on a 0.1 MWh grid; it reaches 98.6% of the exact LP optimum on true test prices. All evaluation uses the exact LP.
- **Grid (validation, picked by capture):**
  - `ridge_raw_w`: alpha ∈ {100, 1000}
  - `spo_w`: alpha ∈ {100, 1000}, lr ∈ {0.003, 0.01, 0.03}, epochs ∈ {10, 30}, mu ∈ {0.05, 0.3}
  - weekly refits, as before

## Hypotheses (test window)

- **P1, profit training works:** `spo_w` earns more than `ridge_raw_w`, with the CI of the daily € difference excluding 0.
  - *Prediction:* small positive, CI likely touching 0. In-sample the gain was only about 1.5 points.
- **P2, the real bar:** `spo_w` earns more than the trailing 28-day profile, with the CI excluding 0.
  - *Prediction:* not shown.
- **P3, the cost:** `spo_w` MAE is higher than `ridge_raw_w`. Training for decisions trades accuracy for profit.
  - *Prediction:* holds.

Everything is reported whatever the outcome.

## Frozen v3 config (added after validation, before the test run)

- **Grid change:** the grid was extended with alpha = 30 because alpha = 100 sat on the edge. alpha = 100 is now interior.
- **Step size:** the chosen lr = 0.03 is the top of the grid. The earlier probe (lr 0.3–3, capture 49–61%) brackets it from above.
- **Chosen:** `spo_w` = {alpha 100, lr 0.03, epochs 10, mu 0.05}, validation capture 67.3% (MAE 35.4). `ridge_raw_w` = {alpha 100}, validation capture 65.6% (MAE 34.9).
- `results/v3/chosen_config.json` SHA-256: `b33ed35c5a9f26b58df21925566e863576e5d0d48a44bf6759433580b5c6ce56`
