# What Is a Price Forecast Worth to a Battery? A Pre-Registered, Decision-Focused Study of Day-Ahead Forecasting for Baltic Flexibility, with a Quantum-Optimisation Benchmark for mFRR Bid Assembly

<div class="authors">
<b>Mouadh Abderrahmane Assal</b><br>
AiQC — AI &amp; Quantum Community · Tlemcen, Algeria · thereviewboost@gmail.com<br>
Code, data and all pre-registrations: https://github.com/mouadhassal/flexcast-baltic · October 2026
</div>

<div class="abstract" markdown="1">
**Abstract.**

*The question.* Flexibility aggregators schedule batteries and flexible loads against electricity prices. They usually judge forecasts by statistical error, but what matters is the money the schedule earns. We measure that money directly, for a 1 MW / 2 MWh battery in the Estonian day-ahead market, from April to October 2026.

*Protocol.* We use only information available at the 12:00 CET gate closure. The evaluation is a rolling origin, settled at realised prices, with a pre-registered, locked test window of 191 days and block-bootstrap confidence intervals. We compare naive rules, LEAR, LightGBM, an echo-state network and their ensemble. A simulated quantum reservoir is compared against a same-size random-feature null control. We then add archived numerical weather forecasts, and finally train the forecast directly on battery profit with the SPO+ loss, using a batched dynamic-programming solver.

*Results.*

1. Forecasts cut error 27–31% against "same as yesterday" and double battery profit, from 31% to 65–67% of the perfect-foresight optimum.
2. But a model-free 28-day hourly profile also reaches 65.2%. Accuracy gains largely fail to turn into euros in summer. They do turn into euros in a wind-driven winter: +€16.0/day per MW, 95% CI [4.0, 25.4].
3. Training on profit instead of squared error raises a linear model's earnings by +€6.56/day per MW, CI [3.53, 9.61], at slightly worse accuracy. The same loss destabilises a variance-stabilised model, a negative result we report.
4. The quantum reservoir does not beat random features.

*Quantum benchmark.* We also pose mFRR bid assembly across a portfolio as a QUBO and benchmark exact search, simulated annealing, QAOA (p ≤ 3, noiseless) and MILP. QAOA lifts the probability of sampling the optimal bid only about 3× above random guessing, and its success falls from 100% to 10% between 8 and 16 assets. Classical MILP proves optimality for 2,000-asset portfolios in under 0.5 s.

*Conclusion.* For aggregators, the remaining value lies in decision-focused training, real asset data and balancing markets, not in more accurate forecasts or near-term quantum hardware.
</div>

## 1. Introduction

The Baltic power system changed fast between 2024 and 2026:

- **Balancing platforms:** the Baltic states joined the European balancing platforms (MARI, PICASSO).
- **New capacity market:** a Baltic Balancing Capacity Market opened in February 2025.
- **Synchronisation:** the grid synchronised with Continental Europe on 9 February 2025.
- **Quarter-hour products:** the single day-ahead coupling (SDAC) moved to quarter-hour products on 1 October 2025.
- **Price extremes:** balancing prices in the region have repeatedly hit the ±€10,000/MWh platform cap.

For an aggregator that turns batteries, heat pumps and industrial loads into market-ready flexibility, three things decide revenue: the forecast, the dispatch logic, and the bid built from many small assets.

Price-forecasting research usually reports MAE or RMSE. An asset owner is paid for decisions. A battery's day-ahead profit depends mostly on the *ordering* of hours, cheap versus dear, not on price levels. So a forecast can improve on MAE and earn nothing more. This paper measures that gap directly and asks three questions:

1. **What is a day-ahead forecast worth in euros** to a small Estonian battery, against honest baselines, including a model-free profile?
2. **Does training the forecast on profit**, rather than on price error, recover the value that accuracy misses?
3. **Is quantum computing useful today** for the two places an aggregator might try it: as a forecaster (quantum reservoir computing), and as an optimiser for combinatorial bid assembly (QAOA)?

**Contributions.**

1. A fully reproducible, pre-registered evaluation on public data. It uses only information available at the 12:00 CET gate closure, has a unit-tested no-look-ahead guarantee, and records SHA-256-frozen pre-registrations for each of five experiments.
2. Evidence that a 28-day hourly profile matches sophisticated models in summer day-ahead arbitrage value, while models beat it significantly in a wind-driven winter.
3. A practical SPO+ implementation for battery scheduling with a vectorised dynamic-programming oracle. It yields a statistically significant profit gain on a linear model, and a documented instability on an asinh-link model.
4. Two quantum null results with proper controls: a quantum reservoir against same-size random features, and QAOA against classical solvers on a realistic aggregator QUBO. Both come with an open benchmark harness.

## 2. Background and related work

**Price forecasting.**

- *Benchmark.* LEAR is a LASSO-estimated autoregressive model fitted per hour on lagged prices and calendar effects. Lago et al. (2021) established it as the open benchmark in electricity price forecasting (EPF).
- *Transforms.* Variance-stabilising transforms such as the area hyperbolic sine are standard for spiky, possibly negative prices (Uniejewski, Weron & Ziel, 2018).
- *Gradient boosting and neural models* are common strong competitors.

**Value of forecasts for storage.** Recent work reports that forecast-driven arbitrage captures between about 66% and 98% of perfect-foresight revenue, depending on market and design. It also reports that ranking quality (e.g. Kendall's τ between forecast and realised hour orderings) predicts value better than MAE (arXiv 2604.12082; arXiv 2608.08377).

**Decision-focused learning.** SPO+ (Elmachtoub & Grigas, 2022) is a convex surrogate for the regret of a downstream linear optimisation. Its subgradient needs an optimisation oracle per training sample.

**Quantum reservoir computing (QRC)** uses the dynamics of a quantum system as a fixed nonlinear feature map with a trained linear readout. Fair evaluation needs size-matched classical random-feature controls.

**Quantum optimisation.**

- *Algorithms.* QAOA and quantum annealing target QUBO problems.
- *Energy pilots.* EDF and Pasqal have tested neutral-atom approaches for EV smart charging. Oak Ridge and IonQ have tested unit commitment. Public evidence of advantage on realistic sizes is lacking.

## 3. Market setting and data

All market data comes from Elering's public API: Nord Pool day-ahead prices for EE, FI, LV and LT, and Estonian system actuals (consumption, production, renewables, solar, AC balance), January 2023 to 8 October 2026.

- **Hourly target.** Quarter-hour prices are averaged to hours.
- **Market days.** These follow CET/CEST, as in SDAC.
- **DST.** On the 25-hour day the repeated hour is averaged; on the 23-hour day the missing hour is interpolated.

**Weather inputs.** Weather enters only as forecasts available before gate closure. We use Open-Meteo's Previous Runs API at a fixed 48-hour lead (`_previous_day2`), so every value for delivery day $D$ was issued no later than $D-2$, 23:00 CET. (The 24-hour fields would leak into the late hours of $D$, so they are not used.)

- **Wind:** four sites (two Finnish wind clusters, West Estonia, the Lithuanian coast), mapped through a turbine-curve proxy $\min\{1, ((v-3)/9)_+\}^3$ with cut-out at 25 m/s.
- **Temperature and solar radiation:** Tallinn, Helsinki and Vilnius.
- **Coverage:** the archive is complete from March 2024.

**Windows.**

| window | dates | days |
|---|---|---|
| validation | 1 Oct 2025 – 31 Mar 2026 | 182 |
| test (locked) | 1 Apr – 8 Oct 2026 | 191 |
| live | from 9 Oct 2026 | growing |

## 4. Methods

### 4.1 Information set and features

For delivery day $D$, the features contain:

- EE prices for $D-1$, $D-2$ and $D-7$ (24 values each)
- FI prices for $D-1$
- $D-1$ summary statistics and cross-zone spreads
- system actuals for the full day $D-2$
- calendar (weekday, Estonian public holidays, season)
- a flag for the post-synchronisation regime
- in v2 onward, 24-hour blocks and daily means of the weather forecasts

A unit test changes every input of day $D$ and checks that the features of day $D$ do not move.

### 4.2 Forecasting models

Prices are modelled in the variance-stabilised space

$$z = \operatorname{asinh}\!\left(\frac{p - \tilde p}{\mathrm{MAD}}\right), \qquad p = \tilde p + \mathrm{MAD}\,\sinh(z),$$

where $\tilde p$ is the training-window median.

| model | description |
|---|---|
| naive | "same as yesterday" (D−1) and "same as last week" (D−7) |
| LEAR | 24 per-hour LASSO models (LARS-AIC), asinh-transformed lags, average of 364- and 728-day windows |
| LightGBM | one model on a day×hour table, L1 loss; quantile versions (q10, q90) for uncertainty |
| ridge | ridge regression on standardised features |
| ESN | leaky echo-state network, 400 units, run over the day sequence, ridge readout |
| ensemble | equal-weight mean of LEAR, LightGBM and ESN, members fixed in advance |

**Quantum reservoir** (`pca8_qrc`). Inputs are compressed to 8 principal components $x\in[0,1]^8$. They are loaded as $\bigotimes_i R_Y(\pi x_i)\lvert 0\rangle$ and evolved under a fixed transverse-field Ising Hamiltonian

$$H = \sum_{i<j} J_{ij} Z_i Z_j + h \sum_i X_i, \qquad J_{ij}\sim \mathcal U(-1,1),$$

for three steps of $e^{-iH\tau}$. Each step is read out as $\langle Z_i\rangle$ and $\langle Z_iZ_j\rangle$, giving 108 features, simulated exactly. The null control `pca8_rff` uses 108 random cosine features of the same inputs. Both share the same ridge readout, so any difference comes from the feature map alone.

### 4.3 Battery and settlement

The battery is 1 MW / 2 MWh, with 90% round-trip efficiency ($\eta=\sqrt{0.9}$ each way), 50% state of charge at the start and end of each day, and a €5/MWh throughput cost $\kappa$. For a price vector $p\in\mathbb R^{24}$, the schedule solves the linear program

$$\min_{c,d,s}\; \sum_t p_t (c_t - d_t) + \tfrac{\kappa}{2}\sum_t (c_t + d_t) \quad \text{s.t.}\quad s_t = s_{t-1} + \eta c_t - d_t/\eta,\; 0\le c_t,d_t\le 1,\; 0\le s_t\le 2,\; s_{24}=s_0.$$

Forecast-based schedules are settled at realised prices. **Capture** is realised profit divided by perfect-foresight profit.

### 4.4 Training for profit: SPO+

Let $x^*(q)$ be the optimal net-charge schedule for a price vector $q$, and write the decision cost as $q^\top x + g(x)$, where $g$ is the throughput cost. The SPO+ loss for a forecast $\hat p$ of the true prices $p$ is

$$\ell_{\mathrm{SPO+}}(\hat p, p) = \max_x\{(p - 2\hat p)^\top x - g(x)\} + 2\hat p^\top x^*(p) - z^*(p),$$

with subgradient

$$\nabla_{\hat p}\,\ell = 2\big(x^*(p) - x^*(2\hat p - p)\big).$$

Each training step needs $x^*$ for every day in the batch. We replace the LP in that inner loop with a **batched dynamic program** on a 0.1 MWh state-of-charge grid. It solves all days at once in milliseconds and reaches 98.6% of the exact LP optimum on true test prices; a unit test bounds the gap. Evaluation always uses the exact LP.

The forecaster is a linear map from standardised features, warm-started at the ridge solution. It is fine-tuned with Adam on SPO+, plus a scaled squared-error term (weight $\mu$), plus an L2 pull to the warm start. Two versions are tested:

- **v3:** in raw price space.
- **v4:** in asinh space, with the subgradient passed through $\partial p/\partial z = \mathrm{MAD}\cosh z$.

### 4.5 Protocol

- **Rolling origin.** Models are refitted every 7 days on all prior history. Feature maps are frozen before each window.
- **Tuning.** Hyperparameters are chosen on validation only, by MAE (v1, v2) or by capture (v3, v4). Grids are widened when an optimum falls on an edge, with the exceptions reported.
- **Pre-registration.** Every experiment's hypotheses and predictions are written to a pre-registration file before its test run. The SHA-256 of that file is stored with the results.
- **Confidence intervals.** These come from a moving-block bootstrap over days (block 7, 5,000 resamples) on daily paired differences.
- **Window reuse.** The v2–v4 experiments reuse the v1 test window and say so in their pre-registrations. The `live` command scores all frozen models on days after 8 October 2026 as a clean test.

## 5. Results

### 5.1 Forecast accuracy and value (v1)

| model | MAE €/MWh | MAE gain vs naive (95% CI) | capture | € gain/day vs naive (95% CI) | losing days |
|---|---|---|---|---|---|
| ensemble | **30.7** | 11.6 [8.2, 15.1] | 65.0% | 72.6 [53.2, 101.9] | 13 |
| ESN | 31.4 | 10.9 [7.2, 14.8] | **65.5%** | 73.7 [55.0, 101.5] | 12 |
| LEAR | 31.6 | 10.7 [6.9, 14.5] | 64.8% | 72.1 [51.9, 101.9] | 19 |
| ridge | 32.2 | 10.0 [6.4, 13.7] | 64.8% | 72.1 [53.9, 99.5] | 11 |
| LightGBM | 33.6 | 8.7 [5.9, 11.5] | 57.5% | 56.5 [38.6, 83.9] | 15 |
| pca8_rff | 33.7 | 8.5 [5.4, 11.7] | 63.9% | 70.1 [51.5, 97.4] | 13 |
| pca8_qrc | 34.9 | 7.4 [3.7, 10.8] | 64.4% | 71.3 [52.7, 97.2] | 14 |
| naive D−1 | 42.3 | – | 31.2% | – | 44 |

![](../figures/skill_ci.png)
<p class="caption">Figure 1. MAE reduction against "same as yesterday" on the locked test window, with 95% block-bootstrap intervals.</p>

**What the table shows.**

- **Every model beats naive** on accuracy and on profit, with intervals excluding zero (pre-registered H1 and H2 held).
- **Profit roughly doubles**, from €12.8k to €26.6k over 191 days.
- **The quantum reservoir is worse than random features** by €1.19/MWh, CI [0.42, 2.24] (pre-registered H3 held). It is indistinguishable from a linear readout on the same 8 inputs.
- **The trust gate failed.** It re-priced each schedule at per-hour pessimistic quantiles and idled the battery on all 191 days (H4 failed). The LightGBM 80% interval covered only 56.5% of outcomes.

### 5.2 The profile benchmark

*Exploratory.*

| benchmark | capture |
|---|---|
| trailing 28-day hourly profile, no model | **65.2%** |
| best v1 model | 65.0% (difference −€0.39/day, CI [−6.1, 5.3]) |
| month-hour average known in advance (oracle) | 69.5% |

The best models are 23% more accurate than the profile (MAE 30.7 against 39.9) but earn the same. Their mean daily Kendall τ between forecast and realised hour rankings is 0.52, against 0.51 for the profile. The literature associates near-perfect capture with τ ≈ 0.85–0.95.

### 5.3 Weather forecasts (v2) and seasonality

Weather forecasts improved accuracy significantly: −€1.61/MWh on the same rows, CI [0.19, 2.80]. The ensemble MAE fell to €29.3/MWh, 31% better than naive. But euros did not move significantly, either against the profile (+€2.33/day, CI [−2.92, 7.20]) or against the no-weather model.

The seasonal split is the key finding:

| | winter (validation, 182 days) | summer (test, 191 days) |
|---|---|---|
| models vs profile, € per day | **+16.0 [4.0, 25.4]** | +2.3 [−2.9, 7.2] |
| capture, model / profile | 67.8% / 62.6% | 66.3% / 65.2% |

In summer the solar-shaped daily curve is regular, and a profile captures it. In winter, wind makes daily shapes irregular, and only models track them. The winter figure comes from the tuning window, so it is mildly optimistic.

### 5.4 Training for profit (v3, v4)

| model | MAE | capture | € vs its error-trained start (95% CI) |
|---|---|---|---|
| v3 ridge, raw prices, squared error | 30.4 | 61.7% | – |
| **v3 same model, SPO+** | 30.8 | **64.8%** | **+6.56 [3.53, 9.61]** |
| v4 ridge, asinh space, squared error | 31.0 | 66.4% | – |
| v4 same model, SPO+ | 35.6 | 64.4% | −4.36 [−9.61, 0.32] |
| v2 linear (asinh), for reference | 31.4 | **67.1%** | – |

**v3 (raw prices).** Changing only the training loss raised profit significantly (pre-registered P1 held) at a small accuracy cost: MAE +€0.35, CI [0.17, 0.55] (P3 held). On a 1 MW battery that is about €2,400 a year from the loss function alone. Because the raw-price base was weak, SPO+ still only tied the profile (P2: −€0.94/day, CI [−6.2, 3.8]).

**v4 (asinh space).** The asinh-link version was numerically unstable.

- On validation, 9 of 18 configurations diverged: one reached MAE 467,834 €/MWh, and diverged forecasts broke the LP.
- The best stable configuration lost to its own warm start on test (Q1 failed).
- *Cause:* $\cosh z$ multiplies the SPO+ subgradient on price spikes.
- *Fixes, not tried here so as not to tune on the test window:* gradient clipping, a bounded link, or SPO+ applied in $z$-space against a linearised decision.

![](../figures/study_summary.png)
<p class="caption">Figure 2. Share of perfect-foresight profit captured on the test window by each step of the study. The dashed line is the model-free 28-day profile.</p>

### 5.5 Quantum optimisation for mFRR bid assembly

An aggregator must build an mFRR bid of size $T$ from on/off assets with power $P_i$ and activation cost $c_i$. We pose it as the QUBO

$$\min_{x\in\{0,1\}^n} \; f(x) = \sum_i c_i P_i x_i + \lambda\Big(\sum_i P_i x_i - T\Big)^2 .$$

**Instances.**

- *Size and portfolio:* $T$ = 1 MW, on seeded synthetic portfolios mixing heat pumps, commercial HVAC, industrial loads and small batteries.
- *Penalty:* $\lambda$ is chosen so that missing the target by the smallest asset always costs more than any single activation.
- *Count:* 50 instances, 10 at each of $n$ = 8, 10, 12, 14 and 16.
- *Quality of the optimum:* every optimum delivers 0.99–1.03 MW.

**Solvers.**

1. **Exhaustive search**, giving ground truth.
2. **Simulated annealing**, 1,000 sweeps × 20 restarts.
3. **QAOA**, $p$ = 1–3, simulated with an exact noiseless statevector (checked against dense matrices). Angles are optimised with COBYLA and INTERP warm starts; we then keep the best of 1,000 sampled bitstrings.
4. **MILP** (HiGHS) on the constrained form, for scaling.

| assets $n$ | SA solves to optimum | QAOA p=3 solves to optimum | QAOA p=3: P(optimum) | random guessing | MILP median time |
|---|---|---|---|---|---|
| 8 | 100% | 100% | 1.0×10⁻² | 3.9×10⁻³ | – |
| 12 | 100% | 50% | 7.4×10⁻⁴ | 2.4×10⁻⁴ | – |
| 16 | 80% | **10%** | 5.0×10⁻⁵ | 1.5×10⁻⁵ | – |
| 2,000 | – | – | – | – | **0.32 s (all proven optimal)** |

![](../figures/quantum_bids.png)
<p class="caption">Figure 3. Left: share of instances solved to optimum. Centre: QAOA amplifies the probability of the optimal bid by about 3× over uniform sampling, at every size, while both decay exponentially. Right: exact MILP solve times up to 2,000 assets.</p>

All three pre-registered predictions held:

- **A:** simulated annealing solved 96% of all instances.
- **B:** QAOA's success fell to 10% at 16 assets.
- **C:** MILP proved optimality for 2,000 assets in under 0.5 s.

QAOA's amplification of about 3× is constant in $n$, so it cannot keep up with a search space that doubles per asset, even without noise. Its expected energy is 70–233× the optimum, because the penalty term dominates the landscape.

## 6. Discussion: what this means for an aggregator

1. **Measure forecasts in euros, against a profile.** A model that "beats naive" by a wide margin can still be worth nothing over a 28-day profile, as in summer here. Seasonal value differs: winter wind is where models earn.
2. **Train for the decision, carefully.** SPO+ turned a forecast that was worse on MAE into one that earned €6.56/day more. That is the first change in this study to move euros rather than error. But the loss interacts badly with nonlinear links, so it needs stabilisation before production.
3. **The money is beyond public day-ahead data.** An industry index puts total revenue for a 2-hour Estonian battery near €152k/MW/year (Clean Horizon, August 2026), mostly from balancing. Our day-ahead arbitrage is about €51k/MW/year annualised, against a €78k perfect-foresight ceiling. The next gains are in balancing markets, real asset constraints and 15-minute resolution.
4. **Quantum: measure, don't assume.** Neither QRC nor QAOA helped at realistic sizes. The value today is the harness: the same QUBO and benchmark can be rerun on real hardware, or at larger depth, as devices improve, giving a factual answer to "should we use quantum?"

## 7. Limitations

- **Market scope.** Day-ahead only, at hourly resolution. No intraday, balancing, bid curves or grid fees.
- **Battery model.** Daily state-of-charge reset.
- **Weather.** A 7-point public proxy from one archived model.
- **Test window.** One window, spring to autumn. Experiments v2–v4 reuse it, which the pre-registrations say. Clean evidence will come from `run.py live`.
- **Quantum instances.** Synthetic and noiseless, a best case for QAOA. QAOA depth was limited to $p \le 3$.

## 8. Conclusion

A careful, pre-registered evaluation shows four things:

1. Day-ahead forecasts double a small Baltic battery's arbitrage profit over persistence.
2. Much of that comes from the daily shape, which a model-free profile already captures in summer.
3. Decision-focused training is the lever that moved euros once accuracy stopped doing so.
4. Quantum methods, tested with honest controls, are not yet useful for forecasting or for bid assembly at aggregator scale.

**Next steps.** The next experiments are stabilised SPO+ on stronger models, real asset data, and the mFRR leg, where Baltic prices reach ±€10,000/MWh.

**Reproducibility.**

- **Forecasting experiments:** `python run.py <stage>` for v1 (`validate`, `test`), v2 (`validate_w`, `test_w`), v3 (`validate_p`, `test_p`), v4 (`validate_v4`, `test_v4`), and `live`.
- **Exploratory tables:** `python extras.py`.
- **Quantum benchmark:** `python quantum_bids.py`.
- **Integrity:** all raw API responses are cached in the repository, the pre-registration SHA-256s are stored with the results, and the 8 unit tests run with `pytest`.

## References

1. Lago, J., Marcjasz, G., De Schutter, B., Weron, R. (2021). Forecasting day-ahead electricity prices: A review of state-of-the-art algorithms, best practices and an open-access benchmark. *Applied Energy* 293.
2. Uniejewski, B., Weron, R., Ziel, F. (2018). Variance stabilizing transformations for electricity spot price forecasting. *IEEE Transactions on Power Systems* 33(2).
3. Elmachtoub, A. N., Grigas, P. (2022). Smart "Predict, then Optimize". *Management Science* 68(1).
4. Farhi, E., Goldstone, J., Gutmann, S. (2014). A Quantum Approximate Optimization Algorithm. arXiv:1411.4028.
5. Zhou, L., Wang, S.-T., Choi, S., Pichler, H., Lukin, M. D. (2020). Quantum Approximate Optimization Algorithm: Performance, mechanism, and implementation on near-term devices. *Physical Review X* 10, 021067.
6. Fujii, K., Nakajima, K. (2017). Harnessing disordered-ensemble quantum dynamics for machine learning. *Physical Review Applied* 8, 024030.
7. Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks. GMD Report 148.
8. *When Forecast Accuracy Fails: Rank Correlation and Decision Quality in Multi-Market Battery Storage Optimization.* arXiv:2604.12082 (2026).
9. *Price Information Is Not Enough: Ordering and Decision Rules in Storage Bidding.* arXiv:2608.08377 (2026).
10. *A Comparative Analysis of Price Forecasting Methods for Maximizing Battery Storage Profits.* *Energies* 18(13):3309 (2025).
11. ESS News (24 Sep 2026). Battery revenues rise everywhere in Europe other than Baltics in August 2026 (Clean Horizon index).
12. Elering AS. Dashboard API, https://dashboard.elering.ee. Open-Meteo, Previous Runs API, https://open-meteo.com.
