# Pre-registration: quantum optimisation (QAOA) for mFRR bid assembly

Written before any solver in `quantum_bids.py` was run. The script records the SHA-256 of this file.

## Problem

An aggregator such as Fusebox must build an mFRR bid of target size T (here 1.0 MW) out of many small flexible assets. Each asset is on/off, has power P_i (MW) and an activation cost c_i (€/MWh: wear, comfort, lost production).

Choose x ∈ {0,1}ⁿ to minimise

  f(x) = Σᵢ cᵢ Pᵢ xᵢ + λ (Σᵢ Pᵢ xᵢ − T)²

The second term penalises under- and over-delivery against the bid, because both create imbalance. This is a QUBO, the native input of quantum annealers and QAOA.

## Instances (synthetic, seeded, documented in code)

- **Portfolio mix:** heat pumps 0.02–0.10 MW, commercial HVAC 0.05–0.30 MW, industrial loads 0.20–0.80 MW, small batteries 0.10–0.50 MW.
- **Costs:** activation costs 5–120 €/MWh, depending on asset type.
- **Penalty:** λ is set so that missing the target by the smallest asset's power always costs more than any single activation.
- **Instances:** n ∈ {8, 10, 12, 14, 16} assets, 10 seeds each, 50 in total.
- **Large-scale check:** n up to 2,000 for the classical MILP.

## Solvers

1. **Exhaustive search** over all 2ⁿ states. This gives the ground truth for n ≤ 16.
2. **Simulated annealing** (classical heuristic), 1,000 sweeps, 20 restarts.
3. **QAOA**, depth p = 1, 2, 3, exact statevector simulation.
   - Angles are optimised with COBYLA on the energy expectation, warm-started layer by layer (INTERP).
   - Then 1,000 bitstrings are sampled and the best is kept. This is a noiseless, best case for QAOA.
4. **MILP** (scipy/HiGHS) on the constrained form: minimise cost subject to T ≤ Σ Pᵢ xᵢ ≤ T + 0.05. Run for scaling to n = 2,000.

## Metrics

- **Hit rate:** the share of instances where the solver's best solution equals the exhaustive optimum, within 1e-9 relative.
- **QAOA sampling probability:** the probability of sampling the optimum.
- **Approximation ratio:** f(best) / f(optimum).
- **Wall time.**

## Predictions

- **A:** simulated annealing hits the optimum in ≥ 90% of instances.
- **B:** QAOA (p ≤ 3, noiseless, 1,000 shots) has a lower hit rate than simulated annealing at n = 16. Its probability of sampling the optimum falls as n grows.
- **C:** the MILP solves n = 2,000 assets in under 1 second per instance.
- **Overall:** quantum optimisation is a **no-go for production today** on this problem. The deliverable is a benchmark harness that can be rerun as hardware improves.

Everything is reported whatever the outcome.
