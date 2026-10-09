"""Quantum vs classical for mFRR bid assembly (pre-registered in PREREGISTRATION_quantum.md).

    python quantum_bids.py      # about 10-20 minutes on a laptop; writes results/quantum/
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
from flexcast import qubo  # noqa: E402

OUT = ROOT / "results" / "quantum"
SIZES = (8, 10, 12, 14, 16)
SEEDS = range(10)
SHOTS = 1000
REL = 1e-9


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256((ROOT / "PREREGISTRATION_quantum.md").read_bytes()).hexdigest()
    rows = []
    for n in SIZES:
        for seed in SEEDS:
            inst = qubo.make_instance(n, seed=1000 * n + seed)
            x_opt, e_opt, t_ex, E = qubo.exhaustive(inst)
            hit = lambda e: abs(e - e_opt) <= REL * max(1.0, abs(e_opt))
            _, e_sa, t_sa = qubo.simulated_annealing(inst, seed=seed)
            t0 = time.perf_counter()
            qa = qubo.QAOA(inst, E).optimise(p_max=3, seed=seed)
            t_q = time.perf_counter() - t0
            opt_idx = int(np.argmin(E))
            row = {"n": n, "seed": seed, "f_opt": e_opt, "delivered_mw": float(x_opt @ inst.P),
                   "exhaustive_s": t_ex, "sa_hit": hit(e_sa), "sa_ratio": e_sa / e_opt, "sa_s": t_sa,
                   "qaoa_total_s": t_q}
            for p, (_, _, probs) in qa.items():
                j, e_q = qubo.sample_best(probs, E, SHOTS, seed=seed)
                row[f"qaoa_p{p}_p_opt"] = float(probs[opt_idx])
                row[f"qaoa_p{p}_hit"] = hit(e_q)
                row[f"qaoa_p{p}_ratio"] = e_q / e_opt
                row[f"qaoa_p{p}_expected_ratio"] = float(probs @ E) / e_opt
            row["uniform_p_opt"] = 1 / 2 ** n
            rows.append(row)
            print(f"n={n:2d} seed={seed} SA hit={row['sa_hit']} "
                  f"QAOA p3 hit={row['qaoa_p3_hit']} p_opt={row['qaoa_p3_p_opt']:.4f} "
                  f"(uniform {row['uniform_p_opt']:.5f})", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "instances.csv", index=False)

    agg = df.groupby("n").agg(
        sa_hit_rate=("sa_hit", "mean"),
        qaoa_p1_hit_rate=("qaoa_p1_hit", "mean"), qaoa_p2_hit_rate=("qaoa_p2_hit", "mean"),
        qaoa_p3_hit_rate=("qaoa_p3_hit", "mean"),
        qaoa_p3_p_opt_median=("qaoa_p3_p_opt", "median"), uniform_p_opt=("uniform_p_opt", "first"),
        qaoa_p3_best_of_1000_ratio_median=("qaoa_p3_ratio", "median"),
        qaoa_p3_expected_ratio_median=("qaoa_p3_expected_ratio", "median"),
        sa_seconds_median=("sa_s", "median"), qaoa_sim_seconds_median=("qaoa_total_s", "median"),
    )
    agg["qaoa_p3_amplification_vs_uniform"] = agg["qaoa_p3_p_opt_median"] / agg["uniform_p_opt"]
    agg.to_csv(OUT / "summary_by_n.csv")

    scale = []
    for n in (50, 200, 500, 1000, 2000):
        for seed in range(5):
            inst = qubo.make_instance(n, seed=7 * n + seed)
            res, t = qubo.milp_constrained(inst)
            scale.append({"n": n, "seed": seed, "status": res.status, "seconds": t,
                          "cost_eur_per_h": float(res.fun) if res.x is not None else np.nan})
    sc = pd.DataFrame(scale)
    sc.to_csv(OUT / "milp_scaling.csv", index=False)
    summary = {"preregistration_quantum_sha256": sha, "shots": SHOTS,
               "by_n": json.loads(agg.reset_index().to_json(orient="records")),
               "milp_scaling_max_seconds_by_n": sc.groupby("n")["seconds"].max().to_dict(),
               "milp_all_optimal": bool((sc["status"] == 0).all())}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    print(agg.round(4).to_string())
    print(sc.groupby("n")["seconds"].agg(["median", "max"]).round(4).to_string())


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = pd.read_csv(OUT / "summary_by_n.csv")
    sc = pd.read_csv(OUT / "milp_scaling.csv").groupby("n")["seconds"].median()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    ax = axes[0]
    ax.plot(agg.n, 100 * agg.sa_hit_rate, "o-", label="simulated annealing (classical)")
    for p, m in ((1, "s--"), (2, "^--"), (3, "D-")):
        ax.plot(agg.n, 100 * agg[f"qaoa_p{p}_hit_rate"], m, label=f"QAOA p={p}, best of {SHOTS} shots")
    ax.set_xlabel("assets (qubits)"); ax.set_ylabel("instances solved to optimum (%)"); ax.set_ylim(-5, 105)
    ax.legend(fontsize=7); ax.set_title("Who finds the best bid?", fontsize=10)
    ax = axes[1]
    ax.semilogy(agg.n, agg.qaoa_p3_p_opt_median, "D-", label="QAOA p=3")
    ax.semilogy(agg.n, agg.uniform_p_opt, "k:", label="random guessing (1/2^n)")
    ax.set_xlabel("assets (qubits)"); ax.set_ylabel("probability of sampling the optimum")
    ax.legend(fontsize=7); ax.set_title("QAOA lifts odds ~3x, but both decay", fontsize=10)
    ax = axes[2]
    ax.loglog(sc.index, sc.values, "o-", color="tab:green")
    ax.set_xlabel("assets"); ax.set_ylabel("median seconds to proven optimum")
    ax.set_title("Exact classical MILP (HiGHS)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "quantum_bids.png", dpi=150)
    plt.close(fig)
    print("wrote figures/quantum_bids.png")


if __name__ == "__main__":
    figure() if "--figure" in sys.argv else main()
