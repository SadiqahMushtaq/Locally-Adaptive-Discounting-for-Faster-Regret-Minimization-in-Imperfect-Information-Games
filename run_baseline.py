"""Step 6: reproduce the baseline. CFR vs CFR+ vs DCFR on Kuhn and Leduc poker.

Records exploitability of the AVERAGE policy at log-spaced iteration counts,
saves a CSV and a log-log plot into ./results/.

Usage:
    python run_baseline.py                       # defaults
    python run_baseline.py --iters-leduc 2000    # longer Leduc run
"""
import argparse
import csv
import os
import time

import matplotlib

matplotlib.use("Agg")  # no display needed (works in WSL)
import matplotlib.pyplot as plt
import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr, discounted_cfr, exploitability

# DCFRSolver defaults to the paper's recommended alpha=3/2, beta=0, gamma=2.
SOLVERS = {
    "CFR": cfr.CFRSolver,
    "CFR+": cfr.CFRPlusSolver,
    "DCFR": discounted_cfr.DCFRSolver,
}


def checkpoints(n_iters, n_points=30):
    """Log-spaced iteration counts in [1, n_iters]."""
    pts = np.logspace(0, np.log10(n_iters), n_points)
    return sorted({int(round(p)) for p in pts} | {n_iters})


def run(game_name, n_iters):
    game = pyspiel.load_game(game_name)
    cps = set(checkpoints(n_iters))
    rows = []
    for label, solver_cls in SOLVERS.items():
        solver = solver_cls(game)
        train_time = 0.0
        for i in range(1, n_iters + 1):
            t0 = time.perf_counter()
            solver.evaluate_and_update_policy()
            train_time += time.perf_counter() - t0
            if i in cps:
                expl = exploitability.exploitability(game, solver.average_policy())
                rows.append(
                    dict(game=game_name, algo=label, iteration=i,
                         exploitability=expl, train_seconds=train_time)
                )
                print(f"{game_name:12s} {label:4s} iter {i:6d}  expl {expl:.3e}  "
                      f"({train_time:.1f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters-kuhn", type=int, default=2000)
    ap.add_argument("--iters-leduc", type=int, default=1000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_rows = []
    all_rows += run("kuhn_poker", args.iters_kuhn)
    all_rows += run("leduc_poker", args.iters_leduc)

    csv_path = os.path.join(args.out, "baseline_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, game_name in zip(axes, ["kuhn_poker", "leduc_poker"]):
        for label in SOLVERS:
            pts = [r for r in all_rows if r["game"] == game_name and r["algo"] == label]
            ax.plot([r["iteration"] for r in pts],
                    [r["exploitability"] for r in pts], marker="o", ms=3, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("iterations")
        ax.set_ylabel("exploitability (average policy)")
        ax.set_title(game_name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    png_path = os.path.join(args.out, "baseline_exploitability.png")
    fig.savefig(png_path, dpi=150)
    print(f"\nSaved {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
