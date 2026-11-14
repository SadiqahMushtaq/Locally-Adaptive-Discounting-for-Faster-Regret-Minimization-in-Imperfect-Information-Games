"""Ablation: toggle each of the three CFR+ switches on OpenSpiel's _CFRSolver.

Answers "which CFR+ change matters most?". Writes results/ablation_<game>.csv and .png.

Usage:
    python ablation.py                                # Kuhn, 2000 iters
    python ablation.py --game leduc_poker --iters 300 # ~8 runs x 300 iters, slower
"""
import argparse
import csv
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability

from run_baseline import checkpoints


def label(alt, lin, rmp):
    parts = [n for n, on in (("alt", alt), ("lin", lin), ("RM+", rmp)) if on]
    return "+".join(parts) or "vanilla (simultaneous)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="kuhn_poker")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    game = pyspiel.load_game(args.game)
    cps = set(checkpoints(args.iters))
    rows = []
    for alt, lin, rmp in itertools.product([False, True], repeat=3):
        name = label(alt, lin, rmp)
        solver = cfr._CFRSolver(game, alternating_updates=alt,
                                linear_averaging=lin, regret_matching_plus=rmp)
        for i in range(1, args.iters + 1):
            solver.evaluate_and_update_policy()
            if i in cps:
                expl = exploitability.exploitability(game, solver.average_policy())
                rows.append(dict(game=args.game, config=name, alternating=alt,
                                 linear=lin, rm_plus=rmp, iteration=i,
                                 exploitability=expl))
        print(f"{name:24s} final expl {rows[-1]['exploitability']:.3e}", flush=True)

    stem = os.path.join(args.out, f"ablation_{args.game}")
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in dict.fromkeys(r["config"] for r in rows):
        pts = [r for r in rows if r["config"] == name]
        ax.plot([r["iteration"] for r in pts], [r["exploitability"] for r in pts],
                marker="o", ms=2, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("iterations")
    ax.set_ylabel("exploitability (average policy)")
    ax.set_title(f"CFR+ ablation: {args.game}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=150)
    print(f"\nSaved {stem}.csv and {stem}.png")


if __name__ == "__main__":
    main()
