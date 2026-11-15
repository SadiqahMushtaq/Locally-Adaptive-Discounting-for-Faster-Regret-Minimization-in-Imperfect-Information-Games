"""Paper figures and tables from the saved CSVs (no new experiments).

Reads results/mix/*.csv (written by tune_lad.py) and produces
    results/figures/convergence.{png,pdf}   exploitability vs iterations, 5 games
    paper/figures/convergence.pdf           same, for LaTeX
and prints the final-iteration table (exploitability and training seconds).
"""
import csv
import glob
import os
import shutil
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

GAMES = [  # (csv stem, title)
    ("kuhn_poker", "Kuhn poker"),
    ("leduc_poker", "Leduc poker"),
    ("liars_dice_dice_sides3", "Liar's dice (3 faces)"),
    ("liars_dice_dice_sides4", "Liar's dice (4 faces)"),
    ("turn_based_simultaneous_game_gamegoofspiel_num_cards4_imp_infoTrue_points_orderdescending",
     "Goofspiel (4 cards)"),
]
ALGOS = [  # (name in CSV, label in paper)
    ("CFR+", "CFR+"),
    ("DCFR", "DCFR"),
    ("PCFR+", "PCFR+"),
    ("LAD-lin-lam0.99-avg2", "LAD-CFR+"),
    ("MIX-global", "MIX-CFR (global)"),
    ("MIX", "MIX-CFR (local)"),
]


def load(stem):
    data = defaultdict(list)
    with open(os.path.join("results", "mix", stem + ".csv")) as f:
        for r in csv.DictReader(f):
            data[r["algo"]].append((int(r["iteration"]), float(r["exploitability"]),
                                    float(r["train_seconds"])))
    return data


def main():
    os.makedirs("results/figures", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    axes = axes.ravel()
    print(f"{'game':24s}" + "".join(f"{lab:>18s}" for _, lab in ALGOS))
    for ax, (stem, title) in zip(axes, GAMES):
        data = load(stem)
        cells = []
        for name, label in ALGOS:
            pts = sorted(data[name])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], label=label,
                    lw=2.2 if name == "MIX" else 1.3,
                    ls="--" if name == "MIX-global" else "-")
            it, e, sec = pts[-1]
            cells.append(f"{e:.2e} ({sec:4.0f}s)")
        print(f"{title:24s}" + "".join(f"{c:>18s}" for c in cells))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("iterations")
        ax.set_ylabel("exploitability")
        ax.grid(True, which="both", alpha=0.3)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc="center", fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig("results/figures/convergence.png", dpi=150)
    fig.savefig("results/figures/convergence.pdf")
    os.makedirs("paper/figures", exist_ok=True)
    shutil.copy("results/figures/convergence.pdf", "paper/figures/convergence.pdf")
    print("Saved results/figures/convergence.{png,pdf} and paper/figures/convergence.pdf")


if __name__ == "__main__":
    main()
