"""Sanity checks for pcfr_plus.py.

1. predict=False, avg_power=1 must give EXACTLY OpenSpiel's CFR+ average policy.
2. Quick exploitability comparison: CFR+ vs PCFR+ (quadratic averaging, as in the
   paper) vs PCFR+ with linear averaging (isolates the effect of the prediction).

Usage:
    python test_pcfr_plus.py                    # Kuhn 2000 iters, Leduc 100 iters
    python test_pcfr_plus.py --iters-leduc 1000
"""
import argparse

import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability

from pcfr_plus import PCFRPlusSolver


def check_matches_cfr_plus(game_name, n_iters):
    game = pyspiel.load_game(game_name)
    ref = cfr.CFRPlusSolver(game)
    ours = PCFRPlusSolver(game, predict=False, avg_power=1)
    for _ in range(n_iters):
        ref.evaluate_and_update_policy()
        ours.evaluate_and_update_policy()
    same = np.array_equal(ref.average_policy().action_probability_array,
                          ours.average_policy().action_probability_array)
    print(f"[check 1] {game_name}: no prediction + linear avg identical to CFR+ "
          f"after {n_iters} iters: {same}")
    assert same


def compare(game_name, n_iters):
    game = pyspiel.load_game(game_name)
    solvers = {"CFR+": cfr.CFRPlusSolver(game),
               "PCFR+": PCFRPlusSolver(game),
               "PCFR+ lin": PCFRPlusSolver(game, avg_power=1)}
    for label, s in solvers.items():
        for _ in range(n_iters):
            s.evaluate_and_update_policy()
        expl = exploitability.exploitability(game, s.average_policy())
        print(f"[check 2] {game_name:12s} {label:9s} iter {n_iters}  expl {expl:.3e}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters-kuhn", type=int, default=2000)
    ap.add_argument("--iters-leduc", type=int, default=100)
    args = ap.parse_args()

    check_matches_cfr_plus("kuhn_poker", 500)
    check_matches_cfr_plus("leduc_poker", 20)
    compare("kuhn_poker", args.iters_kuhn)
    compare("leduc_poker", args.iters_leduc)


if __name__ == "__main__":
    main()
