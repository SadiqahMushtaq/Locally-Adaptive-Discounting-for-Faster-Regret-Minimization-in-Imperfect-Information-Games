"""Sanity checks for lad_cfr.py.

1. mode=0.0 (volatility forced to 0) must give EXACTLY the same average policy as
   OpenSpiel's CFR+ -> proves the only thing we changed is the discount.
2. Volatilities stay in [0, 1].
3. Quick exploitability comparison: CFR+ vs LAD-CFR+ (local) vs LAD-CFR+ (global).

Usage:
    python test_lad_cfr.py                    # Kuhn 2000 iters, Leduc 100 iters
    python test_lad_cfr.py --iters-leduc 1000
"""
import argparse

import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability

from lad_cfr import LADCFRPlusGlobalSolver, LADCFRPlusSolver


def check_matches_cfr_plus(game_name, n_iters):
    game = pyspiel.load_game(game_name)
    ref = cfr.CFRPlusSolver(game)
    ours = LADCFRPlusSolver(game, mode=0.0)
    for _ in range(n_iters):
        ref.evaluate_and_update_policy()
        ours.evaluate_and_update_policy()
    same = np.array_equal(ref.average_policy().action_probability_array,
                          ours.average_policy().action_probability_array)
    print(f"[check 1] {game_name}: v=0 identical to CFR+ after {n_iters} iters: {same}")
    assert same


def compare(game_name, n_iters):
    game = pyspiel.load_game(game_name)
    solvers = {"CFR+": cfr.CFRPlusSolver(game),
               "LAD-CFR+": LADCFRPlusSolver(game),
               "LAD-global": LADCFRPlusGlobalSolver(game)}
    for label, s in solvers.items():
        for _ in range(n_iters):
            s.evaluate_and_update_policy()
        expl = exploitability.exploitability(game, s.average_policy())
        line = f"[check 3] {game_name:12s} {label:10s} iter {n_iters}  expl {expl:.3e}"
        if isinstance(s, LADCFRPlusSolver):
            v = np.array(list(s.volatilities().values()))
            assert np.all((v >= 0) & (v <= 1)), "volatility out of [0,1]"
            line += f"   v: mean {v.mean():.2f}  min {v.min():.2f}  max {v.max():.2f}"
        print(line, flush=True)
    print("[check 2] all volatilities in [0, 1]")


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
