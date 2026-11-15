"""Sanity checks for mix_cfr.py: with a single expert, MIX-CFR must reproduce
that expert's existing solver (up to floating-point rounding, because regrets
are summed in a different order)."""
import numpy as np
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability

from mix_cfr import MixCFRSolver
from pcfr_plus import PCFRPlusSolver


def same(game_name, ref, ours, n_iters):
    game = pyspiel.load_game(game_name)
    a, b = ref(game), ours(game)
    for _ in range(n_iters):
        a.evaluate_and_update_policy()
        b.evaluate_and_update_policy()
    diff = np.abs(a.average_policy().action_probability_array
                  - b.average_policy().action_probability_array).max()
    ea = exploitability.exploitability(game, a.average_policy())
    eb = exploitability.exploitability(game, b.average_policy())
    return diff, ea, eb


for g, n in [("kuhn_poker", 300), ("leduc_poker", 30)]:
    d, ea, eb = same(g, cfr.CFRPlusSolver,
                     lambda gm: MixCFRSolver(gm, experts=("cfr+",), avg_power=1), n)
    print(f"{g}: MIX[cfr+] vs CFR+   max|diff| {d:.1e}  expl {ea:.6e} vs {eb:.6e}")
    assert d < 1e-8
    d, ea, eb = same(g, PCFRPlusSolver,
                     lambda gm: MixCFRSolver(gm, experts=("pcfr+",), avg_power=2), n)
    print(f"{g}: MIX[pcfr+] vs PCFR+ max|diff| {d:.1e}  expl {ea:.6e} vs {eb:.6e}")
    assert d < 1e-8

game = pyspiel.load_game("kuhn_poker")
for scope in ("local", "global"):
    for meta in ("rm+", "prm+"):
        s = MixCFRSolver(game, scope=scope, meta=meta)
        for _ in range(2000):
            s.evaluate_and_update_policy()
        e = exploitability.exploitability(game, s.average_policy())
        w = {k: round(float(v), 2) for k, v in s.mean_weights().items()}
        print(f"kuhn 2000  MIX {scope:6s} {meta:4s} expl {e:.3e}  weights {w}")
print("all checks passed")
