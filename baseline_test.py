"""Step 3: smoke test. If exploitability drops toward 0, your install works."""
import pyspiel
from open_spiel.python.algorithms import cfr, exploitability

game = pyspiel.load_game("kuhn_poker")
solver = cfr.CFRPlusSolver(game)

for i in range(1, 1001):
    solver.evaluate_and_update_policy()
    if i % 100 == 0:
        expl = exploitability.exploitability(game, solver.average_policy())
        print(f"iter {i:5d}  exploitability {expl:.6f}")
