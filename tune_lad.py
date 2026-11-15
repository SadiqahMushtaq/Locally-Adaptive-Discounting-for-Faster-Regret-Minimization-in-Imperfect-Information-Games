"""Tuning sweep: baselines + many LAD-CFR+ settings, run in parallel.

Every config runs `--iters` iterations on one game; exploitability of the average
policy is recorded at log-spaced checkpoints. Results go to
results/tuning/<game>.csv and a sorted summary is printed.

Usage:
    python tune_lad.py --game kuhn_poker  --iters 2000
    python tune_lad.py --game leduc_poker --iters 1000 --workers 8
    python tune_lad.py --game leduc_poker --only CFR+ LAD-geo-1.5-0.5
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import pyspiel
from open_spiel.python.algorithms import cfr, discounted_cfr, exploitability

from lad_cfr import LADCFRPlusSolver
from mix_cfr import MixCFRSolver
from pcfr_plus import PCFRPlusSolver
from run_baseline import checkpoints

GEO = dict(mapping="geometric")

# name -> (solver class, kwargs). LAD kwargs are documented in lad_cfr.py.
CONFIGS = {
    # ---- baselines
    "CFR+": (cfr.CFRPlusSolver, {}),
    "DCFR": (discounted_cfr.DCFRSolver, {}),
    "PCFR+": (PCFRPlusSolver, {}),
    "DCFR+": (LADCFRPlusSolver, dict(mode=1.0)),                   # RM+ with DCFR alpha=1.5 everywhere
    "DCFR+avg2": (LADCFRPlusSolver, dict(mode=1.0, avg_power=2)),
    "PDCFR+": (LADCFRPlusSolver, dict(mode=1.0, avg_power=2, predict=True)),
    # ---- original LAD-CFR+ (linear mapping: stable -> CFR+, volatile -> DCFR)
    "LAD-lin": (LADCFRPlusSolver, {}),
    "LAD-lin-global": (LADCFRPlusSolver, dict(mode="global")),
    "LAD-lin-lam0.5": (LADCFRPlusSolver, dict(lam=0.5)),
    "LAD-lin-lam0.99": (LADCFRPlusSolver, dict(lam=0.99)),
    "LAD-lin-p0.5": (LADCFRPlusSolver, dict(v_power=0.5)),
    "LAD-lin-a0.5": (LADCFRPlusSolver, dict(alpha=0.5)),
    "LAD-lin-avg2": (LADCFRPlusSolver, dict(avg_power=2)),
    # ---- round 2: controls/tuning around t^2 averaging
    "CFR+avg2": (LADCFRPlusSolver, dict(mode=0.0, avg_power=2)),   # CFR+ with t^2 averaging
    "LAD-lin-global-avg2": (LADCFRPlusSolver, dict(mode="global", avg_power=2)),
    "LAD-lin-lam0.99-avg2": (LADCFRPlusSolver, dict(lam=0.99, avg_power=2)),
    "LAD-lin-lam0.5-avg2": (LADCFRPlusSolver, dict(lam=0.5, avg_power=2)),
    "LAD-lin-p0.5-avg2": (LADCFRPlusSolver, dict(v_power=0.5, avg_power=2)),
    "LAD-lin-a1.0-avg2": (LADCFRPlusSolver, dict(alpha=1.0, avg_power=2)),
    "LAD-lin-a2.0-avg2": (LADCFRPlusSolver, dict(alpha=2.0, avg_power=2)),
    # ---- round 3: fair global control for the round-2 winner, longer memory
    "LAD-lin-lam0.99-global-avg2": (LADCFRPlusSolver, dict(lam=0.99, mode="global", avg_power=2)),
    "LAD-lin-lam0.999-avg2": (LADCFRPlusSolver, dict(lam=0.999, avg_power=2)),
    # ---- geometric mapping: everyone discounted like DCFR, volatile infosets more
    "LAD-geo-1.5-0.5": (LADCFRPlusSolver, dict(GEO, alpha_stable=1.5, alpha_volatile=0.5)),
    "LAD-geo-1.5-1.0": (LADCFRPlusSolver, dict(GEO, alpha_stable=1.5, alpha_volatile=1.0)),
    "LAD-geo-2.0-0.5": (LADCFRPlusSolver, dict(GEO, alpha_stable=2.0, alpha_volatile=0.5)),
    "LAD-geo-3.0-1.0": (LADCFRPlusSolver, dict(GEO, alpha_stable=3.0, alpha_volatile=1.0)),
    "LAD-geo-1.5-0.5-p0.5": (LADCFRPlusSolver, dict(GEO, alpha_stable=1.5, alpha_volatile=0.5, v_power=0.5)),
    "LAD-geo-1.5-0.5-avg2": (LADCFRPlusSolver, dict(GEO, alpha_stable=1.5, alpha_volatile=0.5, avg_power=2)),
    "LAD-geo-1.5-0.5-global": (LADCFRPlusSolver, dict(GEO, alpha_stable=1.5, alpha_volatile=0.5, mode="global")),
    "LAD-geo-2.0-1.0": (LADCFRPlusSolver, dict(GEO, alpha_stable=2.0, alpha_volatile=1.0)),
    "LAD-geo-3.0-1.0-lam0.99": (LADCFRPlusSolver, dict(GEO, alpha_stable=3.0, alpha_volatile=1.0, lam=0.99)),
    "LAD-geo-3.0-1.0-global": (LADCFRPlusSolver, dict(GEO, alpha_stable=3.0, alpha_volatile=1.0, mode="global")),
    "LAD-lin-lam0.99-global": (LADCFRPlusSolver, dict(lam=0.99, mode="global")),
    "LAD-geo-REVERSED":(LADCFRPlusSolver, dict(GEO, alpha_stable=0.5, alpha_volatile=1.5)),  # sanity: wrong direction
    # ---- on top of Predictive CFR+
    # ---- MIX-CFR: per-infoset selection among CFR+, DCFR, PCFR+ (mix_cfr.py)
    "MIX": (MixCFRSolver, {}),
    "MIX-global": (MixCFRSolver, dict(scope="global")),
    "MIX-prm": (MixCFRSolver, dict(meta="prm+")),
    "MIX-no-dcfr": (MixCFRSolver, dict(experts=("cfr+", "pcfr+"))),
    "MIX-no-pcfr": (MixCFRSolver, dict(experts=("cfr+", "dcfr"))),
    "MIX-no-cfr+": (MixCFRSolver, dict(experts=("dcfr", "pcfr+"))),
    "LAD-P-lin": (LADCFRPlusSolver, dict(avg_power=2, predict=True)),
    "LAD-P-geo-1.5-0.5": (LADCFRPlusSolver, dict(GEO, alpha_stable=1.5, alpha_volatile=0.5, avg_power=2, predict=True)),
    "LAD-P-geo-3.0-1.0": (LADCFRPlusSolver, dict(GEO, alpha_stable=3.0, alpha_volatile=1.0, avg_power=2, predict=True)),
    "LAD-P-lin-lam0.99": (LADCFRPlusSolver, dict(lam=0.99, avg_power=2, predict=True)),
    "LAD-P-geo-1.5-0.5-global":(LADCFRPlusSolver, dict(GEO, alpha_stable=1.5, alpha_volatile=0.5, avg_power=2, predict=True, mode="global")),
}


def run_one(job):
    game_name, label, n_iters = job
    cls, kwargs = CONFIGS[label]
    game = pyspiel.load_game(game_name)
    solver = cls(game, **kwargs)
    cps = set(checkpoints(n_iters))
    rows, train = [], 0.0
    for i in range(1, n_iters + 1):
        t0 = time.perf_counter()
        solver.evaluate_and_update_policy()
        train += time.perf_counter() - t0
        if i in cps:
            rows.append(dict(game=game_name, algo=label, iteration=i,
                             exploitability=exploitability.exploitability(game, solver.average_policy()),
                             train_seconds=round(train, 2)))
    print(f"  done {label:26s} expl {rows[-1]['exploitability']:.3e}  ({train:.0f}s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", help="subset of config names")
    ap.add_argument("--out", default="results/tuning")
    ap.add_argument("--tag", default="", help="suffix for the CSV name")
    args = ap.parse_args()

    labels = args.only or list(CONFIGS)
    jobs = [(args.game, lab, args.iters) for lab in labels]
    with Pool(args.workers) as pool:
        results = pool.map(run_one, jobs, chunksize=1)
    rows = [r for res in results for r in res]

    os.makedirs(args.out, exist_ok=True)
    safe = args.game.replace("(", "_").replace(")", "").replace(",", "_").replace("=", "")
    path = os.path.join(args.out, f"{safe}{'_' + args.tag if args.tag else ''}.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    final = {r["algo"]: r["exploitability"] for r in rows if r["iteration"] == args.iters}
    ref = final.get("CFR+")
    print(f"\n{args.game} @ {args.iters} iterations (sorted):")
    for lab, e in sorted(final.items(), key=lambda kv: kv[1]):
        rel = f"  ({ref / e:5.2f}x vs CFR+)" if ref else ""
        print(f"  {lab:26s} {e:.3e}{rel}")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
