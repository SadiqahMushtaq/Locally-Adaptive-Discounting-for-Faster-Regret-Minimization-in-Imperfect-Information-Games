# Step 5: Reading OpenSpiel's `cfr.py`

Checked against the installed version: open_spiel 2.0.2, Python 3.14, file
`~/venvs/cfr/lib/python3.14/site-packages/open_spiel/python/algorithms/cfr.py`.
To view it: `python -c "import open_spiel.python.algorithms.cfr as m; print(m.__file__)"`.

## Things to find

- [x] `_InfoStateNode` (L35-50): per-infoset storage. Holds `legal_actions`,
      `index_in_tabular_policy`, `cumulative_regret` and `cumulative_policy` (both
      `defaultdict(float)` keyed by action). The current policy is NOT stored here; it lives
      in the solver's `TabularPolicy` (`self._current_policy`).
- [x] Base class `_CFRSolverBase.__init__` (L144-193) takes
      `alternating_updates, linear_averaging, regret_matching_plus`.
      `_CFRSolver` (L402) adds `evaluate_and_update_policy`.
- [x] `CFRPlusSolver` (L470-511): all three switches on.
      **`CFRSolver` (L514-528): RM+ off, linear averaging off, but alternating updates ON**
      (the docstring says alternating is "far more efficient"). So the baseline CFR vs CFR+
      comparison differs in only two of the three changes.
- [x] `evaluate_and_update_policy()` (L446-467). With alternating updates it loops over
      players: walk the tree updating only that player, apply the RM+ reset if enabled, then
      recompute the current policy, so player 1 sees player 0's new policy in the same
      iteration. With simultaneous updates there is one walk with `player=None`, then a single
      reset and policy update.
- [x] Tree walk `_compute_counterfactual_regret_for_player` (L258-363). Regret is accumulated
      at L352-356: `cfr_regret = cf_reach * (child_value - state_value)`, then
      `cumulative_regret[a] += cfr_regret`. Infosets of other players are skipped at L343.
- [x] Regret matching `_regret_matching` (L375-399): positive part of cumulative regret,
      normalized, or uniform if all are ≤ 0. RM+ is NOT in here. It is a separate pass,
      `_apply_regret_matching_plus_reset` (L53-75), that clips cumulative regrets to 0 after
      each tree walk. (It has to happen per infoset after the walk, because the walk visits
      histories, and one infoset gets several additions.)
- [x] Average policy accumulation (L357-361): `cumulative_policy[a] += w * reach_prob * pi(a)`
      with `w = self._iteration` under linear averaging, else `w = 1`. It is normalized lazily
      in `_update_average_policy` (L100-123), which `average_policy()` calls.
- [x] `average_policy()` (L238-256) is what converges to Nash; measure exploitability on it.
      `current_policy()` (L227) need not converge for CFR/CFR+.

## Map the three CFR+ changes to code

| CFR+ change | Where it lives | What differs from vanilla CFR |
|---|---|---|
| Regret matching+ | `_apply_regret_matching_plus_reset` L53-75, called at L456-457 / L465-466 | cumulative regret floored at 0 after each player's walk |
| Alternating updates | `evaluate_and_update_policy` L449-458; skip at L342-344 | **on in BOTH solvers here** |
| Linear averaging | L357-361 (`self._iteration * reach_prob * action_prob`) | average policy weighted by iteration t |

## Where my modification would go

The hook is `_update_current_policy` (L78-97) / `_regret_matching` (L375). Predictive CFR+
(Farina, Kroer & Sandholm 2021) changes the step that turns regrets into the next policy:
instead of `RM+(R_t)`, play `RM+(R_t + m_{t+1})`, where the prediction `m_{t+1}` is the most
recent instantaneous regret `r_t`. To do this:

1. In `my_cfr.py`, subclass `_CFRSolver`. Add a `last_regret` dict to each node (or keep a
   parallel dict keyed by info state).
2. Override `_compute_counterfactual_regret_for_player` (copy it) so that it ALSO accumulates
   this iteration's instantaneous regret into `last_regret` (reset it before each walk).
3. Keep `cumulative_regret` floored at 0 (RM+ reset as before), but compute the *played*
   policy from `max(R_t + r_t, 0)` instead of `R_t`.
4. Usual DCFR/PCFR+ averaging is quadratic (w = t^2). Try both t and t^2.

Do NOT edit the installed package.

## Answers

1. **Current or average policy during play?** Each iteration's tree walk uses the CURRENT
   policy (`_get_infostate_policy` reads `self._current_policy`, L365-372). The average is
   only what you evaluate.
2. **Which change matters most?** Run the ablation (see `ablation.py`). Alternating updates
   are already on for both baselines.
3. **Info states:** Kuhn = 12, Leduc = 936 (`len(TabularPolicy(game).state_lookup)`).
