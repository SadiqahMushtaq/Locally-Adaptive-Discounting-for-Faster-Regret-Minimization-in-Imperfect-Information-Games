"""Predictive CFR+ (Farina, Kroer & Sandholm, AAAI 2021).

Regrets are accumulated exactly as in CFR+:
    Q^t_I = [ Q^{t-1}_I + r^t_I ]^+
but the strategy played next is computed from Q PLUS a guess of the next
instantaneous regret. The guess is simply the last one, r^t_I:
    sigma^{t+1}_I(a)  proportional to  [ Q^t_I(a) + r^t_I(a) ]^+

The original paper averages strategies with weight t^2 ("quadratic averaging");
avg_power=2 reproduces that, avg_power=1 gives CFR+'s linear averaging.
With predict=False and avg_power=1 this solver is exactly CFR+ (tested).
"""
import numpy as np
from open_spiel.python.algorithms import cfr


class PCFRPlusSolver(cfr._CFRSolver):  # pylint: disable=protected-access

    def __init__(self, game, predict=True, avg_power=2):
        super().__init__(game,
                         alternating_updates=True,
                         linear_averaging=True,
                         regret_matching_plus=True)
        self._predict = predict
        self._avg_power = avg_power
        self._infosets_of = [set() for _ in range(self._num_players)]
        self._collect_owners(self._root_node)
        self._infosets_of = [sorted(s) for s in self._infosets_of]

    def _collect_owners(self, state):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                self._collect_owners(state.child(action))
            return
        player = state.current_player()
        self._infosets_of[player].add(state.information_state_string(player))
        for action in state.legal_actions():
            self._collect_owners(state.child(action))

    def evaluate_and_update_policy(self):
        self._iteration += 1
        t = self._iteration
        for player in range(self._num_players):
            keys = self._infosets_of[player]

            # 1) Remember Q^{t-1} so we can recover r^t after the walk.
            before = {}
            for k in keys:
                node = self._info_state_nodes[k]
                before[k] = np.array([node.cumulative_regret[a] for a in node.legal_actions])

            # 2) Standard tree walk. OpenSpiel's linear averaging adds
            #    self._iteration * reach * sigma to the average policy, so we
            #    temporarily set it to t^p to get weight t^p (p=2: quadratic).
            self._iteration = t ** self._avg_power
            self._compute_counterfactual_regret_for_player(
                self._root_node,
                policies=None,
                reach_probabilities=np.ones(self._num_players + 1),
                player=player)
            self._iteration = t

            # 3) r^t = (after walk, before clipping) - Q^{t-1}.
            r = {}
            for k in keys:
                node = self._info_state_nodes[k]
                after = np.array([node.cumulative_regret[a] for a in node.legal_actions])
                r[k] = after - before[k]

            # 4) RM+ clip: Q^t = [Q^{t-1} + r^t]^+.
            cfr._apply_regret_matching_plus_reset(self._info_state_nodes)

            # 5) New strategy for this player from Q^t + prediction r^t.
            #    (Only this player's info sets change; the others keep theirs.)
            for k in keys:
                node = self._info_state_nodes[k]
                target = np.array([node.cumulative_regret[a] for a in node.legal_actions])
                if self._predict:
                    target = target + r[k]
                probs = cfr._regret_matching(
                    dict(zip(node.legal_actions, target)), node.legal_actions)
                state_policy = self._current_policy.policy_for_key(k)
                for a, p in probs.items():
                    state_policy[a] = p
