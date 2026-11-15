"""Per-information-set selection of CFR update rules ("MIX-CFR").

Every information set I runs K local regret minimizers ("experts") side by side,
all fed the same counterfactual values:
    cfr+   Q = [Q + r_k]^+,                       sigma_k ~ [Q]^+
    dcfr   R = R + r_k, then R>0 *= t^a/(t^a+1), R<0 *= t^b/(t^b+1),  sigma_k ~ [R]^+
    pcfr+  Q = [Q + r_k]^+,                       sigma_k ~ [Q + r_k]^+   (prediction)
where r_k(a) = v(I,a) - <sigma_k, v(I,.)> is expert k's OWN instantaneous regret.

A meta learner (regret matching+ over the experts) mixes them:
    sigma_I = sum_k w_k sigma_k,   w ~ [W]^+,   W = [W + g]^+,   g_k = <sigma_k, r>
g_k is how much better expert k's strategy would have done than the mixture
(<sigma, r> = 0 for the mixture itself). Because r = r_k + g_k for every k,
the regret of the mixture at I is at most (regret of expert k at I) + (meta regret
against k) for every k, i.e. close to the best expert AT THAT INFORMATION SET.

scope="local":  one meta learner per information set (the method)
scope="global": one meta learner for the whole game (ablation)
meta="prm+":    predictive meta learner, w ~ [W + g_last]^+  (often faster)
"""
import numpy as np
from open_spiel.python.algorithms import cfr


def _rm(x):
    p = np.maximum(x, 0.0)
    s = p.sum()
    return p / s if s > 0 else np.full(len(x), 1.0 / len(x))


class MixCFRSolver(cfr._CFRSolver):  # pylint: disable=protected-access

    EXPERTS = ("cfr+", "dcfr", "pcfr+")

    def __init__(self, game, experts=EXPERTS, scope="local", meta="rm+",
                 avg_power=2, alpha=1.5, beta=0.0):
        super().__init__(game, alternating_updates=True, linear_averaging=True,
                         regret_matching_plus=False)
        assert all(e in self.EXPERTS for e in experts), experts
        assert scope in ("local", "global") and meta in ("rm+", "prm+")
        self._experts = tuple(experts)
        self._scope, self._meta = scope, meta
        self._avg_power, self._alpha, self._beta = avg_power, alpha, beta

        self._infosets_of = [set() for _ in range(self._num_players)]
        self._collect_owners(self._root_node)
        self._infosets_of = [sorted(s) for s in self._infosets_of]

        K = len(self._experts)
        self._st = {}
        for k, node in self._info_state_nodes.items():
            n = len(node.legal_actions)
            self._st[k] = dict(
                R=np.zeros((K, n)),                 # each expert's regret table
                sig=np.full((K, n), 1.0 / n),       # each expert's current strategy
                W=np.zeros(K), g=np.zeros(K))       # local meta learner
        self._gW = [np.zeros(K) for _ in range(self._num_players)]   # global meta
        self._gg = [np.zeros(K) for _ in range(self._num_players)]

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

    def meta_weights(self, info_state, player=None):
        """Current mixture weights over the experts at an information set."""
        if self._scope == "global":
            W, g = self._gW[player], self._gg[player]
        else:
            W, g = self._st[info_state]["W"], self._st[info_state]["g"]
        return _rm(W + g) if self._meta == "prm+" else _rm(W)

    def _update_expert(self, e, row, R, r_k, t):
        if e == "cfr+":
            R[row] = np.maximum(R[row] + r_k, 0.0)
            return _rm(R[row])
        if e == "pcfr+":
            R[row] = np.maximum(R[row] + r_k, 0.0)
            return _rm(R[row] + r_k)
        # dcfr
        R[row] = R[row] + r_k
        ta, tb = t ** self._alpha, t ** self._beta
        R[row] = np.where(R[row] > 0, R[row] * ta / (ta + 1), R[row] * tb / (tb + 1))
        return _rm(R[row])

    def evaluate_and_update_policy(self):
        self._iteration += 1
        t = self._iteration
        for player in range(self._num_players):
            keys = self._infosets_of[player]
            # 1) Zero the scratch regret table; the walk then fills it with r^t.
            for k in keys:
                self._info_state_nodes[k].cumulative_regret.clear()
            self._iteration = t ** self._avg_power   # average-strategy weight t^p
            self._compute_counterfactual_regret_for_player(
                self._root_node, policies=None,
                reach_probabilities=np.ones(self._num_players + 1), player=player)
            self._iteration = t

            # 2) Update every expert and compute meta gains g_k = <sigma_k, r>.
            g_sum = np.zeros(len(self._experts))
            for k in keys:
                node, st = self._info_state_nodes[k], self._st[k]
                r = np.array([node.cumulative_regret.get(a, 0.0) for a in node.legal_actions])
                g = st["sig"] @ r
                for i, e in enumerate(self._experts):
                    st["sig"][i] = self._update_expert(e, i, st["R"], r - g[i], t)
                if self._scope == "local":
                    st["W"] = np.maximum(st["W"] + g, 0.0)
                    st["g"] = g
                else:
                    g_sum += g
            if self._scope == "global":
                self._gW[player] = np.maximum(self._gW[player] + g_sum, 0.0)
                self._gg[player] = g_sum

            # 3) Play the mixture.
            for k in keys:
                node, st = self._info_state_nodes[k], self._st[k]
                sigma = self.meta_weights(k, player) @ st["sig"]
                state_policy = self._current_policy.policy_for_key(k)
                for a, p in zip(node.legal_actions, sigma):
                    state_policy[a] = p

    def mean_weights(self):
        """Average meta weight per expert over all information sets (for logging)."""
        ws = []
        for p in range(self._num_players):
            for k in self._infosets_of[p]:
                ws.append(self.meta_weights(k, p))
        return dict(zip(self._experts, np.mean(ws, axis=0)))
