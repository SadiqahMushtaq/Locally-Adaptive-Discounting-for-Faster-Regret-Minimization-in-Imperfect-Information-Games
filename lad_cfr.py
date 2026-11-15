"""LAD-CFR+: Locally Adaptive Discounted CFR+.

CFR+ keeps all past regret:      Q^t = [ Q^{t-1} + r^t ]^+
DCFR shrinks it by a fixed rule: Q^t = [ d_t * Q^{t-1} + r^t ]^+   (same d_t everywhere)
LAD-CFR+ lets every information set I pick its own discount gamma_I from the
"volatility" v_I of its recent regrets:

    Q^t_I = [ gamma_I * Q^{t-1}_I + r^t_I ]^+
    v_I   = EMA ||r^t - r^{t-1}||^2 / (2 * EMA(||r^t||^2 + ||r^{t-1}||^2))   in [0, 1]
            v = 0: regrets point the same way every iteration (stable)
            v = 1: regrets flip sign every iteration (oscillating)

Two ways to turn v_I into gamma_I (d(a, s) = s^a / (s^a + 1), s = t - 1):

  mapping="linear" (original):  gamma_I = 1 - (1 - d(alpha, s)) * v_I^p
        stable -> 1 (CFR+),  volatile -> d(alpha) (DCFR-style)
  mapping="geometric":          gamma_I = d(alpha_stable, s)^(1 - v_I^p) * d(alpha_volatile, s)^(v_I^p)
        stable -> DCFR discount with alpha_stable, volatile -> stronger discount
        with alpha_volatile (smaller alpha = stronger discount).
        alpha_stable=None means "no discount" for stable infosets.

Modes:
    mode="local"   one volatility per information set (LAD-CFR+, our method)
    mode="global"  one volatility shared by the whole game (game-wide ablation)
    mode=<number>  fixed volatility everywhere. mode=0.0 is exactly CFR+;
                   mode=1.0 with mapping="linear" is DCFR's positive discount with RM+.

Other options:
    lam        EMA memory (0.9 = roughly the last 10 observations)
    v_power    p above; p < 1 makes small volatilities count more
    avg_power  average-strategy weight t^avg_power (1 = CFR+ linear, 2 = DCFR/PCFR+)
    predict    play from [Q + r]^+ instead of [Q]^+ (Predictive CFR+ on top)

Defaults reproduce the first version of the method (linear mapping, alpha=1.5,
lam=0.9, p=1, linear averaging, no prediction).
"""
import math

import numpy as np
from open_spiel.python.algorithms import cfr


def _d(alpha, s):
    """DCFR discount s^a / (s^a + 1); alpha=None or inf means no discount."""
    if alpha is None or math.isinf(alpha):
        return 1.0
    if s <= 0:
        return 0.0
    x = s ** alpha
    return x / (x + 1.0)


class LADCFRPlusSolver(cfr._CFRSolver):  # pylint: disable=protected-access

    def __init__(self, game, mode="local", mapping="linear", alpha=1.5,
                 alpha_stable=None, alpha_volatile=0.5, lam=0.9, v_power=1.0,
                 avg_power=1, predict=False):
        super().__init__(game,
                         alternating_updates=True,
                         linear_averaging=True,
                         regret_matching_plus=True)
        if not (mode in ("local", "global") or isinstance(mode, (int, float))):
            raise ValueError(f"unknown mode {mode!r}")
        if mapping not in ("linear", "geometric"):
            raise ValueError(f"unknown mapping {mapping!r}")
        self._mode = mode
        self._mapping = mapping
        self._alpha = alpha
        self._alpha_stable = alpha_stable
        self._alpha_volatile = alpha_volatile
        self._lam = lam
        self._v_power = v_power
        self._avg_power = avg_power
        self._predict = predict

        # Which information sets belong to which player (we discount only the
        # player whose regrets are about to be updated).
        self._infosets_of = [set() for _ in range(self._num_players)]
        self._collect_owners(self._root_node)
        self._infosets_of = [sorted(s) for s in self._infosets_of]

        # Volatility statistics.
        self._prev_r = {}    # info_state -> last non-zero instantaneous regret vector
        self._ema_num = {}   # info_state -> EMA of ||r^t - r^{t-1}||^2
        self._ema_den = {}   # info_state -> EMA of ||r^t||^2 + ||r^{t-1}||^2
        self._g_num = 0.0    # same two EMAs, pooled over the whole game
        self._g_den = 0.0

    # ------------------------------------------------------------------ setup
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

    # ------------------------------------------------------------- volatility
    def volatility(self, info_state):
        """Current volatility v_I in [0, 1] used to set the discount at I."""
        if not isinstance(self._mode, str):
            return float(self._mode)
        if self._mode == "global":
            num, den = self._g_num, self._g_den
        else:
            num, den = self._ema_num.get(info_state, 0.0), self._ema_den.get(info_state, 0.0)
        if den <= 0.0:
            return 0.0  # no evidence yet
        return min(1.0, num / (2.0 * den))

    def volatilities(self):
        """{info_state: v_I} for every information set (for logging/plots)."""
        return {k: self.volatility(k) for k in self._info_state_nodes}

    def discount(self, info_state, s):
        """gamma_I applied to I's cumulative regret before iteration s + 1."""
        v = self.volatility(info_state) ** self._v_power
        if self._mapping == "linear":
            return 1.0 - (1.0 - _d(self._alpha, s)) * v
        # geometric blend of the two DCFR discounts
        g_s, g_v = _d(self._alpha_stable, s), _d(self._alpha_volatile, s)
        if g_s == 0.0 or g_v == 0.0:
            return 0.0  # only at s = 0, when all regrets are still 0 anyway
        return g_s ** (1.0 - v) * g_v ** v

    def _observe(self, info_state, r):
        """Feed one instantaneous regret vector r^t_I into the statistics.

        Returns this iteration's (num, den) contribution, used by the global mode.
        An all-zero r means I was not reached this iteration (no information),
        so it is skipped instead of being treated as a regret change.
        """
        rr = float(r @ r)
        if rr == 0.0:
            return 0.0, 0.0
        prev = self._prev_r.get(info_state)
        self._prev_r[info_state] = r
        if prev is None:
            return 0.0, 0.0  # need two observations to measure a change
        diff = r - prev
        num = float(diff @ diff)
        den = rr + float(prev @ prev)
        lam = self._lam
        self._ema_num[info_state] = lam * self._ema_num.get(info_state, 0.0) + (1 - lam) * num
        self._ema_den[info_state] = lam * self._ema_den.get(info_state, 0.0) + (1 - lam) * den
        return num, den

    # --------------------------------------------------------------- one step
    def evaluate_and_update_policy(self):
        self._iteration += 1
        t = self._iteration
        # DCFR multiplies the regret accumulated after iteration t-1 by d_{t-1}
        # before iteration t's regret is added; we use the same timing.
        s = t - 1

        pooled_num = pooled_den = 0.0
        for player in range(self._num_players):
            keys = self._infosets_of[player]

            # 1) Discount this player's cumulative regrets, each with its own gamma_I,
            #    and remember the result so we can recover r^t after the tree walk.
            before = {}
            for k in keys:
                node = self._info_state_nodes[k]
                reg = node.cumulative_regret
                gamma = self.discount(k, s)
                if gamma != 1.0:
                    for a in node.legal_actions:
                        reg[a] *= gamma
                before[k] = np.array([reg[a] for a in node.legal_actions])

            # 2) Standard CFR tree walk: adds r^t_I to cumulative_regret and
            #    self._iteration * reach * sigma to cumulative_policy. Setting
            #    self._iteration = t^p for the walk gives averaging weight t^p.
            self._iteration = t ** self._avg_power
            self._compute_counterfactual_regret_for_player(
                self._root_node,
                policies=None,
                reach_probabilities=np.ones(self._num_players + 1),
                player=player)
            self._iteration = t

            # 3) r^t_I = (after walk) - (before walk); update volatility statistics.
            r = {}
            for k in keys:
                node = self._info_state_nodes[k]
                after = np.array([node.cumulative_regret[a] for a in node.legal_actions])
                r[k] = after - before[k]
                num, den = self._observe(k, r[k])
                pooled_num += num
                pooled_den += den

            # 4) RM+ clip, then the new current policy.
            cfr._apply_regret_matching_plus_reset(self._info_state_nodes)
            if not self._predict:
                cfr._update_current_policy(self._current_policy, self._info_state_nodes)
            else:
                # Predictive: sigma proportional to [Q + r]^+, this player only.
                for k in keys:
                    node = self._info_state_nodes[k]
                    target = np.array([node.cumulative_regret[a]
                                       for a in node.legal_actions]) + r[k]
                    probs = cfr._regret_matching(
                        dict(zip(node.legal_actions, target)), node.legal_actions)
                    state_policy = self._current_policy.policy_for_key(k)
                    for a, p in probs.items():
                        state_policy[a] = p

        if pooled_den > 0.0:
            lam = self._lam
            self._g_num = lam * self._g_num + (1 - lam) * pooled_num
            self._g_den = lam * self._g_den + (1 - lam) * pooled_den


class LADCFRPlusGlobalSolver(LADCFRPlusSolver):
    """Ablation: same rule, but one game-wide volatility instead of one per infoset."""

    def __init__(self, game, **kwargs):
        super().__init__(game, mode="global", **kwargs)
