import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from poke_env import (
    AccountConfiguration,
    MaxBasePowerPlayer,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle import AbstractBattle
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import ObsType
from poke_env.player.player import Player
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType

from showdown_gym.base_environment import BaseShowdownEnv


# A simplified version of simple heuristics player that doesn't do hazards
# class ModifiedSimpleHeuristicsPlayer(Player):
#     SPEED_TIER_COEFICIENT = 0.1
#     HP_FRACTION_COEFICIENT = 0.4
#     SWITCH_OUT_MATCHUP_THRESHOLD = -2

#     def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
#         score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
#         score -= max(
#             [mon.damage_multiplier(t) for t in opponent.types if t is not None]
#         )
#         if mon.base_stats["spe"] > opponent.base_stats["spe"]:
#             score += self.SPEED_TIER_COEFICIENT
#         elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
#             score -= self.SPEED_TIER_COEFICIENT

#         score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
#         score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

#         return score

#     def _should_dynamax(self, battle: AbstractBattle, n_remaining_mons: int):
#         if battle.can_dynamax:
#             # Last full HP mon
#             if (
#                 len([m for m in battle.team.values() if m.current_hp_fraction == 1])
#                 == 1
#                 and battle.active_pokemon.current_hp_fraction == 1
#             ):
#                 return True
#             # Matchup advantage and full hp on full hp
#             if (
#                 self._estimate_matchup(
#                     battle.active_pokemon, battle.opponent_active_pokemon
#                 )
#                 > 0
#                 and battle.active_pokemon.current_hp_fraction == 1
#                 and battle.opponent_active_pokemon.current_hp_fraction == 1
#             ):
#                 return True
#             if n_remaining_mons == 1:
#                 return True
#         return False

#     def _should_switch_out(self, battle: AbstractBattle):
#         active = battle.active_pokemon
#         opponent = battle.opponent_active_pokemon
#         # If there is a decent switch in...
#         if [
#             m
#             for m in battle.available_switches
#             if self._estimate_matchup(m, opponent) > 0
#         ]:
#             # ...and a 'good' reason to switch out
#             if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
#                 return True
#             if (
#                 active.boosts["atk"] <= -3
#                 and active.stats["atk"] >= active.stats["spa"]
#             ):
#                 return True
#             if (
#                 active.boosts["spa"] <= -3
#                 and active.stats["atk"] <= active.stats["spa"]
#             ):
#                 return True
#             if (
#                 self._estimate_matchup(active, opponent)
#                 < self.SWITCH_OUT_MATCHUP_THRESHOLD
#             ):
#                 return True
#         return False

#     def _stat_estimation(self, mon: Pokemon, stat: str):
#         # Stats boosts value
#         if mon.boosts[stat] > 1:
#             boost = (2 + mon.boosts[stat]) / 2
#         else:
#             boost = 2 / (2 - mon.boosts[stat])
#         return ((2 * mon.base_stats[stat] + 31) + 5) * boost

#     def choose_move(self, battle: AbstractBattle):
#         if isinstance(battle, DoubleBattle):
#             return self.choose_random_doubles_move(battle)

#         # Main mons shortcuts
#         active = battle.active_pokemon
#         opponent = battle.opponent_active_pokemon

#         if active is None or opponent is None:
#             return self.choose_random_move(battle)

#         # Rough estimation of damage ratio
#         physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(
#             opponent, "def"
#         )
#         special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(
#             opponent, "spd"
#         )

#         if battle.available_moves and (
#             not self._should_switch_out(battle) or not battle.available_switches
#         ):
#             n_remaining_mons = len(
#                 [m for m in battle.team.values() if m.fainted is False]
#             )

#             move = max(
#                 battle.available_moves,
#                 key=lambda m: m.base_power
#                 * (1.5 if m.type in active.types else 1)
#                 * (
#                     physical_ratio
#                     if m.category == MoveCategory.PHYSICAL
#                     else special_ratio
#                 )
#                 * m.accuracy
#                 * m.expected_hits
#                 * opponent.damage_multiplier(m),
#             )
#             return self.create_order(
#                 move, dynamax=self._should_dynamax(battle, n_remaining_mons)
#             )

#         if battle.available_switches:
#             switches: List[Pokemon] = battle.available_switches
#             return self.create_order(
#                 max(
#                     switches,
#                     key=lambda s: self._estimate_matchup(s, opponent),
#                 )
#             )

#         return self.choose_random_move(battle)


# =============== Safe helpers ===============
def _safe_priority(m: Move) -> float:
    try:
        return float(m.priority)
    except Exception:
        return 0.0


def _safe_accuracy(m: Move) -> float:
    try:
        acc = m.accuracy
        if acc is None:
            return 1.0
        acc = float(acc)
        if acc > 1.0:
            acc /= 100.0
        return float(np.clip(acc, 0.0, 1.0))
    except Exception:
        return 1.0


def _safe_base_power(m: Move) -> float:
    try:
        bp = float(getattr(m, "base_power", 0.0))
        return float(np.clip(bp, 0.0, 200.0))
    except Exception:
        return 0.0


def _safe_expected_hits(m: Move) -> float:
    try:
        return float(np.clip(getattr(m, "expected_hits", 1.0) or 1.0, 1.0, 5.0))
    except Exception:
        return 1.0


def _safe_type(m: Move) -> PokemonType | None:
    try:
        return getattr(m, "type", None)
    except Exception:
        return None


def _safe_base_stat(p: Pokemon | None, key: str) -> int:
    try:
        if p is None:
            return 0
        return int(getattr(p, "base_stats", {}).get(key, 0))
    except Exception:
        return 0


def _safe_boost(p: Pokemon | None, key: str) -> int:
    try:
        if p is None:
            return 0
        return int(getattr(p, "boosts", {}).get(key, 0))
    except Exception:
        return 0


def _is_move_physical(m: Move) -> float:
    try:
        # Robust check across poke-env versions
        cat = getattr(m, "category", None)
        name = str(cat).upper()
        return 1.0 if "PHYSICAL" in name else 0.0
    except Exception:
        return 0.0


def _is_self_boost_setup_move(m: Move, active: Pokemon | None) -> float:
    """Heuristic mirrors SHP: target == 'self' and boosts sum >= 2 and not already at +6."""
    try:
        if getattr(m, "target", None) != "self":
            return 0.0
        boosts = getattr(m, "boosts", None)
        if not boosts:
            return 0.0
        if sum(v for v in boosts.values() if isinstance(v, (int, float))) < 2:
            return 0.0
        if active is None:
            return 1.0
        # not already at +6 for any boosted stat
        for s, v in boosts.items():
            if v > 0 and _safe_boost(active, s) >= 6:
                return 0.0
        return 1.0
    except Exception:
        return 0.0


def _type_effectiveness(mtype: PokemonType | None, target: Pokemon | None) -> float:
    try:
        if mtype is None or target is None:
            return 1.0
        mult = 1.0
        gen = getattr(target, "battle", None)
        gen = getattr(gen, "gen", 9)
        for t in (getattr(target, "type_1", None), getattr(target, "type_2", None)):
            if t is None:
                continue
            mult *= float(mtype.damage_multiplier(t, gen))
        if mult <= 0.0:
            return 0.0
        # snap to canonical values
        for v in (0.5, 1.0, 2.0, 4.0):
            if abs(mult - v) < 1e-6:
                return v
        return float(np.clip(mult, 0.0, 4.0))
    except Exception:
        return 1.0


def _opp_damage_to_mon_max(opp: Pokemon | None, mon: Pokemon | None) -> float:
    """Max damage multiplier that opponent's types deal to our mon."""
    try:
        if opp is None or mon is None:
            return 1.0
        types = [
            t for t in (getattr(opp, "type_1", None), getattr(opp, "type_2", None)) if t
        ]
        if not types:
            return 1.0
        gen = getattr(mon, "battle", None)
        gen = getattr(gen, "gen", 9)
        vals = []
        for t in types:
            vals.append(float(t.damage_multiplier(getattr(mon, "type_1", None), gen)))
            if getattr(mon, "type_2", None) is not None:
                vals[-1] *= float(
                    t.damage_multiplier(getattr(mon, "type_2", None), gen)
                )
        return float(np.clip(max(vals), 0.0, 4.0)) if vals else 1.0
    except Exception:
        return 1.0


# Hazards (include both correct and the typo variant found in the provided SHP code)
_ENTRY_HAZARDS_IDS = {"spikes", "stealhrock", "stickyweb", "toxicspikes"}
_ANTI_HAZARDS_IDS = {"rapidspin", "defog"}


def _is_entry_hazard_move(m: Move) -> float:
    try:
        return 1.0 if getattr(m, "id", "") in _ENTRY_HAZARDS_IDS else 0.0
    except Exception:
        return 0.0


def _is_anti_hazard_move(m: Move) -> float:
    try:
        return 1.0 if getattr(m, "id", "") in _ANTI_HAZARDS_IDS else 0.0
    except Exception:
        return 0.0


def _has_condition(side_conditions: Dict[Any, Any], needle: str) -> float:
    """Robustly detect a condition by name from SideCondition keys."""
    try:
        ndl = needle.lower()
        for k in side_conditions.keys():
            s = str(k).lower()
            if ndl in s:
                return 1.0
        return 0.0
    except Exception:
        return 0.0


# =============== Expert order interpretation ===============
def _order_is_switch(order: Any) -> bool:
    try:
        if hasattr(order, "is_switch") and callable(order.is_switch):
            return bool(order.is_switch())
        if hasattr(order, "is_move") and callable(order.is_move):
            return not bool(order.is_move())
        if getattr(order, "switch", None) is not None:
            return True
        if getattr(order, "move", None) is not None:
            return False
        inner = getattr(order, "order", None)
        if inner is not None:
            if hasattr(inner, "base_power") or hasattr(inner, "id"):
                return False
            if hasattr(inner, "species") or hasattr(inner, "name"):
                return True
        s = str(order).lower()
        if "switch" in s:
            return True
        if "move" in s:
            return False
    except Exception:
        pass
    return False


def _match_expert_action_index(
    order: Any, moves: List[Move], switches: List[Pokemon]
) -> int:
    try:
        if _order_is_switch(order):
            chosen_sw = getattr(order, "switch", None) or getattr(
                getattr(order, "order", None), "species", None
            )
            for i, s in enumerate(switches[:6]):
                if s is not None and (
                    s == chosen_sw
                    or getattr(s, "species", None)
                    == getattr(chosen_sw, "species", None)
                ):
                    return i
            return 0 if switches else 6
        else:
            chosen_mv = getattr(order, "move", None) or getattr(
                getattr(order, "order", None), "id", None
            )
            for i, m in enumerate(moves[:4]):
                if m is not None and (
                    m == chosen_mv
                    or getattr(m, "id", None) == getattr(chosen_mv, "id", None)
                    or getattr(m, "id", None) == chosen_mv
                ):
                    return 6 + i
            return 6 if moves else 0
    except Exception:
        return 6 if moves else (0 if switches else 6)


# =============== Environment ===============
class ShowdownEnvironment(BaseShowdownEnv):
    """
    Trains against / imitates a SimpleHeuristicsPlayer (SHP) at the action level (0..9).
      - Action space: 10 (0..5 = switches, 6..9 = moves).
      - Phase A: +1 if agent_action == expert_action else -1.
      - Phase B: win-only shaping (fast wins valued higher).

    Observation includes all features SHP uses:
      * Team / battle: HPs, remaining mons, can_dynamax, full-HP flags
      * Active mon: base stats (atk/def/spa/spd/spe), boosts (atk/def/spa/spd)
      * Opponent: base stats (def/spd/spe)
      * Side conditions: our/opp SR, Spikes, Web, TSpikes (+ our_any)
      * Per-move (×4): power, STAB, eff, acc, priority>0, is_physical, expected_hits,
                       is_entry_hazard, is_anti_hazard, is_self_boost_setup
      * Per-switch (×5): bench_hp, bench_off_eff, bench_def_vuln, speed_gt_opp, bench_spe_base
      * Opp typing one-hots (primary 18, secondary 19)
    """

    # Opp typing one-hot sizes
    _TYPE_ORDER = (
        "NORMAL",
        "FIRE",
        "WATER",
        "ELECTRIC",
        "GRASS",
        "ICE",
        "FIGHTING",
        "POISON",
        "GROUND",
        "FLYING",
        "PSYCHIC",
        "BUG",
        "ROCK",
        "GHOST",
        "DRAGON",
        "DARK",
        "STEEL",
        "FAIRY",
    )
    _N_TYPE_PRIMARY = 18
    _N_TYPE_SECONDARY = 19  # + NONE

    # Layout sizes
    # Move block: [bp_norm, STAB, eff/4, acc, pr>0, is_phys, exp_hits/5, is_hazard, is_anti_hazard, is_self_boost_setup]
    _MOVE_BLOCK = 10
    _N_MOVES = 4  # maps to 6..9

    # Switch block: [bench_hp, off_eff/4, def_vuln/4, speed_gt_opp, bench_spe_base_norm]
    _SW_BLOCK = 5
    _N_SWITCHES = 5  # maps to 0..4 (slot 5 remains spare if you ever expose it)

    # Core extras (beyond the original 4 HP totals):
    #   my_rem, opp_rem, can_dmax, my_full_hp, opp_full_hp (5)
    #   active boosts atk/def/spa/spd (4)
    #   active base atk/def/spa/spd/spe (5)
    #   opp base def/spd/spe (3)
    #   our hazards [sr, spikes, tspikes, web, any] (5)
    #   opp hazards [sr, spikes, tspikes, web] (4)
    _CORE_BASE = 4
    _CORE_EXTRA = 5 + 4 + 5 + 3 + 5 + 4

    _OBS_SIZE = (
        _CORE_BASE
        + _CORE_EXTRA
        + _N_MOVES * _MOVE_BLOCK
        + _N_SWITCHES * _SW_BLOCK
        + _N_TYPE_PRIMARY
        + _N_TYPE_SECONDARY
    )

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self.rl_agent = account_name_one

        # SHP instance for expert supervision
        self._expert_player = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration("expertplayer", None),
        )

        # Minimal training state
        self._last_agent_action: int | None = None  # 0..9 chosen by policy
        self._last_expert_action: int | None = None  # 0..9 chosen by SHP

    # ---------- Action space ----------
    def _get_action_size(self) -> int | None:
        return 10

    def process_action(self, action: np.int64) -> np.int64:
        """Map 0..5 switches / 6..9 moves to concrete order; also record SHP's matching 0..9 action."""
        b: AbstractBattle | None = self.battle1
        a = int(action)

        if b is None or not (0 <= a <= 9):
            self._last_agent_action = self._last_expert_action = None
            return np.int64(-2)

        moves: List[Move] = list(b.available_moves or [])
        switches: List[Pokemon] = list(b.available_switches or [])

        # Agent mapping
        if a < 6:  # switch slot
            concrete = a if switches else (6 if moves else -2)
        else:  # move slot
            mv_idx = a - 6
            if moves:
                concrete = 6 + (mv_idx if mv_idx < len(moves) else 0)
            else:
                concrete = 0 if switches else -2

        # Expert mapping
        try:
            order = self._expert_player.choose_move(b)
            expert_action = _match_expert_action_index(order, moves, switches)
        except Exception:
            expert_action = 6 if moves else (0 if switches else 6)

        self._last_agent_action = a
        self._last_expert_action = expert_action
        return np.int64(concrete)

    def calc_reward(self, battle: AbstractBattle) -> float:
        """+1/-1 action-level imitation in phase A; win-only (time-decayed) afterward."""
        ra, re = self._last_agent_action, self._last_expert_action
        r = 1.0 if (ra is not None and re is not None and ra == re) else -1.0
        return r

    # ---------- Observation / Embedding ----------
    def _observation_size(self) -> int:
        return self._OBS_SIZE

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Observation vector contents (sizes):
          Core (4 + 27 = 31):
            [ my_hp, opp_hp, my_team_hp_avg, opp_team_hp_avg,
              my_rem/6, opp_rem/6, can_dmax, my_full_hp, opp_full_hp,           (5)
              boosts_atk/6, boosts_def/6, boosts_spa/6, boosts_spd/6,           (4)
              base_atk/255, base_def/255, base_spa/255, base_spd/255, base_spe/255, (5)
              opp_base_def/255, opp_base_spd/255, opp_base_spe/255,             (3)
              ours_[sr,spikes,tspikes,web,any], opp_[sr,spikes,tspikes,web]     (9)
            ]
          Moves (4 × 10 = 40):
            each: [ bp/200, STAB, eff/4, acc, pr>0, is_phys, exp_hits/5,
                    is_entry_hazard, is_anti_hazard, is_self_boost_setup ]
          Switches (5 × 5 = 25):
            each: [ bench_hp, off_eff/4, def_vuln/4, speed_gt_opp, bench_spe_base/255 ]
          Opp typing one-hots (18 + 19 = 37)
        Total dims = {_self._OBS_SIZE}.
        """
        me: Pokemon | None = battle.active_pokemon
        opp: Pokemon | None = battle.opponent_active_pokemon

        # --- Core (HPs and totals) ---
        my_hp = float(getattr(me, "current_hp_fraction", 0.0) or 0.0) if me else 0.0
        opp_hp = float(getattr(opp, "current_hp_fraction", 0.0) or 0.0) if opp else 0.0

        my_team_total = (
            float(np.sum([m.current_hp_fraction for m in battle.team.values()])) / 6.0
        )
        opp_team_total = (
            float(
                np.sum([m.current_hp_fraction for m in battle.opponent_team.values()])
            )
            / 6.0
        )

        # Remaining mons (alive)
        my_rem = float(len([m for m in battle.team.values() if not m.fainted])) / 6.0
        opp_rem = (
            float(len([m for m in battle.opponent_team.values() if not m.fainted]))
            / 6.0
        )

        can_dmax = 0.0  # Always equals 0.0 in gen9ubers
        my_full = 1.0 if (me and me.current_hp_fraction == 1) else 0.0
        opp_full = 1.0 if (opp and opp.current_hp_fraction == 1) else 0.0

        # Active boosts and base stats
        boosts_atk = _safe_boost(me, "atk") / 6.0
        boosts_def = _safe_boost(me, "def") / 6.0
        boosts_spa = _safe_boost(me, "spa") / 6.0
        boosts_spd = _safe_boost(me, "spd") / 6.0

        base_atk = _safe_base_stat(me, "atk") / 255.0
        base_def = _safe_base_stat(me, "def") / 255.0
        base_spa = _safe_base_stat(me, "spa") / 255.0
        base_spd = _safe_base_stat(me, "spd") / 255.0
        base_spe = _safe_base_stat(me, "spe") / 255.0

        opp_base_def = _safe_base_stat(opp, "def") / 255.0
        opp_base_spd = _safe_base_stat(opp, "spd") / 255.0
        opp_base_spe = _safe_base_stat(opp, "spe") / 255.0

        # Side conditions
        ours_sc = getattr(battle, "side_conditions", {}) or {}
        opp_sc = getattr(battle, "opponent_side_conditions", {}) or {}

        ours_sr = _has_condition(ours_sc, "stealth")  # stealth rock
        ours_spikes = _has_condition(ours_sc, "spikes")
        ours_tspikes = _has_condition(ours_sc, "toxic")  # toxic spikes
        ours_web = _has_condition(ours_sc, "web")
        ours_any = 1.0 if (ours_sr or ours_spikes or ours_tspikes or ours_web) else 0.0

        opp_sr = _has_condition(opp_sc, "stealth")
        opp_spikes = _has_condition(opp_sc, "spikes")
        opp_tspikes = _has_condition(opp_sc, "toxic")
        opp_web = _has_condition(opp_sc, "web")

        vec: List[float] = [
            my_hp,
            opp_hp,
            my_team_total,
            opp_team_total,
            my_rem,
            opp_rem,
            can_dmax,
            my_full,
            opp_full,
            boosts_atk,
            boosts_def,
            boosts_spa,
            boosts_spd,
            base_atk,
            base_def,
            base_spa,
            base_spd,
            base_spe,
            opp_base_def,
            opp_base_spd,
            opp_base_spe,
            ours_sr,
            ours_spikes,
            ours_tspikes,
            ours_web,
            ours_any,
            opp_sr,
            opp_spikes,
            opp_tspikes,
            opp_web,
        ]

        # --- Moves (pad to exactly 4) ---
        moves: List[Move | None] = list(battle.available_moves or [])[: self._N_MOVES]
        while len(moves) < self._N_MOVES:
            moves.append(None)

        for m in moves:
            if m is None:
                vec += [0.0] * self._MOVE_BLOCK
                continue
            bp_norm = _safe_base_power(m) / 200.0
            acc = _safe_accuracy(m)
            pr = 1.0 if _safe_priority(m) > 0 else 0.0
            is_phys = _is_move_physical(m)
            exp_hits_norm = _safe_expected_hits(m) / 5.0

            t = _safe_type(m)
            stab = 0.0
            if t is not None and me is not None:
                stab = (
                    1.0
                    if (
                        t == getattr(me, "type_1", None)
                        or t == getattr(me, "type_2", None)
                    )
                    else 0.0
                )
            eff = _type_effectiveness(t, opp) if t is not None else 1.0
            eff_norm = float(np.clip(eff / 4.0, 0.0, 1.0))

            is_hazard = _is_entry_hazard_move(m)
            is_anti = _is_anti_hazard_move(m)
            is_self_boost = _is_self_boost_setup_move(m, me)

            vec += [
                bp_norm,
                stab,
                eff_norm,
                acc,
                pr,
                is_phys,
                exp_hits_norm,
                is_hazard,
                is_anti,
                is_self_boost,
            ]

        # --- Switches (pad to exactly 5) ---
        bench: List[Pokemon | None] = list(battle.available_switches or [])[
            : self._N_SWITCHES
        ]
        while len(bench) < self._N_SWITCHES:
            bench.append(None)

        opp_spe = _safe_base_stat(opp, "spe")

        for bm in bench:
            if bm is None:
                vec += [0.0] * self._SW_BLOCK
                continue
            hp = float(getattr(bm, "current_hp_fraction", 0.0) or 0.0)
            # Offensive effectiveness (max of its types vs opp)
            types = [
                t
                for t in (getattr(bm, "type_1", None), getattr(bm, "type_2", None))
                if t
            ]
            eff_off = (
                max((_type_effectiveness(t, opp) for t in types), default=1.0)
                if types
                else 1.0
            )
            eff_off_norm = float(np.clip(eff_off / 4.0, 0.0, 1.0))

            # Defensive vulnerability: how hard opp types hit this mon (max multiplier)
            def_vuln = _opp_damage_to_mon_max(opp, bm)
            def_vuln_norm = float(np.clip(def_vuln / 4.0, 0.0, 1.0))

            spd_gt = 1.0 if _safe_base_stat(bm, "spe") > opp_spe else 0.0
            bench_spe_norm = _safe_base_stat(bm, "spe") / 255.0

            vec += [hp, eff_off_norm, def_vuln_norm, spd_gt, bench_spe_norm]

        # --- Opponent typing one-hots (18 + 19) ---
        t1 = getattr(opp, "type_1", None)
        t2 = getattr(opp, "type_2", None)
        vec += self._type_one_hot_primary(t1)
        vec += self._type_one_hot_secondary(t2)

        arr = np.asarray(vec, dtype=np.float32)
        assert (
            arr.shape[0] == self._OBS_SIZE
        ), f"embed_battle produced {arr.shape[0]} dims, expected {self._OBS_SIZE}"
        return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

    # ---------- Typing one-hots ----------
    def _type_one_hot_primary(self, t: PokemonType | None) -> List[float]:
        vec = [0.0] * self._N_TYPE_PRIMARY
        if t is None:
            return vec
        try:
            name = str(t).split(".")[-1].upper()
            if name in self._TYPE_ORDER:
                vec[self._TYPE_ORDER.index(name)] = 1.0
        except Exception:
            pass
        return vec

    def _type_one_hot_secondary(self, t: PokemonType | None) -> List[float]:
        vec = [0.0] * self._N_TYPE_SECONDARY
        if t is None:
            vec[-1] = 1.0
            return vec
        try:
            name = str(t).split(".")[-1].upper()
            if name in self._TYPE_ORDER:
                vec[self._TYPE_ORDER.index(name)] = 1.0
            else:
                vec[-1] = 1.0
        except Exception:
            vec[-1] = 1.0
        return vec


########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        unique_id = time.strftime("%H%M%S")

        opponent_account = "ot" if not evaluation else "oe"
        opponent_account = f"{opponent_account}_{unique_id}"

        opponent_configuration = AccountConfiguration(opponent_account, None)
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer(
                account_configuration=opponent_configuration
            )
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer(account_configuration=opponent_configuration)
        elif opponent_type == "random":
            opponent = RandomPlayer(account_configuration=opponent_configuration)
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "t1" if not evaluation else "e1"
        account_name_two: str = "t2" if not evaluation else "e2"

        account_name_one = f"{account_name_one}_{unique_id}"
        account_name_two = f"{account_name_two}_{unique_id}"

        team = self._load_team(team_type)

        battle_format = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None
