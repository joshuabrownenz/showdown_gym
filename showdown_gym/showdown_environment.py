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
from poke_env.battle import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType

from showdown_gym.base_environment import BaseShowdownEnv


# A simplified version of simple heuristics player that doesn't do hazards
class ModifiedSimpleHeuristicsPlayer(Player):
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4
    SWITCH_OUT_MATCHUP_THRESHOLD = -2

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def _should_switch_out(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        # If there is a decent switch in...
        if [
            m
            for m in battle.available_switches
            if self._estimate_matchup(m, opponent) > 0
        ]:
            # ...and a 'good' reason to switch out
            if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
                return True
            if (
                active.boosts["atk"] <= -3
                and active.stats["atk"] >= active.stats["spa"]
            ):
                return True
            if (
                active.boosts["spa"] <= -3
                and active.stats["atk"] <= active.stats["spa"]
            ):
                return True
            if (
                self._estimate_matchup(active, opponent)
                < self.SWITCH_OUT_MATCHUP_THRESHOLD
            ):
                return True
        return False

    def _stat_estimation(self, mon: Pokemon, stat: str):
        # Stats boosts value
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost

    def choose_move(self, battle: AbstractBattle):
        # Main mons shortcuts
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        # Rough estimation of damage ratio
        physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(
            opponent, "def"
        )
        special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(
            opponent, "spd"
        )

        if battle.available_moves and (
            not self._should_switch_out(battle) or not battle.available_switches
        ):
            n_remaining_mons = len(
                [m for m in battle.team.values() if m.fainted is False]
            )

            move = max(
                battle.available_moves,
                key=lambda m: m.base_power
                * (1.5 if m.type in active.types else 1)
                * (
                    physical_ratio
                    if m.category == MoveCategory.PHYSICAL
                    else special_ratio
                )
                * m.accuracy
                * m.expected_hits
                * opponent.damage_multiplier(m),
            )
            return self.create_order(move)

        if battle.available_switches:
            switches: List[Pokemon] = battle.available_switches
            return self.create_order(
                max(
                    switches,
                    key=lambda s: self._estimate_matchup(s, opponent),
                )
            )

        return self.choose_random_move(battle)


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


def _shp_best_move_idx(battle: AbstractBattle) -> int | None:
    """Replicates SHP's move scoring to pick the best available move index [0..3]."""
    active: Pokemon | None = battle.active_pokemon
    opponent: Pokemon | None = battle.opponent_active_pokemon
    if active is None or opponent is None:
        return None
    moves: List[Move] = list(battle.available_moves or [])
    if not moves:
        return None

    # SHP's damage ratios
    def _stat_est(mon: Pokemon, stat: str) -> float:
        boost = (
            (2 + mon.boosts[stat]) / 2
            if mon.boosts[stat] > 1
            else 2 / (2 - mon.boosts[stat])
        )
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost

    physical_ratio = _stat_est(active, "atk") / _stat_est(opponent, "def")
    special_ratio = _stat_est(active, "spa") / _stat_est(opponent, "spd")

    def score(m: Move) -> float:
        bp = _safe_base_power(m)
        stab = 1.5 if (_safe_type(m) in (active.type_1, active.type_2)) else 1.0
        ratio = physical_ratio if _is_move_physical(m) > 0.5 else special_ratio
        acc = _safe_accuracy(m)
        hits = _safe_expected_hits(m)
        eff = (
            opponent.damage_multiplier(m)
            if hasattr(opponent, "damage_multiplier")
            else 1.0
        )
        return bp * stab * ratio * acc * hits * eff

    best_i, best_v = 0, -1.0
    for i, m in enumerate(moves[:4]):
        v = score(m)
        if v > best_v:
            best_i, best_v = i, v
    return best_i


def _shp_best_switch_idx(battle: AbstractBattle) -> int | None:
    """Replicates SHP's switch scoring to pick the best available switch index [0..5]."""
    opponent: Pokemon | None = battle.opponent_active_pokemon
    if opponent is None:
        return None
    switches: List[Pokemon] = list(battle.available_switches or [])
    if not switches:
        return None

    def estimate_matchup(mon: Pokemon, opp: Pokemon) -> float:
        # same as your ModifiedSimpleHeuristicsPlayer._estimate_matchup
        speed_coef = ModifiedSimpleHeuristicsPlayer.SPEED_TIER_COEFICIENT
        hp_coef = ModifiedSimpleHeuristicsPlayer.HP_FRACTION_COEFICIENT
        score = max(
            [opp.damage_multiplier(t) for t in mon.types if t is not None]
        ) - max([mon.damage_multiplier(t) for t in opp.types if t is not None])
        if mon.base_stats["spe"] > opp.base_stats["spe"]:
            score += speed_coef
        elif opp.base_stats["spe"] > mon.base_stats["spe"]:
            score -= speed_coef
        score += mon.current_hp_fraction * hp_coef
        score -= opp.current_hp_fraction * hp_coef
        return float(score)

    best_i, best_v = 0, -1e9
    for i, s in enumerate(switches[:6]):
        v = estimate_matchup(s, opponent)
        if v > best_v:
            best_i, best_v = i, v
    return best_i


# =============== Environment ===============
class ShowdownEnvironment(BaseShowdownEnv):
    """
    Trains against / imitates a SimpleHeuristicsPlayer (SHP) at the action level (0..9).

    Observation includes only the information that ModifiedSimpleHeuristicsPlayer uses:
      Core:
        - Active & opponent HP fractions
        - Active boosts: atk/def/spa/spd
        - Active base stats: atk/def/spa/spd/spe
        - Opponent base stats: def/spd/spe
      Per-move (×4):
        - base_power/200, STAB (0/1), effectiveness vs opp /4, accuracy (0..1),
          priority>0 (0/1), is_physical (0/1), expected_hits/5
      Per-switch (×5):
        - bench_hp, bench_off_eff/4, bench_def_vuln/4, bench_faster_than_opp (0/1),
          bench_spe_base/255
    """

    # Layout sizes (STRICTLY what the heuristic uses)
    _MOVE_BLOCK = 7
    _N_MOVES = 4  # maps to 6..9

    _SW_BLOCK = 5
    _N_SWITCHES = 5  # maps to 0..4

    # Core:
    #   my_hp, opp_hp (2)
    #   boosts_atk/6, boosts_def/6, boosts_spa/6, boosts_spd/6 (4)
    #   base_atk/255, base_def/255, base_spa/255, base_spd/255, base_spe/255 (5)
    #   opp_base_def/255, opp_base_spd/255, opp_base_spe/255 (3)
    _CORE_SIZE = 2 + 4 + 5 + 3

    _OBS_SIZE = _CORE_SIZE + _N_MOVES * _MOVE_BLOCK + _N_SWITCHES * _SW_BLOCK

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
        # create expert player with a random numeric suffix to avoid account name collisions
        expert_suffix = int(np.random.randint(0, 1_000_000))
        expert_account = f"expertplayer{expert_suffix}"
        self._expert_player = ModifiedSimpleHeuristicsPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(expert_account, None),
        )

        # Minimal training state
        self._last_agent_action: int | None = None  # 0..9 chosen by policy
        self._last_expert_action: int | None = None  # 0..9 chosen by SHP

        self._ep_total_intents: int = (
            0  # how many intent labels we produced this episode
        )
        self._ep_match_intents: int = 0  # how many times agent intent == expert intent
        self._ep_last_turn: int = 0  # latest battle.turn seen this episode

    def _reset_episode_counters(self) -> None:
        self._ep_total_intents = 0
        self._ep_match_intents = 0
        self._ep_last_turn = 0

    def reset(self, seed=None, options=None):
        response = super().reset(seed=seed, options=options)
        self._reset_episode_counters()
        return response

    # ---------- Action space ----------
    def _get_action_size(self) -> int | None:
        return 2

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            b = self.battle1

            # Existing basics
            team_hp = (
                float(np.sum([m.current_hp_fraction for m in b.team.values()])) / 6.0
            )
            opp_hp = (
                float(np.sum([m.current_hp_fraction for m in b.opponent_team.values()]))
                / 6.0
            )
            fainted_self = int(np.sum([int(m.fainted) for m in b.team.values()]))
            fainted_opp = int(
                np.sum([int(m.fainted) for m in b.opponent_team.values()])
            )
            info[agent]["win"] = b.won
            info[agent]["team_hp"] = team_hp
            info[agent]["opp_hp"] = opp_hp
            info[agent]["fainted_self"] = fainted_self
            info[agent]["fainted_opp"] = fainted_opp

            # Intents on the last decision (for debugging)
            info[agent]["intent_agent"] = getattr(self, "_last_agent_intent", None)
            info[agent]["intent_expert"] = getattr(self, "_last_expert_intent", None)
            info[agent]["intent_match"] = int(
                (getattr(self, "_last_agent_intent", None) is not None)
                and (getattr(self, "_last_expert_intent", None) is not None)
                and (self._last_agent_intent == self._last_expert_intent)
            )

            # --- Episode aggregates (what you’ll use in post-processing) ---
            info[agent]["ep_intent_total"] = int(self._ep_total_intents)
            info[agent]["ep_intent_matches"] = int(self._ep_match_intents)
            info[agent]["ep_imitation_accuracy"] = (
                float(self._ep_match_intents) / float(self._ep_total_intents)
                if self._ep_total_intents > 0
                else 0.0
            )
            # turns for this episode (battle.turn tends to be 1-based; we just emit last seen)
            info[agent]["ep_turns"] = int(self._ep_last_turn)

            # Optional: mark opponent you evaluated against if you know it externally
            # info[agent]["opponent"] = "MaxBasePower"  # uncomment if you run that eval

        return info

    def process_action(self, action: np.int64) -> np.int64:
        """
        Action 0 -> ATTACK: execute SHP's best move.
        Action 1 -> SWITCH: execute SHP's best switch.
        Reward compares agent intent vs SHP intent (move vs switch) on this state.
        """
        b: AbstractBattle | None = self.battle1
        a = int(action)
        assert a in (0, 1), f"Action must be 0 (ATTACK) or 1 (SWITCH), got {a}"

        if b is None:
            self._last_agent_intent = None
            self._last_expert_intent = None
            return np.int64(-2)  # Default if no battle

        moves: List[Move] = list(b.available_moves or [])
        switches: List[Pokemon] = list(b.available_switches or [])

        # --- Expert intent from SHP's actual decision on this state ---
        expert_order = self._expert_player.choose_move(b)
        expert_is_switch = _order_is_switch(expert_order)
        self._last_expert_intent = "SWITCH" if expert_is_switch else "ATTACK"

        # --- Agent intent from action ---
        self._last_agent_intent = "ATTACK" if a == 0 else "SWITCH"

        # --- Build the concrete action to execute based on the agent's intent ---
        mv_idx = _shp_best_move_idx(b)
        sw_idx = _shp_best_switch_idx(b)

        if self._last_agent_intent == "ATTACK":
            if mv_idx is not None and moves:
                concrete = 6 + int(mv_idx)  # move slot
            elif sw_idx is not None and switches:
                concrete = int(sw_idx)  # fallback to switch
            else:
                concrete = -2  # no-ops
        else:  # SWITCH
            if sw_idx is not None and switches:
                concrete = int(sw_idx)  # switch slot
            elif mv_idx is not None and moves:
                concrete = 6 + int(mv_idx)  # fallback to move
            else:
                concrete = -2

        # Invariants (helpful during dev)
        assert self._last_agent_intent in ("ATTACK", "SWITCH")
        assert self._last_expert_intent in ("ATTACK", "SWITCH")

        # Update episode counters (count only when both intents are available)
        if (
            getattr(self, "_last_agent_intent", None) is not None
            and getattr(self, "_last_expert_intent", None) is not None
        ):
            self._ep_total_intents += 1
            if self._last_agent_intent == self._last_expert_intent:
                self._ep_match_intents += 1

        # Track last seen turn (helpful for mean turns / win)
        try:
            if self.battle1 is not None:
                t = int(getattr(self.battle1, "turn", 0) or 0)
                if t > self._ep_last_turn:
                    self._ep_last_turn = t
        except Exception:
            pass

        return np.int64(concrete)

    def calc_reward(self, battle: AbstractBattle) -> float:
        ai = getattr(self, "_last_agent_intent", None)
        ei = getattr(self, "_last_expert_intent", None)
        if ai is None or ei is None:
            return 0.0
        return 1.0 if ai == ei else -1.0

    # ---------- Observation / Embedding ----------
    def _observation_size(self) -> int:
        return self._OBS_SIZE

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Observation vector contents (sizes):
          Core (14):
            [ my_hp, opp_hp,
              boosts_atk/6, boosts_def/6, boosts_spa/6, boosts_spd/6,
              base_atk/255, base_def/255, base_spa/255, base_spd/255, base_spe/255,
              opp_base_def/255, opp_base_spd/255, opp_base_spe/255 ]
          Moves (4 × 7 = 28):
            each: [ bp/200, STAB, eff/4, acc, pr>0, is_phys, exp_hits/5 ]
          Switches (5 × 5 = 25):
            each: [ bench_hp, off_eff/4, def_vuln/4, speed_gt_opp, bench_spe_base/255 ]
          Total dims = {_self._OBS_SIZE}.
        """
        me: Pokemon | None = battle.active_pokemon
        opp: Pokemon | None = battle.opponent_active_pokemon

        # --- Core ---
        my_hp = float(getattr(me, "current_hp_fraction", 0.0) or 0.0) if me else 0.0
        opp_hp = float(getattr(opp, "current_hp_fraction", 0.0) or 0.0) if opp else 0.0

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

        vec: List[float] = [
            my_hp,
            opp_hp,
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

            # STAB and effectiveness require only types (used by the heuristic)
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

            vec += [bp_norm, stab, eff_norm, acc, pr, is_phys, exp_hits_norm]

        # --- Switches (pad to exactly 5) ---
        bench: List[Pokemon | None] = list(battle.available_switches or [])[
            : self._N_SWITCHES
        ]
        while len(bench) < self._N_SWITCHES:
            bench.append(None)

        opp_spe_raw = _safe_base_stat(opp, "spe")

        for bm in bench:
            if bm is None:
                vec += [0.0] * self._SW_BLOCK
                continue

            hp = float(getattr(bm, "current_hp_fraction", 0.0) or 0.0)

            # Offensive effectiveness: max over bench types vs opponent (same notion SHP uses)
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

            # Defensive vulnerability: how hard opp's types hit bench (max multiplier)
            def_vuln = _opp_damage_to_mon_max(opp, bm)
            def_vuln_norm = float(np.clip(def_vuln / 4.0, 0.0, 1.0))

            spd_gt = 1.0 if _safe_base_stat(bm, "spe") > opp_spe_raw else 0.0
            bench_spe_norm = _safe_base_stat(bm, "spe") / 255.0

            vec += [hp, eff_off_norm, def_vuln_norm, spd_gt, bench_spe_norm]

        arr = np.asarray(vec, dtype=np.float32)
        assert (
            arr.shape[0] == self._OBS_SIZE
        ), f"embed_battle produced {arr.shape[0]} dims, expected {self._OBS_SIZE}"
        return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)


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
