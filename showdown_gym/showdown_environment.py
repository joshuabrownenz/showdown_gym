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


# ---------- Safe helpers ----------
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
        for v in (0.5, 1.0, 2.0, 4.0):
            if abs(mult - v) < 1e-6:
                return v
        return float(np.clip(mult, 0.0, 4.0))
    except Exception:
        return 1.0


# ---------- Interpret SimpleHeuristicsPlayer orders ----------
def _order_is_switch(order: Any) -> bool:
    """Best-effort: is the expert order a switch? (portable across poke-env versions)."""
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
    """
    Map the expert's BattleOrder to our 0..9 discrete action.
      0..5 = switch[i] (we only ever have up to 5 real switches; 5 is a spare slot)
      6..9 = move[i] among the first 4 available moves
    Fallbacks default to the first legal option of that type.
    """
    try:
        if _order_is_switch(order):
            # Try identity match first
            chosen_sw = getattr(order, "switch", None) or getattr(
                getattr(order, "order", None), "species", None
            )
            # Identity compare: object equality is OK in practice for available_switches
            for i, s in enumerate(
                switches[:6]
            ):  # safe bound (we expose up to 6 switch slots: 0..5)
                if s is not None and (
                    s == chosen_sw
                    or getattr(s, "species", None)
                    == getattr(chosen_sw, "species", None)
                ):
                    return i  # 0..5
            return (
                0 if switches else 6
            )  # fallback to a switch slot if any, else move slot as degenerate case
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
                    return 6 + i  # 6..9
            return 6 if moves else 0  # fallback: first move if any, else first switch
    except Exception:
        # Conservative default: choose first move slot (6) if possible
        return 6 if moves else (0 if switches else 6)


# ---------- Environment ----------
class ShowdownEnvironment(BaseShowdownEnv):
    """
    Level-1 Curriculum: Imitation against SimpleHeuristicsPlayer.
      * Action space: 10 (0..5 switch, 6..9 move).
      * Phase A reward: action-level imitation (+1 match, -1 mismatch).
      * Phase B reward: speed-weighted win-only.
      * Observation: 76 dims (core HP, 4×move features, 5×switch features, opp typing one-hots).
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
    _MOVE_BLOCK = 5  # [bp_norm, STAB, eff/4, acc, pr>0]
    _N_MOVES = 4  # maps to 6..9
    _SW_BLOCK = 3  # [bench_hp, bench_off_STAB_eff/4, speed_gt_opp]
    _N_SWITCHES = 5  # maps to 0..4 (we expose 0..5 actions; 5 is spare)

    _OBS_SIZE = (
        4
        + _N_MOVES * _MOVE_BLOCK
        + _N_SWITCHES * _SW_BLOCK
        + _N_TYPE_PRIMARY
        + _N_TYPE_SECONDARY
    )  # 76

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

        # Instantiate SimpleHeuristicsPlayer as the supervision "expert"
        self._expert_player = SimpleHeuristicsPlayer(
            battle_format=battle_format,
            account_configuration=AccountConfiguration(account_name_two, None),
        )

        # Minimal training state
        self._last_agent_action: int | None = None  # 0..9 (what the policy chose)
        self._last_expert_action: int | None = None  # 0..9 (what expert would do)

    # --------------------------
    # Action space: 10 actions
    # --------------------------
    def _get_action_size(self) -> int | None:
        return 10

    def process_action(self, action: np.int64) -> np.int64:
        """
        Map [0..5]=switch, [6..9]=move to *concrete* Showdown action.
        Record the expert's matching 0..9 action using SimpleHeuristicsPlayer.
        """
        b: AbstractBattle | None = self.battle1
        a = int(action)

        if b is None or not (0 <= a <= 9):
            self._last_agent_action = self._last_expert_action = None
            return np.int64(-2)

        moves: List[Move] = list(b.available_moves or [])
        switches: List[Pokemon] = list(b.available_switches or [])

        # --- Agent mapping to concrete action with simple fallbacks ---
        if a < 6:  # switch slot
            concrete = a if switches else (6 if moves else -2)
        else:  # move slot
            mv_idx = a - 6
            if moves:
                concrete = 6 + (mv_idx if mv_idx < len(moves) else 0)
            else:
                concrete = 0 if switches else -2

        # --- Expert's discrete 0..9 action via heuristics ---
        try:
            order = self._expert_player.choose_move(b)
            expert_action = _match_expert_action_index(order, moves, switches)
        except Exception:
            expert_action = 6 if moves else (0 if switches else 6)

        # Persist for reward
        self._last_agent_action = a
        self._last_expert_action = expert_action

        return np.int64(concrete)

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Phase A: action-level imitation
            r = +1.0 if agent_action == expert_action else -1.0
        """
        ra, re = self._last_agent_action, self._last_expert_action
        r = 1.0 if (ra is not None and re is not None and ra == re) else -1.0

        return r

    # --------------------------
    # Observation / Embedding
    # --------------------------
    def _observation_size(self) -> int:
        return self._OBS_SIZE

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        76 dims:
          Core 4:
            [ my_hp, opp_hp, my_team_total_hp, opp_team_total_hp ]
          Moves 20:
            4 × [ bp_norm, STAB, eff/4, acc, priority>0 ]
          Switches 15:
            5 × [ bench_hp, bench_off_STAB_eff/4, speed_gt_opp ]
          Opp typing 37:
            [ primary_type_onehot(18), secondary_type_onehot(19 with NONE) ]

        This gives the policy everything needed to pick among actions 0..9:
          - For 6..9 (moves): per-move power, accuracy, priority, STAB, effectiveness.
          - For 0..5 (switches): bench HP, offensive effectiveness vs opp, speed edge.
        """
        me: Pokemon | None = battle.active_pokemon
        opp: Pokemon | None = battle.opponent_active_pokemon

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

        for name, val in (
            ("my_hp", my_hp),
            ("opp_hp", opp_hp),
            ("my_team_total", my_team_total),
            ("opp_team_total", opp_team_total),
        ):
            assert np.isfinite(val), f"{name} not finite: {val}"
            assert -1e-6 <= val <= 1.0 + 1e-6, f"{name} out of [0,1]: {val}"

        vec: List[float] = [my_hp, opp_hp, my_team_total, opp_team_total]

        # ---- Moves (pad to exactly 4) ----
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
            for name, val in (
                ("bp_norm", bp_norm),
                ("acc", acc),
                ("pr", pr),
                ("eff_norm", eff_norm),
            ):
                assert np.isfinite(val), f"move feature {name} not finite: {val}"
                assert (
                    -1e-6 <= val <= 1.0 + 1e-6
                ), f"move feature {name} out of [0,1]: {val}"
            vec += [bp_norm, stab, eff_norm, acc, pr]

        # ---- Switches (pad to exactly 5) ----
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
            # offensive STAB eff vs opp (max over its two types)
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
            spd_gt = 1.0 if _safe_base_stat(bm, "spe") > opp_spe else 0.0
            for name, val in (
                ("hp", hp),
                ("eff_off_norm", eff_off_norm),
                ("spd_gt", spd_gt),
            ):
                assert np.isfinite(val), f"switch feature {name} not finite: {val}"
                assert (
                    -1e-6 <= val <= 1.0 + 1e-6
                ), f"switch feature {name} out of [0,1]: {val}"
            vec += [hp, eff_off_norm, spd_gt]

        # ---- Opponent typing one-hots (18 + 19) ----
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
