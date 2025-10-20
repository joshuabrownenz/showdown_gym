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
from poke_env.battle import Pokemon
from poke_env.battle import Move
from poke_env.battle import PokemonType

from showdown_gym.base_environment import BaseShowdownEnv


class ShowdownEnvironment(BaseShowdownEnv):
    """
    Ultra-minimal embedding (8 dims). High-level actions:
      0 = ATTACK (env chooses best move)
      1 = SWITCH (env chooses best switch)

    Observation (8 dims):
      [ my_hp, opp_hp, my_team_total_hp, opp_team_total_hp,
        best_move_expected, any_priority_move,
        best_switch_stab_eff_vs_opp, best_switch_sr_damage ]
    """

    _OBS_SIZE = 8

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
        hp_bonus_weight: float = 0.25,  # small terminal bonus for our team HP remaining
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self.rl_agent = account_name_one
        self.hp_bonus_weight = float(hp_bonus_weight)

    # --------------------------
    # Action space: 2 actions
    #   0 = ATTACK (env picks best move)
    #   1 = SWITCH (env picks best switch)
    # --------------------------
    def _get_action_size(self) -> int | None:
        return 2

    def process_action(self, action: np.int64) -> np.int64:
        """
        Maps high-level action to Showdown action id:
          0 -> return move id (6..9) by expected-damage argmax
          1 -> return switch id (0..5) by bench heuristic
        Fallbacks ensure a legal action when possible.
        """
        a = int(action)
        assert a in (0, 1), f"High-level action must be 0 or 1, got {a}"

        battle: AbstractBattle | None = self.battle1
        if battle is None:
            return np.int64(-2)

        moves: List[Move] = list(battle.available_moves or [])
        switches: List[Pokemon] = list(battle.available_switches or [])

        # ATTACK
        if a == 0:
            if moves:
                best_idx = self._argmax_move_expected(
                    moves, battle.active_pokemon, battle.opponent_active_pokemon
                )
                assert (
                    0 <= best_idx < min(4, len(moves))
                ), f"best move idx out of range: {best_idx}, moves={len(moves)}"
                return np.int64(6 + int(best_idx))  # 6..9 = move ids
            # no moves -> try to switch
            if switches:
                best_sw = self._argmax_switch_offense(
                    switches, battle.opponent_active_pokemon, battle.side_conditions
                )
                assert (
                    0 <= best_sw < len(switches)
                ), f"best switch idx out of range: {best_sw}, switches={len(switches)}"
                return np.int64(best_sw)
            return np.int64(-2)

        # SWITCH
        if switches:
            best_sw = self._argmax_switch_offense(
                switches, battle.opponent_active_pokemon, battle.side_conditions
            )
            assert (
                0 <= best_sw < len(switches)
            ), f"best switch idx out of range: {best_sw}, switches={len(switches)}"
            return np.int64(best_sw)
        # no switches -> attack if possible
        if moves:
            best_idx = self._argmax_move_expected(
                moves, battle.active_pokemon, battle.opponent_active_pokemon
            )
            assert (
                0 <= best_idx < min(4, len(moves))
            ), f"best move idx out of range: {best_idx}, moves={len(moves)}"
            return np.int64(6 + int(best_idx))
        return np.int64(-2)

    # --------------------------
    # Logging / additional info
    # --------------------------
    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()
        if self.battle1 is not None:
            agent = self.possible_agents[0]
            b = self.battle1

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

        return info

    # --------------------------
    # Reward: single, win-centric terminal reward
    #   r = (1 if win else 0) + hp_bonus_weight * (our_team_total_hp / 6)
    #   (no per-step reward)
    # --------------------------
    def calc_reward(self, battle: AbstractBattle) -> float:
        if not battle.finished:
            return 0.0

        # Big reward for winning
        r = 1.0 if battle.won else 0.0

        # Small bonus for our team HP remaining (normalized)
        team_hp_total = (
            float(np.sum([m.current_hp_fraction for m in battle.team.values()])) / 6.0
        )
        assert np.isfinite(team_hp_total), f"team_hp_total not finite: {team_hp_total}"
        team_hp_total = float(np.clip(team_hp_total, 0.0, 1.0))
        r += self.hp_bonus_weight * team_hp_total
        return float(r)

    # --------------------------
    # Observation / Embedding (8 dims)
    # --------------------------
    def _observation_size(self) -> int:
        return self._OBS_SIZE

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        [ my_hp, opp_hp, my_team_total_hp, opp_team_total_hp,
          best_move_expected, any_priority_move,
          best_switch_stab_eff_vs_opp, best_switch_sr_damage ]
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

        # Bounds & finiteness checks
        for name, val in (
            ("my_hp", my_hp),
            ("opp_hp", opp_hp),
            ("my_team_total", my_team_total),
            ("opp_team_total", opp_team_total),
        ):
            assert np.isfinite(val), f"{name} not finite: {val}"
            assert 0.0 - 1e-6 <= val <= 1.0 + 1e-6, f"{name} out of [0,1]: {val}"

        best_expected, any_prio = self._best_move_expected_and_priority(battle, me, opp)
        best_offense, best_sr = self._best_bench_heuristics(battle, me, opp)

        for name, val in (
            ("best_expected", best_expected),
            ("any_prio", any_prio),
            ("best_offense", best_offense),
            ("best_sr", best_sr),
        ):
            assert np.isfinite(val), f"{name} not finite: {val}"
            assert -1e-6 <= val <= 1.0 + 1e-6, f"{name} out of [0,1]: {val}"

        vec = np.array(
            [
                my_hp,
                opp_hp,
                my_team_total,
                opp_team_total,
                best_expected,
                any_prio,
                best_offense,
                best_sr,
            ],
            dtype=np.float32,
        )
        assert (
            vec.shape[0] == self._OBS_SIZE
        ), f"embed_battle produced {vec.shape[0]} dims, expected {self._OBS_SIZE}"
        return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)

    # --------------------------
    # Micro-decision helpers
    # --------------------------
    def _argmax_move_expected(
        self, moves: List[Move], me: Pokemon | None, opp: Pokemon | None
    ) -> int:
        best_idx = 0
        best_val = -1.0
        limit = min(4, len(moves))  # move ids 6..9
        for i in range(limit):
            m = moves[i]
            bp = self._safe_base_power(m)  # 0..200 -> normalized inside
            acc = self._safe_accuracy(m)  # 0..1
            eff = self._type_effectiveness(self._safe_type(m), opp)  # 0..4
            stab_mult = 1.5 if self._safe_stab(m) else 1.0
            expected = (bp / 200.0) * acc * eff * stab_mult / 6.0
            expected = float(np.clip(expected, 0.0, 1.0))
            if expected > best_val:
                best_val = expected
                best_idx = i
        assert (
            0 <= best_idx < limit or limit == 0
        ), f"_argmax_move_expected idx invalid: {best_idx}, limit={limit}"
        return best_idx

    def _argmax_switch_offense(
        self,
        switches: List[Pokemon],
        opp: Pokemon | None,
        side_conditions: Dict[Any, Any],
    ) -> int:
        assert len(switches) > 0, "_argmax_switch_offense called with empty switches"
        sr_on_our_side = self._stealth_rock_active(side_conditions)
        best_idx = 0
        best_val = -1.0
        for i, mon in enumerate(switches):
            off = (1.5 * self._stab_eff_vs_opp(mon, opp)) / 6.0  # [0,1]
            sr = self._expected_sr_damage(mon) if sr_on_our_side else 0.0  # [0,1]
            hp = float(getattr(mon, "current_hp_fraction", 0.0) or 0.0)
            off = float(np.clip(off, 0.0, 1.0))
            sr = float(np.clip(sr, 0.0, 1.0))
            hp = float(np.clip(hp, 0.0, 1.0))
            val = off - 0.25 * sr + 0.05 * hp
            if val > best_val:
                best_val = val
                best_idx = i
        assert (
            0 <= best_idx < len(switches)
        ), f"_argmax_switch_offense idx invalid: {best_idx}, n={len(switches)}"
        return best_idx

    def _best_move_expected_and_priority(
        self, battle: AbstractBattle, me: Pokemon | None, opp: Pokemon | None
    ) -> Tuple[float, float]:
        moves = list(battle.available_moves or [])
        if not moves:
            return 0.0, 0.0

        best_expected = 0.0
        any_prio = 0.0
        for m in moves:
            bp = self._safe_base_power(m)  # 0..200
            acc = self._safe_accuracy(m)  # 0..1
            eff = self._type_effectiveness(self._safe_type(m), opp)  # 0..4
            stab_mult = 1.5 if self._safe_stab(m) else 1.0
            expected = float(
                np.clip((bp / 200.0) * acc * eff * stab_mult / 6.0, 0.0, 1.0)
            )
            best_expected = max(best_expected, expected)

            pr = self._safe_priority(m)  # numeric priority (can be 0)
            if pr > 0:
                any_prio = 1.0

        return best_expected, any_prio

    def _best_bench_heuristics(
        self, battle: AbstractBattle, me: Pokemon | None, opp: Pokemon | None
    ) -> Tuple[float, float]:
        bench = list(battle.available_switches or [])
        if not bench:
            return 0.0, 0.0

        sr_on_our_side = self._stealth_rock_active(battle.side_conditions)
        best_offense = 0.0
        best_sr_dmg = 0.0
        for mon in bench:
            eff = self._stab_eff_vs_opp(mon, opp)  # 0..4
            off_norm = float(np.clip((1.5 * eff) / 6.0, 0.0, 1.0))
            if off_norm > best_offense:
                best_offense = off_norm
                best_sr_dmg = self._expected_sr_damage(mon) if sr_on_our_side else 0.0
        return best_offense, best_sr_dmg

    # --------------------------
    # Safe accessors (avoid KeyError/None)
    # --------------------------
    def _safe_priority(self, m: Move) -> float:
        try:
            # poke-env Move.priority may raise KeyError
            pr = m.priority  # triggers property -> may KeyError
            return float(pr)
        except Exception:
            return 0.0

    def _safe_accuracy(self, m: Move) -> float:
        try:
            acc = m.accuracy  # may be None or >1.0
            if acc is None:
                return 1.0
            acc = float(acc)
            if acc > 1.0:
                acc /= 100.0
            return float(np.clip(acc, 0.0, 1.0))
        except Exception:
            return 1.0

    def _safe_base_power(self, m: Move) -> float:
        try:
            bp = float(getattr(m, "base_power", 0.0))
            return float(np.clip(bp, 0.0, 200.0))
        except Exception:
            return 0.0

    def _safe_stab(self, m: Move) -> bool:
        try:
            return bool(getattr(m, "stab", False))
        except Exception:
            return False

    def _safe_type(self, m: Move) -> PokemonType | None:
        try:
            return getattr(m, "type", None)
        except Exception:
            return None

    # --------------------------
    # Small helpers
    # --------------------------
    def _stab_eff_vs_opp(self, mon: Pokemon | None, opp: Pokemon | None) -> float:
        if mon is None or opp is None:
            return 1.0
        eff_best = 1.0
        for t in (getattr(mon, "type_1", None), getattr(mon, "type_2", None)):
            if t is None:
                continue
            eff_best = max(eff_best, self._type_effectiveness(t, opp))
        assert np.isfinite(eff_best), f"_stab_eff_vs_opp not finite: {eff_best}"
        return float(np.clip(eff_best, 0.0, 4.0))

    def _stealth_rock_active(self, side_conditions: Dict[Any, Any]) -> bool:
        try:
            for k in (side_conditions or {}).keys():
                name = str(k).upper()
                if "STEALTHROCK" in name or "STEALTH_ROCK" in name:
                    return True
        except Exception:
            pass
        return False

    def _expected_sr_damage(self, mon: Pokemon | None) -> float:
        if mon is None:
            return 0.0
        try:
            mult = 1.0
            rock = PokemonType.ROCK
            gen = getattr(mon, "battle", None)
            gen = getattr(gen, "gen", 9)
            for t in (getattr(mon, "type_1", None), getattr(mon, "type_2", None)):
                if t is None:
                    continue
                dm = float(rock.damage_multiplier(t, gen))
                mult *= dm
            dmg = float(np.clip(0.125 * mult, 0.0, 1.0))
            assert np.isfinite(dmg), f"_expected_sr_damage not finite: {dmg}"
            return dmg
        except Exception:
            return 0.0

    def _type_effectiveness(
        self, mtype: PokemonType | None, opp: Pokemon | None
    ) -> float:
        """Return effectiveness multiplier (0, 0.5, 1, 2, 4); clamps other values to [0,4]."""
        try:
            if mtype is None or opp is None:
                return 1.0
            mult = 1.0
            ot1 = getattr(opp, "type_1", None)
            ot2 = getattr(opp, "type_2", None)
            gen = getattr(opp, "battle", None)
            gen = getattr(gen, "gen", 9)
            for t in (ot1, ot2):
                if t is None:
                    continue
                dm = float(mtype.damage_multiplier(t, gen))
                mult *= dm
            if mult <= 0.0:
                return 0.0
            # snap near-canonical values
            for v in (0.5, 1.0, 2.0, 4.0):
                if abs(mult - v) < 1e-6:
                    mult = v
                    break
            mult = float(np.clip(mult, 0.0, 4.0))
            assert np.isfinite(mult), f"_type_effectiveness not finite: {mult}"
            return mult
        except Exception:
            return 1.0


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
