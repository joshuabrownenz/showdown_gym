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

from showdown_gym.base_environment import BaseShowdownEnv


# no change needed to imports above
from poke_env.battle import Pokemon
from poke_env.battle import Move
from poke_env.battle import PokemonType


class ShowdownEnvironment(BaseShowdownEnv):
    """
    Super-slim embedding focused on the core decision:
      - Attack now (choose best expected-damage + priority) vs
      - Switch (bench heuristic: hp, offensive STAB effectiveness vs opp, SR damage risk)

    Observation (27 dims):
      [ my_hp, opp_hp, my_team_total_hp, opp_team_total_hp,           # 4
        move1(2), move2(2), move3(2), move4(2),                       # 8
        bench1(3), bench2(3), bench3(3), bench4(3), bench5(3) ]       # 15
    """

    # ---- Simple move encoding ----
    _MOVE_BLOCK = 2
    _N_MOVES = 4

    # ---- Bench heuristic block (hp, stab_eff_vs_opp, expected_sr_damage) ----
    _BENCH_SLOTS = 5
    _BENCH_BLOCK = 3

    # ---- Final observation size ----
    _OBS_SIZE = 4 + _N_MOVES * _MOVE_BLOCK + _BENCH_SLOTS * _BENCH_BLOCK  # 27

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
        reward_mode: str = "hp_delta",  # dense & fast by default
        shaping_weights: Dict[str, float] | None = None,
    ):
        """
        reward_mode: "hp_delta" | "terminal_only" | "potential_v1" | "mixed"
        shaping_weights: optional overrides for potential terms (very small set here)
        """
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self.rl_agent = account_name_one
        self.reward_mode = reward_mode

        self.shaping_weights = {
            "team_hp_adv": 0.7,  # (our_total_hp - opp_total_hp)
            "fainted_adv": 0.3,  # (their_fainted - our_fainted)/6
            "ko_bonus": 0.2,  # bonus if we KO since last step
            "ko_malus": -0.2,  # penalty if we get KO'd since last step
            "step_cost": 0.0,  # set to -0.005 for mild anti-stall
        }
        if shaping_weights:
            self.shaping_weights.update(shaping_weights)

        self._last_turn = -1

    # --------------------------
    # Action space (unchanged)
    # --------------------------
    def _get_action_size(self) -> int | None:
        return None  # default 26 actions

    def process_action(self, action: np.int64) -> np.int64:
        return action

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
    # Rewards (curriculum-ready)
    # --------------------------
    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Reward modes:
          - hp_delta:        sum drop in opponent total HP since prior step (dense)
          - terminal_only:   +1 win / -1 loss (+ optional small step cost)
          - potential_v1:    Φ with only team HP advantage and fainted advantage; KO bonuses
          - mixed:           hp_delta + 0.5 * potential_v1
        """
        mode = self.reward_mode.lower()
        if mode == "hp_delta":
            return self._r_hp_delta(battle)
        elif mode == "terminal_only":
            return self._r_terminal_only(battle)
        elif mode == "potential_v1":
            return self._r_potential(battle)
        elif mode == "mixed":
            return self._r_hp_delta(battle) + 0.5 * self._r_potential(battle)
        else:
            return self._r_hp_delta(battle)

    def _r_terminal_only(self, battle: AbstractBattle) -> float:
        w = self.shaping_weights
        reward = 0.0
        if battle.finished:
            reward += 1.0 if battle.won else -1.0
        reward += w.get("step_cost", 0.0)
        return reward

    def _r_hp_delta(self, battle: AbstractBattle) -> float:
        prior_battle = self._get_prior_battle(battle)
        if prior_battle is None:
            return 0.0

        def total_hp(b: AbstractBattle, mine: bool) -> float:
            mons = b.team.values() if mine else b.opponent_team.values()
            return float(np.sum([m.current_hp_fraction for m in mons])) / 6.0

        curr_opp = total_hp(battle, mine=False)
        prev_opp = total_hp(prior_battle, mine=False)
        return float(prev_opp - curr_opp)

    def _r_potential(self, battle: AbstractBattle) -> float:
        """
        Potential-based shaping (very small Φ):
          Φ(s) = w1*(our_hp - opp_hp) + w2*((their_fainted - our_fainted)/6), clipped to [-1,1].
        """
        w = self.shaping_weights
        gamma = (
            self.train_config.get("gamma", 0.99)
            if hasattr(self, "train_config")
            else 0.99
        )

        reward = 0.0
        if battle.finished:
            reward += 1.0 if battle.won else -1.0

        phi_curr = self._potential_phi(battle, w)
        prior_battle = self._get_prior_battle(battle)
        if prior_battle is not None:
            phi_prev = self._potential_phi(prior_battle, w)
            reward += gamma * phi_curr - phi_prev

            # KO bonuses since last step
            curr_self_fainted = int(
                np.sum([int(m.fainted) for m in battle.team.values()])
            )
            curr_opp_fainted = int(
                np.sum([int(m.fainted) for m in battle.opponent_team.values()])
            )
            prev_self_fainted = int(
                np.sum([int(m.fainted) for m in prior_battle.team.values()])
            )
            prev_opp_fainted = int(
                np.sum([int(m.fainted) for m in prior_battle.opponent_team.values()])
            )

            if curr_opp_fainted > prev_opp_fainted:
                reward += w["ko_bonus"]
            if curr_self_fainted > prev_self_fainted:
                reward += w["ko_malus"]

        reward += w.get("step_cost", 0.0)
        self._last_turn = battle.turn
        return float(reward)

    def _potential_phi(self, battle: AbstractBattle, w: Dict[str, float]) -> float:
        our_hp = (
            float(np.sum([m.current_hp_fraction for m in battle.team.values()])) / 6.0
        )
        opp_hp = (
            float(
                np.sum([m.current_hp_fraction for m in battle.opponent_team.values()])
            )
            / 6.0
        )
        team_hp_adv = float(np.clip(our_hp - opp_hp, -1.0, 1.0))

        our_fainted = int(np.sum([int(m.fainted) for m in battle.team.values()])) / 6.0
        opp_fainted = (
            int(np.sum([int(m.fainted) for m in battle.opponent_team.values()])) / 6.0
        )
        fainted_adv = float(np.clip(opp_fainted - our_fainted, -1.0, 1.0))

        phi = w["team_hp_adv"] * team_hp_adv + w["fainted_adv"] * fainted_adv
        return float(np.clip(phi, -1.0, 1.0))

    # --------------------------
    # Observation / Embedding
    # --------------------------
    def _observation_size(self) -> int:
        return self._OBS_SIZE  # 27

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        27 dims total:
          [ my_hp, opp_hp, my_team_total_hp, opp_team_total_hp,
            move1(2), move2(2), move3(2), move4(2),
            bench1(3), bench2(3), bench3(3), bench4(3), bench5(3) ]
        """
        vec: List[float] = []

        me: Pokemon | None = battle.active_pokemon
        opp: Pokemon | None = battle.opponent_active_pokemon

        # --- Core HP features ---
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

        vec += [my_hp, opp_hp, my_team_total, opp_team_total]

        # --- Per-move expected-damage blocks (pad to 4) ---
        moves: List[Move] = list(battle.available_moves or [])[: self._N_MOVES]
        while len(moves) < self._N_MOVES:
            moves.append(None)

        for m in moves:
            vec += self._encode_move_block(m, me, opp)

        # --- Bench heuristics (up to 5), pad with None ---
        bench = [mon for mon in battle.team.values() if mon is not me]
        bench = bench[: self._BENCH_SLOTS]
        while len(bench) < self._BENCH_SLOTS:
            bench.append(None)

        sr_on_our_side = self._stealth_rock_active(battle.side_conditions)

        for bm in bench:
            # hp
            hp = float(getattr(bm, "current_hp_fraction", 0.0) or 0.0) if bm else 0.0

            # offensive STAB effectiveness vs opponent (max over its two types, as if it had a STAB move of that type)
            stab_eff = self._bench_stab_effectiveness_vs_opp(bm, opp)

            # expected SR damage fraction if we switch in (0 if no SR on our side)
            sr_dmg = self._expected_sr_damage(bm) if sr_on_our_side else 0.0

            vec += [hp, stab_eff, sr_dmg]

        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape[0] != self._OBS_SIZE:
            raise ValueError(
                f"embed_battle produced {arr.shape[0]} dims, expected {self._OBS_SIZE}"
            )
        return np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)

    # --------------------------
    # Move block (2 dims): expected damage score & priority
    # --------------------------
    def _encode_move_block(
        self, m: Move | None, me: Pokemon | None, opp: Pokemon | None
    ) -> List[float]:
        """
        [ expected_score, priority_gt0 ]
          expected_score = base_power_norm * accuracy * effectiveness * STAB / 6  (normalized to [0,1])
            - base_power_norm = min(base_power, 200) / 200
            - accuracy ∈ [0,1] (100% -> 1.0)
            - effectiveness ∈ {0, .5, 1, 2, 4}
            - STAB = 1.5 if STAB else 1.0
            - divide by 6 (= 1.5 * 4) to map max case to 1.0
        """
        if m is None:
            return [0.0, 0.0]

        # base power norm
        try:
            bp = float(max(0.0, min(200.0, getattr(m, "base_power", 0.0)))) / 200.0
        except Exception:
            bp = 0.0

        # accuracy -> [0,1]
        try:
            acc = getattr(m, "accuracy", 1.0)
            if acc is None:
                acc = 1.0
            acc = float(acc)
            if acc > 1.0:
                acc /= 100.0
            acc = float(np.clip(acc, 0.0, 1.0))
        except Exception:
            acc = 1.0

        # effectiveness vs current opponent
        eff = self._type_effectiveness(getattr(m, "type", None), opp)  # 0..4

        # STAB flag (poke-env usually provides m.stab)
        try:
            stab_mult = 1.5 if bool(getattr(m, "stab", False)) else 1.0
        except Exception:
            stab_mult = 1.0

        expected = bp * acc * eff * stab_mult
        # normalize by max possible (1 * 1 * 4 * 1.5 = 6)
        expected_norm = float(np.clip(expected / 6.0, 0.0, 1.0))

        # priority flag
        try:
            pr = 1.0 if getattr(m, "priority", 0) > 0 else 0.0
        except Exception:
            pr = 0.0

        return [expected_norm, pr]

    def _type_effectiveness(
        self, mtype: PokemonType | None, opp: Pokemon | None
    ) -> float:
        """Return effectiveness multiplier (0, 0.5, 1, 2, 4) best-effort."""
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
            if abs(mult - 0.5) < 1e-6:
                return 0.5
            if abs(mult - 1.0) < 1e-6:
                return 1.0
            if abs(mult - 2.0) < 1e-6:
                return 2.0
            if abs(mult - 4.0) < 1e-6:
                return 4.0
            return float(np.clip(mult, 0.0, 4.0))
        except Exception:
            return 1.0

    # --------------------------
    # Bench heuristics
    # --------------------------
    def _bench_stab_effectiveness_vs_opp(
        self, mon: Pokemon | None, opp: Pokemon | None
    ) -> float:
        """
        Offensive heuristic: if this bench mon attacked with a STAB-type move,
        what is the best type effectiveness vs the current opponent?
        Returns a normalized scalar in [0,1]: (1.5 * max_eff) / 6.
        """
        if mon is None or opp is None:
            return 0.0
        types = (getattr(mon, "type_1", None), getattr(mon, "type_2", None))
        best = 1.0
        gen = getattr(opp, "battle", None)
        gen = getattr(gen, "gen", 9)
        try:
            for t in types:
                if t is None:
                    continue
                # treat as if attacking with a move of type t
                eff = self._type_effectiveness(
                    t, opp
                )  # reuses same method (OK because it only multiplies)
                best = max(best, eff)
        except Exception:
            best = 1.0
        # include STAB multiplier (1.5) and normalize by 6
        return float(np.clip((1.5 * best) / 6.0, 0.0, 1.0))

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
        """
        Expected fraction of HP lost if Stealth Rock is up on OUR side:
          damage = 0.125 * rock_effectiveness_vs_mon  (already ∈ [0,1] in practice)
        """
        if mon is None:
            return 0.0
        try:
            mult = 1.0
            t1 = getattr(mon, "type_1", None)
            t2 = getattr(mon, "type_2", None)
            gen = getattr(mon, "battle", None)
            gen = getattr(gen, "gen", 9)
            rock = PokemonType.ROCK
            for t in (t1, t2):
                if t is None:
                    continue
                dm = float(rock.damage_multiplier(t, gen))
                mult *= dm
            dmg = 0.125 * mult
            return float(np.clip(dmg, 0.0, 1.0))
        except Exception:
            return 0.0


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
