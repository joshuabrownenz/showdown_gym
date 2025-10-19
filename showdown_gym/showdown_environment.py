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
    Simplified embedding (no status/weather/terrain) with reward curriculum.
    """

    # --------------------------
    # Constants / helper tables
    # --------------------------
    # Type order (18 types)
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

    _TYPE_TO_ENUM = {name: PokemonType.from_name(name) for name in _TYPE_ORDER}

    # Embedding sizes (keep in sync with _build_observation)
    _N_TYPE_PRIMARY = 18
    _N_TYPE_SECONDARY = 19  # includes "NONE" slot for mono types
    _MOVE_BLOCK = 27  # per move (defined in _encode_move)
    _N_MOVES = 4

    # Computed final size (update if you add/remove features)
    # My active:
    #   hp(1) + boosts(6) + typing(18+19=37) + tera(2) + known flags(2) = 48
    # Moves: 4 * 27 = 108
    # Opponent active:
    #   hp(1) + typing(37) = 38
    # Hazards/screens (no weather/terrain): 14
    # Team summaries: 5
    # Tempo bits: 1
    # TOTAL = 48 + 108 + 38 + 14 + 5 + 1 = 214
    _OBS_SIZE = 214

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
        reward_mode: str = "potential_v1",
        shaping_weights: Dict[str, float] | None = None,
    ):
        """
        reward_mode: "hp_delta" | "terminal_only" | "potential_v1" | "mixed"
        shaping_weights: optional weights for potential terms
        """
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self.rl_agent = account_name_one

        self.reward_mode = reward_mode
        # Default shaping weights (bounded; no status term anymore)
        self.shaping_weights = {
            "team_hp_adv": 0.60,
            "fainted_adv": 0.30,
            "hazard_adv": 0.08,
            "tempo": 0.02,
            "ko_bonus": 0.20,
            "ko_malus": -0.20,
            "supereff_hit": 0.05,
            "step_cost": 0.0,  # set negative for anti-stall if desired
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
          - hp_delta:        sum drop in opponent total HP
          - terminal_only:   +1 win / -1 loss (+ optional small step cost)
          - potential_v1:    potential-based shaping (HP/fainted/hazards/tempo) + event bonuses
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
        """
        Dense: reward is decrease in opponent total team HP since last step.
        """
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
        Potential-based shaping without status/weather/terrain.
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

            # Event bonuses: KO since last step
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

            # Tiny positive feedback for landing a super-effective hit (best-effort heuristic)
            if self._last_turn == prior_battle.turn:
                try:
                    me = battle.active_pokemon
                    opp = battle.opponent_active_pokemon
                    if opp and me and prior_battle.available_moves:
                        effs = [
                            self._type_effectiveness(m.type, opp)
                            for m in prior_battle.available_moves
                            if hasattr(m, "type")
                        ]
                        if len(effs) > 0 and max(effs) >= 2.0:
                            reward += w["supereff_hit"]
                except Exception:
                    pass

        reward += w.get("step_cost", 0.0)
        self._last_turn = battle.turn
        return float(reward)

    def _potential_phi(self, battle: AbstractBattle, w: Dict[str, float]) -> float:
        """
        Φ(s) = weighted, bounded sum of:
          - team HP advantage
          - fainted advantage
          - hazards advantage
          - tempo (trapped/screens proxy)
        """
        # Team HP advantage
        our_hp = (
            float(np.sum([m.current_hp_fraction for m in battle.team.values()])) / 6.0
        )
        opp_hp = (
            float(
                np.sum([m.current_hp_fraction for m in battle.opponent_team.values()])
            )
            / 6.0
        )
        team_hp_adv = np.clip(our_hp - opp_hp, -1.0, 1.0)

        # Fainted advantage
        our_fainted = int(np.sum([int(m.fainted) for m in battle.team.values()])) / 6.0
        opp_fainted = (
            int(np.sum([int(m.fainted) for m in battle.opponent_team.values()])) / 6.0
        )
        fainted_adv = np.clip(opp_fainted - our_fainted, -1.0, 1.0)

        # Hazards advantage
        hz_self = self._hazard_tuple(battle.side_conditions)
        hz_opp = self._hazard_tuple(battle.opponent_side_conditions)
        sr_s, sp_s, ts_s, web_s, screens_s = hz_self
        sr_o, sp_o, ts_o, web_o, screens_o = hz_opp
        hazard_score_self = (
            (1.0 if sr_s else 0.0) + 0.03 * sp_s + 0.03 * ts_s + 0.02 * web_s
        )
        hazard_score_opp = (
            (1.0 if sr_o else 0.0) + 0.03 * sp_o + 0.03 * ts_o + 0.02 * web_o
        )
        hazard_adv = np.clip(hazard_score_self - hazard_score_opp, -1.0, 1.0)

        # Tempo proxy: trapped and screens advantage
        me = battle.active_pokemon
        tempo = 0.0
        try:
            trapped = float(bool(getattr(me, "trapped", False)))
            tempo -= 0.25 * trapped
        except Exception:
            pass
        scr_self = sum(screens_s) / 8.0 if screens_s else 0.0
        scr_opp = sum(screens_o) / 8.0 if screens_o else 0.0
        tempo += np.clip(scr_self - scr_opp, -1.0, 1.0) * 0.25
        tempo = float(np.clip(tempo, -1.0, 1.0))

        phi = (
            w["team_hp_adv"] * team_hp_adv
            + w["fainted_adv"] * fainted_adv
            + w["hazard_adv"] * hazard_adv
            + w["tempo"] * tempo
        )
        return float(np.clip(phi, -1.0, 1.0))

    # --------------------------
    # Observation / Embedding
    # --------------------------
    def _observation_size(self) -> int:
        return self._OBS_SIZE  # 214 after simplification

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        """
        Simplified, fixed-size vector (214 dims):
          - My active: hp(1), boosts(6), typing(18+19), tera(2), known flags(2)
          - 4 moves x 27 each
          - Opp active: hp(1), typing(18+19)
          - Hazards/screens (self+opp): 14
          - Team summaries (alive_self, alive_opp, tot_hp_self, tot_hp_opp, turn_norm): 5
          - Tempo: trapped(1)
        """
        vec: List[float] = []

        # ----- My active -----
        me: Pokemon | None = battle.active_pokemon
        vec += [self._hp_frac(me)]
        vec += self._boosts_vector(me)  # 6 stats normalized

        t1, t2 = self._types_tuple(me)
        vec += self._type_one_hot_primary(t1)
        vec += self._type_one_hot_secondary(t2)

        vec += [float(getattr(battle, "can_tera", False))]
        vec += [float(getattr(battle, "tera_used", False))]

        vec += [float(bool(getattr(me, "item", None)))]
        vec += [float(bool(getattr(me, "ability", None)))]

        # ----- Moves (pad to 4) -----
        moves: List[Move] = list(battle.available_moves or [])[: self._N_MOVES]
        while len(moves) < self._N_MOVES:
            moves.append(None)

        opp: Pokemon | None = battle.opponent_active_pokemon
        for m in moves:
            vec += self._encode_move(m, opp)

        # ----- Opponent active -----
        vec += [self._hp_frac(opp)]
        ot1, ot2 = self._types_tuple(opp)
        vec += self._type_one_hot_primary(ot1)
        vec += self._type_one_hot_secondary(ot2)

        # ----- Hazards / screens (no weather/terrain) -----
        vec += self._encode_hazards_and_screens(battle)

        # ----- Team summaries -----
        my_team = list(battle.team.values())
        op_team = list(battle.opponent_team.values())
        alive_self = float(sum(1 - int(m.fainted) for m in my_team)) / 6.0
        alive_opp = float(sum(1 - int(m.fainted) for m in op_team)) / 6.0
        tot_hp_self = float(np.sum([m.current_hp_fraction for m in my_team])) / 6.0
        tot_hp_opp = float(np.sum([m.current_hp_fraction for m in op_team])) / 6.0
        turn_norm = np.tanh(battle.turn / 100.0)

        vec += [alive_self, alive_opp, tot_hp_self, tot_hp_opp, turn_norm]

        # ----- Tempo bits -----
        vec += [float(bool(getattr(me, "trapped", False)))]

        arr = np.asarray(vec, dtype=np.float32)
        if arr.shape[0] != self._OBS_SIZE:
            raise ValueError(
                f"embed_battle produced {arr.shape[0]} dims, expected {self._OBS_SIZE}"
            )
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        return arr

    # --------------------------
    # Helper encoders
    # --------------------------
    def _hp_frac(self, mon: Pokemon | None) -> float:
        try:
            return float(mon.current_hp_fraction) if mon is not None else 0.0
        except Exception:
            return 0.0

    def _boosts_vector(self, mon: Pokemon | None) -> List[float]:
        # Use six core combat-related entries; map [-6,+6] -> [0,1]
        vals = []
        keys = ("atk", "defense", "spa", "spd", "spe", "evasion")  # evasion as 6th slot
        for k in keys:
            try:
                stage = (
                    getattr(mon.boosts, k, 0)
                    if mon and getattr(mon, "boosts", None)
                    else 0
                )
            except Exception:
                stage = 0
            vals.append((stage + 6) / 12.0)
        return vals

    def _types_tuple(
        self, mon: Pokemon | None
    ) -> Tuple[PokemonType | None, PokemonType | None]:
        try:
            if mon is None:
                return None, None
            return getattr(mon, "type_1", None), getattr(mon, "type_2", None)
        except Exception:
            return None, None

    def _type_one_hot_primary(self, t: PokemonType | None) -> List[float]:
        vec = [0.0] * self._N_TYPE_PRIMARY
        try:
            if t is None:
                return vec
            name = str(t).split(".")[-1].upper()
            if name in self._TYPE_ORDER:
                vec[self._TYPE_ORDER.index(name)] = 1.0
        except Exception:
            pass
        return vec

    def _type_one_hot_secondary(self, t: PokemonType | None) -> List[float]:
        vec = [0.0] * self._N_TYPE_SECONDARY
        if t is None:
            vec[-1] = 1.0  # NONE slot
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

    def _encode_move(self, m: Move | None, opp: Pokemon | None) -> List[float]:
        """
        Per-move 27-dim encoding:
          [base_power, accuracy, cat_onehot(3), priority>0, pp_frac, type_onehot(18), STAB, eff_vs_opp]
        """
        out = []
        if m is None:
            return [0.0] * self._MOVE_BLOCK

        # base power (clip to 200)
        try:
            bp = float(max(0.0, min(200.0, getattr(m, "base_power", 0.0)))) / 200.0
        except Exception:
            bp = 0.0
        out.append(bp)

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
        out.append(acc)

        # category one-hot
        cat = str(getattr(m, "category", "")).upper()
        cats = ("PHYSICAL", "SPECIAL", "STATUS")
        out += [1.0 if cat.endswith(c) else 0.0 for c in cats]

        # priority > 0
        try:
            out.append(1.0 if getattr(m, "priority", 0) > 0 else 0.0)
        except Exception:
            out.append(0.0)

        # PP fraction
        try:
            cur_pp = float(getattr(m, "current_pp", 0))
            max_pp = float(getattr(m, "max_pp", 1) or 1)
            out.append(float(np.clip(cur_pp / max_pp, 0.0, 1.0)))
        except Exception:
            out.append(0.0)

        # type one-hot (18)
        try:
            t = getattr(m, "type", None)
            out += self._type_one_hot_primary(t)
        except Exception:
            out += [0.0] * self._N_TYPE_PRIMARY

        # STAB (best-effort via move.stab if present)
        try:
            stab = bool(getattr(m, "stab", False))
            out.append(1.0 if stab else 0.0)
        except Exception:
            out.append(0.0)

        # effectiveness vs opponent typing
        eff = self._type_effectiveness(getattr(m, "type", None), opp)
        out.append(float(np.clip(eff / 4.0, 0.0, 1.0)))

        return out

    def _type_effectiveness(
        self, mtype: PokemonType | None, opp: Pokemon | None
    ) -> float:
        """Return effectiveness multiplier (0, 0.5, 1, 2, 4) best-effort."""
        try:
            if mtype is None or opp is None:
                return 1.0
            mult = 1.0
            ot1, ot2 = self._types_tuple(opp)
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

    def _hazard_tuple(
        self, side_conditions: Dict[Any, Any]
    ) -> Tuple[bool, int, int, bool, Tuple[int, int, int]]:
        """
        Returns: (sr, spikes_layers, tspikes_layers, sticky_web, (reflect, light_screen, aurora_veil))
        """
        sr = False
        spikes = 0
        tspikes = 0
        web = False
        reflect = 0
        light_screen = 0
        aurora_veil = 0
        try:
            for k, v in (side_conditions or {}).items():
                name = str(k).upper()
                if "STEALTHROCK" in name or "STEALTH_ROCK" in name:
                    sr = True
                elif "SPIKES" in name and "TOXIC" not in name:
                    spikes = int(v or 1)
                elif "TOXICSPIKES" in name or "TOXIC_SPIKES" in name:
                    tspikes = int(v or 1)
                elif "STICKYWEB" in name or "STICKY_WEB" in name:
                    web = True
                elif "REFLECT" in name:
                    reflect = int(v or 0)
                elif "LIGHTSCREEN" in name or "LIGHT_SCREEN" in name:
                    light_screen = int(v or 0)
                elif "AURORAVEIL" in name or "AURORA_VEIL" in name:
                    aurora_veil = int(v or 0)
        except Exception:
            pass
        return (sr, spikes, tspikes, web, (reflect, light_screen, aurora_veil))

    def _encode_hazards_and_screens(self, battle: AbstractBattle) -> List[float]:
        """
        14 dims total (no weather/terrain):
          our: SR(1) Spikes(1) TSpikes(1) Web(1) + screens turns/8 (3)  => 7
          opp: SR(1) Spikes(1) TSpikes(1) Web(1) + screens turns/8 (3)  => 7
        """
        vec: List[float] = []
        self_hz = self._hazard_tuple(battle.side_conditions)
        opp_hz = self._hazard_tuple(battle.opponent_side_conditions)

        # Self hazards + screens
        vec += [
            1.0 if self_hz[0] else 0.0,
            float(self_hz[1]) / 3.0,
            float(self_hz[2]) / 2.0,
            1.0 if self_hz[3] else 0.0,
        ]
        vec += [c / 8.0 for c in self_hz[4]]

        # Opp hazards + screens
        vec += [
            1.0 if opp_hz[0] else 0.0,
            float(opp_hz[1]) / 3.0,
            float(opp_hz[2]) / 2.0,
            1.0 if opp_hz[3] else 0.0,
        ]
        vec += [c / 8.0 for c in opp_hz[4]]
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
