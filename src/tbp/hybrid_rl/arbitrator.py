"""Arbitrator: decides which action source to use per step.

Logic:
  1. Calibration phase (10 ep per source × 3 sources): forced Q/SAC/heuristic
  2. Normal phase: scoring = confidence × track_record[level]
     - ML track < heuristic track × 0.75: heuristic fallback
     - Q >> SAC: Q type + SAC params
     - Q > SAC, types differ: Q type + Q params
     - Q > SAC, types agree: blend params (50/50)
     - SAC > Q: SAC type + SAC params
"""

import logging
from collections import Counter, defaultdict, deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .experience_extractor import ExperienceExtractor
from .sac_actor import SACActorNetwork
from .rl_goal_approach_controller import (
    RLGoalApproachController,
    RunningQStats,
)

logger = logging.getLogger(__name__)


def sac_to_discrete(action_type: int, action_params: np.ndarray) -> int:
    if action_type == 0:
        angle = float(action_params[0]) % 360.0
        directions = [0, 45, 90, 135, 180, 225, 270, 315]
        best_idx = 0
        best_diff = 360.0
        for i, d in enumerate(directions):
            diff = abs(angle - d)
            diff = min(diff, 360 - diff)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx
    elif action_type == 1:
        step = float(action_params[0])
        if step < 0:
            return 9
        elif abs(step) <= 3.0:
            return 19
        else:
            return 8
    elif action_type == 2:
        angle = float(action_params[0])
        if abs(angle) > 10.0:
            return 22 if angle >= 0 else 23
        return 10 if angle >= 0 else 11
    elif action_type == 3:
        angle = float(action_params[0])
        if abs(angle) > 10.0:
            return 20 if angle >= 0 else 21
        return 12 if angle >= 0 else 13
    elif action_type == 4:
        return 14 if float(action_params[0]) >= 0 else 15
    elif action_type == 5:
        return 16
    elif action_type == 6:
        return 17
    elif action_type == 7:
        return 18
    return 0


class Arbitrator:
    """Decides which action source to use per step.

    Logic:
      1. Q-confident override (high confidence + spread > 3.0):
         - Q type == SAC type → SAC params (Q confirms SAC)
         - Q type != SAC type → heuristic (conflict resolution)
      2. Track record scoring:
         - ML below heuristic and not improving → heuristic
         - Heuristic budget = gap between heuristic and ML track
         - Otherwise → SAC (default)
    """

    def __init__(
        self,
        controller: RLGoalApproachController,
        sac_actor: Optional[SACActorNetwork] = None,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        param_mean: Optional[np.ndarray] = None,
        param_std: Optional[np.ndarray] = None,
        min_eval_per_source: int = 5,
    ):
        self.controller = controller
        self.sac_actor = sac_actor
        self.state_mean = state_mean
        self.state_std = state_std
        self.param_mean = param_mean
        self.param_std = param_std

        self.stats = {
            "q_store_chosen": 0,
            "sac_chosen": 0,
            "blend_chosen": 0,
            "heuristic_chosen": 0,
            "total_decisions": 0,
        }

        self.q_proposed_actions = defaultdict(int)
        self.sac_proposed_actions = defaultdict(int)
        self.q_chosen_actions = defaultdict(int)
        self.sac_chosen_actions = defaultdict(int)
        self.blend_chosen_actions = defaultdict(int)
        self.heuristic_chosen_actions = defaultdict(int)

        self.agreement_count = 0
        self.both_proposed_count = 0

        self.q_confidence_history = deque(maxlen=100)
        self.q_spread_history = deque(maxlen=100)

        self._current_episode_sources = []

        # Per-level track record
        self._min_eval_per_source = min_eval_per_source
        self._level_q_results: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self._level_sac_results: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self._level_heuristic_results: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        self._level_blend_results: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )

        # Level state (no calibration)
        self._current_level = 0
        self._is_calibrating = False
        self._calibration_counter = 0
        self._calibration_source = None
        self._episodes_on_level = 0

        self._param_dims = ExperienceExtractor.get_param_dims()
        self._type_names = ExperienceExtractor.get_type_names()

        # Running Q statistics
        self._running_q_stats_free = RunningQStats(warmup=200)
        self._running_q_stats_surface = RunningQStats(warmup=200)
        self._warmup_running_stats()

        # Strategic SAC references
        self._sac_strategic_detach: Optional[Any] = None
        self._sac_strategic_direction: Optional[Any] = None

        # Arbitrage-only stats
        self._arbitrage_stats = {
            "q_store_chosen": 0,
            "sac_chosen": 0,
            "blend_chosen": 0,
            "heuristic_chosen": 0,
            "total_decisions": 0,
        }

        # Heuristic budget tracking per level
        self._level_total_decisions: Dict[int, int] = defaultdict(int)
        self._level_heuristic_decisions: Dict[int, int] = defaultdict(int)
        self._heuristic_eps_min = 0.1

    def _warmup_running_stats(self):
        """Warm up RunningQStats from existing Q-store points."""
        MAX_WARMUP_POINTS = 10000

        for store, stats in [
            (self.controller.q_store_free, self._running_q_stats_free),
            (self.controller.q_store_surface, self._running_q_stats_surface),
        ]:
            if not store.points:
                continue

            pids = list(store.points.keys())
            if len(pids) > MAX_WARMUP_POINTS:
                pids = np.random.choice(
                    pids, size=MAX_WARMUP_POINTS, replace=False
                )

            for pid in pids:
                stats.update(store.points[pid].q_values)

            logger.info(
                f"Arbitrator warmup {store.name}: "
                f"mean={stats.mean:.3f}, std={stats.std:.3f}, "
                f"n={stats.count}, points_sampled={len(pids)}, "
                f"points_total={len(store.points)}"
            )

    def _get_heuristic_eps(self, ml_track: float, h_track: float) -> float:
        """Dynamic heuristic epsilon = gap between heuristic and ML.

        h_track=0.8, best_ml=0.6 → eps=0.2 (20% heuristic)
        h_track=0.8, best_ml=0.85 → eps=0.05 (ML better, minimum)
        """
        gap = h_track - ml_track
        return max(gap, self._heuristic_eps_min)

    def _get_level_tracks(self, level: int):
        """Compute ML and heuristic tracks for a level."""
        q_track = self._get_track(self._level_q_results[level])
        sac_track = self._get_track(self._level_sac_results[level])
        b_track = self._get_track(self._level_blend_results[level])
        h_track = max(self._get_track(self._level_heuristic_results[level]), 0.8)

        ml_tracks = []
        if len(self._level_q_results[level]) >= self._min_eval_per_source:
            ml_tracks.append(q_track)
        if len(self._level_sac_results[level]) >= self._min_eval_per_source:
            ml_tracks.append(sac_track)
        if len(self._level_blend_results[level]) >= self._min_eval_per_source:
            ml_tracks.append(b_track)
        best_ml_track = max(ml_tracks) if ml_tracks else 0.5
        worst_ml_track = min(ml_tracks) if ml_tracks else 0.5

        return q_track, sac_track, b_track, h_track, best_ml_track, worst_ml_track, len(ml_tracks) > 0
    
    def _is_ml_trend_increasing(self, level: int) -> bool:
        """Detect if any ML source is trending upward."""
        for results in [
            self._level_q_results[level],
            self._level_sac_results[level],
            self._level_blend_results[level],
        ]:
            if len(results) < 10:
                continue
            mid = len(results) // 2
            results_list = list(results)
            first_half = sum(results_list[:mid]) / max(mid, 1)
            second_half = sum(results_list[mid:]) / max(len(results_list) - mid, 1)
            if second_half > first_half + 0.05:
                return True
        return False

    def start_episode(self, level: int):
        if level != self._current_level:
            self._current_level = level
            self._episodes_on_level = 0
        self._episodes_on_level += 1
        self._current_episode_sources = []

    def decide(
        self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> Tuple[int, np.ndarray, str]:
        self.stats["total_decisions"] += 1
        level = self._current_level
        has_sac = self.sac_actor is not None

        self._level_total_decisions[level] += 1

        # === Get proposals ===
        q_action, q_confidence, q_spread = self._get_q_action(state)
        q_type = ExperienceExtractor.DISCRETE_TO_PSAC[q_action][0]
        q_params = self._discrete_to_params(q_action)

        if has_sac:
            sac_type, sac_params = self._get_sac_action_continuous(state)
        else:
            sac_type, sac_params = q_type, q_params.copy()

        # Track proposals
        q_name = self._type_names.get(q_type, f"type_{q_type}")
        self.q_proposed_actions[q_name] += 1
        self.q_confidence_history.append(q_confidence)
        self.q_spread_history.append(q_spread)
        if has_sac:
            sac_name = self._type_names.get(sac_type, f"type_{sac_type}")
            self.sac_proposed_actions[sac_name] += 1
            self.both_proposed_count += 1
            if q_type == sac_type:
                self.agreement_count += 1

        # === 1. Q-confident override ===
        q_conf_mean = (
            float(np.mean(self.q_confidence_history))
            if self.q_confidence_history
            else 0.0
        )
        q_conf_threshold = min(max(q_conf_mean * 0.9, 0.5), 1.0)

        if q_confidence >= q_conf_threshold and q_spread > 3.0:
            if q_type == sac_type:
                # 1.1 Types agree: Q confirms SAC → use SAC params
                self._record_decision("blend")
                self.blend_chosen_actions[q_name] += 1
                self._current_episode_sources.append("blend")
                return sac_type, sac_params, (
                    f"q_confirms_sac("
                    f"conf={q_confidence:.2f},"
                    f"spread={q_spread:.1f})"
                )
            else:
                # 1.2 Types differ: conflict → heuristic decides
                h_action = self._get_heuristic_action(
                    state, current_pose, sensor_data
                )
                h_type = ExperienceExtractor.DISCRETE_TO_PSAC[h_action][0]
                h_params = self._discrete_to_params(h_action)
                self._record_decision("heuristic")
                self._level_heuristic_decisions[level] += 1
                h_name = self._type_names.get(h_type, f"type_{h_type}")
                self.heuristic_chosen_actions[h_name] += 1
                self._current_episode_sources.append("heuristic")
                return h_type, h_params, (
                    f"q_sac_conflict("
                    f"q={q_name},sac={sac_name},"
                    f"conf={q_confidence:.2f},"
                    f"spread={q_spread:.1f})"
                )

        # === 2. Track record scoring ===
        q_track, sac_track, b_track, h_track, best_ml_track, worst_ml_track, has_enough_data = (
            self._get_level_tracks(level)
        )

        ml_trend_not_increasing = True
        if has_enough_data:
            ml_trend_not_increasing = not self._is_ml_trend_increasing(level)

        ml_below_heuristic = worst_ml_track < h_track

        use_heuristic = (
            has_enough_data
            and ml_below_heuristic
            # and ml_trend_not_increasing
        )

        if use_heuristic:
            total_on_level = max(self._level_total_decisions[level], 1)
            heuristic_on_level = self._level_heuristic_decisions[level]
            heuristic_ratio = heuristic_on_level / total_on_level
            current_eps = self._get_heuristic_eps(worst_ml_track, h_track)

            if heuristic_ratio < current_eps:
                h_action = self._get_heuristic_action(
                    state, current_pose, sensor_data
                )
                h_type = ExperienceExtractor.DISCRETE_TO_PSAC[h_action][0]
                h_params = self._discrete_to_params(h_action)
                self._record_decision("heuristic")
                self._level_heuristic_decisions[level] += 1
                h_name = self._type_names.get(h_type, f"type_{h_type}")
                self.heuristic_chosen_actions[h_name] += 1
                self._current_episode_sources.append("heuristic")
                return h_type, h_params, (
                    f"heuristic(ml_low,"
                    f"best_ml={best_ml_track:.2f},"
                    f"ht={h_track:.2f},"
                    f"eps={current_eps:.3f},"
                    f"used={heuristic_ratio:.3f})"
                )

        # === 3. SAC default ===
        if has_sac:
            sac_n = self._type_names.get(sac_type, f"type_{sac_type}")
            self._record_decision("sac")
            self.sac_chosen_actions[sac_n] += 1
            self._current_episode_sources.append("sac")
            return sac_type, sac_params, (
                f"sac(qt={q_track:.2f},"
                f"st={sac_track:.2f},"
                f"bt={b_track:.2f},"
                f"ht={h_track:.2f},"
                f"h_eps={self._get_heuristic_eps(worst_ml_track, h_track):.3f})"
            )

        # Fallback Q (no SAC)
        self._record_decision("q_store")
        self.q_chosen_actions[q_name] += 1
        self._current_episode_sources.append("q_store")
        return q_type, q_params, "q_fallback"

    def _record_decision(self, source: str):
        self.stats[f"{source}_chosen"] += 1
        self._arbitrage_stats[f"{source}_chosen"] += 1
        self._arbitrage_stats["total_decisions"] += 1

    def _get_track(self, results: deque) -> float:
        if len(results) < self._min_eval_per_source:
            return 0.5
        return sum(results) / len(results)

    def _discrete_to_params(self, discrete_action: int) -> np.ndarray:
        _, params_fn = ExperienceExtractor.DISCRETE_TO_PSAC[discrete_action]
        raw_params = params_fn(self.controller.config)
        padded = np.zeros(3, dtype=np.float32)
        padded[: len(raw_params)] = raw_params
        return padded

    def _get_sac_action_continuous(
        self, state: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        state_norm = state
        if self.state_mean is not None:
            state_norm = (
                (state - self.state_mean) / (self.state_std + 1e-8)
            )

        on_object = state[11] > 0.5

        self.sac_actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(
                state_norm.astype(np.float32)
            ).unsqueeze(0)

            type_logits, _, _ = self.sac_actor(state_t)

            if not on_object:
                type_logits[0, 0] = -1e9
                type_logits[0, 7] = -1e9
            if on_object:
                type_logits[0, 1] = -1e9
            if self.controller._consecutive_detach_count >= 3:
                type_logits[0, 7] = -1e9

            temperature = 0.3
            type_probs = torch.softmax(
                type_logits / temperature, dim=-1
            )
            type_probs = type_probs.clamp(min=1e-8)
            type_probs = type_probs / type_probs.sum(dim=-1, keepdim=True)
            type_dist = torch.distributions.Categorical(type_probs)
            action_type_t = type_dist.sample()
            action_type = action_type_t[0].item()

            _, ap, _, _ = self.sac_actor.sample_eval(
                state_t, temperature=temperature
            )
            dim = self._param_dims.get(action_type, 0)
            if dim > 0:
                _, param_mus, param_log_stds = self.sac_actor(state_t)
                if action_type in param_mus:
                    mu = param_mus[action_type][0]
                    log_std = param_log_stds[action_type][0]
                    std = (log_std.exp() * temperature).clamp(min=1e-6)
                    normal = torch.distributions.Normal(mu, std)
                    raw_sample = normal.rsample()
                    squashed = torch.tanh(raw_sample)

                    scale, center = self.sac_actor._get_scale_center(action_type)
                    if scale is not None:
                        scaled = squashed * scale[:dim] + center[:dim]
                    else:
                        scaled = squashed

                    padded = np.zeros(3, dtype=np.float32)
                    padded[:dim] = scaled.numpy()
                else:
                    padded = np.zeros(3, dtype=np.float32)
            else:
                padded = np.zeros(3, dtype=np.float32)

        return action_type, padded

    def _get_sac_params_for_type(
        self, state: np.ndarray, forced_type: int
    ) -> np.ndarray:
        if forced_type == 7:
            return np.zeros(3, dtype=np.float32)

        state_norm = state
        if self.state_mean is not None:
            state_norm = (state - self.state_mean) / (self.state_std + 1e-8)

        self.sac_actor.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(
                state_norm.astype(np.float32)
            ).unsqueeze(0)
            _, param_mus, _ = self.sac_actor(state_t)

            dim = self._param_dims.get(forced_type, 0)
            if (
                dim > 0
                and forced_type in param_mus
                and self.param_mean is not None
            ):
                raw_mu = param_mus[forced_type][0]
                squashed = torch.tanh(raw_mu)
                from .sac_actor import ACTION_PARAM_BOUNDS
                bounds = ACTION_PARAM_BOUNDS.get(forced_type, [])
                if bounds:
                    for i, (lo, hi) in enumerate(bounds[:dim]):
                        center = (hi + lo) / 2.0
                        scale = (hi - lo) / 2.0
                        squashed[i] = squashed[i] * scale + center
                params = squashed.numpy()[:dim]
                padded = np.zeros(3, dtype=np.float32)
                padded[:dim] = params
                return padded

        return np.zeros(3, dtype=np.float32)

    def _get_q_action(self, state):
        store = self.controller._select_store(state)
        running_stats = (
            self._running_q_stats_surface
            if state[11] > 0.5
            else self._running_q_stats_free
        )

        if store.next_id == 0:
            return 0, 0.0, 0.0

        q_values, confidence_info = (
            store.get_q_values_with_confidence(state)
        )
        confidence = confidence_info["overall"]

        if np.max(np.abs(q_values)) > 1e-8:
            running_stats.update(q_values)

        V = float(np.mean(q_values))
        if running_stats.is_warmed_up:
            V_norm = running_stats.normalize_value(V)
            if V_norm >= 0:
                boost = float(np.clip(V_norm * 0.15, 0.0, 0.3))
                q_confidence = min(confidence * (1.0 + boost), 0.95)
            else:
                penalty = float(np.clip(-V_norm * 0.3, 0.0, 0.5))
                q_confidence = confidence * (1.0 - penalty)
        else:
            q_confidence = confidence

        q_values = self.controller.apply_action_mask(q_values, state)

        on_object = state[11] > 0.5
        if on_object:
            sensor_proxy = self._get_sensor_proxy()
            if sensor_proxy is not None:
                same_side = sensor_proxy.get("same_side", True)
                path_blocked = sensor_proxy.get("path_blocked", False)
                if (
                    (not same_side or path_blocked)
                    and self.controller._can_detach(state)
                ):
                    t_state = (
                        self.controller
                        ._compute_detach_transition_state(
                            state, sensor_proxy,
                            movement_efficiency=(
                                self.controller
                                ._compute_movement_efficiency(window=20)
                            ),
                        )
                    )
                    s_q = (
                        self.controller.strategic_detach
                        .get_q_values(t_state)
                    )
                    if (
                        self.controller.strategic_detach.next_id > 0
                        and s_q[1] > s_q[0]
                    ):
                        detach_idx = self.controller.action_space.IDX_DETACH
                        q_values[detach_idx] = np.max(q_values) + 1.0

        if self.controller._consecutive_detach_count >= 3:
            q_values[self.controller.action_space.IDX_DETACH] = -1e9

        valid = q_values > -1e8
        if valid.sum() > 1:
            q_spread = float(
                np.max(q_values[valid]) - np.min(q_values[valid])
            )
        else:
            q_spread = 0.0

        eps = self.controller._get_current_epsilon()
        temperature = max(np.clip(0.5 * eps, 0.01, 0.5), 0.05)

        v = q_values.copy()
        v[~valid] = -1e9
        v = v / temperature
        v = v - np.max(v)
        exp_v = np.exp(v)
        probs = exp_v / exp_v.sum()
        q_action = int(np.random.choice(len(probs), p=probs))

        return q_action, q_confidence, q_spread

    def _get_sensor_proxy(self) -> Optional[dict]:
        if self.controller._prev_sensor_data is not None:
            return self.controller._prev_sensor_data
        return None

    def _get_heuristic_action(
        self,
        state: np.ndarray,
        current_pose: np.ndarray,
        sensor_data: Dict[str, Any],
    ) -> int:
        return self.controller._choose_action_heuristic(
            state=state,
            current_pose=current_pose,
            sensor_data=sensor_data,
        )

    def on_episode_end(self, success: bool):
        level = self._current_level
        counts = Counter(self._current_episode_sources)
        if counts:
            dominant = counts.most_common(1)[0][0]
            if dominant == "q_store":
                self._level_q_results[level].append(success)
            elif dominant == "blend":
                self._level_blend_results[level].append(success)
            elif dominant == "sac":
                self._level_sac_results[level].append(success)
            elif dominant == "heuristic":
                self._level_heuristic_results[level].append(success)
        self._current_episode_sources = []

    @property
    def q_success_rate(self) -> float:
        all_results = []
        for results in self._level_q_results.values():
            all_results.extend(results)
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    @property
    def sac_success_rate(self) -> float:
        all_results = []
        for results in self._level_sac_results.values():
            all_results.extend(results)
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    @property
    def heuristic_success_rate(self) -> float:
        all_results = []
        for results in self._level_heuristic_results.values():
            all_results.extend(results)
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    @property
    def blend_success_rate(self) -> float:
        all_results = []
        for results in self._level_blend_results.values():
            all_results.extend(results)
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    def get_stats(self) -> Dict[str, Any]:
        total = max(self.stats["total_decisions"], 1)

        agreement_rate = (
            self.agreement_count / max(self.both_proposed_count, 1)
        )
        q_conf_mean = (
            float(np.mean(self.q_confidence_history))
            if self.q_confidence_history
            else 0.0
        )
        q_spread_mean = (
            float(np.mean(self.q_spread_history))
            if self.q_spread_history
            else 0.0
        )

        def _top_actions(action_dict, n=15):
            sorted_actions = sorted(
                action_dict.items(), key=lambda x: -x[1]
            )
            total_actions = max(sum(action_dict.values()), 1)
            return {
                name: {
                    "count": count,
                    "rate": round(count / total_actions, 3),
                }
                for name, count in sorted_actions[:n]
            }

        level_stats = {}
        for level in sorted(
            set(
                list(self._level_q_results.keys())
                + list(self._level_sac_results.keys())
                + list(self._level_blend_results.keys())
                + list(self._level_heuristic_results.keys())
            )
        ):
            total_lvl = max(self._level_total_decisions.get(level, 0), 1)
            h_lvl = self._level_heuristic_decisions.get(level, 0)
            q_r, s_r, b_r, h_r, best_ml, worst_ml, _ = self._get_level_tracks(level)
            level_stats[f"level_{level}"] = {
                "q_rate": round(q_r, 3),
                "sac_rate": round(s_r, 3),
                "blend_rate": round(b_r, 3),
                "heuristic_rate": round(h_r, 3),
                "q_evals": len(self._level_q_results[level]),
                "sac_evals": len(self._level_sac_results[level]),
                "blend_evals": len(self._level_blend_results[level]),
                "heuristic_evals": len(self._level_heuristic_results[level]),
                "heuristic_budget_used": round(h_lvl / total_lvl, 3),
                "heuristic_eps": round(self._get_heuristic_eps(worst_ml, h_r), 4),
                "ml_trend_increasing": self._is_ml_trend_increasing(level),
            }

        return {
            "total_decisions": self.stats["total_decisions"],
            "q_store_rate": round(self.stats["q_store_chosen"] / total, 3),
            "sac_rate": round(self.stats["sac_chosen"] / total, 3),
            "blend_rate": round(self.stats.get("blend_chosen", 0) / total, 3),
            "heuristic_rate": round(
                self.stats["heuristic_chosen"] / total, 3
            ),
            "q_success_rate": round(self.q_success_rate, 3),
            "sac_success_rate": round(self.sac_success_rate, 3),
            "heuristic_success_rate": round(self.heuristic_success_rate, 3),
            "blend_success_rate": round(self.blend_success_rate, 3),
            "agreement_rate": round(agreement_rate, 3),
            "q_confidence_mean": round(q_conf_mean, 3),
            "q_spread_mean": round(q_spread_mean, 3),
            "is_calibrating": False,
            "current_level": self._current_level,
            "per_level_track_record": level_stats,
            "arbitrage_only": {
                "total_decisions": self._arbitrage_stats["total_decisions"],
                "q_store_rate": round(
                    self._arbitrage_stats["q_store_chosen"]
                    / max(self._arbitrage_stats["total_decisions"], 1),
                    3,
                ),
                "sac_rate": round(
                    self._arbitrage_stats["sac_chosen"]
                    / max(self._arbitrage_stats["total_decisions"], 1),
                    3,
                ),
                "blend_rate": round(
                    self._arbitrage_stats["blend_chosen"]
                    / max(self._arbitrage_stats["total_decisions"], 1),
                    3,
                ),
                "heuristic_rate": round(
                    self._arbitrage_stats["heuristic_chosen"]
                    / max(self._arbitrage_stats["total_decisions"], 1),
                    3,
                ),
            },
            "q_proposed_top": _top_actions(self.q_proposed_actions),
            "sac_proposed_top": _top_actions(self.sac_proposed_actions),
            "q_chosen_top": _top_actions(self.q_chosen_actions),
            "sac_chosen_top": _top_actions(self.sac_chosen_actions),
            "blend_chosen_top": _top_actions(self.blend_chosen_actions),
        }
    