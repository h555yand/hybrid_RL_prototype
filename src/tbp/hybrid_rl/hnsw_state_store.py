"""
HNSW-based State Store for Q-Learning.

Replaces traditional hash-table Q-table with continuous state space
using Hierarchical Navigable Small World graphs for fast KNN lookup
and Gaussian kernel interpolation for Q-value estimation.
"""

import os
import numpy as np
import hnswlib
import logging
from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict
from .config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class StatePoint:
    """Single point in state space with associated Q-values.

    Attributes:
        norm_state: Normalized state coordinates stored for index rebuild.
        q_values: Q-values for each discrete action.
        visit_count: How many times this point was accessed/updated.
        last_step: Global step when this point was last touched.
    """
    raw_state: np.ndarray           # NEW: исходный state (13D)
    norm_state: np.ndarray
    q_values: np.ndarray
    visit_count: int = 0
    last_step: int = 0
    on_object: int = 0              # NEW: если фильтровать/диагностировать

class HNSWStateStore:
    """Q-value store using HNSW index for nearest neighbor lookup.

    Key properties:
        - Linear memory growth (only visited states stored)
        - Smooth Q-function (kernel interpolation, not step function)
        - Local generalization (similar states share Q-values)

    State flow:
        get_q_values(state):
            state → normalize → KNN search → kernel interpolation → Q-values

        update_q_value(state, action, td_target):
            state → normalize → KNN search
            → if near existing point: update it
            → else: insert new point with interpolated init

    Args:
        state_dim: Dimensionality of state vector.
        num_actions: Number of discrete actions.
        max_points: Maximum points before eviction triggers.
        k_neighbors: Number of neighbors for KNN interpolation.
        sigma: Gaussian kernel bandwidth. Controls smoothness.
            Too small = no generalization (like tabular).
            Too large = over-smoothing (loses detail).
        insert_threshold: L2 distance below which we update existing
            point instead of inserting new one.
        evict_fraction: Fraction of points to remove when full.
        adaptive_sigma: If True, sigma adapts to local point density.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        name: str = "not defined"
    ):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.name = name
        self.state_dim = self.config["state_dim"]
        self.num_actions = self.config["num_actions"]
        self.max_points = self.config["max_points"]
        self.k_neighbors = self.config["k_neighbors"]
        self.sigma = self.config["sigma"]
        self.insert_threshold = self.config["insert_threshold"]
        self.evict_fraction = self.config["evict_fraction"]
        self.adaptive_sigma = self.config["adaptive_sigma"]
        self.auto_calibrate = self.config["auto_calibrate"]
        self.calibration_percentile = self.config["calibration_percentile"]
        self.min_calibration_samples = self.config["min_calibration_samples"]
        
        # Статистика для калибровки
        self._nn_distances = deque(maxlen=2000)
        self._is_calibrated = False
        self._calibration_interval = 500  # пересчитывать каждые N шагов

        self.min_weight_threshold = self.config["min_weight_threshold"]

        # HNSW index
        self.norm_warmup_steps = int(self.config.get("norm_warmup_steps", 5000))
        self.norm_min_std = float(self.config.get("norm_min_std", 1e-4))
        self.rebuild_on_freeze = bool(self.config.get("rebuild_on_freeze", True))
        self._norm_frozen = False
        self._freeze_done = False  # чтобы не фризить повторно

        self._init_index()

        # Point storage: id → StatePoint
        self.points: Dict[int, StatePoint] = {}
        self.next_id: int = 0
        self.global_step: int = 0
        self._updates_existing_count: int = 0
        self._insert_count: int = 0
        # Incremental eviction state
        self._deleted_count: int = 0
        self._rebuild_threshold: float = 0.3  # rebuild when ghost ratio > 30%

        # Running normalization statistics
        self._state_mean = np.zeros(self.config["state_dim"])
        self._state_std = np.ones(self.config["state_dim"])
        self._state_buffer: deque = deque(maxlen=max(self.norm_warmup_steps, 5000))
        self._norm_update_interval: int = 50
        self._norm_min_samples: int = 50

    def _update_normalization(self, state: np.ndarray):
        # всегда копим
        self._state_buffer.append(state.copy())

        # если уже заморожено — ничего не делаем
        if self._norm_frozen:
            return

        n = len(self._state_buffer)
        if n < self._norm_min_samples:
            return

        # можно оставить ваш интервал (каждые 50), но freeze будет на warmup_steps
        if n % self._norm_update_interval != 0:
            return

        buf = np.array(self._state_buffer)
        new_mean = buf.mean(axis=0)
        new_std = np.maximum(buf.std(axis=0), self.norm_min_std)

        self._state_mean = new_mean
        self._state_std = new_std

        # freeze условие
        if (not self._freeze_done) and n >= self.norm_warmup_steps:
            self._norm_frozen = True
            self._freeze_done = True
            logger.info(f"Normalization frozen on {n} samples for {self.name}")

            if self.rebuild_on_freeze and self.next_id > 0:
                self._rebuild_index_with_renorm()   # реализуем ниже

    # Calibration #############
    def _record_distance(self, nearest_distance_sq: float):
        """Записать расстояние до ближайшего соседа."""
        self._nn_distances.append(np.sqrt(max(nearest_distance_sq, 0)))
    
    def _maybe_recalibrate(self):
        """Пересчитать threshold если накопилось достаточно данных."""
        if not self.auto_calibrate:
            return
        
        n = len(self._nn_distances)
        if n < self.min_calibration_samples:
            return
        
        if n % self._calibration_interval != 0:
            return
        
        distances = np.array(self._nn_distances)
        
        # Основная идея:
        # Если типичное расстояние между последовательными шагами = d_step
        # То threshold ≈ d_step × multiplier
        #
        # Малый percentile расстояний ≈ "шаг между соседними состояниями"
        
        new_threshold = np.percentile(
            distances, self.calibration_percentile
        )
        
        # Ограничиваем разумным диапазоном
        new_threshold = np.clip(new_threshold, 0.01, 5.0)
        
        old_threshold = self.insert_threshold
        self.insert_threshold = new_threshold
        
        if not self._is_calibrated or abs(new_threshold - old_threshold) > 0.01:
            logger.info(
                f"Insert threshold: {old_threshold:.4f} → {new_threshold:.4f} "
                f"(from {n} samples, "
                f"median_nn_dist={np.median(distances):.4f}, "
                f"mean_nn_dist={np.mean(distances):.4f})"
            )
        
        self._is_calibrated = True

    # ══════════════════════════════════════════════════════════
    # INDEX MANAGEMENT
    # ══════════════════════════════════════════════════════════

    def _init_index(self):
        """Create or recreate HNSW index.

        HNSW parameters:
            space='l2': squared L2 distance (hnswlib convention)
            M=16: number of connections per layer (higher = more accurate,
                  more memory)
            ef_construction=200: build-time search depth (higher = better
                  index quality, slower build)
            ef=50: query-time search depth (higher = more accurate queries,
                  slower search)
        """
        self._index = hnswlib.Index(space="l2", dim=self.state_dim)
        self._index.init_index(
            max_elements=self.max_points,
            ef_construction=200,
            M=16,
            allow_replace_deleted=True,
        )
        self._index.set_ef(50)

    # ══════════════════════════════════════════════════════════
    # NORMALIZATION
    # ══════════════════════════════════════════════════════════

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize raw state for HNSW distance computation.

        Without normalization, features with larger scale (e.g. position
        error in mm: 0-100) dominate distance over smaller-scale features
        (e.g. curvature: 0-5), making KNN effectively ignore small features.

        Returns:
            Normalized state where each dimension has approximately
            zero mean and unit variance.
        """
        return (state - self._state_mean) / (self._state_std + 1e-8)

    def _rebuild_index_with_renorm(self):
        """Rebuild HNSW index after normalization is frozen/changed.

        Recomputes norm_state from raw_state for all ACTIVE points,
        keeping point IDs unchanged (important if index labels == ids).
        """
        logger.info(
            "Rebuilding HNSW index with re-normalization: points=%d", len(self.points)
        )
        if len(self.points) > self.max_points:
            self.max_points = len(self.points) + 1000
        self._init_index()
        self._deleted_count = 0

        if not self.points:
            return

        ids = np.array(list(self.points.keys()), dtype=np.int64)
        data = np.vstack([self._normalize(self.points[i].raw_state) for i in ids])

        # update stored norm_state too
        for j, pid in enumerate(ids):
            self.points[int(pid)].norm_state = data[j].copy()

        self._index.add_items(data, ids)
        logger.info(f"Rebuild with re-normalization done for {self.name}")

    # ══════════════════════════════════════════════════════════
    # GET Q-VALUES
    # ══════════════════════════════════════════════════════════

    def get_q_values(self, state):
        """Get Q-values for a state via KNN interpolation.

        This is the main query method. For a given state:

        1. If index is empty → return zeros (optimistic initialization)
        2. Find K nearest neighbors in HNSW
        3. If nearest neighbor is very close (< insert_threshold)
           → return that point's Q-values (exact match)
        4. Otherwise → Gaussian kernel weighted average of neighbors'
           Q-values

        The Gaussian kernel ensures smooth interpolation:
            w_i = exp(-dist_i / (2σ²))
            Q(s) = Σ w_i * Q(s_i) / Σ w_i

        Note: hnswlib returns SQUARED L2 distances, so we use
        dist directly (not dist²) in the kernel formula.

        Args:
            state: Raw (unnormalized) state vector [state_dim].

        Returns:
            Q-values array [num_actions]. Interpolated from neighbors
            or zeros if no data.
        """

        if self.next_id == 0:
            return np.zeros(self.num_actions)
        
        norm_state = self._normalize(state)
        k = min(self.k_neighbors, self.next_id)
        
        labels, distances = self._index.knn_query(
            norm_state.reshape(1, -1), k=k
        )
        labels = labels[0]
        distances = distances[0]
        
        # Точное совпадение
        if distances[0] < self.insert_threshold ** 2:
            point = self.points[labels[0]]
            point.visit_count += 1
            point.last_step = self.global_step
            return point.q_values.copy()
        
        # ═══ НОВОЕ: проверка достоверности ═══
        sigma = self._get_sigma(distances)
        weights = self._gaussian_kernel(distances, sigma)
        weight_sum = weights.sum()
        
        # Если суммарный вес слишком мал → мы далеко от всех точек
        # → возвращаем нули (честное "не знаю")
        if weight_sum < self.min_weight_threshold:
            return np.zeros(self.num_actions)
        
        # ═══ НОВОЕ: confidence-weighted interpolation ═══
        # Вместо полной нормализации, используем confidence
        # confidence = насколько мы доверяем интерполяции
        #
        # confidence = 1.0 когда соседи рядом (weight_sum большой)
        # confidence → 0.0 когда соседи далеко (weight_sum маленький)
        
        max_possible_weight = k * 1.0  # если все соседи на distance=0
        confidence = min(weight_sum / max_possible_weight, 1.0)
        
        # Нормализуем веса для интерполяции
        weights /= weight_sum
        
        q_values = np.zeros(self.num_actions)
        for i, label in enumerate(labels):
            q_values += weights[i] * self.points[label].q_values
        
        # Масштабируем по confidence
        # Далеко от известных точек → Q-values ближе к нулю
        q_values *= confidence
        
        return q_values

    # ══════════════════════════════════════════════════════════
    # UPDATE Q-VALUES
    # ══════════════════════════════════════════════════════════

    def update_q_value(
        self,
        state,
        action,
        td_target,
        alpha=0.1,
        count_visit: bool = True,
    ):
        """Обновлённый метод с калибровкой.
        Update Q(state, action) toward td_target.

        Two cases:
        1. Existing point nearby (distance < insert_threshold):
           → Standard Q-update on that point
           → Q(s,a) += α * (td_target - Q(s,a))

        2. No nearby point:
           → Insert new point
           → Initialize Q-values by interpolating from neighbors
           → Apply Q-update for current action

        This means the store grows organically: dense where the agent
        visits often, sparse where it rarely goes.

        Args:
            state: Raw state vector [state_dim].
            action: Action index (0 to num_actions-1).
            td_target: Target value = reward + γ * max_a Q(s', a).
            alpha: Learning rate for Q-update.
            count_visit: If True (default), increment visit_count and update
                last_step. Pass False for offline/backup passes so that
                synthetic Q-refinements do not inflate the eviction score
                (which uses visit_count × recency). last_step is still
                refreshed so successful-path points stay eviction-fresh.
        """

        self.global_step += 1
        self._update_normalization(state)
        norm_state = self._normalize(state)
        
        if self.next_id == 0:
            self._insert_point(state, norm_state, action, td_target, alpha)
            return
        
        k = min(self.k_neighbors, self.next_id)
        labels, distances = self._index.knn_query(
            norm_state.reshape(1, -1), k=k
        )
        labels = labels[0]
        distances = distances[0]
        
        # Записываем расстояние для калибровки
        self._record_distance(distances[0])
        self._maybe_recalibrate()
        
        # Используем текущий (возможно обновлённый) threshold
        if distances[0] < self.insert_threshold ** 2:
            point = self.points[labels[0]]
            point.q_values[action] += alpha * (
                td_target - point.q_values[action]
            )
            if count_visit:
                point.visit_count += 1
            point.last_step = self.global_step  # always refresh recency
            self._updates_existing_count += 1
        else:
            if count_visit:
                self._insert_point(state, norm_state, action, td_target, alpha)
            # else: backup hit an unknown region — skip insertion;
            # we only refine states the agent actually visited online.

    # ══════════════════════════════════════════════════════════
    # POINT INSERTION
    # ══════════════════════════════════════════════════════════
    def _insert_point(self, raw_state: np.ndarray, norm_state: np.ndarray, action: int, td_target: float, alpha: float):
        """Insert a new state point into the HNSW index.

        Q-values are initialized by interpolating from existing neighbors
        (if any), then the current Q-update is applied. This gives new
        points a reasonable starting estimate instead of zeros.

        Triggers eviction if index is at capacity.
        """
        if len(self.points) >= self.max_points:
            self._evict_points()

        # Initialize Q-values from neighbors
        q_init = self._interpolate_q_init(norm_state)

        # Apply current Q-update
        q_init[action] += alpha * (td_target - q_init[action])

        # Create and store point
        point_id = self.next_id
        point = StatePoint(
            raw_state=np.array(raw_state, copy=True),
            norm_state=norm_state.copy(),
            q_values=q_init,
            visit_count=1,
            last_step=self.global_step,
            on_object=int(raw_state[9] > 0.5),
        )

        do_replace = self._deleted_count > 0
        self._index.add_items(
            norm_state.reshape(1, -1),
            np.array([point_id]),
            replace_deleted=do_replace,
        )
        if do_replace:
            self._deleted_count -= 1
        self.points[point_id] = point
        self.next_id += 1
        self._insert_count += 1

        if self.next_id % 1000 == 0:
            logger.info(
                f"HNSW store {self.name}: {self.next_id} points, "
                f"step {self.global_step}"
            )

    def _interpolate_q_init(self, norm_state: np.ndarray) -> np.ndarray:
        """Initialize Q-values for new point from existing neighbors.

        Uses same Gaussian kernel interpolation as get_q_values.
        Returns zeros if index is empty or all neighbors too far.
        """
        if self.next_id == 0:
            return np.zeros(self.num_actions)

        k = min(self.k_neighbors, self.next_id)
        labels, distances = self._index.knn_query(
            norm_state.reshape(1, -1), k=k
        )

        sigma = self._get_sigma(distances[0])
        weights = self._gaussian_kernel(distances[0], sigma)
        weight_sum = weights.sum()

        if weight_sum < 1e-10:
            return np.zeros(self.num_actions)

        weights /= weight_sum

        q_init = np.zeros(self.num_actions)
        for i, label in enumerate(labels[0]):
            q_init += weights[i] * self.points[label].q_values

        return q_init

    # ══════════════════════════════════════════════════════════
    # KERNEL
    # ══════════════════════════════════════════════════════════

    def _gaussian_kernel(
        self,
        squared_distances: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """Compute Gaussian kernel weights from squared L2 distances.

        w_i = exp(-d_i² / (2σ²))

        hnswlib returns squared L2 distances, so we use them directly.

        Args:
            squared_distances: Array of squared L2 distances from hnswlib.
            sigma: Kernel bandwidth.

        Returns:
            Unnormalized weights (caller must normalize).
        """
        return np.exp(-squared_distances / (2.0 * sigma ** 2))

    def _get_sigma(self, distances: np.ndarray) -> float:
        """Get kernel bandwidth, optionally adaptive.

        Fixed sigma: use self.sigma always.

        Adaptive sigma: set sigma proportional to median neighbor
        distance. This automatically adjusts to local point density:
            - Dense region → small sigma → precise interpolation
            - Sparse region → large sigma → broader generalization

        Args:
            distances: Squared L2 distances to neighbors.

        Returns:
            Sigma value to use for kernel computation.
        """
        if not self.adaptive_sigma:
            return self.sigma

        # Convert squared distances to actual distances
        actual_distances = np.sqrt(np.maximum(distances, 0))
        median_dist = np.median(actual_distances)

        # sigma = half the median distance, with floor
        adaptive = max(median_dist * 0.5, 0.1)

        # Blend with base sigma to prevent extreme values
        return 0.7 * adaptive + 0.3 * self.sigma

    """
    median_dist * 0.5
    Идея: взять характерный локальный масштаб (медиану дистанций до соседей) и сделать ядро уже этого масштаба.
    При    σ    ≈    0.5    ⋅    median_dist
    σ≈0.5⋅median_dist ближайшие соседи доминируют, но не один-единственный сосед. Это баланс между:
    слишком маленьким    σ    σ (почти tabular, шумно),    слишком большим    σ    σ (пересглаживание, теряется локальная структура).
    Нижняя граница 0.1
    Нужна как предохранитель, когда точки очень плотные и median_dist почти ноль.
    ) становятся почти нулевыми для всех, кроме точного совпадения:
    interpolation становится «рваной»,
    чаще получаются вырожденные веса,
    хуже обобщение.
    Почему это обычно работает в вашем коде:
    дальше еще есть смешивание с базовым sigma (0.7 adaptive + 0.3 base), это дополнительно стабилизирует.
    поэтому 0.5 и 0.1 — рабочие стартовые значения для широкого диапазона плотностей.
    Если хотите тюнить:
    Более гладко: увеличить 0.5 до 0.7–1.0.
    Более локально: уменьшить 0.5 до 0.3–0.4.
    Если много почти одинаковых состояний: поднять floor с 0.1 до 0.15–0.2.
    Если нужно тоньше различать очень близкие состояния: опустить floor до 0.05 (но осторожно со стабильностью).
    """
    # ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
    # EVICTION
    # ══════════════════════════════════════════════════════════

    def _evict_points(self):
        """Remove least useful points via incremental mark_deleted.

        Uses hnswlib mark_deleted() which is O(1) per point — the
        point stays in the index graph but is excluded from search
        results. Much faster than full rebuild for large indices.

        Full rebuild triggers automatically when ghost ratio exceeds
        _rebuild_threshold (default 30%), because too many ghosts
        degrade search quality and waste memory.

        Usefulness score:
            score = visit_count × recency
            recency = 1 / (1 + global_step - last_step)
        """
        n_active = len(self.points)
        n_evict = max(1, int(n_active * self.evict_fraction))

        logger.info(
            f"HNSW eviction: removing {n_evict} of {n_active} active points"
        )

        # Score and sort: lowest score = first to evict
        scores = self._compute_eviction_scores()
        sorted_ids = sorted(scores, key=scores.get)

        for pid in sorted_ids[:n_evict]:
            self._index.mark_deleted(pid)
            del self.points[pid]
            self._deleted_count += 1

        # Full rebuild when ghost ratio hurts search quality
        ghost_ratio = self._deleted_count / max(self.next_id, 1)
        if ghost_ratio > self._rebuild_threshold:
            logger.info(
                f"Ghost ratio {ghost_ratio:.1%} > "
                f"{self._rebuild_threshold:.0%}, triggering full rebuild"
            )
            self._rebuild_from_active()
            return

        # Ensure index has room for next insertion
        if self.next_id >= self._index.get_max_elements():
            new_max = self.next_id + max(1000, int(self.max_points * 0.1))
            self._index.resize_index(new_max)
            logger.debug(
                f"HNSW index resized to {new_max} elements"
            )

        logger.info(
            f"HNSW eviction complete (incremental): "
            f"{len(self.points)} active, {self._deleted_count} ghosts"
        )

    def _compute_eviction_scores(self) -> Dict[int, float]:
        """Score each active point by usefulness.

        score = visit_count × recency
        recency = 1 / (global_step - last_step + 1)

        Lower score → evicted first.
        """
        scores = {}
        for pid, point in self.points.items():
            age = self.global_step - point.last_step + 1
            recency = 1.0 / age
            scores[pid] = point.visit_count * recency
        return scores

    def _rebuild_index(self, surviving_points: Dict[int, StatePoint]):
        """Rebuild HNSW index from scratch with surviving points.

        All points get new sequential IDs. Resets ghost/deleted state.

        Args:
            surviving_points: Dict of points that survived eviction.
        """
        if len(self.points) > self.max_points:
            self.max_points = len(self.points) + 1000
        self._init_index()
        self.points = {}
        self.next_id = 0
        self._deleted_count = 0

        for old_id, point in surviving_points.items():
            new_id = self.next_id

            self._index.add_items(
                point.norm_state.reshape(1, -1),
                np.array([new_id]),
            )

            self.points[new_id] = StatePoint(
                raw_state=point.raw_state.copy(),
                norm_state=point.norm_state.copy(),
                q_values=point.q_values.copy(),
                visit_count=point.visit_count,
                last_step=point.last_step,
                on_object=point.on_object,
            )
            self.next_id += 1

    def _rebuild_from_active(self):
        """Full rebuild using only currently active points.

        Called when ghost ratio exceeds threshold. Compacts the index
        by removing all ghost slots and reassigning sequential IDs.
        """
        surviving = dict(self.points)
        self._rebuild_index(surviving)
        logger.info(
            f"HNSW full rebuild: {len(self.points)} points, "
            f"next_id reset to {self.next_id}"
        )

    # ══════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Return diagnostic statistics for logging/monitoring.

        Useful for tracking:
            - Memory growth (num_points)
            - Learning progress (q_magnitude growing)
            - Coverage (visit distribution)
            - Normalization health (state_mean/std)

        Returns:
            Dictionary with diagnostic values.
        """
        total_updates = self._updates_existing_count + self._insert_count
        
        if not self.points:
            return {
                "num_points": 0,
                "global_step": self.global_step,
                "update_hit_rate": 0.0,
                "active_to_created_ratio": 0.0,
            }

        visit_counts = [p.visit_count for p in self.points.values()]
        q_magnitudes = [
            np.max(np.abs(p.q_values)) for p in self.points.values()
        ]
        q_spreads = [
            np.max(p.q_values) - np.min(p.q_values)
            for p in self.points.values()
        ]

        stats = {
            # ... существующие метрики ...
            "num_points": len(self.points),
            "global_step": self.global_step,
            "updates_existing": self._updates_existing_count,
            "inserts": self._insert_count,
            "update_hit_rate": (
                self._updates_existing_count / max(total_updates, 1)
            ),
            "active_to_created_ratio": (
                len(self.points) / max(self.next_id, 1)
            ),
            "visits_mean": float(np.mean(visit_counts)),
            "visits_max": int(np.max(visit_counts)),
            "visits_median": float(np.median(visit_counts)),
            "q_magnitude_mean": float(np.mean(q_magnitudes)),
            "q_magnitude_max": float(np.max(q_magnitudes)),
            "q_spread_mean": float(np.mean(q_spreads)),
            "q_spread_max": float(np.max(q_spreads)),
            "state_mean": self._state_mean.tolist(),
            "state_std": self._state_std.tolist(),
            
            # Threshold диагностика
            "insert_threshold": self.insert_threshold,
            "is_calibrated": self._is_calibrated,
            "points_per_update_ratio": (
                self.next_id / max(self.global_step, 1)
            ),
            # ratio ≈ 0.01 → слишком мало точек (threshold большой)
            # ratio ≈ 1.0  → каждый шаг новая точка (threshold маленький)
            # ratio ≈ 0.1-0.3 → хорошо
            "adaptive_sigma": self.adaptive_sigma,
        }
        
        if self._nn_distances:
            distances = np.array(self._nn_distances)
            stats.update({
                "nn_distance_median": float(np.median(distances)),
                "nn_distance_p10": float(np.percentile(distances, 10)),
                "nn_distance_p90": float(np.percentile(distances, 90)),
            })
        
        return stats

    def get_nearest_points_info(
        self,
        state: np.ndarray,
        k: Optional[int] = None,
    ) -> list:
        """Get info about nearest neighbors for debugging.

        Useful for understanding WHY a particular Q-value was returned:
        which neighbors contributed and with what weights.

        Args:
            state: Raw state vector.
            k: Number of neighbors (defaults to self.k_neighbors).

        Returns:
            List of dicts with neighbor info, sorted by distance.
        """
        if self.next_id == 0:
            return []

        if k is None:
            k = self.k_neighbors
        k = min(k, self.next_id)

        norm_state = self._normalize(state)
        labels, distances = self._index.knn_query(
            norm_state.reshape(1, -1), k=k
        )
        labels = labels[0]
        distances = distances[0]

        sigma = self._get_sigma(distances)
        weights = self._gaussian_kernel(distances, sigma)
        weight_sum = weights.sum()
        if weight_sum > 1e-10:
            normalized_weights = weights / weight_sum
        else:
            normalized_weights = np.zeros_like(weights)

        result = []
        for i, label in enumerate(labels):
            point = self.points[label]
            result.append({
                "point_id": int(label),
                "squared_distance": float(distances[i]),
                "distance": float(np.sqrt(max(distances[i], 0))),
                "weight": float(normalized_weights[i]),
                "q_values": point.q_values.tolist(),
                "best_action": int(np.argmax(point.q_values)),
                "visit_count": point.visit_count,
                "last_step": point.last_step,
            })

        return result

    # ══════════════════════════════════════════════════════════
    # PERSISTENCE
    # ══════════════════════════════════════════════════════════

    def save(self, filepath: str):
        """Save store state to disk.

        Saves:
            - All points (norm_state, q_values, metadata)
            - Normalization statistics
            - Configuration

        HNSW index is rebuilt on load (not saved separately)
        because it depends on hnswlib version/platform.

        Args:
            filepath: Path to .npz file.
        """
        if not self.points:
            np.savez(
                filepath,
                config=np.array([
                    self.state_dim,
                    self.num_actions,
                    self.max_points,
                    self.k_neighbors,
                    self.global_step,
                    self.next_id,
                ]),
                sigma=np.array([self.sigma]),
                state_mean=self._state_mean,
                state_std=self._state_std,
            )
            return

        # Pack all points into arrays
        ids = sorted(self.points.keys())
        norm_states = np.array([self.points[i].norm_state for i in ids])
        q_values = np.array([self.points[i].q_values for i in ids])
        visit_counts = np.array([self.points[i].visit_count for i in ids])
        last_steps = np.array([self.points[i].last_step for i in ids])
        raw_states = np.array([self.points[i].raw_state for i in ids])

        np.savez(
            filepath,
            config=np.array([
                self.state_dim,
                self.num_actions,
                self.max_points,
                self.k_neighbors,
                self.global_step,
                self.next_id,
            ]),
            sigma=np.array([self.sigma]),
            state_mean=self._state_mean,
            state_std=self._state_std,
            point_ids=np.array(ids),
            norm_states=norm_states,
            q_values=q_values,
            visit_counts=visit_counts,
            last_steps=last_steps,
            raw_states=raw_states,
        )

        logger.info(f"HNSW store saved: {len(ids)} points to {filepath}")

    def save_with_index(self, filepath: str):
        """Save store state AND native HNSW index to disk.

        Produces two files:
            - <filepath>.npz  — points, config, normalization (same as save())
            - <filepath>.hnsw — native hnswlib binary index

        On load_with_index the binary index is memory-mapped directly,
        avoiding O(N log N) rebuild.  Falls back to rebuild if the .hnsw
        file is missing or incompatible (different hnswlib version).

        Args:
            filepath: Base path. Extensions are appended automatically.
                      e.g. "model/store" → "model/store.npz" + "model/store.hnsw"
        """
        # base = filepath.removesuffix(".npz")
        base = filepath[:-4] if filepath.endswith(".npz") else filepath
        npz_path = base + ".npz"
        hnsw_path = base + ".hnsw"

        # 1. Save points & config via existing method
        self.save(npz_path)

        # 2. Save native HNSW index binary
        if self.next_id > 0:
            self._index.save_index(hnsw_path)
            logger.info(
                f"HNSW index saved: {self.next_id} points to {hnsw_path}"
            )

        # 3. Save calibration state that vanilla save() skips
        calibration_path = base + ".cal.npz"
        cal_data = {
            "insert_threshold": np.array([self.insert_threshold]),
            "is_calibrated": np.array([self._is_calibrated]),
            "deleted_count": np.array([self._deleted_count]),
        }
        if self._nn_distances:
            cal_data["nn_distances"] = np.array(self._nn_distances)
        np.savez(calibration_path, **cal_data)

    @classmethod
    def load_with_index(
        cls, filepath: str, extra_cfg
    ) -> "HNSWStateStore":
        """Load store with pre-built HNSW index (fast path).

        Expects files produced by save_with_index():
            - <filepath>.npz  — points & config
            - <filepath>.hnsw — native index binary
            - <filepath>.cal.npz — calibration state (optional)

        If the .hnsw file is missing or fails to load, falls back to
        the standard load() which rebuilds the index from points.

        Args:
            filepath: Base path (same as passed to save_with_index).
            extra_cfg: Override config parameters.

        Returns:
            Restored HNSWStateStore.
        """
        # base = filepath.removesuffix(".npz")
        base = filepath[:-4] if filepath.endswith(".npz") else filepath
        npz_path = base + ".npz"
        hnsw_path = base + ".hnsw"
        calibration_path = base + ".cal.npz"

        # --- Fallback: no native index → rebuild from points ----------
        if not os.path.exists(hnsw_path):
            logger.warning(
                f"No HNSW index file at {hnsw_path}, "
                "falling back to rebuild from points"
            )
            return cls.load(npz_path, extra_cfg)

        # --- Fast path: load .npz metadata + native index -------------
        data = np.load(npz_path, allow_pickle=False)

        config = data["config"]
        state_dim = int(config[0])
        num_actions = int(config[1])
        max_points = int(config[2])
        k_neighbors = int(config[3])
        global_step = int(config[4])
        next_id = int(config[5])

        store_cfg = {
            "state_dim": state_dim,
            "num_actions": num_actions,
            "max_points": max_points,
            "k_neighbors": k_neighbors,
            "sigma": float(data["sigma"][0]),
        }
        # store_cfg.update(extra_cfg)
        store_cfg.update(extra_cfg or {})

        store = cls(config=store_cfg)
        store.global_step = global_step
        store._state_mean = data["state_mean"]
        store._state_std = data["state_std"]

        # Load native HNSW index
        try:
            store._index = hnswlib.Index(space="l2", dim=state_dim)
            store._index.load_index(
                hnsw_path,
                max_elements=store_cfg["max_points"],
                allow_replace_deleted=True,
            )
            store._index.set_ef(50)
        except Exception as exc:
            logger.warning(
                f"Failed to load HNSW index ({exc}), "
                "falling back to rebuild from points"
            )
            return cls.load(npz_path, extra_cfg)

        # Restore StatePoint objects (without re-inserting into index)
        if "norm_states" in data:
            norm_states = data["norm_states"]
            q_values = data["q_values"]
            visit_counts = data["visit_counts"]
            last_steps = data["last_steps"]
            raw_states = data["raw_states"] if "raw_states" in data else None

            # Use saved IDs to match native HNSW index labels.
            # Without this, sequential 0..N-1 IDs mismatch the HNSW
            # labels after incremental eviction (mark_deleted), causing
            # KeyError or silent wrong Q-value reads.
            if "point_ids" in data:
                point_ids = data["point_ids"]
            else:
                # Legacy files without point_ids: assume sequential.
                # Safe only if no evictions happened before save.
                point_ids = np.arange(len(norm_states))

            for i in range(len(norm_states)):
                pid = int(point_ids[i])
                store.points[pid] = StatePoint(
                    raw_state=raw_states[i] if raw_states is not None else norm_states[i].copy(),
                    norm_state=norm_states[i],
                    q_values=q_values[i],
                    visit_count=int(visit_counts[i]),
                    last_step=int(last_steps[i]),
                    on_object=int((raw_states[i][9] if raw_states is not None else 0.0) > 0.5)
                )
            store.next_id = next_id
            store._norm_frozen = True
            store._freeze_done = True

        # Restore calibration state
        if os.path.exists(calibration_path):
            try:
                cal = np.load(calibration_path, allow_pickle=False)
                store.insert_threshold = float(cal["insert_threshold"][0])
                store._is_calibrated = bool(cal["is_calibrated"][0])
                if "nn_distances" in cal:
                    store._nn_distances = deque(
                        cal["nn_distances"].tolist(), maxlen=2000
                    )
                if "deleted_count" in cal:
                    store._deleted_count = int(cal["deleted_count"][0])
            except Exception as exc:
                logger.warning(f"Could not restore calibration: {exc}")

        logger.info(
            f"HNSW store loaded (fast): {store.next_id} points "
            f"from {npz_path} + {hnsw_path}"
        )
        return store

    @classmethod
    def load(cls, filepath: str, extra_cfg) -> "HNSWStateStore":
        """Load store state from disk and rebuild HNSW index.

        Args:
            filepath: Path to .npz file saved by save().
            extra_cfg: Override any config parameter
                (e.g. max_points=100000 to increase capacity).

        Returns:
            Restored HNSWStateStore with rebuilt index.
        """
        data = np.load(filepath, allow_pickle=False)

        config = data["config"]
        state_dim = int(config[0])
        num_actions = int(config[1])
        max_points = int(config[2])
        k_neighbors = int(config[3])
        global_step = int(config[4])

        # Create store with saved or overridden config
        store_cfg = {
            "state_dim": state_dim,
            "num_actions": num_actions,
            "max_points": max_points,
            "k_neighbors": k_neighbors,
            "sigma": float(data["sigma"][0]),
        }
        # store_cfg.update(extra_cfg)
        store_cfg.update(extra_cfg or {})

        store = cls(config=store_cfg)
        store.global_step = global_step
        store._state_mean = data["state_mean"]
        store._state_std = data["state_std"]

        # Restore points and rebuild index
        if "norm_states" in data:
            norm_states = data["norm_states"]
            q_values = data["q_values"]
            visit_counts = data["visit_counts"]
            last_steps = data["last_steps"]
            raw_states = data["raw_states"] if "raw_states" in data else None

            if "point_ids" in data:
                point_ids = data["point_ids"]
            else:
                # Legacy files without point_ids: assume sequential.
                # Safe only if no evictions happened before save.
                point_ids = np.arange(len(norm_states))

            for i in range(len(norm_states)):
                pid = int(point_ids[i])
                store.points[pid] = StatePoint(
                    raw_state=raw_states[i] if raw_states is not None else norm_states[i].copy(),
                    norm_state=norm_states[i],
                    q_values=q_values[i],
                    visit_count=int(visit_counts[i]),
                    last_step=int(last_steps[i]),
                    on_object=int((raw_states[i][9] if raw_states is not None else 0.0) > 0.5)
                )

                store._index.add_items(
                    norm_states[i].reshape(1, -1),
                    np.array([pid], dtype=np.int64),
                )
            store.next_id = (max(store.points.keys()) + 1) if store.points else 0
            store._norm_frozen = True
            store._freeze_done = True

            logger.info(
                f"HNSW store loaded: {store.next_id} points from {filepath}"
            )

        return store
