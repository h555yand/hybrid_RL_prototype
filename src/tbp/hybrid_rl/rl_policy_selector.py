# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

"""RLPolicySelector: routes goals to RL navigation or other policies.

Drop-in replacement for DistantPolicySelector. Routes:
- GSG goals → RLGoalPolicy (incremental RL navigation)
- SM goals → LookAtGoal (salience-driven)
- No goals → default exploration policy

All existing Monty behavior is preserved — the RL module only
activates when the hypothesis-testing policy generates a goal state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from tbp.monty.cmp import Goal, Message
from tbp.monty.context import RuntimeContext
from tbp.monty.experiment.motor_system import ExperimentMotorSystem
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import (
    MotorPolicy,
    MotorPolicyResult,
    PolicyStatus,
)
from tbp.monty.frameworks.models.motor_policy_selectors import (
    MotorPolicySelector,
    highest_confidence_goal,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tbp.monty.memento import Memento

from tbp.hybrid_rl.rl_goal_policy import RLGoalPolicy

if TYPE_CHECKING:
    from tbp.monty.frameworks.models.salience.motor_policy import LookAtGoal

logger = logging.getLogger(__name__)


class RLPolicySelector(MotorPolicySelector):
    """Routes goals to RL navigation or fallback policies.

    Mirrors DistantPolicySelector logic but replaces JumpToGoal
    with RLGoalPolicy for GSG goals.

    Key difference from DistantPolicySelector:
    - JumpToGoal: 2 calls (jump + check undo)
    - RLGoalPolicy: N calls (one per navigation step),
      status=IN_PROGRESS until goal reached or timeout

    During RL navigation, new GSG goals are passed through
    (goal may change mid-navigation). SM goals and default
    exploration are blocked until navigation completes.
    """

    def __init__(
        self,
        rl_goal_policy: RLGoalPolicy,
        look_at_goal: LookAtGoal,
        default: MotorPolicy,
    ):
        """Initialize selector.

        Args:
            rl_goal_policy: RL navigation policy (replaces JumpToGoal).
            look_at_goal: Salience-driven look-at policy for SM goals.
            default: Default exploration policy (random walk etc).
        """
        self._rl_goal = rl_goal_policy
        self._look_at_goal = look_at_goal
        self._default = default
        self._is_navigating: bool = False

        # Telemetry
        self._selected_policies: list[MotorPolicy] = []
        self._selected_goals: list[Goal | None] = []

    def fixme_provide_motor_system(
        self, motor_system: ExperimentMotorSystem
    ) -> None:
        self._rl_goal.fixme_provide_motor_system(motor_system)
        self._look_at_goal.fixme_provide_motor_system(motor_system)
        self._default.fixme_provide_motor_system(motor_system)

    def reset(self) -> None:
        self._rl_goal.reset()
        self._look_at_goal.reset()
        self._default.reset()
        self._is_navigating = False
        self._selected_policies = []
        self._selected_goals = []

    def state_dict(self) -> Memento:
        return {
            "rl_goal": self._rl_goal.state_dict(),
            "look_at_goal": self._look_at_goal.state_dict(),
            "default": self._default.state_dict(),
            "is_navigating": self._is_navigating,
        }

    def __call__(
        self,
        ctx: RuntimeContext,
        observations: Observations,
        state: MotorSystemState,
        percept: Message,
        goals: Sequence[Goal],
    ) -> MotorPolicyResult:
        """Select policy and return actions.

        Priority:
        1. Continue RL navigation (if active)
        2. Start RL navigation for new GSG goal
        3. LookAtGoal for SM goals
        4. Default exploration

        Args:
            ctx: Runtime context.
            observations: Environment observations.
            state: Motor system state.
            percept: CMP Message from first sensor module.
            goals: Sequence of goals from LMs and SMs.

        Returns:
            MotorPolicyResult from selected policy.
        """
        gsg_goals = [g for g in goals if g.sender_type == "GSG"]

        # ═══ 1. Continue RL navigation ═══
        if self._is_navigating:
            # Pass current GSG goal (may have changed) or None
            goal = (
                highest_confidence_goal(gsg_goals) if gsg_goals else None
            )
            result = self._rl_goal(
                ctx, observations, state, percept, goal,
            )
            self._is_navigating = (
                result.status == PolicyStatus.IN_PROGRESS
            )
            self._update_telemetry(self._rl_goal, goal)

            if result.actions or result.status == PolicyStatus.IN_PROGRESS:
                return result
            # Navigation finished — fall through to check for new goals

        # ═══ 2. New GSG goal → start RL navigation ═══
        if gsg_goals:
            goal = highest_confidence_goal(gsg_goals)
            result = self._rl_goal(
                ctx, observations, state, percept, goal,
            )
            self._is_navigating = (
                result.status == PolicyStatus.IN_PROGRESS
            )
            self._update_telemetry(self._rl_goal, goal)
            return result

        # ═══ 3. SM goals → look_at_goal ═══
        sm_goals = [g for g in goals if g.sender_type == "SM"]
        if sm_goals:
            goal = highest_confidence_goal(sm_goals)
            self._is_navigating = False
            result = self._look_at_goal(
                ctx, observations, state, percept, goal,
            )
            self._update_telemetry(self._look_at_goal, goal)
            return result

        # ═══ 4. Default exploration ═══
        self._is_navigating = False
        result = self._default(
            ctx, observations, state, percept, None,
        )
        self._update_telemetry(self._default, None)
        return result

    def _update_telemetry(
        self,
        policy: MotorPolicy,
        goal: Goal | None,
    ) -> None:
        self._selected_policies.append(policy)
        self._selected_goals.append(goal)
