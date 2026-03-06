# recovery.py

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Protocol, Any


class SafetyAction(Enum):
    STOP = auto()
    BACK_UP = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    BACK_UP_THEN_TURN_LEFT = auto()
    BACK_UP_THEN_TURN_RIGHT = auto()
    WAIT_AND_RETRY = auto()


@dataclass(frozen=True)
class RecoveryConfig:
    reverse_speed: int = 40
    reverse_duration_s: float = 0.35

    turn_speed: int = 45
    turn_duration_s: float = 0.30

    wait_duration_s: float = 0.25

    clear_threshold: float = 0.40
    max_recovery_attempts: int = 2

    post_action_stop: bool = True


class RecoveryPolicy(Protocol):
    async def handle_obstacle(self, robot: Any, sample: Any) -> None:
        ...


def _sample_value(sample: Any, name: str, default: float = 0.0) -> float:
    v = getattr(sample, name, default)
    if v is None:
        return default
    return float(v)


class StopRecoveryPolicy:
    async def handle_obstacle(self, robot: Any, sample: Any) -> None:
        await robot.stop()


class StubRecoveryPolicy:
    async def handle_obstacle(self, robot: Any, sample: Any) -> None:
        print(f"[recovery] stub invoked with sample={sample}")
        await robot.stop()


class HeuristicRecoveryPolicy:
    """
    A simple baseline recovery policy.

    Assumptions:
    - Higher proximity values mean "closer obstacle"
    - robot.stop() exists
    - robot.drive(speed, turn) exists and is non-blocking for the commanded motion
      followed by asyncio.sleep(...)
    - Positive speed means forward, negative means backward
    - Positive/negative turn sign may need swapping for your platform
    """

    def __init__(self, cfg: Optional[RecoveryConfig] = None):
        self.cfg = cfg or RecoveryConfig()
        self._attempt_count = 0

    async def handle_obstacle(self, robot: Any, sample: Any) -> None:
        action = self.choose_action(sample)
        print(f"[recovery] chosen action = {action.name}, sample={sample}")

        try:
            await self.execute_action(robot, action, sample)
        except Exception:
            # Safety fallback
            await robot.stop()
            raise

    def choose_action(self, sample: Any) -> SafetyAction:
        left = _sample_value(sample, "left")
        right = _sample_value(sample, "right")
        rear = _sample_value(sample, "rear")

        max_side = max(left, right)

        # If rear is hot and sides are mild, don't reverse into danger.
        if rear >= max_side and rear >= self.cfg.clear_threshold:
            if left > right:
                return SafetyAction.TURN_RIGHT
            if right > left:
                return SafetyAction.TURN_LEFT
            return SafetyAction.STOP

        # Front/side obstacle handling.
        if left > right:
            return SafetyAction.BACK_UP_THEN_TURN_RIGHT
        if right > left:
            return SafetyAction.BACK_UP_THEN_TURN_LEFT

        # Symmetric / straight-on obstacle.
        if self._attempt_count >= self.cfg.max_recovery_attempts:
            return SafetyAction.STOP

        return SafetyAction.BACK_UP

    async def execute_action(self, robot: Any, action: SafetyAction, sample: Any) -> None:
        if action == SafetyAction.STOP:
            await robot.stop()
            return

        if action == SafetyAction.WAIT_AND_RETRY:
            await robot.stop()
            await asyncio.sleep(self.cfg.wait_duration_s)
            return

        if action == SafetyAction.BACK_UP:
            self._attempt_count += 1
            await self._back_up(robot)
            return

        if action == SafetyAction.TURN_LEFT:
            self._attempt_count += 1
            await self._turn_left(robot)
            return

        if action == SafetyAction.TURN_RIGHT:
            self._attempt_count += 1
            await self._turn_right(robot)
            return

        if action == SafetyAction.BACK_UP_THEN_TURN_LEFT:
            self._attempt_count += 1
            await self._back_up(robot)
            await asyncio.sleep(0.05)
            await self._turn_left(robot)
            return

        if action == SafetyAction.BACK_UP_THEN_TURN_RIGHT:
            self._attempt_count += 1
            await self._back_up(robot)
            await asyncio.sleep(0.05)
            await self._turn_right(robot)
            return

        await robot.stop()

    async def _back_up(self, robot: Any) -> None:
        await robot.drive(-self.cfg.reverse_speed, 0)
        await asyncio.sleep(self.cfg.reverse_duration_s)
        if self.cfg.post_action_stop:
            await robot.stop()

    async def _turn_left(self, robot: Any) -> None:
        await robot.drive(0, -self.cfg.turn_speed)
        await asyncio.sleep(self.cfg.turn_duration_s)
        if self.cfg.post_action_stop:
            await robot.stop()

    async def _turn_right(self, robot: Any) -> None:
        await robot.drive(0, self.cfg.turn_speed)
        await asyncio.sleep(self.cfg.turn_duration_s)
        if self.cfg.post_action_stop:
            await robot.stop()

    def reset(self) -> None:
        self._attempt_count = 0