# phy.py

from __future__ import annotations

import asyncio
import dataclasses
from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar


T = TypeVar("T")


class SensorPHY(Generic[T]):
    """
    Single-owner BLE sensor pump.

    Important design rule:
    - No asyncio primitives are created in __init__.
    - They are bound lazily inside the running event loop when start() is called.

    This allows DashDriver / DashRobot / SensorPHY objects to be instantiated
    at module scope, before asyncio.run(...), without cross-loop failures.
    """

    def __init__(self, robot, *, history_size: int = 256):
        self._robot = robot
        self._history: Deque[T] = deque(maxlen=history_size)

        self._latest: Optional[T] = None
        self._latest_seq: int = 0

        # Lazy-bound asyncio state
        self._stop_evt: Optional[asyncio.Event] = None
        self._new_sample_cv: Optional[asyncio.Condition] = None
        self._task: Optional[asyncio.Task] = None

        # Per-consumer stream cursors
        self._streams: List["SensorStream[T]"] = []

    def _bind_loop_if_needed(self) -> None:
        """
        Create loop-bound asyncio primitives only inside a running loop.
        """
        if self._stop_evt is None:
            self._stop_evt = asyncio.Event()
        if self._new_sample_cv is None:
            self._new_sample_cv = asyncio.Condition()

    def start(self) -> None:
        """
        Start the single BLE pump task. Safe to call multiple times.
        Must be called from inside a running asyncio loop.
        """
        self._bind_loop_if_needed()

        if self._task is not None and not self._task.done():
            return

        assert self._stop_evt is not None
        self._stop_evt.clear()

        self._task = asyncio.create_task(self._run())

        def _report_task_result(t: asyncio.Task) -> None:
            try:
                exc = t.exception()
            except asyncio.CancelledError:
                print("PHY cancelled")
                return

            if exc is not None:
                print(f"PHY crashed: {exc!r}")

        self._task.add_done_callback(_report_task_result)

    async def stop(self) -> None:
        """
        Stop the BLE pump task.
        """
        if self._stop_evt is not None:
            self._stop_evt.set()

        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def _run(self) -> None:
        """
        Only place in the system that is allowed to call robot.next_sensor().
        """
        assert self._stop_evt is not None
        assert self._new_sample_cv is not None

        print("BLE PHY Running..")

        try:
            while not self._stop_evt.is_set():
                s = await self._robot.next_sensor()

                # ProximitySample is frozen, so create a normalized copy.
                s = dataclasses.replace(
                    s,
                    left=float(s.left) / 255.0,
                    right=float(s.right) / 255.0,
                    rear=float(s.rear) / 255.0,
                )
                self._latest = s
                self._latest_seq += 1
                self._history.append(s)

                async with self._new_sample_cv:
                    self._new_sample_cv.notify_all()

                for st in list(self._streams):
                    st._notify_new_seq(self._latest_seq)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"BLE stream has crashed with message: {e!r}")
            raise

    def latest_nowait(self) -> Optional[T]:
        return self._latest

    async def latest(self, *, wait: bool = True) -> T:
        """
        Return the most recent sample.
        If wait=True and no sample has arrived yet, wait for the first one.
        """
        if self._latest is not None:
            return self._latest

        if not wait:
            raise RuntimeError("No sensor sample available yet.")

        self._bind_loop_if_needed()
        assert self._stop_evt is not None
        assert self._new_sample_cv is not None

        async with self._new_sample_cv:
            while self._latest is None and not self._stop_evt.is_set():
                await self._new_sample_cv.wait()

        if self._latest is None:
            raise asyncio.CancelledError("PHY stopped before first sample arrived.")

        return self._latest

    def history_snapshot(self) -> list[T]:
        return list(self._history)

    def createStream(self) -> "SensorStream[T]":
        """
        Create a per-consumer stream cursor. Each consumer gets its own
        'next sample' view without stealing samples from other consumers.
        """
        self._bind_loop_if_needed()
        st = SensorStream(self)
        self._streams.append(st)
        return st

    # Optional alias if you prefer `stream()`
    def stream(self) -> "SensorStream[T]":
        return self.createStream()

    def _remove_stream(self, st: "SensorStream[T]") -> None:
        try:
            self._streams.remove(st)
        except ValueError:
            pass


class SensorStream(Generic[T]):
    """
    Per-consumer stream cursor over SensorPHY.
    Each stream waits for a new sample after the last one it observed.
    """

    def __init__(self, phy: SensorPHY[T]):
        self._phy = phy
        self._last_seen_seq = 0
        self._wake = asyncio.Event()

    def _notify_new_seq(self, seq: int) -> None:
        self._wake.set()

    async def next(self) -> T:
        """
        Wait for the next unread sample for this consumer.
        """
        # First call: if no samples yet, wait for the first.
        if self._phy._latest is None:
            await self._phy.latest(wait=True)
            self._last_seen_seq = self._phy._latest_seq
            self._wake.clear()
            return self._phy._latest  # type: ignore[return-value]

        # Otherwise wait until a strictly newer sample arrives.
        while self._phy._latest_seq <= self._last_seen_seq:
            self._wake.clear()
            await self._wake.wait()

        self._last_seen_seq = self._phy._latest_seq
        return self._phy._latest  # type: ignore[return-value]

    async def close(self) -> None:
        self._phy._remove_stream(self)