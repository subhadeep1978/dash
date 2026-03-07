"""
dash_ble_async.py

A single-event-loop (asyncio-only) Dash controller built on top of Bleak.

Design goals
-----------
1) One asyncio loop (no background threads).
2) Continuous sensor sampling via BLE notifications while commands are issued.
3) Simple, readable, well-documented primitives you can build:
   - a kid-friendly scripting layer
   - a control loop (KF / planner)
   - logging + calibration runs

How it works
------------
- We scan for a BLE device whose advertised name matches `name` (default: "Dash").
- We connect with BleakClient.
- We start notifications on Dash's sensor characteristic.
- The notify callback parses proximity (left/right/rear) and pushes samples into an asyncio.Queue.
- Your code can:
  - await robot.get_latest_sensor() (non-blocking)
  - or run a control loop that consumes the stream and sends drive commands.

Important note
--------------
Dash is a BLE peripheral; your code runs on the host (Mac, Pi, etc.). There is no
official “upload code to Dash” capability.

Protocol note
-------------
This module uses UUIDs/command IDs widely used by open-source Dash tooling.
They match what the `bleak-dash` project uses for the relevant characteristics
and commands. Those protocol constants are not official Wonder Workshop docs.

Tested environment
------------------
- macOS + Python 3.9+ + bleak
"""

from __future__ import annotations

import random
import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Callable, Deque, Dict, Any
from collections import deque
import matplotlib.colors as mcolors
from bleak import BleakClient, BleakScanner

import phy
import services.safety as safety
import services.recovery as recovery
from simulated_robot import SimulatedClient

# =============================================================================
#                           DASH BLE PROTOCOL CONSTANTS
# =============================================================================
# These UUIDs/command IDs are commonly used by Dash reverse-engineered tooling.
# If Wonder Workshop changes firmware/protocol, these may need adjustment.

ROBOT_SERVICE_UUID = "af237777-879d-6186-1f49-deca0e85d9c1"

# Command characteristic: write bytes([cmd_id]) + payload
COMMAND1_CHAR_UUID = "af230002-879d-6186-1f49-deca0e85d9c1"

# Sensor notify characteristic (streaming sensor packets)
SENSOR1_CHAR_UUID = "af230006-879d-6186-1f49-deca0e85d9c1"

# Command IDs
CMD_DRIVE             = 0x02       # Minimal drive/stop command ID
CMD_RESET             = 0xC8       # Reset/enable sensors (mode byte)

CMD_TAIL_BRIGHTNESS   = 0x04
CMD_NECK_COLOR        = 0x03       # Neck RGB
CMD_HEAD_BUTTON_COLOR = 0x0D
CMD_LEFT_EAR_COLOR    = 0x0B       # left ear (RGB)
CMD_RIGHT_EAR_COLOR   = 0x0C       # right ear (RGB)
CMD_EYE_COLOR         = 0x08
CMD_EAR_BUTTON_COLOR  = 0x09
CMD_EYE_BRIGHTNESS    = 0x08       # 1 byte

# SIMULATE:
#   True  ==> If Robot is not discovered, start a fake client to flush the rest of the architecture
#   False ==> If Robot is not discovered, throw an error
SIMULATE = False


# =============================================================================
# Utilities
# =============================================================================
def color_to_rgb(color: str):
    """
    Converts a color name string to an RGB tuple (floats in 0-1 range).

    Args:
    color_name: A string representing the color (e.g., 'red', 'navy', '#00ff00').

    Returns:
    A tuple of (R, G, B) float values in the range [0, 1].
    """
    try:
        rgb_tuple = mcolors.to_rgb(color)
        rgb_tuple = tuple([int(c * 255) for c in rgb_tuple])
        return rgb_tuple
    except ValueError as e:
        return (0,0,0)     # OFF
    return None    

# =============================================================================
#                                   DATA TYPES
# =============================================================================
@dataclass(frozen=True)
class ProximitySample:
    """
    ProximitySample

    A timestamped snapshot of Dash proximity readings.

    Fields
    ------
    t : float
        Host receive time (seconds since epoch). This includes BLE/OS latency.
    left, right, rear : int
        Raw proximity intensity (unitless, typically 0..255-ish).
    raw : bytes
        Raw sensor packet payload (for debugging / future decoding).
    """
    t: float
    left: int
    right: int
    rear: int
    raw: bytes


@dataclass
class RobotLimits:
    """
    RobotLimits

    Safety clamps for user-facing motion commands. These are not physical units;
    they are command magnitudes for Dash's motor controller.

    Notes
    -----
    - Keep conservative values for kid mode.
    - You can loosen them for experiments.
    """
    max_speed: int = 300
    max_turn: int = 300


# =============================================================================
#                              LOW-LEVEL ASYNC DRIVER
# =============================================================================
class DashDriver:
    """
    DashDriver

    Low-level async driver that:
      - scans + connects to Dash over BLE (BleakClient)
      - starts sensor notifications
      - provides async methods to send commands
      - publishes sensor samples to an asyncio.Queue

    This class is "protocol aware":
      - knows UUIDs
      - knows a minimal parse of proximity bytes
      - knows minimal command payloads

    You typically don't expose this directly to kids. Wrap it in a higher-level
    Robot API or a runtime that provides safe primitives (drive_for, turn_for, etc.).
    """

    def __init__(
        self,
        name: str = "Dash",
        *,
        sensor_queue_maxsize: int = 2000,
        history_len: int = 2000,
    ) -> None:
        """
        __init__

        Parameters
        ----------
        name : str
            BLE advertised device name to match (case-insensitive).
        sensor_queue_maxsize : int
            Max queued sensor samples; prevents unbounded memory growth if consumer is slow.
        history_len : int
            Rolling in-memory history length for quick inspection/calibration.
        """
        self.name = name
        self.client: Optional[BleakClient] = None
        self.sensor_queue_maxsize                         = sensor_queue_maxsize
        self.sensor_queue: asyncio.Queue[ProximitySample] = None
        self._history: Deque[ProximitySample] = deque(maxlen=history_len)

        self._notify_started = False


    # -------------------------------------------------------------------------
    # Discover target
    # -------------------------------------------------------------------------
    async def discover(self, timeout):
        if SIMULATE: return self.name 

        # Discover real robot
        devices = await BleakScanner.discover(timeout=timeout)

        target = None
        for d in devices:
            if (d.name or "").strip().lower() == self.name.lower():
                target = d
                break

        return target    

    # -------------------------------------------------------------------------
    # Create a bleak client
    # -------------------------------------------------------------------------
    def create_client(self, target):
        if SIMULATE:
            self.client = SimulatedClient()
        else:
            self.client = BleakClient(target)


    # -------------------------------------------------------------------------
    # Connection lifecycle
    # -------------------------------------------------------------------------
    async def scan_and_connect(self, timeout: float = 6.0) -> None:
        """
        scan_and_connect

        Scan for BLE devices and connect to the first whose `device.name` matches `self.name`.

        Notes
        -----
        - BLE peripherals often allow only a single connection at a time.
          Fully quit the Wonder app on iPad/iPhone if it is connected.
        - On macOS, the identifier may not be a MAC address; that's normal.

        Raises
        ------
        RuntimeError
            If no matching device found or connection fails.
        """
        if not self.sensor_queue:
            self.sensor_queue: asyncio.Queue[ProximitySample] = asyncio.Queue(maxsize=self.sensor_queue_maxsize)            

        # Discover the target
        target = await self.discover(timeout)

        if target is None:
            raise RuntimeError(
                f"Dash not found (name='{self.name}'). "
                f"Tips: power on Dash; keep it close; quit Wonder app fully."
            )

        # Create a client for the target
        self.create_client(target)

        await self.client.connect()

        # Some devices behave better if reset/enable is issued after connect.
        # Mode=4 is commonly used by existing Dash tooling; it's safe to keep.
        try:
            await self.reset(mode=4)
        except Exception:
            # Don't fail hard if reset is unsupported on some firmware.
            pass

        await self.start_sensor_stream()

    async def disconnect(self) -> None:
        """
        disconnect

        Stop sensor notifications and disconnect BLE client.
        Safe to call multiple times.
        """
        if not self.client:
            return
        try:
            if self.client.is_connected and self._notify_started:
                try:
                    await self.client.stop_notify(SENSOR1_CHAR_UUID)
                except Exception:
                    pass
                self._notify_started = False
            if self.client.is_connected:
                await self.client.disconnect()
        finally:
            self.client = None

    def is_connected(self) -> bool:
        """
        is_connected

        Returns True if BleakClient exists and is connected.
        """
        return bool(self.client and self.client.is_connected)

    # -------------------------------------------------------------------------
    # Sensor streaming
    # -------------------------------------------------------------------------
    async def start_sensor_stream(self) -> None:
        """
        start_sensor_stream

        Subscribe to Dash's sensor notification characteristic. Each notification
        triggers `_on_sensor_notify`, which parses proximity and pushes samples
        into:
          - `self.sensor_queue` (for async consumption)
          - `self._history` (rolling buffer)

        This runs concurrently with everything else in the same asyncio loop.
        """
        client = self._require_client()
        if self._notify_started:
            return
        await client.start_notify(SENSOR1_CHAR_UUID, self._on_sensor_notify)
        self._notify_started = True

    def get_history(self) -> list[ProximitySample]:
        """
        get_history

        Return a snapshot list of recent sensor samples (rolling buffer).
        Useful for calibration and debugging.
        """
        return list(self._history)

    def _on_sensor_notify(self, _char: Any, data: bytearray) -> None:
        """
        _on_sensor_notify

        BLE notification callback. Runs in Bleak's callback context but still
        within the event loop.

        Parses proximity as:
          right = bytes[6]
          left  = bytes[7]
          rear  = bytes[8]

        Notes
        -----
        - This is a minimal decode; the packet contains more fields.
        - We timestamp on receipt; this includes transmission + OS scheduling latency.
        - We use `put_nowait` to avoid blocking the callback; if queue is full, we drop.
        """
        b = bytes(data)
        t = time.time()

        left = right = rear = 0
        if len(b) >= 9:
            right = b[6]
            left = b[7]
            rear = b[8]

        sample = ProximitySample(t=t, left=left, right=right, rear=rear, raw=b)
        self._history.append(sample)

        # Non-blocking enqueue; drop if consumer is slow.
        if self.sensor_queue:
            try:
                self.sensor_queue.put_nowait(sample)
            except asyncio.QueueFull:
                pass

    def normalize_colors(self, r,g,b): 
        return int(max(0, min(255, r))), int(max(0, min(255, g))), int(max(0, min(255, b)))

    # -------------------------------------------------------------------------
    # Command writes
    # -------------------------------------------------------------------------
    async def reset(self, mode: int = 4) -> None:
        """
        reset

        Send a reset/enable command. Commonly used after connect.

        Parameters
        ----------
        mode : int
            A small mode byte used by existing Dash tooling. Defaults to 4.
        """
        await self._write_command(CMD_RESET, bytes([int(mode) & 0xFF]))
    
    async def set_neck_color(self, r: int, g: int, b: int) -> None:
        r,g,b = self.normalize_colors(r,g,b)
        await self._write_command(CMD_NECK_COLOR, bytes([r, g, b]))

    async def set_ear_color(self, left_rgb, right_rgb=None) -> None:
        if right_rgb is None:  right_rgb = left_rgb
        lr, lg, lb = self.normalize_colors(*left_rgb)
        rr, rg, rb = self.normalize_colors(*right_rgb)
        await self._write_command(CMD_LEFT_EAR_COLOR,  bytes([lr, lg, lb]))
        await self._write_command(CMD_RIGHT_EAR_COLOR, bytes([rr, rg, rb]))

    async def set_head_color(self, r: int, g: int, b: int) -> None:
        r,g,b = self.normalize_colors(r,g,b)
        await self._write_command(CMD_HEAD_BUTTON_COLOR, bytes([r, g, b]))

    async def set_eye_brightness(self, brightness):
        brightness = max(0, min(255, int(brightness)))
        await self._write_command(CMD_EYE_BRIGHTNESS, bytes([brightness]))        

    async def stop(self) -> None:
        """
        stop

        Stop motion immediately (best-effort).
        """
        await self._write_command(CMD_DRIVE, bytes([0, 0, 0]))

    async def drive_velocity(self, speed: int, turn: int = 0) -> None:
        """
        drive_velocity

        Set a velocity-like motor command. This is NOT a "move for duration" command.
        It persists until overwritten by another drive/stop command.
        
        Note: 
            Reverse motion is not guaranteed to be straight with the current minimal
            Dash drive packet. Use only for short recovery maneuvers, not precise travel.

        Parameters
        ----------
        speed : int
            Signed speed command. Positive forward, negative backward.
        turn : int
            Signed turn command (implementation depends on protocol). In this minimal
            module we keep turning support conservative; for robust turning you may
            implement a dedicated spin/turn packet once you validate the format.

        Notes
        -----
        The minimal payload used here matches a common "works for forward/stop" pattern.
        Turning may require further protocol refinement.
        """
        speed = int(max(-2048, min(2048, speed)))
        turn = int(max(-2048, min(2048, turn)))

        def enc16(x: int) -> int:
            # Some Dash tooling uses an offset encoding for negative values.
            return (0x8000 + abs(x)) if x < 0 else x

        s = enc16(speed)
        
        # Minimal 3-byte payload; this is sufficient for forward/backward + stop.
        # Turning is intentionally conservative here.
        payload = bytes([s & 0xFF, (s >> 8) & 0xFF, 0x00])
        
        await self._write_command(CMD_DRIVE, payload)

    async def _write_command(self, cmd_id: int, payload: bytes) -> None:
        """
        _write_command

        Low-level write to Dash command characteristic.

        Packet format:
          [cmd_id] + payload

        Raises
        ------
        RuntimeError
            If not connected.
        """
        client = self._require_client()
        msg = bytes([cmd_id]) + payload
        await client.write_gatt_char(COMMAND1_CHAR_UUID, msg)

    def _require_client(self) -> BleakClient:
        """
        _require_client

        Return connected client or raise.
        """
        if not self.client or not self.client.is_connected:
            raise RuntimeError("Dash is not connected.")
        return self.client


# =============================================================================
#                         HIGH-LEVEL ROBOT API (ASYNC)
# =============================================================================
class DashRobot:
    """
    DashRobot

    A higher-level async API that:
      - wraps DashDriver
      - provides kid-friendly primitives:
          eyes()   -> mapped to neck color for now
          drive()  -> velocity command (persists)
          drive_for()
          turn_for()
      - offers convenient sensor access

    This API stays inside a single asyncio loop, and sensor notifications keep
    arriving while drive_for/turn_for are running because those functions `await`
    (they do not block the event loop).
    """

    def __init__(self, name: str, *, limits: RobotLimits = RobotLimits()) -> None:
        self.d = DashDriver(name)
        self.limits = limits
        self._phy = phy.SensorPHY(self, history_size=256)
        self._obstacle_cfg                       = safety.ObstacleConfig()
        self._stop_evt                           = None
        self._guard_task: Optional[asyncio.Task] = None
        self.recovery_policy                     = recovery.StopRecoveryPolicy()

    
    # -------------------------------------------------------------------------
    # Safety
    # -------------------------------------------------------------------------
    def _start_guardrails(self) -> None:
        # idempotent
        if self._guard_task is not None and not self._guard_task.done():
            return

        self._stop_evt.clear()

        def _on_trigger(sample) -> None:
            print("🛑 Obstacle detected! Stopping.")

        # Start PHY first (so guard has data)
        self._phy.start()            

        self._guard_task = asyncio.create_task(
            safety.stop_on_obstacle(
                self, 
                self._phy, 
                self._stop_evt, 
                self._obstacle_cfg, 
                on_trigger=_on_trigger,
                recovery_policy = self.recovery_policy
            )
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    async def connect(self, timeout: float = 6.0) -> None:
        """
        connect

        Scan and connect, and start sensor streaming.
        """
        if not self._stop_evt: 
            self._stop_evt = asyncio.Event()

        await self.d.scan_and_connect(timeout=timeout)
        # self._start_guardrails()

    async def close(self) -> None:
        """
        close

        Stop and disconnect safely.
        """
        try:
            await self.d.stop()
        except Exception:
            pass
        await self.d.disconnect()

    # -------------------------------------------------------------------------
    # Kid-friendly outputs
    # -------------------------------------------------------------------------
    async def neck(self, color: str) -> None:
        """
        neck
        """
        await self.d.set_neck_color(  *color_to_rgb(color) )

    async def ear(self, lcolor , rcolor = None) -> None:
        """
        ear
        """
        rcolor = lcolor if not rcolor else rcolor
        await self.d.set_ear_color(  left_rgb = color_to_rgb(lcolor), right_rgb = color_to_rgb(rcolor) )

    async def head(self, color: str ) -> None:
        """
        head
        """
        await self.d.set_head_color(  *color_to_rgb(color) )

    async def eye(self, b: str ) -> None:
        # brightness = "high", "medium", "low", "off"
        brightness = None
        if b == "high"   : brightness = 255
        if b == "medium" : brightness = 128
        if b == "low"    : brightness = 64
        if b == "off"    : brightness = 0
        assert brightness != None, f"BRIGHTNESS VALUE CAN ONLY BE high, medium, low or off. YOU HAVE TOLD ME {b}"

        await self.d.set_eye_brightness (brightness)

    async def stop(self) -> None:
        """
        stop

        Stop motion.
        """
        await self.d.stop()

    async def drive(self, speed: int, turn: int = 0) -> None:
        """
        drive

        Velocity command that persists until overwritten.

        Parameters
        ----------
        speed : int
            Signed speed. We'll clamp to limits.max_speed.
        turn : int
            Signed turn. We'll clamp to limits.max_turn.

        Notes
        -----
        Turning support is intentionally conservative at the driver layer.
        """

        # Truncate the absolute speed. Then apply sign
        max_speed = self.limits.max_speed
        abs_speed = min(abs(speed), max_speed)
        final_speed = abs_speed if speed >= 0 else -abs_speed

        # Truncate the absolute turn. Then apply sign
        max_turn  = self.limits.max_turn
        abs_turn  = min( abs(turn), max_turn )
        final_turn= abs_turn if turn >= 0 else -abs_turn

        await self.d.drive_velocity(speed=final_speed, turn=final_turn)

    async def drive_for(self, speed: int, duration_s: float, direction: str = "forward") -> None:
        """
        drive_for

        Drive straight for a duration, then stop.

        Parameters
        ----------
        speed : int
            Positive magnitude (kid-friendly). We'll apply direction and clamp.
        duration_s : float
            Seconds to drive.
        direction : str
            "forward" or "backward" relative to the eyes/front of Dash.

        Implementation detail
        ---------------------
        Uses asyncio.sleep(), not time.sleep(), so the event loop remains alive and
        sensor notifications continue to stream during the motion.
        """
        direction = direction.strip().lower()
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")

        mag = abs(int(speed))
        signed_speed = mag if direction == "forward" else -mag

        await self.drive(signed_speed, 0)
        await asyncio.sleep(max(0.0, float(duration_s)))
        await self.stop()

    async def turn_for(self, turn_speed: int, duration_s: float, direction: str = "right") -> None:
        """
        turn_for

        Turn in place for a duration, then stop.

        Parameters
        ----------
        turn_speed : int
            Positive magnitude (kid-friendly). We'll apply direction and clamp.
        duration_s : float
            Seconds to turn.
        direction : str
            "left" or "right" relative to the eyes/front of Dash.

        Notes
        -----
        This calls drive(speed=0, turn=...), which may need protocol refinement
        for perfect spins on all firmware versions. If spin behavior is weak,
        you can implement a dedicated spin packet once validated.
        """
        direction = direction.strip().lower()
        if direction not in ("left", "right"):
            raise ValueError("direction must be 'left' or 'right'")

        mag = abs(int(turn_speed))
        signed_turn = mag if direction == "right" else -mag

        await self.drive(0, signed_turn)
        await asyncio.sleep(max(0.0, float(duration_s)))
        await self.stop()

    async def light_show(self):
        # Blinky ear    
        await robot.ear("green", "blue")
        await asyncio.sleep(0.2)
        await robot.ear("blue", "green")
        await asyncio.sleep(0.2)

        # Blinky head
        await robot.head("yellow")
        await asyncio.sleep(0.2)
        await robot.head("purple")
        
        # Turn off neck and eye
        await asyncio.sleep(0.2)
        await robot.neck("black")
        await robot.eye("off")

    # -------------------------------------------------------------------------
    # Sensors
    # -------------------------------------------------------------------------
    async def next_sensor(self, timeout: Optional[float] = None) -> Optional[ProximitySample]:
        """
        next_sensor

        Await the next sensor sample from the queue.

        Parameters
        ----------
        timeout : float | None
            If provided, return None on timeout.

        Returns
        -------
        ProximitySample | None
        """
        try:
            if timeout is None:
                return await self.d.sensor_queue.get()
            return await asyncio.wait_for(self.d.sensor_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def get_latest_sensor(self) -> Optional[ProximitySample]:
        """
        get_latest_sensor

        Non-blocking fetch of the most recent sample currently available in the queue.
        If multiple samples are queued, it drains to the most recent and returns that.

        Returns
        -------
        ProximitySample | None
        """
        latest: Optional[ProximitySample] = None
        while True:
            try:
                latest = self.d.sensor_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        return latest


# =============================================================================
#                                 EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    
    robot  = DashRobot("Dash")

    # ====================================================================
    # Basic demostration of robot motion
    # ====================================================================
    async def motion_demo_task(robot, stop_evt: asyncio.Event) -> None:
        """
        Foreground motion logic.
        Ends when done, or exits early if stop_evt is set.
        """
        try:
            while not stop_evt.is_set():
                await robot.drive_for(120, 1.0, "forward")
                await asyncio.sleep(0.2)

                await robot.drive_for(120, 1.0, "backward")
                await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            try:
                await robot.stop()
            except Exception:
                pass
            raise

    # ====================================================================
    # Foreground logger consuming from PHY-owned latest().
    # ====================================================================
    async def sensor_logger_task(robot, stop_evt: asyncio.Event, hz: float = 5.0) -> None:
        period = 1.0 / hz
        try:
            while not stop_evt.is_set():
                s = await robot.get_latest_sensor()
                print(f"[logger] {s}")
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            raise

    # ====================================================================
    # Orchestrate the session
    # ====================================================================
    async def run_session(robot) -> None:
        """
        One full session:
        1) connect
        2) auto-start PHY + safety inside robot.connect()
        3) run motion + logger in parallel
        4) graceful shutdown
        """
        session_stop_evt = asyncio.Event()

        try:
            # T1: serial setup
            await robot.connect(timeout=6.0)
            print("Robot found and connected")

            # T2/T3: background services are assumed to be started inside robot.connect()
            # e.g. PHY task and safety task launched by robot._start_guardrails()
            # If additional background tasks are needed:
            #   asyncio.create_task( BG() )

            # T4/T5: foreground tasks in parallel
            await asyncio.gather(
                # motion_demo_task(robot, session_stop_evt),
                sensor_logger_task(robot, session_stop_evt, hz=5.0),
            )

        except KeyboardInterrupt:
            print("KeyboardInterrupt received")
            session_stop_evt.set()
            raise

        finally:
            # T6: graceful shutdown
            session_stop_evt.set()
            try:
                await robot.stop()
            except Exception:
                pass
            await robot.close()
            print("Robot closed cleanly")


    # Launch
    asyncio.run(run_session(robot))
