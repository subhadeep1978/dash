import asyncio
import math
import time
from dataclasses import dataclass


@dataclass
class SimPose:
    x: float = 150.0
    y: float = 150.0
    theta: float = 0.0   # radians, 0 means facing +x


class SimulatedClient:
    """
    Simulated bleak client for Dash-style testing.

    Packet format:
      - proximity callback expects bytes 6,7,8 to hold:
          right = b[6]
          left  = b[7]
          rear  = b[8]

    Motion model:
      - write_gatt_char() interprets CMD_DRIVE payload in the same minimal way
        your current driver does: a single signed speed field
      - the robot moves in a simple rectangular room
      - proximity values are synthesized from distances to walls
    """

    def __init__(self):
        self.notifier = None
        self._notify_task = None
        self._physics_task = None
        self._running = False

        # Connection state
        self._connected = False

        # Simple world: rectangular room
        self.room_x_min = 0.0
        self.room_x_max = 300.0
        self.room_y_min = 0.0
        self.room_y_max = 300.0

        # Robot pose
        self.pose = SimPose()

        # Commanded motion state
        self.cmd_speed = 0.0   # signed scalar "firmware speed"
        self.cmd_turn = 0.0    # optional future extension; currently ignored by packet format

        # Simulation gains
        self.linear_gain = 0.35     # distance units per second per command unit
        self.turn_bias = 0.0035     # built-in curvature to mimic imperfect reverse/forward behavior
        self.dt = 0.05              # physics step
        self.notify_period = 0.10   # BLE notify rate

    async def connect(self):
        print("[SIM] connect")
        self._connected = True

    async def disconnect(self):
        print("[SIM] disconnect")
        self._connected = False
        self._running = False

        tasks = [t for t in [self._notify_task, self._physics_task] if t is not None]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._notify_task = None
        self._physics_task = None

    @property
    def is_connected(self):
        return self._connected

    async def stop_notify(self, uuid):
        print(f"[SIM] stop_notify uuid={uuid}")
        self._running = False
        tasks = [t for t in [self._notify_task, self._physics_task] if t is not None]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._notify_task = None
        self._physics_task = None

    async def start_notify(self, uuid, callback):
        """
        Install the BLE notification callback and start:
          1) physics loop
          2) notification loop

        The callback is invoked as:
            callback(characteristic, data: bytearray)
        """
        print(f"[SIM] start_notify uuid={uuid}")
        self.notifier = callback
        self._running = True

        async def physics_loop():
            try:
                while self._running:
                    self._step_physics(self.dt)
                    await asyncio.sleep(self.dt)
            except asyncio.CancelledError:
                pass

        async def notify_loop():
            try:
                while self._running:
                    pkt = self._make_fake_sensor_packet()
                    self.notifier(uuid, pkt)
                    await asyncio.sleep(self.notify_period)
            except asyncio.CancelledError:
                pass

        self._physics_task = asyncio.create_task(physics_loop())
        self._notify_task = asyncio.create_task(notify_loop())

    async def write_gatt_char(self, uuid, payload):
        """
        Simulate writes from the driver.

        Assumes your driver sends:
            msg = bytes([cmd_id]) + payload

        and for CMD_DRIVE=0x02 with 3-byte payload:
            payload = [speed_lo, speed_hi, 0x00]

        We decode the signed speed and update internal simulated motion state.
        """
        msg = bytes(payload)
        print(f"[SIM WRITE] uuid={uuid} bytes={list(msg)}")

        if len(msg) < 1:
            return

        cmd_id = msg[0]

        # You said CMD_DRIVE is 0x02 in your stack
        if cmd_id == 0x02:
            if len(msg) >= 3:
                lo = msg[1]
                hi = msg[2]
                enc = lo | (hi << 8)
                speed = self._dec16_offset(enc)
                self.cmd_speed = float(speed)
                # current minimal packet ignores turn
                print(f"[SIM] decoded drive speed={speed}")
            else:
                print("[SIM] drive packet too short")

        # If you have a known stop command, handle it here too.
        # If stop is just drive speed = 0, this may not be needed.

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _dec16_offset(self, x: int) -> int:
        """
        Inverse of your enc16:
            x >= 0   -> x
            x < 0    -> 0x8000 + abs(x)
        """
        if x & 0x8000:
            return -(x & 0x7FFF)
        return x

    def _step_physics(self, dt: float) -> None:
        """
        Very simple kinematic model.
        The current firmware packet gives only one signed speed field,
        so we simulate a translational command plus a small curvature bias.

        Bias is useful because it lets you reproduce "reverse traces a circle"
        behavior even without full wheel control.
        """
        v = self.linear_gain * self.cmd_speed

        # Built-in curvature bias:
        # reverse curves more than forward, to mimic the real issue you observed
        if self.cmd_speed < 0:
            omega = -0.35 + self.turn_bias * self.cmd_speed
        elif self.cmd_speed > 0:
            omega = 0.05 + self.turn_bias * self.cmd_speed
        else:
            omega = 0.0

        self.pose.theta += omega * dt

        dx = v * math.cos(self.pose.theta) * dt
        dy = v * math.sin(self.pose.theta) * dt

        new_x = self.pose.x + dx
        new_y = self.pose.y + dy

        # Clip to walls
        self.pose.x = min(max(new_x, self.room_x_min), self.room_x_max)
        self.pose.y = min(max(new_y, self.room_y_min), self.room_y_max)

    def _make_fake_sensor_packet(self) -> bytearray:
        """
        Build a fake packet with proximity bytes at:
            b[6] = right
            b[7] = left
            b[8] = rear

        All other bytes are filler for now.
        """
        right, left, rear = self._compute_proximity_bytes()

        b = bytearray(9)
        b[0] = 0x00
        b[1] = 0x00
        b[2] = 0x00
        b[3] = 0x00
        b[4] = 0x00
        b[5] = 0x00
        b[6] = right
        b[7] = left
        b[8] = rear

        return b

    def _compute_proximity_bytes(self):
        """
        Convert distances to walls into synthetic proximity values [0..255].

        We use three rays:
          - left  : theta + 45 deg
          - right : theta - 45 deg
          - rear  : theta + 180 deg
        """
        left_dist = self._distance_to_wall(self.pose.x, self.pose.y, self.pose.theta + math.pi / 4.0)
        right_dist = self._distance_to_wall(self.pose.x, self.pose.y, self.pose.theta - math.pi / 4.0)
        rear_dist = self._distance_to_wall(self.pose.x, self.pose.y, self.pose.theta + math.pi)

        left = self._distance_to_proximity_byte(left_dist)
        right = self._distance_to_proximity_byte(right_dist)
        rear = self._distance_to_proximity_byte(rear_dist)

        return right, left, rear

    def _distance_to_wall(self, x: float, y: float, theta: float) -> float:
        """
        Ray-box intersection distance to the nearest wall in a rectangular room.
        """
        dx = math.cos(theta)
        dy = math.sin(theta)

        eps = 1e-9
        candidates = []

        # Intersections with x = const walls
        if abs(dx) > eps:
            t1 = (self.room_x_min - x) / dx
            t2 = (self.room_x_max - x) / dx
            if t1 > 0:
                y1 = y + t1 * dy
                if self.room_y_min <= y1 <= self.room_y_max:
                    candidates.append(t1)
            if t2 > 0:
                y2 = y + t2 * dy
                if self.room_y_min <= y2 <= self.room_y_max:
                    candidates.append(t2)

        # Intersections with y = const walls
        if abs(dy) > eps:
            t3 = (self.room_y_min - y) / dy
            t4 = (self.room_y_max - y) / dy
            if t3 > 0:
                x3 = x + t3 * dx
                if self.room_x_min <= x3 <= self.room_x_max:
                    candidates.append(t3)
            if t4 > 0:
                x4 = x + t4 * dx
                if self.room_x_min <= x4 <= self.room_x_max:
                    candidates.append(t4)

        if not candidates:
            return 1e6

        return min(candidates)

    def _distance_to_proximity_byte(self, d: float) -> int:
        """
        Map distance to a fake sensor byte.
        Near wall -> high proximity.
        Far from wall -> low proximity.
        """
        max_range = 120.0  # beyond this, treat as far
        d = max(0.0, min(d, max_range))
        proximity = int(round(255.0 * (1.0 - d / max_range)))
        return max(0, min(255, proximity))