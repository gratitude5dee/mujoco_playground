# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stream sim2sim rollout state as JSON-Lines RobotFrames.

A rollout of a joystick policy in C MuJoCo already knows everything a map-scale
telemetry consumer wants: where the pelvis is, how fast it is going, which way it
faces, and where the gait clock is. What it does *not* have is a place on Earth,
because the sim floor is a local plane at the origin. This module supplies both
halves: it projects the sim's own ENU displacement onto a geodetic origin, and it
serializes the result as one JSON object per line.

The wire format is the RobotFrame v1 schema used by God's Eye View
(`src/data/robotFrame.js`); its `mujoco-g1` bridge provider validates every line
before anything is ingested, so this module is deliberately permissive about
sinks and strict about field shapes.

Nothing here writes back into the sim, and nothing here is on by default: the
play scripts construct an exporter only when `--telemetry` is passed.

Usage from a play script (see `play_g1_joystick.py`):

    exporter = make_exporter("stdout", robot_id="g1-01")
    ...
    exporter.publish(exporter.frame_from_state(model, data, controller))

Sinks:
    "stdout"              one JSON object per line on stdout
    "/tmp/gev-g1.sock"    connect to a listening unix socket
    "127.0.0.1:8765"      connect to a listening TCP socket
"""

from __future__ import annotations

import dataclasses
import json
import math
import queue
import re
import socket
import sys
import threading
import time
from typing import Any, Optional, Protocol, Sequence

import mujoco
import numpy as np

# Everest Base Camp, the tail of the Khumbu route GEV ships as its demo track.
# Arbitrary but useful: it puts the rollout somewhere with real terrain relief,
# which is where a ground-clamped robot renderer is worth looking at.
DEFAULT_ORIGIN_LAT = 28.0026
DEFAULT_ORIGIN_LON = 86.8528
DEFAULT_ORIGIN_ALT_M = 5364.0

# The consumer's robot-id grammar (gods-eye-view `src/data/robotFrame.js`).
# Checked here so a bad `--telemetry_robot_id` fails at startup instead of
# having every frame silently rejected downstream.
_ROBOT_ID_RE = re.compile(r"^[0-9a-z~_-]{1,16}$")

_WGS84_A = 6378137.0
_WGS84_E2 = 6.69437999014e-3

# Below this the policy is holding position rather than travelling, and above it
# the joystick envelope is being used as a run. Both are reporting thresholds
# only — they do not feed back into the controller.
_STAND_SPEED_MPS = 0.08
_RUN_SPEED_MPS = 1.2
# A pelvis this low is squatting rather than standing; the G1 `knees_bent`
# keyframe stands around 0.75 m.
_SQUAT_PELVIS_M = 0.55


class Sink(Protocol):
  """Anything that accepts one serialized line."""

  def write_line(self, line: str) -> None:
    ...

  def close(self) -> None:
    ...


class StdoutSink:
  """JSON Lines on stdout, for `… | node tools/robot-bridge/bridge.mjs`."""

  def write_line(self, line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()

  def close(self) -> None:
    sys.stdout.flush()


MAX_PORT = 65535


def parse_socket_target(target: str) -> tuple[int, Any]:
  """Resolve a `--telemetry` socket spec to an (address family, address) pair.

  A numeric suffix means TCP, and it is range-checked here rather than left to
  `socket.connect`, which raises `OverflowError` — not `OSError` — on a port
  outside 0..65535 and so would escape the exporter's failure containment.
  """
  host, _, port = target.rpartition(":")
  if port.isdigit():
    port_number = int(port)
    if not 1 <= port_number <= MAX_PORT:
      raise ValueError(
          f"telemetry TCP port must be 1..{MAX_PORT}, received {target!r}"
      )
    return socket.AF_INET, ((host or "127.0.0.1"), port_number)
  return socket.AF_UNIX, target


class ThreadedSink:
  """Runs a blocking sink on a worker thread behind a bounded queue.

  Writes are issued from `mjcb_control`, where a stalled consumer — an unread
  pipe, a socket whose window is full, a TCP connect that has to time out —
  would hold up the physics step itself. The control thread therefore only ever
  does a non-blocking `put`, and the *telemetry* is what degrades under
  backpressure: the oldest queued frame is discarded to make room for the
  newest, because a map consumer wants the latest pose, not a replay of a
  backlog.
  """

  def __init__(self, sink: Sink, max_queued: int = 8):
    self._sink = sink
    self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=max_queued)
    self._worker = threading.Thread(
        target=self._drain, name="robotframe-telemetry", daemon=True
    )
    self.dropped = 0
    # Set when `close` gives up on a worker stuck in a blocking write.
    self.abandoned = False
    self._worker.start()

  @property
  def sink(self) -> Sink:
    """The wrapped blocking sink, for status reporting and tests."""
    return self._sink

  def _drain(self) -> None:
    while True:
      line = self._queue.get()
      if line is None:
        return
      try:
        self._sink.write_line(line)
      except (OSError, ValueError):
        self.dropped += 1

  def write_line(self, line: str) -> None:
    while True:
      try:
        self._queue.put_nowait(line)
        return
      except queue.Full:
        try:
          self._queue.get_nowait()
          self.dropped += 1
        except queue.Empty:
          # The worker drained it between the two calls; retry the put.
          continue

  def close(self) -> None:
    try:
      self._queue.put_nowait(None)
    except queue.Full:
      # Make room for the sentinel; a queued frame at shutdown is worth less
      # than a clean join.
      try:
        self._queue.get_nowait()
        self._queue.put_nowait(None)
      except (queue.Empty, queue.Full):
        pass
    self._worker.join(timeout=2.0)
    if self._worker.is_alive():
      # The worker is still inside a blocking write, and the sink's own `close`
      # would queue behind it — a full stdout pipe has no timeout to fall back
      # on. The worker is a daemon, so leaving it where it is costs a possibly
      # unflushed final frame and nothing else; hanging the rollout's shutdown
      # would cost more.
      self.abandoned = True
      return
    self._sink.close()


class SocketSink:
  """JSON Lines to a listening local socket, reconnecting when it drops.

  The bridge provider is the listener, so the sim may be started, stopped and
  restarted freely; a bridge that is not up yet simply means dropped lines, never
  a dead rollout.
  """

  def __init__(self, target: str, reconnect_interval_s: float = 1.0):
    # Validated eagerly so a malformed target fails at startup rather than once
    # per frame inside the control callback.
    parse_socket_target(target)
    self._target = target
    self._reconnect_interval_s = reconnect_interval_s
    self._socket: Optional[socket.socket] = None
    self._next_attempt_at = 0.0
    self.dropped = 0

  def _address(self) -> tuple[int, Any]:
    return parse_socket_target(self._target)

  def _connect(self) -> None:
    now = time.monotonic()
    if self._socket is not None or now < self._next_attempt_at:
      return
    self._next_attempt_at = now + self._reconnect_interval_s
    family, address = self._address()
    try:
      connection = socket.socket(family, socket.SOCK_STREAM)
      connection.settimeout(1.0)
      connection.connect(address)
      self._socket = connection
    except OSError:
      self._socket = None

  def write_line(self, line: str) -> None:
    self._connect()
    if self._socket is None:
      self.dropped += 1
      return
    try:
      self._socket.sendall((line + "\n").encode("utf-8"))
    except OSError:
      self.dropped += 1
      try:
        self._socket.close()
      finally:
        self._socket = None

  def close(self) -> None:
    if self._socket is not None:
      try:
        self._socket.close()
      finally:
        self._socket = None


def open_sink(spec: str, threaded: bool = True) -> Sink:
  """Build a sink from a `--telemetry` value.

  Wrapped in a `ThreadedSink` by default so no write can block the control
  callback; pass `threaded=False` for synchronous behaviour in tests.
  """
  if not spec:
    raise ValueError("empty telemetry target")
  sink: Sink = StdoutSink() if spec == "stdout" else SocketSink(spec)
  return ThreadedSink(sink) if threaded else sink


def _round_list(values: Any, digits: int = 4) -> list[float]:
  """Serialize an array compactly: full float64 text triples the line length."""
  return [round(float(v), digits) for v in np.asarray(values).reshape(-1)]


def _sensor(data: mujoco.MjData, name: str) -> Optional[np.ndarray]:
  """Read a named sensor, or None on a model that does not carry it."""
  try:
    return np.asarray(data.sensor(name).data, dtype=float)
  except (KeyError, ValueError):
    return None


def _gravity_in_site(
    model: mujoco.MjModel, data: mujoco.MjData, site: str
) -> Optional[np.ndarray]:
  """Gravity direction expressed in a site's frame, as the policy sees it."""
  try:
    site_id = model.site(site).id
  except (KeyError, ValueError):
    return None
  rotation = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
  return rotation.T @ np.array([0.0, 0.0, -1.0])


def wrap_deg(angle_deg: float) -> float:
  """Folds an angle into [0, 360), snapping float dust away from the seam.

  Plain `% 360.0` turns a hair under a full turn into 359.999999999999994, which
  reads as "pointing south-by-a-whisker" downstream instead of due north.
  """
  wrapped = angle_deg % 360.0
  if wrapped > 360.0 - 1e-9 or wrapped < 1e-9:
    return 0.0
  return wrapped


def yaw_from_quat(quat: Sequence[float]) -> float:
  """Yaw in radians about +Z from a MuJoCo (w, x, y, z) quaternion."""
  w, x, y, z = (float(v) for v in quat)
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def euler_from_quat(quat: Sequence[float]) -> tuple[float, float, float]:
  """(roll, pitch, yaw) in radians from a MuJoCo (w, x, y, z) quaternion."""
  w, x, y, z = (float(v) for v in quat)
  roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  # Clamped so a gimbal-adjacent pose yields ±90° instead of a domain error.
  pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
  return roll, pitch, yaw_from_quat(quat)


@dataclasses.dataclass(frozen=True)
class GeodeticOrigin:
  """Anchors the sim's local ENU plane to a point on the ellipsoid.

  MuJoCo's world x/y is a metric tangent plane, which is exactly what a local
  ENU frame is, so the projection is a per-latitude metres-per-degree scaling
  rather than anything that needs a projection library. It is accurate to well
  under a metre over the kilometre-scale walks a rollout produces, and it never
  drifts, because it reads the sim's absolute position instead of integrating
  velocity.
  """

  lat: float = DEFAULT_ORIGIN_LAT
  lon: float = DEFAULT_ORIGIN_LON
  alt_m: float = DEFAULT_ORIGIN_ALT_M

  def to_lat_lon(self, north_m: float, east_m: float) -> tuple[float, float]:
    lat_rad = math.radians(self.lat)
    sin_lat = math.sin(lat_rad)
    denom = 1.0 - _WGS84_E2 * sin_lat * sin_lat
    meridian = _WGS84_A * (1.0 - _WGS84_E2) / (denom**1.5)
    normal = _WGS84_A / math.sqrt(denom)
    lat = self.lat + math.degrees(north_m / meridian)
    lon = self.lon + math.degrees(east_m / (normal * math.cos(lat_rad)))
    # Keep both inside the wire schema's ranges even for an absurd walk.
    lat = max(-90.0, min(90.0, lat))
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat, lon

  @classmethod
  def parse(cls, spec: Optional[str]) -> "GeodeticOrigin":
    """Parse `lat,lon[,alt]`; an empty spec yields the default origin."""
    if not spec:
      return cls()
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) not in (2, 3):
      raise ValueError(f"origin must be 'lat,lon[,alt]', received {spec!r}")
    lat, lon = float(parts[0]), float(parts[1])
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
      raise ValueError(f"origin out of range: {spec!r}")
    alt = float(parts[2]) if len(parts) == 3 else DEFAULT_ORIGIN_ALT_M
    return cls(lat=lat, lon=lon, alt_m=alt)


def classify_gait(speed_mps: float, pelvis_height_m: float) -> str:
  """Map rollout state onto the wire schema's gait FSM states.

  The joystick policy has no FSM of its own — it is one continuous controller —
  so this is a *description* of what the robot is doing, not a state the
  controller is in. Keeping it here rather than in the consumer means the
  thresholds live next to the sim that justifies them.
  """
  if pelvis_height_m < _SQUAT_PELVIS_M:
    return "squat"
  if speed_mps < _STAND_SPEED_MPS:
    return "stand"
  if speed_mps > _RUN_SPEED_MPS:
    return "run"
  return "walk"


class RobotFrameExporter:
  """Serialize control-step state as RobotFrames onto a sink."""

  def __init__(
      self,
      sink: Sink,
      robot_id: str = "g1-01",
      origin: Optional[GeodeticOrigin] = None,
      rate_hz: float = 10.0,
      gait_freq_hz: float = 1.5,
      provenance: str = "live-g1",
      confidence: float = 0.9,
      include_obs: bool = False,
  ):
    self._sink = sink
    self._include_obs = include_obs
    self._robot_id = robot_id
    self._origin = origin or GeodeticOrigin()
    self._min_interval_s = 1.0 / rate_hz if rate_hz and rate_hz > 0 else 0.0
    self._gait_freq_hz = gait_freq_hz
    self._provenance = provenance
    self._confidence = confidence
    self._last_publish_at = 0.0
    self.published = 0
    self.skipped = 0
    self.errors = 0

  def frame_from_state(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      controller: Optional[Any] = None,
  ) -> dict[str, Any]:
    """Build one RobotFrame from the live sim state.

    `controller` is the `OnnxController` when one is available: its private gait
    clock is the phase the policy is actually acting on, which is the whole point
    of forwarding a phase at all. Everything else is read from `data`, so this
    works for any humanoid whose free joint is `qpos[0:7]`.
    """
    position = np.asarray(data.qpos[0:3], dtype=float)
    roll, pitch, yaw = euler_from_quat(data.qpos[3:7])
    # The sim plane is read as ENU — x east, y north — which is the mapping that
    # keeps MuJoCo's right-handed, z-up world right-handed on the map. (Sending
    # x to north instead would mirror the world and reverse every turn.)
    lat, lon = self._origin.to_lat_lon(
        north_m=float(position[1]), east_m=float(position[0])
    )
    # World-frame linear velocity of the free joint, rotated into the pelvis
    # frame: forward is +x, left is +y — the same convention the policy's own
    # `local_linvel_pelvis` observation uses, so the consumer needs no second
    # rule for reading it.
    world_vel = np.asarray(data.qvel[0:3], dtype=float)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    forward = cos_yaw * world_vel[0] + sin_yaw * world_vel[1]
    left = -sin_yaw * world_vel[0] + cos_yaw * world_vel[1]
    speed_mps = math.hypot(forward, left)
    # Yaw is counter-clockwise from east; a compass heading is clockwise from
    # north.
    heading_deg = wrap_deg(90.0 - math.degrees(yaw))

    phase = None
    if controller is not None:
      clock = getattr(controller, "_phase", None)
      if clock is not None and len(np.atleast_1d(clock)) > 0:
        phase = float(np.atleast_1d(clock)[0]) % (2.0 * math.pi)

    frame: dict[str, Any] = {
        "v": 1,
        "id": self._robot_id,
        "t": int(time.time() * 1000),
        "pose": {
            "lat": lat,
            "lon": lon,
            "altM": self._origin.alt_m + float(position[2]),
            "headingDeg": heading_deg,
            "pitchDeg": math.degrees(pitch),
            "rollDeg": math.degrees(roll),
        },
        "datum": "wgs84-ellipsoid",
        # The sim knows its own pose exactly; `fused` with a sub-centimetre
        # accuracy is the honest description of a rollout, and it keeps the
        # consumer from having to special-case a source it cannot render.
        "fix": {"source": "fused", "hAccM": 0.01, "vAccM": 0.01},
        "gait": {
            "fsm": classify_gait(speed_mps, float(position[2])),
            "cadenceHz": self._gait_freq_hz,
            "strideM": None,
            "phase": phase,
        },
        "vel": {
            "speedMps": speed_mps,
            "courseDeg": wrap_deg(
                heading_deg - math.degrees(math.atan2(left, forward))
            ),
        },
        "provenance": {
            "source": self._provenance,
            "label": _PROVENANCE_LABELS[self._provenance],
            "confidence": self._confidence,
        },
    }
    if self._include_obs:
      frame["sim"] = self._obs_block(model, data, controller)
    return frame

  def _obs_block(
      self,
      model: mujoco.MjModel,
      data: mujoco.MjData,
      controller: Optional[Any],
  ) -> dict[str, Any]:
    """The policy's own view of the robot, for local inspection only.

    This is the same state `OnnxController.get_obs` feeds the network. It is off
    by default and *not* part of the wire schema: the relay caps an ingest body
    at 256 KB across a batch of up to 200 frames, and ~23 joints x 2 arrays per
    frame would push whole batches over that cap and get them rejected, so the
    bridge strips this block before forwarding. Turn it on when recording a
    rollout or debugging a policy against the map, not for a live map feed.
    """
    block: dict[str, Any] = {"jointVelRadps": _round_list(data.qvel[6:])}
    # The network is fed joint angles *relative to the default pose*, so that is
    # what is serialized; the absolute angles are recoverable by adding the
    # controller's default angles back, which are constant for a rollout.
    default_angles = getattr(controller, "_default_angles", None)
    if default_angles is not None:
      block["jointAnglesRad"] = _round_list(
          np.asarray(data.qpos[7:], dtype=float)
          - np.asarray(default_angles, dtype=float)
      )
      block["defaultAnglesRad"] = _round_list(default_angles)
    else:
      block["jointAnglesRad"] = _round_list(data.qpos[7:])
    linvel = _sensor(data, "local_linvel_pelvis")
    if linvel is not None:
      block["linvelPelvis"] = _round_list(linvel)
    gyro = _sensor(data, "gyro_pelvis")
    if gyro is not None:
      block["gyroPelvis"] = _round_list(gyro)
    gravity = _gravity_in_site(model, data, "imu_in_pelvis")
    if gravity is not None:
      block["gravityPelvis"] = _round_list(gravity)
    if controller is not None:
      clock = getattr(controller, "_phase", None)
      if clock is not None:
        block["gaitPhaseRad"] = _round_list(np.atleast_1d(clock))
      command = getattr(controller, "_last_command", None)
      if command is not None:
        # (forward, lateral, yaw-rate) as the operator asked for it.
        block["command"] = _round_list(np.atleast_1d(command))
      # The action term of the observation is the *previous* action, and
      # `_last_action` already holds this step's prediction by the time
      # telemetry runs — so `_obs_last_action`, snapshotted inside `get_obs`, is
      # what makes a recorded frame reproducible.
      last_action = getattr(controller, "_obs_last_action", None)
      if last_action is None:
        last_action = getattr(controller, "_last_action", None)
      if last_action is not None:
        block["lastAction"] = _round_list(np.atleast_1d(last_action))
    return block

  def publish(self, frame: dict[str, Any]) -> bool:
    """Emit one frame, rate-limited. Never raises into the control callback."""
    now = time.monotonic()
    if (
        self._min_interval_s
        and now - self._last_publish_at < self._min_interval_s
    ):
      self.skipped += 1
      return False
    self._last_publish_at = now
    try:
      self._sink.write_line(json.dumps(frame, separators=(",", ":")))
    except (OSError, TypeError, ValueError):
      # A telemetry problem must never stop the robot mid-stride: the sim keeps
      # running and the failure shows up as a count.
      self.errors += 1
      return False
    self.published += 1
    return True

  def close(self) -> None:
    self._sink.close()


_PROVENANCE_LABELS = {
    "live-g1": "LIVE",
    "phone": "PHONE PROXY",
    "replay": "REPLAY",
    "synthetic": "SIMULATED",
}


def make_exporter(
    telemetry: Optional[str],
    robot_id: str = "g1-01",
    origin: Optional[str] = None,
    rate_hz: float = 10.0,
    gait_freq_hz: float = 1.5,
    provenance: str = "live-g1",
    include_obs: bool = False,
) -> Optional[RobotFrameExporter]:
  """Build an exporter, or None when telemetry is off (the default)."""
  if not telemetry:
    return None
  if not _ROBOT_ID_RE.match(robot_id):
    raise ValueError(
        f"robot id must match {_ROBOT_ID_RE.pattern}, received {robot_id!r}"
    )
  if provenance not in _PROVENANCE_LABELS:
    raise ValueError(
        f"provenance must be one of {sorted(_PROVENANCE_LABELS)}, "
        f"received {provenance!r}"
    )
  return RobotFrameExporter(
      open_sink(telemetry),
      robot_id=robot_id,
      origin=GeodeticOrigin.parse(origin),
      rate_hz=rate_hz,
      gait_freq_hz=gait_freq_hz,
      provenance=provenance,
      include_obs=include_obs,
  )
