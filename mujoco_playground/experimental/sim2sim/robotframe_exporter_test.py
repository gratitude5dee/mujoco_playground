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

"""Tests for the sim2sim RobotFrame exporter."""

import json
import math
import threading
import time

from absl.testing import absltest
import mujoco
import numpy as np

from mujoco_playground.experimental.sim2sim import robotframe_exporter

# A free-floating body is all `frame_from_state` reads: qpos[0:7] is the pose and
# qvel[0:3] the world-frame velocity, exactly as for the G1 pelvis.
_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 0.75">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class _RecordingSink:

  def __init__(self):
    self.lines = []
    self.closed = False

  def write_line(self, line):
    self.lines.append(line)

  def close(self):
    self.closed = True


class _BrokenSink(_RecordingSink):

  def write_line(self, line):
    raise OSError("bridge went away")


class _FakeController:
  """Stands in for OnnxController.

  Carries the same private attributes the real controller exposes and builds its
  observation the same way, so a test can compare what the exporter serializes
  against the slices the policy is actually fed.
  """

  def __init__(self, phase, n_joints=0):
    self._phase = np.array([phase, phase + np.pi])
    self._default_angles = np.arange(n_joints, dtype=float) * 0.1
    self._last_action = np.full(n_joints, -0.25)
    self._last_command = np.zeros(3)

  def get_obs(self, model, data):
    del model
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
    return np.hstack([
        self._last_command,
        joint_angles,
        joint_velocities,
        self._last_action,
        phase,
    ]).astype(np.float32)


def _state(
    pos=(0.0, 0.0, 0.75), quat=(1.0, 0.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0)
):
  model = mujoco.MjModel.from_xml_string(_XML)
  data = mujoco.MjData(model)
  data.qpos[0:3] = pos
  data.qpos[3:7] = quat
  data.qvel[0:3] = vel
  mujoco.mj_forward(model, data)
  return model, data


# Two hinges past the free joint, so qpos[7:] / qvel[6:] are non-empty and the
# serialized observation can be checked against real slices.
_JOINTED_XML = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 0.75">
      <freejoint/>
      <geom type="sphere" size="0.1"/>
      <body name="thigh" pos="0 0 -0.2">
        <joint name="hip" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.04"/>
        <body name="shank" pos="0 0 -0.2">
          <joint name="knee" type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.04"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _jointed_state(joint_angles=(0.3, -0.7), joint_vels=(0.1, -0.2)):
  model = mujoco.MjModel.from_xml_string(_JOINTED_XML)
  data = mujoco.MjData(model)
  data.qpos[0:3] = (0.0, 0.0, 0.75)
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  data.qpos[7:] = joint_angles
  data.qvel[6:] = joint_vels
  mujoco.mj_forward(model, data)
  return model, data


def _yaw_quat(yaw_rad):
  return (math.cos(yaw_rad / 2.0), 0.0, 0.0, math.sin(yaw_rad / 2.0))


class GeodeticOriginTest(absltest.TestCase):

  def test_local_plane_maps_to_metres_on_the_ellipsoid(self):
    origin = robotframe_exporter.GeodeticOrigin(lat=0.0, lon=0.0, alt_m=0.0)
    lat, lon = origin.to_lat_lon(north_m=1000.0, east_m=0.0)
    # A kilometre north is ~0.009° of latitude anywhere on Earth.
    self.assertAlmostEqual(lat, 0.00904, places=4)
    self.assertAlmostEqual(lon, 0.0)
    lat, lon = origin.to_lat_lon(north_m=0.0, east_m=1000.0)
    self.assertAlmostEqual(lat, 0.0)
    self.assertGreater(lon, 0.0)

  def test_east_degrees_stretch_with_latitude(self):
    at_equator = robotframe_exporter.GeodeticOrigin(lat=0.0, lon=0.0)
    up_north = robotframe_exporter.GeodeticOrigin(lat=60.0, lon=0.0)
    _, equator_lon = at_equator.to_lat_lon(0.0, 1000.0)
    _, north_lon = up_north.to_lat_lon(0.0, 1000.0)
    self.assertGreater(north_lon, equator_lon * 1.9)

  def test_positions_stay_inside_the_wire_schema_ranges(self):
    origin = robotframe_exporter.GeodeticOrigin(lat=0.0, lon=179.999)
    lat, lon = origin.to_lat_lon(north_m=0.0, east_m=10_000.0)
    self.assertLessEqual(-180.0, lon)
    self.assertLess(lon, 180.0)
    self.assertLessEqual(abs(lat), 90.0)

  def test_parse(self):
    self.assertEqual(
        robotframe_exporter.GeodeticOrigin.parse("1.5,-2.5,300"),
        robotframe_exporter.GeodeticOrigin(1.5, -2.5, 300.0),
    )
    default = robotframe_exporter.GeodeticOrigin.parse(None)
    self.assertEqual(default.lat, robotframe_exporter.DEFAULT_ORIGIN_LAT)
    self.assertEqual(
        robotframe_exporter.GeodeticOrigin.parse("10,20").alt_m,
        robotframe_exporter.DEFAULT_ORIGIN_ALT_M,
    )
    for bad in ("10", "10,20,30,40", "91,0", "0,181", "north,east"):
      with self.assertRaises(ValueError):
        robotframe_exporter.GeodeticOrigin.parse(bad)


class OrientationTest(absltest.TestCase):

  def test_yaw_and_euler_round_trip(self):
    for yaw_deg in (0.0, 45.0, 90.0, -170.0):
      quat = _yaw_quat(math.radians(yaw_deg))
      self.assertAlmostEqual(
          math.degrees(robotframe_exporter.yaw_from_quat(quat)),
          yaw_deg,
          places=6,
      )
      roll, pitch, yaw = robotframe_exporter.euler_from_quat(quat)
      self.assertAlmostEqual(roll, 0.0, places=6)
      self.assertAlmostEqual(pitch, 0.0, places=6)
      self.assertAlmostEqual(math.degrees(yaw), yaw_deg, places=6)

  def test_pitch_is_clamped_rather_than_raising(self):
    # A straight-down pose is a numerically singular asin argument.
    half = math.sqrt(0.5)
    _, pitch, _ = robotframe_exporter.euler_from_quat((half, 0.0, -half, 0.0))
    self.assertAlmostEqual(math.degrees(pitch), -90.0, places=4)


class ClassifyGaitTest(absltest.TestCase):

  def test_describes_what_the_rollout_is_doing(self):
    self.assertEqual(robotframe_exporter.classify_gait(0.0, 0.75), "stand")
    self.assertEqual(robotframe_exporter.classify_gait(0.6, 0.75), "walk")
    self.assertEqual(robotframe_exporter.classify_gait(2.4, 0.75), "run")
    # A collapsed pelvis is not a fast stand, whatever the speed says.
    self.assertEqual(robotframe_exporter.classify_gait(0.6, 0.3), "squat")


class FrameFromStateTest(absltest.TestCase):

  def _exporter(self, sink=None, **kwargs):
    return robotframe_exporter.RobotFrameExporter(
        sink or _RecordingSink(), rate_hz=0.0, **kwargs
    )

  def test_frame_shape_matches_the_v1_wire_schema(self):
    model, data = _state()
    frame = self._exporter().frame_from_state(model, data, _FakeController(0.4))
    self.assertEqual(frame["v"], 1)
    self.assertEqual(frame["id"], "g1-01")
    self.assertEqual(frame["datum"], "wgs84-ellipsoid")
    self.assertEqual(frame["fix"]["source"], "fused")
    self.assertEqual(
        frame["provenance"],
        {"source": "live-g1", "label": "LIVE", "confidence": 0.9},
    )
    self.assertAlmostEqual(frame["gait"]["phase"], 0.4)
    self.assertEqual(frame["gait"]["cadenceHz"], 1.5)
    self.assertGreater(frame["t"], 1_600_000_000_000)
    # Serializable as one line, which is the entire transport contract.
    self.assertEqual(json.loads(json.dumps(frame))["id"], "g1-01")

  def test_position_is_read_absolutely_so_it_cannot_drift(self):
    origin = robotframe_exporter.GeodeticOrigin(lat=10.0, lon=20.0, alt_m=100.0)
    exporter = self._exporter(origin=origin)
    model, data = _state(pos=(0.0, 1000.0, 0.9))
    frame = exporter.frame_from_state(model, data)
    self.assertGreater(frame["pose"]["lat"], 10.0, "sim +y walks north")
    self.assertAlmostEqual(frame["pose"]["lon"], 20.0, places=6)
    self.assertAlmostEqual(frame["pose"]["altM"], 100.9, places=6)
    _, east = _state(pos=(1000.0, 0.0, 0.9))
    east_frame = exporter.frame_from_state(model, east)
    self.assertGreater(east_frame["pose"]["lon"], 20.0, "sim +x walks east")
    self.assertAlmostEqual(east_frame["pose"]["lat"], 10.0, places=6)

  def test_heading_is_compass_degrees(self):
    exporter = self._exporter()
    # Yaw 0 faces sim +x, which is east; yaw grows counter-clockwise while a
    # compass heading grows clockwise.
    model, facing_east = _state(quat=_yaw_quat(0.0))
    self.assertAlmostEqual(
        exporter.frame_from_state(model, facing_east)["pose"]["headingDeg"],
        90.0,
    )
    _, facing_north = _state(quat=_yaw_quat(math.radians(90.0)))
    self.assertAlmostEqual(
        exporter.frame_from_state(model, facing_north)["pose"]["headingDeg"],
        0.0,
        places=6,
    )

  def test_velocity_is_reported_in_the_pelvis_frame(self):
    # Facing north-east (yaw 45) at 1 m/s due east: the motion splits evenly
    # between forward and the robot's right, and the course is still due east.
    model, data = _state(
        quat=_yaw_quat(math.radians(45.0)), vel=(1.0, 0.0, 0.0)
    )
    frame = self._exporter().frame_from_state(model, data)
    self.assertAlmostEqual(frame["vel"]["speedMps"], 1.0, places=6)
    self.assertAlmostEqual(frame["pose"]["headingDeg"], 45.0, places=6)
    self.assertAlmostEqual(frame["vel"]["courseDeg"], 90.0, places=4)
    self.assertEqual(frame["gait"]["fsm"], "walk")
    # Drifting to the robot's left turns the course counter-clockwise instead.
    _, north = _state(quat=_yaw_quat(math.radians(45.0)), vel=(0.0, 1.0, 0.0))
    self.assertAlmostEqual(
        self._exporter().frame_from_state(model, north)["vel"]["courseDeg"],
        0.0,
        places=4,
    )

  def test_phase_is_absent_without_a_controller(self):
    model, data = _state()
    frame = self._exporter().frame_from_state(model, data)
    self.assertIsNone(frame["gait"]["phase"])

  def test_phase_is_wrapped_into_a_positive_cycle(self):
    model, data = _state()
    frame = self._exporter().frame_from_state(
        model, data, _FakeController(-0.5)
    )
    self.assertGreaterEqual(frame["gait"]["phase"], 0.0)
    self.assertLess(frame["gait"]["phase"], 2 * math.pi)


class ObsBlockTest(absltest.TestCase):

  def test_the_raw_observation_is_off_by_default(self):
    model, data = _state()
    frame = robotframe_exporter.RobotFrameExporter(
        _RecordingSink(), rate_hz=0.0
    ).frame_from_state(model, data, _FakeController(0.0))
    self.assertNotIn("sim", frame)

  def test_the_raw_observation_carries_policy_inputs_when_asked(self):
    model, data = _state()
    controller = _FakeController(0.25)
    controller._last_command = np.array([0.5, -0.25, 1.0])
    exporter = robotframe_exporter.RobotFrameExporter(
        _RecordingSink(), rate_hz=0.0, include_obs=True
    )
    block = exporter.frame_from_state(model, data, controller)["sim"]
    self.assertEqual(block["command"], [0.5, -0.25, 1.0])
    self.assertEqual(block["gaitPhaseRad"], [0.25, round(0.25 + math.pi, 4)])
    # A free body has no actuated joints, so both arrays are simply empty.
    self.assertEqual(block["jointAnglesRad"], [])
    self.assertEqual(block["jointVelRadps"], [])
    # The G1's IMU sensors are absent here; the block degrades instead of
    # raising, so a model without them still streams.
    self.assertNotIn("gyroPelvis", block)
    self.assertNotIn("gravityPelvis", block)
    self.assertIsInstance(json.loads(json.dumps(block)), dict)

  def test_the_serialized_observation_matches_the_policy_input_slices(self):
    model, data = _jointed_state()
    controller = _FakeController(0.25, n_joints=2)
    controller._last_command = np.array([0.5, -0.25, 1.0])
    block = robotframe_exporter.RobotFrameExporter(
        _RecordingSink(), rate_hz=0.0, include_obs=True
    ).frame_from_state(model, data, controller)["sim"]
    obs = controller.get_obs(model, data)
    # Layout of the fake's (and the real controller's) trailing observation:
    # command(3) | joint angles(2) | joint velocities(2) | last action(2).
    np.testing.assert_allclose(block["command"], obs[0:3], atol=1e-4)
    np.testing.assert_allclose(block["jointAnglesRad"], obs[3:5], atol=1e-4)
    np.testing.assert_allclose(block["jointVelRadps"], obs[5:7], atol=1e-4)
    np.testing.assert_allclose(block["lastAction"], obs[7:9], atol=1e-4)
    # Absolute angles remain recoverable from the offsets that were sent.
    np.testing.assert_allclose(
        np.asarray(block["jointAnglesRad"])
        + np.asarray(block["defaultAnglesRad"]),
        data.qpos[7:],
        atol=1e-4,
    )


class PublishTest(absltest.TestCase):

  def test_rate_limiting_drops_rather_than_blocks(self):
    sink = _RecordingSink()
    exporter = robotframe_exporter.RobotFrameExporter(sink, rate_hz=1.0)
    model, data = _state()
    frame = exporter.frame_from_state(model, data)
    self.assertTrue(exporter.publish(frame))
    for _ in range(9):
      self.assertFalse(exporter.publish(frame))
    self.assertLen(sink.lines, 1)
    self.assertEqual(exporter.published, 1)
    self.assertEqual(exporter.skipped, 9)

  def test_a_broken_sink_never_stops_the_robot(self):
    exporter = robotframe_exporter.RobotFrameExporter(
        _BrokenSink(), rate_hz=0.0
    )
    model, data = _state()
    self.assertFalse(exporter.publish(exporter.frame_from_state(model, data)))
    self.assertEqual(exporter.errors, 1)
    self.assertEqual(exporter.published, 0)

  def test_close_closes_the_sink(self):
    sink = _RecordingSink()
    robotframe_exporter.RobotFrameExporter(sink).close()
    self.assertTrue(sink.closed)


class ThreadedSinkTest(absltest.TestCase):

  def test_a_slow_sink_does_not_block_the_caller(self):
    release = threading.Event()

    class _StalledSink(_RecordingSink):

      def write_line(self, line):
        release.wait(5.0)
        super().write_line(line)

    stalled = _StalledSink()
    sink = robotframe_exporter.ThreadedSink(stalled, max_queued=2)
    started = time.monotonic()
    for index in range(50):
      sink.write_line(f'{{"n":{index}}}')
    elapsed = time.monotonic() - started
    # The worker is parked inside the first write for seconds; the control
    # thread must not have waited on it.
    self.assertLess(elapsed, 1.0)
    self.assertGreater(sink.dropped, 0, "backpressure drops the oldest frames")
    release.set()
    sink.close()
    self.assertTrue(stalled.closed)
    self.assertLessEqual(len(stalled.lines), 3)

  def test_frames_reach_the_wrapped_sink_and_close_joins(self):
    recording = _RecordingSink()
    sink = robotframe_exporter.ThreadedSink(recording)
    sink.write_line('{"n":1}')
    sink.close()
    self.assertEqual(recording.lines, ['{"n":1}'])
    self.assertTrue(recording.closed)

  def test_a_failing_write_is_counted_not_raised(self):
    sink = robotframe_exporter.ThreadedSink(_BrokenSink())
    sink.write_line('{"n":1}')
    sink.close()
    self.assertEqual(sink.dropped, 1)

  def test_open_sink_wraps_blocking_sinks_by_default(self):
    threaded = robotframe_exporter.open_sink("stdout")
    self.assertIsInstance(threaded, robotframe_exporter.ThreadedSink)
    self.assertIsInstance(threaded.sink, robotframe_exporter.StdoutSink)
    threaded.close()


class MakeExporterTest(absltest.TestCase):

  def test_telemetry_is_off_by_default(self):
    self.assertIsNone(robotframe_exporter.make_exporter(None))
    self.assertIsNone(robotframe_exporter.make_exporter(""))

  def test_bad_options_fail_at_startup(self):
    with self.assertRaises(ValueError):
      robotframe_exporter.make_exporter("stdout", provenance="imagined")
    with self.assertRaises(ValueError):
      # Rejected downstream by the relay's id grammar, so rejected here.
      robotframe_exporter.make_exporter("stdout", robot_id="G1 Prototype #1")
    with self.assertRaises(ValueError):
      robotframe_exporter.make_exporter("stdout", origin="not-a-place")

  def test_sink_selection(self):
    self.assertIsInstance(
        robotframe_exporter.open_sink("stdout", threaded=False),
        robotframe_exporter.StdoutSink,
    )
    self.assertIsInstance(
        robotframe_exporter.open_sink("/tmp/gev-g1.sock", threaded=False),
        robotframe_exporter.SocketSink,
    )
    self.assertIsInstance(
        robotframe_exporter.open_sink("127.0.0.1:8765", threaded=False),
        robotframe_exporter.SocketSink,
    )
    with self.assertRaises(ValueError):
      robotframe_exporter.open_sink("")

  def test_an_out_of_range_tcp_port_is_rejected_at_startup(self):
    for target in ("localhost:99999", "127.0.0.1:0", "host:65536"):
      with self.assertRaises(ValueError):
        robotframe_exporter.open_sink(target)
    family, address = robotframe_exporter.parse_socket_target("localhost:8765")
    self.assertEqual(address, ("localhost", 8765))
    del family

  def test_a_socket_sink_without_a_listener_drops_instead_of_raising(self):
    sink = robotframe_exporter.SocketSink("/tmp/gev-g1-nonexistent.sock")
    exporter = robotframe_exporter.RobotFrameExporter(sink, rate_hz=0.0)
    model, data = _state()
    # A bridge that is not up yet costs lines, not the rollout.
    exporter.publish(exporter.frame_from_state(model, data))
    self.assertEqual(sink.dropped, 1)
    self.assertEqual(exporter.errors, 0)
    exporter.close()


if __name__ == "__main__":
  absltest.main()
