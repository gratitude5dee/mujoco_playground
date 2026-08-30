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

"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it.

Pass `--telemetry` to additionally stream the rollout as JSON-Lines RobotFrames
for a map-scale viewer; see `robotframe_exporter.py` and this directory's README.
Without it the script behaves exactly as before.
"""

from absl import app
from absl import flags
from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt

from mujoco_playground._src.locomotion.g1 import g1_constants
from mujoco_playground._src.locomotion.g1.base import get_assets
from mujoco_playground.experimental.sim2sim import robotframe_exporter
from mujoco_playground.experimental.sim2sim.gamepad_reader import Gamepad

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"

_TELEMETRY = flags.DEFINE_string(
    "telemetry",
    None,
    "Where to stream JSON-Lines RobotFrames: 'stdout', a unix socket path, or"
    " 'host:port'. Off by default; the viewer is unchanged without it.",
)
_TELEMETRY_ROBOT_ID = flags.DEFINE_string(
    "telemetry_robot_id", "g1-01", "Robot id stamped on every exported frame."
)
_TELEMETRY_ORIGIN = flags.DEFINE_string(
    "telemetry_origin",
    None,
    "Geodetic anchor for the sim's local plane as 'lat,lon[,alt]'. Defaults to"
    " Everest Base Camp.",
)
_TELEMETRY_RATE_HZ = flags.DEFINE_float(
    "telemetry_rate_hz", 10.0, "Maximum exported frames per second."
)
_TELEMETRY_PROVENANCE = flags.DEFINE_enum(
    "telemetry_provenance",
    "live-g1",
    ["live-g1", "synthetic", "replay", "phone"],
    "How the consumer should label these frames. 'live-g1' is the sim2sim"
    " deploy seam (the same ONNX policy that runs on hardware); 'synthetic'"
    " labels a rollout as SIMULATED.",
)
_TELEMETRY_INCLUDE_OBS = flags.DEFINE_bool(
    "telemetry_include_obs",
    False,
    "Also emit the raw policy observation (pelvis linvel, gyro, gravity, joint"
    " angles/velocities, gait phase, command) under a `sim` key. Useful for"
    " recording or debugging a rollout; the map bridge strips it, since a batch"
    " of these exceeds the relay's body limit.",
)


class OnnxController:
  """ONNX controller for the Go-1 robot."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      action_scale: float = 0.5,
      vel_scale_x: float = 1.0,
      vel_scale_y: float = 1.0,
      vel_scale_rot: float = 1.0,
      exporter=None,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)
    self._obs_last_action = self._last_action.copy()

    self._counter = 0
    self._n_substeps = n_substeps
    self._last_command = np.zeros(3, dtype=np.float32)

    self._phase = np.array([0.0, np.pi])
    self._gait_freq = 1.5
    self._phase_dt = 2 * np.pi * self._gait_freq * ctrl_dt

    self._joystick = Gamepad(
        vel_scale_x=vel_scale_x,
        vel_scale_y=vel_scale_y,
        vel_scale_rot=vel_scale_rot,
    )

    # None unless --telemetry was passed; the control path below is otherwise
    # byte-for-byte the original one.
    self._exporter = exporter

  def get_obs(self, model, data) -> np.ndarray:
    linvel = data.sensor("local_linvel_pelvis").data
    gyro = data.sensor("gyro_pelvis").data
    imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    phase = np.concatenate([np.cos(self._phase), np.sin(self._phase)])
    command = self._joystick.get_command()
    # Kept so the optional telemetry exporter can report the operator intent the
    # policy actually saw, instead of re-polling the pad a step later.
    self._last_command = command
    # Likewise for the action term: `_last_action` is overwritten with this
    # step's prediction before telemetry runs, so the value the policy was
    # conditioned on is kept separately.
    self._obs_last_action = np.array(self._last_action, dtype=np.float32)
    obs = np.hstack([
        linvel,
        gyro,
        gravity,
        command,
        joint_angles,
        joint_velocities,
        self._last_action,
        phase,
    ])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      self._last_action = onnx_pred.copy()
      data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
      # Published before the clock advances, so the reported phase is the one
      # this step's action was computed from.
      if self._exporter is not None:
        self._exporter.publish(
            self._exporter.frame_from_state(model, data, self)
        )
      phase_tp1 = self._phase + self._phase_dt
      self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


def load_callback(model=None, data=None, exporter=None):
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      g1_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 1)

  ctrl_dt = 0.02
  sim_dt = 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  policy = OnnxController(
      policy_path=(_ONNX_DIR / "g1_policy.onnx").as_posix(),
      default_angles=np.array(model.keyframe("knees_bent").qpos[7:]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      action_scale=0.5,
      vel_scale_x=1.5,
      vel_scale_y=0.8,
      vel_scale_rot=2 * np.pi,
      exporter=exporter,
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data


def main(argv):
  del argv
  exporter = robotframe_exporter.make_exporter(
      _TELEMETRY.value,
      robot_id=_TELEMETRY_ROBOT_ID.value,
      origin=_TELEMETRY_ORIGIN.value,
      rate_hz=_TELEMETRY_RATE_HZ.value,
      provenance=_TELEMETRY_PROVENANCE.value,
      include_obs=_TELEMETRY_INCLUDE_OBS.value,
  )
  try:
    viewer.launch(
        loader=lambda model=None, data=None: load_callback(
            model, data, exporter=exporter
        )
    )
  finally:
    if exporter is not None:
      exporter.close()


if __name__ == "__main__":
  app.run(main)
