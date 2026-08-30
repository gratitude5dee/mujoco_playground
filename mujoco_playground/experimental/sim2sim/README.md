# Sim2Sim Transfer

In this directory, we demonstrate how to deploy a Go1 joystick controller trained with Playground in native C MuJoCo and interact with it using a joystick.

## Usage

```bash
python play_go1_joystick.py
```

<a href="https://youtu.be/XwF2lkT2gqo" target="_blank">
 <img src="http://img.youtube.com/vi/XwF2lkT2gqo/hqdefault.jpg" alt="Watch the video" width="560" height="315"/>
</a>

## Requirements

We'll need 2 additional dependencies:

1. `onnxruntime` for running the ONNX model.
2. `hidapi` for reading the joystick.

```bash
uv pip install onnxruntime hidapi
```

On macOS, you'll need to install `hidapi` with brew and correctly set the `DYLD_LIBRARY_PATH` environment variable.

```bash
brew install hidapi
export DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/hidapi/0.14.0/lib:$DYLD_LIBRARY_PATH
```

## Joystick

We use a Logitech G F710 Wireless Gamepad in this example. You can buy one on [Amazon](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW) for $40. In principle, you can use any joystick of your choice, but you'll need to modify `gamepad_reader.py` to support it.

<img src="assets/f710.jpg" alt="Logitech F710 Gamepad" width="200">

- Why not use `inputs`? It didn't seem to read any joystick on macOS.
- Why not use `pygame`? PyGame and MuJoCo's viewer don't place nice together. pygame needs to run on the main thread on macOS and we use the managed viewer in MuJoCo which runs the policy in a callback thread.

## Exporting a trained policy

We have a notebook for exporting trained policies to ONNX format. See `mujoco_playground/experimental/brax_network_to_onnx.ipynb`.

## G1: train, deploy, and stream the rollout

`play_g1_joystick.py` is the same story for the Unitree G1 humanoid, and it closes the
loop from RL training all the way to a map-scale viewer.

### 1. Train a policy

```bash
# Flat terrain: the quickest policy that walks.
train-jax-ppo --env_name G1JoystickFlatTerrain
# Rough terrain: slower to train, sturdier on slopes and debris.
train-jax-ppo --env_name G1JoystickRoughTerrain
```

Each run writes checkpoints under `logs/`. Training is where the "self-learning" happens:
PPO improves the joystick policy against the MJX environment until it tracks commanded
velocities without falling.

### 2. Export it to ONNX

Run `mujoco_playground/experimental/brax_network_to_onnx.ipynb` against the checkpoint and
save the result as `mujoco_playground/experimental/sim2sim/onnx/g1_policy.onnx` — the path
`play_g1_joystick.py` loads.

### 3. Play it in native MuJoCo

```bash
python play_g1_joystick.py
```

This is the sim2sim step: the exported ONNX policy — byte-for-byte the artifact you would
flash onto hardware — now drives C MuJoCo instead of MJX, with the gamepad supplying the
velocity command.

### 4. Stream the rollout as RobotFrames (optional)

`--telemetry` turns on `robotframe_exporter.py`, which serializes each control step as one
line of JSON on stdout or a local socket. The record is a
[RobotFrame](https://github.com/gratitude5dee/gods-eye-view/blob/main/src/data/robotFrame.js):
pose (the sim's local ENU plane projected onto WGS84 around a geodetic origin), heading,
speed/course, and a gait descriptor carrying the policy's own gait phase. Without the flag
the script behaves exactly as before.

```bash
# Straight into the viewer's telemetry bridge over a unix socket (recommended:
# stdout also carries import-time warnings from optional dependencies).
# In the gods-eye-view checkout, with GEV_ROBOT_INGEST_TOKEN exported:
node tools/robot-bridge/bridge.mjs --provider mujoco-g1 \
    --socket /tmp/g1-telemetry.sock \
    --ingest https://<your-gev-host>/api/robot/ingest &
python play_g1_joystick.py --telemetry /tmp/g1-telemetry.sock
```

Useful flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--telemetry` | off | `stdout`, a unix socket path, or `host:port`. |
| `--telemetry_robot_id` | `g1-01` | Id stamped on every frame (`[0-9a-z~_-]{1,16}`). |
| `--telemetry_origin` | Everest Base Camp | `lat,lon[,alt]` anchor for the sim's local plane. |
| `--telemetry_rate_hz` | `10` | Publish rate; excess control steps are dropped, never queued. |
| `--telemetry_provenance` | `live-g1` | How the viewer labels the feed (`live-g1` → `LIVE`, `synthetic` → `SIMULATED`). |
| `--telemetry_include_obs` | off | Also emit the raw policy observation under a `sim` key. |

`--telemetry_include_obs` adds the exact state the network consumes — pelvis linear
velocity, gyro, gravity in the IMU frame, joint angles and velocities, both gait phases and
the operator command — which is what you want when recording or debugging a rollout. It is
off by default because the viewer's relay caps an ingest body at 256 KB per batch and the
bridge strips the block anyway; a live map feed does not need it.

Telemetry is strictly outbound: nothing on this channel can command the robot, and a sink
that is down or slow costs dropped lines, never a stumble in the control callback.
