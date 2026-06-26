# `control_test/` — direct_rl controller + control_test ported to MuJoCo

Replicates direct_rl's `control_test` on UniLab's simulator (**mujoco-uni 3.8.0**),
to validate the **drone simulation + PX4/CTBR controller** before building the RL env.

## What is reused verbatim (identical numerics / params / delays)

`dynamics/` is **copied byte-for-byte** from `direct_rl/.../dynamics` (it is pure
`torch` + stdlib). The only edit: the single `isaaclab.utils.math.matrix_from_quat`
dependency (in `air_drag_effect.py`, `position_controller.py`) is replaced by
`dynamics/quat_math.py` — the **same IsaacLab formula**, so the controller behaves
exactly as under Isaac.

- PX4 position+attitude controller (`px4_control.py`) — charpi PX4 params (`config/charpi.yaml`)
- CTBR rate loop + motor 1st-order ZOH + air-drag + allocation (`control_stack.py`, …)
- `IntegerTimeDelay(action_delay_steps=3)`, rate Butterworths (40/30 Hz), position ZOH —
  **all delays identical**
- `circle_trajectory.py`, `helpers.py`, `analyze_log.py`, `plot_log.py`, `config/*.yaml` — copied

## What changed (the simulator boundary only)

| direct_rl (Isaac Sim) | here (MuJoCo) |
|---|---|
| `RigidObject.set_external_force_and_torque(F, M)` (body frame) | `data.xfrc_applied` (rotate body→world; `mj_drone.apply_body_wrench`) |
| `root_pos_w / quat_w / lin_vel_w / ang_vel_b` | `xpos / xquat` + `mj_objectVelocity` (`mj_drone.read_drone_state`) |
| `charpi.usd` + explicit D6 rope chain | `mj_drone.py` builds the MuJoCo drone (mass 1.09, COMBINED_INERTIA) + ball-jointed capsule rope |
| PhysX, GPU, `num_envs` batched | MuJoCo CPU; controllers run on torch-CPU; one env |

Frame conventions match Isaac/direct_rl: ENU world (z up), quaternion (w,x,y,z),
body FLU (z = thrust). The PX4 controller converts to NED/FRD internally, so the
inputs are identical.

## Your charpi model (URDF) — why it isn't loaded as-is, and what is used

You **can** use the real charpi model — and the drone visual now **is** your
`charpi_vision_description` mesh (`--visual mesh`, the default). What is *not* used
directly is the full URDF, because:

- it's a **`.xacro`** (ROS macro), not plain URDF — needs the ROS `xacro` tool to expand,
  and pulls `<gazebo>` / `<ros2control>` plugin tags MuJoCo ignores;
- it's a **46-link articulated** model (base + 4 arms + 4 motors + 4 *spinning* props +
  cameras + landing gear + plates + battery…), with **Y-up, cm-scale, per-link** meshes.
  direct_rl's control_test flew a **single folded rigid body** (charpi.usd "visual" mode),
  so to match it the plant here is one rigid body: **mass 1.09 kg, COMBINED_INERTIA**
  (the URDF's 46 link masses sum to **1.10 kg**, so this is faithful).

So: **dynamics = direct_rl's single rigid body; visual = your real charpi mesh**
(`obj_files/charpi_vision.obj`, loaded directly by MuJoCo at scale 0.01, rotated Y-up→Z-up,
centered on the COM). Override the mesh path with `$CHARPI_MESH`; `--visual simple` falls
back to a light box+arms. The mesh is visual-only (mass 0, no collision) — tracking is
byte-identical with or without it. To use the **full articulated** model instead, expand
the xacro to URDF (`xacro charpi_vision.xacro > charpi.urdf`) and load that — but that is a
different (heavier, multi-body) plant than direct_rl's and is not needed for the controller test.

## Visualization

- **Live (GUI)** — `viz.py` draws the commanded circle (green ring) and the current
  setpoint(s) (red = payload, blue = per-drone) into the viewer each step, so you see
  commanded-vs-achieved in 3-D while it flies. Enabled automatically with `--gui`.
- **Offline (`plot_log.py`)** — the direct_rl plotter, works on these logs unchanged:
  4 panels (XY trajectory vs setpoint, position error + lag, per-axis pos vs setpoint,
  tilt + thrust). `logs/single_circle.png`, `logs/multilift_circle.png` are pre-generated.

```bash
uv run python sandbox/multilift_mjcf/control_test/plot_log.py --log <csv> --cruise <m/s>
```

## Run

```bash
cd /home/carlson/rl_ws/UniLab          # mujoco-uni lives in this uv env

# controller unit test (no sim) — must match direct_rl exactly
uv run python sandbox/multilift_mjcf/control_test/test_px4_control.py

# single drone flying a circle (headless: log + tracking summary)
uv run python sandbox/multilift_mjcf/control_test/px4_circle_single.py --num_envs 1
uv run python sandbox/multilift_mjcf/control_test/px4_circle_single.py --gui        # live viewer

# cable-lift formation carrying the slung payload around a circle
uv run python sandbox/multilift_mjcf/control_test/px4_circle_multilift.py
uv run python sandbox/multilift_mjcf/control_test/px4_circle_multilift.py --gui

# re-analyse any log
uv run python sandbox/multilift_mjcf/control_test/analyze_log.py --log .../logs/single_circle.csv --cruise 2.09
```

`config/{single,multilift,charpi}.yaml` are the direct_rl configs verbatim — edit
trajectory / gains there. `--gui` runs the controller live (real-time paced) so you
can watch and Ctrl-drag to perturb.

## Validated (headless)

- **Controller unit test**: all 9 checks PASS (hover thrust 0.343, frame adapters,
  accel-FF) — bit-identical to direct_rl.
- **Single drone**: flies the circle stably; phase-lag-dominated error
  (~0.13–0.24 m steady, ~60–150 ms effective lag) growing with speed — the cascaded-PX4
  signature direct_rl documents. `hover_thrust=0.260` = charpi `MPC_THR_HOVER`.
- **Multilift**: 4-drone formation carries the 2 kg payload around the circle, **stable**;
  `hover_thrust=0.343` includes the payload share (support 1.59 kg). Horizontal payload
  tracking tight (~0.04 m); see the altitude note below.

## Known modeling difference (drone ↔ cable coupling)

direct_rl's drones are **maximal-coordinate free rigid bodies** connected to the cable
by a D6 joint (6 DOF) — PhysX resolves the coupling. Here the rope is a **reduced-coordinate
ball-jointed chain** and each drone is the **ball-jointed leaf** of that chain, so a drone
cannot translate independently of its cable (only swing it). This is the most stable cable
model (verified in `../passive_test.py`) but it leaves a **steady ~0.2 m payload altitude
sag** under load (the position loop can't push the drone straight up against the constraint).

For a closer match to Isaac's 6-DOF drones (tighter altitude), the alternative is **free-body
drones + a `<spatial>` tendon / `<equality><connect>` cable** (closed loop) — the transfer_plan
Phase-2 "cable model" decision. Easy to add as a `--cable {chain,tendon}` switch if you want
to compare; the controller/driver code is unchanged.
