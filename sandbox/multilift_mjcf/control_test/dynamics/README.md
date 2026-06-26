# `direct_rl.dynamics` — drone dynamics + flight-control stack

Framework-agnostic, batched (multi-env) `torch` modules that turn an RL command
into a body wrench you apply to the drone `RigidObject`. Everything is pure
`torch` over a leading `[N, …]` env axis (no per-env Python loops), so it
vectorizes to thousands of parallel envs on GPU and drops straight into a
`DirectRLEnv._apply_action` or a standalone script.

There are two layers:

1. **Inner CTBR stack** (`control_stack.py`) — the collective-thrust / body-rate
   loop, ported verbatim from `isaac_dreamer`'s `BodyRateControlAction`. This is
   already **PX4-equivalent** at the rate level (PX4 rate PID + Butterworth +
   PX4 mixer + thrust curve + motor model + aerodynamics).
2. **Outer position + attitude controllers** — turn a higher-level command into
   the `(collective-thrust, body-rate)` setpoint the inner stack consumes. Two
   options:
   - **`px4_control.py`** — a faithful port of **PX4 v1.16** `mc_pos_control`
     (position) + `mc_att_control` (attitude). **Command = position / velocity /
     acceleration (PVA).** *This is the recommended controller.*
   - **`position_controller.py`** — a lightweight Butterworth-lag surrogate of
     the outer loop (position command only). Useful as a fast/cheap
     domain-randomization stand-in; not a faithful PX4 model.

## The cascade (PVA → wrench)

```
RL policy
  │  per-drone setpoint:  position p_sp, velocity v_sp, acceleration a_sp, (yaw)
  ▼
PX4PositionControl                         mc_pos_control/PositionControl.cpp
  P-position → PID-velocity → accelerationControl
  → normalized thrust vector  (+ anti-windup, tilt limit, thrust prioritization)
  ▼
thrustToAttitude                           mc_pos_control/ControlMath.cpp
  thrust vector + yaw → desired attitude quaternion q_des, ‖thrust‖
  ▼
PX4AttitudeControl                         mc_att_control/AttitudeControl.cpp
  Brescianini (2013) reduced+full quaternion law → body-rate setpoint
  ▼
CTBRControlStack            (= PX4 rate loop, already ported from isaac_dreamer)
  BodyRateController (PID+Butterworth) → ControlAllocator (mixer + thrust curve)
  → Motor (1st-order ZOH) → AirDragEffect → Allocation
  ▼
body wrench (F, M)  →  RigidObject.set_external_force_and_torque  →  PhysX
```

The drone stays **force-driven** end-to-end, so cable tension from the payload
still acts on it (unlike `write_root_velocity_to_sim`, which would overwrite the
coupling).

## PX4 fidelity & what was ported

`px4_control.py` reproduces the PX4 C++ **line-for-line**, including:

| PX4 source | Ported as |
|---|---|
| `PositionControl::_positionControl` | `PX4PositionControl._position_control` (P-pos, `constrainXY`, vel limits) |
| `PositionControl::_velocityControl` | `PX4PositionControl._velocity_control` (PID, vertical + horizontal anti-windup / ARW) |
| `PositionControl::_accelerationControl` | `PX4PositionControl._acceleration_control` (tilt-limited thrust, hover-thrust map) |
| `ControlMath::thrustToAttitude/bodyzToAttitude` | `_bodyz_to_attitude` |
| `ControlMath::constrainXY / limitTilt / addIfNotNan` | `_constrain_xy / _limit_tilt / _add_if_not_nan` |
| `AttitudeControl::update` | `PX4AttitudeControl.update` (reduced/full attitude, yaw-weight blend, rate FF) |
| `matrix::Quaternion` `canonical / (src,dst) / dcm_z` | `quat_canonical / quat_from_two_vectors / quat_dcm_z` |

Default gains/limits are the **real PX4 parameter defaults** (`MPC_*`, `MC_*`):
`MPC_XY_P=0.95`, `MPC_Z_P=1.0`, `MPC_{XY,Z}_VEL_{P,I,D}_ACC`, `MPC_THR_*`,
`MPC_TILTMAX_AIR=45°`, `MC_{ROLL,PITCH,YAW}_P=6.5/6.5/2.8`, `MC_YAW_WEIGHT=0.4`,
`MC_*RATE_MAX`. All live in `PX4PositionControlCfg` / `PX4AttitudeControlCfg`.

### Frames (ENU/FLU ↔ NED/FRD)

PX4 works in **NED world / FRD body**; IsaacLab works in **ENU world / FLU
body**. The port runs the PX4 math **internally in NED/FRD** (so it stays
byte-faithful) and converts only at the boundary:

- state/setpoint vectors `ENU→NED` via `enu_ned` (`(x,y,z)→(y,x,-z)`),
- the output body-rate `FRD→FLU` via `flu_frd` (`(x,y,z)→(x,-y,-z)`),
- the attitude quaternion via the constant sandwich `q_ned = q_enu2ned ⊗ q_enu ⊗ q_flu2frd`.

All three are involutions / fixed rotations (cheap, no trig per step). The
collective thrust is a magnitude, hence frame-independent.

### Hover-thrust calibration (the one thing you must set)

PX4's `hover_thrust` is the **normalized** collective thrust that holds hover. It
is handed straight to the CTBR mixer (thrust-column gain ≈ 1), so it equals the
mixer command that produces `m_eff·g` newtons. For this drone (≈ 54 N max thrust):

- bare 1.09 kg drone → `hover_thrust ≈ 0.25`
- 1.09 kg + 0.5 kg payload share → `hover_thrust ≈ 0.34`

Set `PX4PositionControlCfg.hover_thrust`, or call
`hover_thrust_for_thrust(m_eff*g)` to compute it from the effective hover weight.

## Usage

```python
from direct_rl.dynamics import (
    CTBRControlStack, DroneControlCfg,
    PX4PositionAttitudeController, PX4ControlCfg, hover_thrust_for_thrust,
)

stack = CTBRControlStack(num_envs, dt=sim_dt, cfg=DroneControlCfg(), device=dev)
cfg = PX4ControlCfg()
cfg.position.hover_thrust = hover_thrust_for_thrust(m_eff * 9.80665)
ctrl = PX4PositionAttitudeController(num_envs, dt=control_dt, cfg=cfg, device=dev)

# ── per CONTROL step (e.g. 100 Hz) — outer position loop ──
thrust = ctrl.update_position(pos_w, vel_w, quat_w,        # current state (ENU/FLU)
                              pos_sp_w, vel_sp_w, acc_sp_w) # RL PVA command (ENU)
stack.set_collective_thrust(thrust)

# ── per SIM step (e.g. 500 Hz) — mid attitude loop + inner rate loop ──
stack.set_body_rate(ctrl.update_attitude(quat_w))          # cascade: att faster than pos
F, M = stack.compute_wrench(omega_b, lin_vel_w, quat_w)    # body-frame wrench
drone.set_external_force_and_torque(F, M)                  # is_global=False
```

`compute(...)` runs both loops in one call if you don't need the cascaded-rate
split. `reset(env_ids)` clears integrators / filters / the finite-difference
acceleration estimate per env at episode boundaries.

> **Note (returned thrust aliases an internal buffer).** `update_position`
> returns `_collective_thrust` directly for efficiency; `set_collective_thrust`
> copies it (via `clamp`). Don't hold the returned tensor across a `reset()` —
> `clone()` it first if you must.

## Tuning knobs (and good domain-randomization targets)

- **Position/velocity gains** — `PX4PositionControlCfg.gain_pos_p`,
  `gain_vel_{p,i,d}`. Lower them for a softer, more cable-friendly outer loop.
- **Hover thrust** — `hover_thrust`; randomize ±10 % to cover payload/mass error.
- **Tilt / velocity / thrust limits** — `tilt_max_deg`, `lim_vel_*`, `lim_thr_*`.
- **Attitude gains** — `PX4AttitudeControlCfg.{roll,pitch,yaw}_p`, `yaw_weight`,
  `*rate_max_deg`.
- **Inner-loop / actuator** (`DroneControlCfg`) — `thrust_const`, `taus_*`,
  `action_delay_steps`, `kp/ki/kd`, etc. — randomize for sim-to-real.

## Validation

All controller tests live in **`control_test/`** (YAML-driven trajectory, GPU,
CSV setpoint-vs-actual logging) — see `control_test/README.md`, which also
explains the circular-tracking lag. In short:

```bash
python control_test/px4_circle_single.py    --headless   # single free drone
python control_test/px4_circle_multilift.py --headless   # N-drone payload lift
python control_test/test_px4_control.py                  # CPU unit test (no sim)
python control_test/analyze_log.py --log control_test/logs/single_circle.csv --cruise 1.05
```

Rate loop = isaac_dreamer params, outer loop = PX4 defaults; reference trajectory
is the trapezoidal-profile circle ported from the real-flight `traj_test` node
(`control_test/circle_trajectory.py`). Both circle tests default to
`--device cuda:0`. The **CPU unit test** (no Isaac Sim — the controller module
imports on bare Python) covers frame round-trips, the quaternion law, hover
thrust, climb, lateral tilt, acceleration feed-forward, and a 1000-env batch.
