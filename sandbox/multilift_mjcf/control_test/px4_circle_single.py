#!/usr/bin/env python
"""Single-drone circle test on MuJoCo (mujoco-uni) — direct_rl/control_test port.

``num_envs`` free drones (no payload) on a grid fly the same trapezoidal-profile
circle in parallel, driven by the SAME controllers as direct_rl:

* outer — PX4PositionAttitudeController, charpi PX4 params (config/charpi.yaml), PVA;
* inner — CTBRControlStack, isaac_dreamer rate params (+ charpi rate overrides).

Identical call order / rates / decimation / delays to direct_rl's Isaac driver;
only the simulator is MuJoCo (body-frame wrench applied via xfrc_applied). Logs a
setpoint-vs-achieved CSV and prints the tracking summary (RMS + effective lag).

    uv run python control_test/px4_circle_single.py                 # headless, logs + summary
    uv run python control_test/px4_circle_single.py --gui           # live viewer (watch + drag)
    uv run python control_test/px4_circle_single.py --num_envs 1
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
from circle_trajectory import CircleTrajectory  # noqa: E402
from dynamics import (  # noqa: E402
    CTBRControlStack,
    DroneControlCfg,
    PX4ControlCfg,
    PX4PositionAttitudeController,
    hover_thrust_for_thrust,
)
from helpers import (  # noqa: E402
    CsvLogger,
    apply_drone_config,
    grid_origins,
    load_config,
    summarize_tracking,
)
from mj_drone import (  # noqa: E402
    DRONE_MASS,
    G,
    apply_body_wrench,
    body_ids,
    build_single_scene,
    read_drone_state,
    seat_drone_meshes,
)
from viz import draw_overlay  # noqa: E402

SIM_DT = 1.0 / 500.0
_COLS = [
    "t",
    "env",
    "x_sp",
    "y_sp",
    "z_sp",
    "x",
    "y",
    "z",
    "vx_sp",
    "vy_sp",
    "vz_sp",
    "vx",
    "vy",
    "vz",
    "err_pos",
    "err_vel",
    "tilt_deg",
    "thrust",
]


def _t(a) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=os.path.join(_HERE, "config", "single.yaml"))
    p.add_argument("--drone_config", default=os.path.join(_HERE, "config", "charpi.yaml"))
    p.add_argument("--log", default=os.path.join(_HERE, "logs", "single_circle.csv"))
    p.add_argument("--num_envs", type=int, default=0, help="override config num_envs (0 = YAML)")
    p.add_argument(
        "--steps", type=int, default=0, help="override control steps (0 = full traj + settle)"
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="launch the interactive viewer instead of headless logging",
    )
    p.add_argument(
        "--visual",
        choices=["mesh", "simple"],
        default="mesh",
        help="drone visual: real charpi mesh (default) or a light box+arms",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    tj, me, ct, rn = (cfg.get(k, {}) for k in ("trajectory", "multi_env", "control", "run"))
    radius = float(tj.get("circle_radius", 1.0))
    period = float(tj.get("circle_duration", 6.0))
    circles = int(tj.get("circle_times", 2))
    ramp = float(tj.get("ramp_up_time", 4.0))
    ramp_dn = float(tj.get("ramp_down_time", ramp))
    init_phase = float(tj.get("circle_init_phase", 0.0))
    base_height = float(tj.get("base_height", 1.5))
    num_envs = args.num_envs or int(me.get("num_envs", 1))
    if args.gui:
        num_envs = 1  # one drone for a clean interactive view
    spacing = float(me.get("env_spacing", 8.0))
    decimation = int(ct.get("decimation", 5))
    hover_arg = float(ct.get("hover_thrust", 0.0))
    settle = float(rn.get("settle_time", 3.0))

    # ── scene: N drones spawned at their circle t=0 point on a grid ──
    origins = grid_origins(num_envs, spacing)
    start_local = np.array(
        [radius * math.cos(init_phase), radius * math.sin(init_phase), base_height]
    )
    spawn = [np.array(o) + start_local for o in origins]
    xml, names = build_single_scene(spawn, SIM_DT, visual=args.visual)
    model = mujoco.MjModel.from_xml_string(xml)
    if args.visual == "mesh":
        seat_drone_meshes(model, center=True)  # free drone: center the mesh on its position
    data = mujoco.MjData(model)
    bids = body_ids(model, names)
    mujoco.mj_forward(model, data)

    # ── controllers (identical params/rates to direct_rl) ──
    dt_outer = decimation * SIM_DT
    drone_cfg, px4_cfg = DroneControlCfg(), PX4ControlCfg()
    apply_drone_config(px4_cfg, drone_cfg, load_config(args.drone_config))
    if hover_arg > 0:
        px4_cfg.position.hover_thrust = hover_arg
    elif px4_cfg.position.hover_thrust <= 0:
        px4_cfg.position.hover_thrust = hover_thrust_for_thrust(DRONE_MASS * G)
    hover_thrust = px4_cfg.position.hover_thrust
    stack = CTBRControlStack(num_envs, dt=SIM_DT, cfg=drone_cfg, device="cpu")
    ctrl = PX4PositionAttitudeController(num_envs, dt=dt_outer, cfg=px4_cfg, device="cpu")
    stack.reset()
    ctrl.reset()

    traj = CircleTrajectory(radius, period, circles, ramp, ramp_dn, init_phase, device="cpu")
    env_origin = _t(np.array(origins))
    hover_world = env_origin + _t(start_local)
    cruise = traj.omega_max * radius
    max_steps = args.steps or int((traj.total_time + settle) / dt_outer)
    print(
        f"[run] {num_envs} drones | r={radius}m period={period}s cruise={cruise:.2f}m/s | "
        f"hover_thrust={hover_thrust:.3f} | {max_steps} steps | {'GUI' if args.gui else 'headless'}"
    )

    logger = None if args.gui else CsvLogger(args.log, _COLS)
    state = {"t": 0.0, "outer": 0, "diverged": False}

    def control_step() -> bool:
        """One 100 Hz control step (position loop + ``decimation`` rate substeps)."""
        pos, quat, linw, _ = read_drone_state(model, data, bids)
        p_off, v_des, a_des = traj.offset_pva(state["t"])
        p_sp = hover_world + p_off
        state["target"] = p_sp[0].numpy()
        thrust = ctrl.update_position(
            _t(pos), _t(linw), _t(quat), p_sp, v_des.expand(num_envs, 3), a_des.expand(num_envs, 3)
        )
        stack.set_collective_thrust(thrust)
        for _ in range(decimation):
            _, quat, linw, omb = read_drone_state(model, data, bids)
            stack.set_body_rate(ctrl.update_attitude(_t(quat)))
            F, M = stack.compute_wrench(_t(omb), _t(linw), _t(quat))
            apply_body_wrench(model, data, bids, quat, F.squeeze(1).numpy(), M.squeeze(1).numpy())
            mujoco.mj_step(model, data)
            state["t"] += SIM_DT
        if not (np.isfinite(data.xpos).all() and np.isfinite(data.qvel).all()):
            state["diverged"] = True
            return False

        if logger is not None:
            pos, quat, _, _ = read_drone_state(model, data, bids)
            v = read_drone_state(model, data, bids)[2]
            p_sp_np = p_sp.numpy()
            vs = v_des.numpy()
            for e in range(num_envs):
                ex = pos[e] - p_sp_np[e]
                qx, qy = quat[e, 1], quat[e, 2]
                tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (qx * qx + qy * qy)))))
                pl = pos[e] - np.array(origins[e])
                sl = p_sp_np[e] - np.array(origins[e])
                logger.write(
                    t=f"{state['t']:.4f}",
                    env=e,
                    x_sp=f"{sl[0]:.4f}",
                    y_sp=f"{sl[1]:.4f}",
                    z_sp=f"{sl[2]:.4f}",
                    x=f"{pl[0]:.4f}",
                    y=f"{pl[1]:.4f}",
                    z=f"{pl[2]:.4f}",
                    vx_sp=f"{vs[0]:.4f}",
                    vy_sp=f"{vs[1]:.4f}",
                    vz_sp=f"{vs[2]:.4f}",
                    vx=f"{v[e, 0]:.4f}",
                    vy=f"{v[e, 1]:.4f}",
                    vz=f"{v[e, 2]:.4f}",
                    err_pos=f"{np.linalg.norm(ex):.4f}",
                    err_vel=f"{np.linalg.norm(v[e] - vs):.4f}",
                    tilt_deg=f"{tilt:.3f}",
                    thrust=f"{float(thrust[e, 0]):.4f}",
                )
        state["outer"] += 1
        return True

    if args.gui:
        circle_center = np.array(origins[0]) + np.array([0.0, 0.0, base_height])
        print(
            "\n[gui] green ring = commanded circle, red sphere = current setpoint.\n"
            "      double-click + Ctrl-drag to perturb the drone. Close the window to exit.\n"
        )
        with mujoco.viewer.launch_passive(model, data) as viewer:
            wall0 = time.time()
            while viewer.is_running() and state["outer"] < max_steps:
                if not control_step():
                    break
                draw_overlay(
                    viewer.user_scn,
                    circle_center,
                    radius,
                    [(state["target"], 0.05, (0.9, 0.2, 0.1, 0.9))],
                )
                viewer.sync()
                lag = (state["outer"] * dt_outer) - (time.time() - wall0)
                if lag > 0:
                    time.sleep(lag)
        print(f"[gui] {'DIVERGED' if state['diverged'] else 'done'} at t={state['t']:.1f}s")
        return

    while state["outer"] < max_steps:
        if not control_step():
            break
    logger.close()
    if state["diverged"]:
        print(f"[FAIL] non-finite at t={state['t']:.2f}s (step {state['outer']}) — DIVERGED.")
    else:
        print(f"[done] {state['outer']} steps ({state['t']:.1f}s), {logger.n} rows -> {args.log}")
        print(summarize_tracking(args.log, cruise_speed=cruise))


if __name__ == "__main__":
    main()
