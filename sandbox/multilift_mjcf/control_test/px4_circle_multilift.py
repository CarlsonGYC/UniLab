#!/usr/bin/env python
"""Cable-lift multi-drone circle test on MuJoCo (mujoco-uni) — direct_rl port.

One ``num_ropes``-drone formation holds a slung payload and flies it around a
trapezoidal-profile circle, driven by the SAME controllers / params / rates /
decimation as direct_rl's Isaac driver:

* outer — PX4PositionAttitudeController, charpi PX4 params, PVA (drone setpoint =
  formation equilibrium + circle offset, so the whole formation translates rigidly);
* inner — CTBRControlStack, isaac_dreamer rate params (+ charpi overrides);
* hover thrust auto-calibrated per drone INCLUDING the payload share
  (support_mass = drone 1.09 kg + payload_mass / num_ropes), exactly as direct_rl.

The drone authority is the controller's body-frame net wrench applied with
``xfrc_applied`` (Isaac ``set_external_force_and_torque`` equivalent). The cables
are the explicit ball-jointed rigid-body chains (the stable model verified in
``passive_test.py``); the coupling is resolved by MuJoCo. Logs PAYLOAD tracking +
mean drone-formation error.

    uv run python control_test/px4_circle_multilift.py            # headless, logs + summary
    uv run python control_test/px4_circle_multilift.py --gui      # live viewer
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
from helpers import CsvLogger, apply_drone_config, load_config, summarize_tracking  # noqa: E402
from mj_drone import (  # noqa: E402
    DRONE_MASS,
    G,
    apply_body_wrench,
    body_ids,
    build_multilift_scene,
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
    "drone_err",
    "max_tilt_deg",
    "thrust_mean",
]


def _t(a) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=os.path.join(_HERE, "config", "multilift.yaml"))
    p.add_argument("--drone_config", default=os.path.join(_HERE, "config", "charpi.yaml"))
    p.add_argument("--log", default=os.path.join(_HERE, "logs", "multilift_circle.csv"))
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--gui", action="store_true")
    p.add_argument(
        "--visual",
        choices=["mesh", "simple"],
        default="mesh",
        help="drone visual: real charpi mesh (default) or a light box+arms",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    tj, sc, ct, rn = (cfg.get(k, {}) for k in ("trajectory", "scene", "control", "run"))
    radius = float(tj.get("circle_radius", 0.5))
    period = float(tj.get("circle_duration", 8.0))
    circles = int(tj.get("circle_times", 2))
    ramp = float(tj.get("ramp_up_time", 4.0))
    ramp_dn = float(tj.get("ramp_down_time", ramp))
    init_phase = float(tj.get("circle_init_phase", 0.0))
    base_height = float(tj.get("base_height", 1.5))
    nd = int(sc.get("num_ropes", 4))
    rope_length = float(sc.get("rope_length", 1.0))
    payload_mass = float(sc.get("payload_mass", 2.0))
    elevation = float(sc.get("elevation_angle", 40.0))
    decimation = int(ct.get("decimation", 5))
    hover_arg = float(ct.get("hover_thrust", 0.0))
    settle = float(rn.get("settle_time", 3.0))

    # ── scene: one formation (payload + nd explicit rope chains + active drones) ──
    xml, meta = build_multilift_scene(
        SIM_DT,
        num_ropes=nd,
        rope_length=rope_length,
        payload_mass=payload_mass,
        elevation_angle_deg=elevation,
        base_height=base_height,
        visual=args.visual,
    )
    model = mujoco.MjModel.from_xml_string(xml)
    if args.visual == "mesh":
        seat_drone_meshes(model, bottom_z=0.0)  # seat drone belly at the cable attach (no clipping)
    data = mujoco.MjData(model)
    drone_bids = body_ids(model, meta["drone_names"])
    payload_bid = body_ids(model, [meta["payload_name"]])[0]
    mujoco.mj_forward(model, data)

    # ── controllers (identical params/rates to direct_rl) ──
    dt_outer = decimation * SIM_DT
    drone_cfg, px4_cfg = DroneControlCfg(), PX4ControlCfg()
    apply_drone_config(px4_cfg, drone_cfg, load_config(args.drone_config))
    support_mass = DRONE_MASS + payload_mass / nd
    hover_thrust = hover_arg if hover_arg > 0 else hover_thrust_for_thrust(support_mass * G)
    px4_cfg.position.hover_thrust = hover_thrust
    stack = CTBRControlStack(nd, dt=SIM_DT, cfg=drone_cfg, device="cpu")
    ctrl = PX4PositionAttitudeController(nd, dt=dt_outer, cfg=px4_cfg, device="cpu")
    stack.reset()
    ctrl.reset()

    eq_world = _t(meta["equilibrium"])  # [nd,3] drone hover positions
    payload_start = _t(meta["payload_pos"])  # [3]
    traj = CircleTrajectory(radius, period, circles, ramp, ramp_dn, init_phase, device="cpu")
    cruise = traj.omega_max * radius
    max_steps = args.steps or int((traj.total_time + settle) / dt_outer)
    print(
        f"[run] 1 formation x {nd} drones | r={radius}m period={period}s cruise={cruise:.2f}m/s | "
        f"payload={payload_mass}kg hover_thrust={hover_thrust:.3f} | {max_steps} steps | "
        f"{'GUI' if args.gui else 'headless'}"
    )

    logger = None if args.gui else CsvLogger(args.log, _COLS)
    state = {"t": 0.0, "outer": 0, "diverged": False}

    def control_step() -> bool:
        pos, quat, linw, _ = read_drone_state(model, data, drone_bids)
        p_off, v_des, a_des = traj.offset_pva(state["t"])
        p_sp = eq_world + p_off
        state["drone_targets"] = p_sp.numpy()
        state["payload_target"] = (payload_start + p_off).numpy()
        thrust = ctrl.update_position(
            _t(pos), _t(linw), _t(quat), p_sp, v_des.expand(nd, 3), a_des.expand(nd, 3)
        )
        stack.set_collective_thrust(thrust)
        for _ in range(decimation):
            _, quat, linw, omb = read_drone_state(model, data, drone_bids)
            stack.set_body_rate(ctrl.update_attitude(_t(quat)))
            F, M = stack.compute_wrench(_t(omb), _t(linw), _t(quat))
            apply_body_wrench(
                model, data, drone_bids, quat, F.squeeze(1).numpy(), M.squeeze(1).numpy()
            )
            mujoco.mj_step(model, data)
            state["t"] += SIM_DT
        pl_pos, _, pl_vel, _ = read_drone_state(model, data, [payload_bid])
        if not (np.isfinite(data.xpos).all() and np.isfinite(data.qvel).all()):
            state["diverged"] = True
            return False

        if logger is not None:
            pos, quat, _, _ = read_drone_state(model, data, drone_bids)
            p_sp_np = p_sp.numpy()
            drone_err = float(np.linalg.norm(pos - p_sp_np, axis=-1).mean())
            tilts = [
                math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)))))
                for q in quat
            ]
            payload_sp = (payload_start + p_off).numpy()
            err_pos = float(np.linalg.norm(pl_pos[0] - payload_sp))
            err_vel = float(np.linalg.norm(pl_vel[0] - v_des.numpy()))
            pl = pl_pos[0]
            vs = v_des.numpy()
            logger.write(
                t=f"{state['t']:.4f}",
                env=0,
                x_sp=f"{payload_sp[0]:.4f}",
                y_sp=f"{payload_sp[1]:.4f}",
                z_sp=f"{payload_sp[2]:.4f}",
                x=f"{pl[0]:.4f}",
                y=f"{pl[1]:.4f}",
                z=f"{pl[2]:.4f}",
                vx_sp=f"{vs[0]:.4f}",
                vy_sp=f"{vs[1]:.4f}",
                vz_sp=f"{vs[2]:.4f}",
                vx=f"{pl_vel[0, 0]:.4f}",
                vy=f"{pl_vel[0, 1]:.4f}",
                vz=f"{pl_vel[0, 2]:.4f}",
                err_pos=f"{err_pos:.4f}",
                err_vel=f"{err_vel:.4f}",
                drone_err=f"{drone_err:.4f}",
                max_tilt_deg=f"{max(tilts):.3f}",
                thrust_mean=f"{float(thrust.mean()):.4f}",
            )
        state["outer"] += 1
        return True

    if args.gui:
        circle_center = payload_start.numpy()
        print(
            "\n[gui] green ring = commanded payload circle, red = payload setpoint, "
            "blue = drone setpoints.\n      double-click + Ctrl-drag to perturb. Close the window to exit.\n"
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
                    [(state["payload_target"], 0.06, (0.9, 0.2, 0.1, 0.9))]
                    + [(dt, 0.035, (0.2, 0.45, 0.9, 0.9)) for dt in state["drone_targets"]],
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
