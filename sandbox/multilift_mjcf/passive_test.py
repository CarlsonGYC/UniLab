#!/usr/bin/env python
"""Passive (force-free) stability test for the multi-rigid-body cable-lift scene.

Ports direct_rl's ``RigidBodyRopes`` (assets/charpi_lift.py) topology to a native
MuJoCo (mujoco-uni 3.8.0) model, so we can confirm the explicit multi-rigid-body
rope + drone + payload coupling is numerically stable in UniLab's simulator BEFORE
building the full RL env.

Topology (a single environment, matching direct_rl):
    payload (free body, the draggable root)
      └─ Rope i:  Link0 ─ Link1 ─ ... ─ Link{N-1} ─ Drone i      (i = 0..num_ropes-1)
    Each link/drone connects to its parent by a 3-DOF BALL joint (inextensible by
    construction — links can swing but not stretch), the MuJoCo analogue of the
    near-inextensible D6 spherical joint chain in charpi_lift.py. The whole assembly
    is ONE kinematic tree rooted at the payload (no closed loops) → cleanest baseline.

Two modes:
    --hold (default): each drone is welded to a draggable mocap body held at its
        formation position, so the cables go taut under the payload weight and you
        can drag the (red) mocap anchors AND the payload to probe the coupled system.
    --drop: no anchors — the whole assembly is free and falls under gravity (pure
        kinematic-tree baseline, zero equality constraints).

NO actuation / NO control force anywhere — this only tests passive solver stability.

Run (from the UniLab repo, which owns the mujoco-uni env):
    uv run python sandbox/multilift_mjcf/passive_test.py                 # interactive viewer (hold)
    uv run python sandbox/multilift_mjcf/passive_test.py --drop          # interactive, free-fall
    uv run python sandbox/multilift_mjcf/passive_test.py --headless      # no GUI, prints stability stats
    uv run python sandbox/multilift_mjcf/passive_test.py --num_ropes 4 --rope_length 1.0 --payload_mass 2.0

Viewer controls (mujoco.viewer):
    double-click   select a body
    Ctrl + drag    apply force (left) / torque (right) to the selected free body
    drag a red mocap sphere to move that drone (hold mode)
    Space pause · '[' / ']' step · 'R' reset
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

# ── geometry constants (verbatim from direct_rl/assets/charpi_lift.py) ──
LINK_TOTAL_LENGTH = 0.20  # joint-to-joint spacing along the rope
LINK_CYL_HEIGHT = 0.19  # capsule cylinder part (+2*radius caps ≈ 0.20 total)
LINK_RADIUS = 0.005
LINK_MASS = 0.008
BODY_BOTTOM_OFFSET = 0.012  # drone attaches 12 mm above the last link end
# drone (charpi.usd "visual" mode: body + props folded in) — scene_builder.py
DRONE_MASS = 1.09
DRONE_INERTIA = (0.004421, 0.006721, 0.010024)
PAYLOAD_HEIGHT_RATIO = 1.0 / 8.0  # payload height = radius / 8


def quat_z_to(d: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) rotating local +z onto world direction ``d``."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])  # 180° about x
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    ang = math.acos(max(-1.0, min(1.0, c)))
    s = math.sin(ang / 2.0)
    return np.array([math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _v(x) -> str:
    return " ".join(f"{float(c):.6g}" for c in x)


def build_mjcf(
    num_ropes: int = 4,
    rope_length: float = 1.0,
    payload_mass: float = 2.0,
    payload_radius: float = 0.15,
    elevation_angle_deg: float = 40.0,
    base_height: float = 1.5,
    hold: bool = True,
    timestep: float = 0.002,
    joint_damping: float = 0.01,
    joint_armature: float = 1e-6,
) -> str:
    """Procedurally build the cable-lift MJCF (one environment), mirroring charpi_lift."""
    num_links = max(1, int(round(rope_length / LINK_TOTAL_LENGTH)))
    payload_height = payload_radius * PAYLOAD_HEIGHT_RATIO
    half_h = payload_height / 2.0
    # solid-cylinder inertia (charpi_lift.py:114-116)
    izz = 0.5 * payload_mass * payload_radius**2
    ixx = (1.0 / 12.0) * payload_mass * (3.0 * payload_radius**2 + payload_height**2)
    beta = math.radians(elevation_angle_deg)
    cos_b, sin_b = math.cos(beta), math.sin(beta)

    ropes_xml: list[str] = []
    welds_xml: list[str] = []
    mocaps_xml: list[str] = []

    for i in range(num_ropes):
        ang = 2.0 * math.pi * i / num_ropes
        rho = np.array(
            [payload_radius * math.cos(ang), payload_radius * math.sin(ang), half_h]
        )  # attach point (payload frame)
        d = np.array([math.cos(ang) * cos_b, math.sin(ang) * cos_b, sin_b])  # rope direction
        q_link = quat_z_to(d)  # capsule axis ‖ d
        cap = (
            f'<geom type="capsule" fromto="0 0 0 0 0 {LINK_CYL_HEIGHT:.6g}" '
            f'size="{LINK_RADIUS:.6g}" mass="{LINK_MASS:.6g}" '
            f'contype="0" conaffinity="0" rgba="0.85 0.85 0.85 1"/>'
        )
        jnt = f'<joint type="ball" damping="{joint_damping:.6g}" armature="{joint_armature:.6g}"/>'

        # build the nested link chain from the drone (leaf) outward, then wrap inward
        # drone leaf: level it (counter-rotate the inherited rope tilt) so the box sits flat
        q_drone = quat_conj(q_link)
        drone = (
            f'<body name="rope{i}_drone" pos="0 0 {LINK_TOTAL_LENGTH + BODY_BOTTOM_OFFSET:.6g}" '
            f'quat="{_v(q_drone)}">'
            f"{jnt}"
            f'<geom type="box" size="0.06 0.06 0.02" mass="{DRONE_MASS:.6g}" '
            f'rgba="0.2 0.45 0.8 1"/>'
            f'<inertial pos="0 0 0" mass="{DRONE_MASS:.6g}" diaginertia="{_v(DRONE_INERTIA)}"/>'
            f"</body>"
        )
        inner = drone
        for j in range(num_links - 1, 0, -1):
            inner = (
                f'<body name="rope{i}_link{j}" pos="0 0 {LINK_TOTAL_LENGTH:.6g}">'
                f"{jnt}{cap}{inner}</body>"
            )
        link0 = (
            f'<body name="rope{i}_link0" pos="{_v(rho)}" quat="{_v(q_link)}">'
            f"{jnt}{cap}{inner}</body>"
        )
        ropes_xml.append(link0)

        if hold:
            # drone world position (payload starts level at (0,0,base_height))
            drone_w = (
                np.array([0.0, 0.0, base_height])
                + rho
                + d * (num_links * LINK_TOTAL_LENGTH + BODY_BOTTOM_OFFSET)
            )
            mocaps_xml.append(
                f'<body name="mocap{i}" mocap="true" pos="{_v(drone_w)}">'
                f'<geom type="sphere" size="0.03" contype="0" conaffinity="0" rgba="0.9 0.1 0.1 0.5"/>'
                f"</body>"
            )
            welds_xml.append(f'<weld body1="rope{i}_drone" body2="mocap{i}"/>')

    equality = f"<equality>{''.join(welds_xml)}</equality>" if welds_xml else ""

    return f"""<mujoco model="multilift_passive">
  <option timestep="{timestep:.6g}" integrator="implicitfast" cone="elliptic"/>
  <compiler angle="radian" autolimits="true"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <map force="0.005"/>
  </visual>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.3 0.3 0.35 1"/>
    <body name="payload" pos="0 0 {base_height:.6g}">
      <freejoint/>
      <geom type="cylinder" size="{payload_radius:.6g} {half_h:.6g}" rgba="0.22 0.43 0.55 1"/>
      <inertial pos="0 0 0" mass="{payload_mass:.6g}" diaginertia="{ixx:.6g} {ixx:.6g} {izz:.6g}"/>
      {"".join(ropes_xml)}
    </body>
    {"".join(mocaps_xml)}
  </worldbody>
  {equality}
</mujoco>
"""


def report_stability(model, data, steps: int) -> None:
    import mujoco

    maxv = 0.0
    nan_step = -1
    warn0 = data.warning.number.copy()
    for k in range(steps):
        mujoco.mj_step(model, data)
        v = float(np.abs(data.qvel).max()) if model.nv else 0.0
        maxv = max(maxv, v)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            nan_step = k
            break
    warns = data.warning.number - warn0
    print(f"  steps run        : {steps if nan_step < 0 else nan_step}")
    print(f"  finite           : {nan_step < 0}")
    print(f"  max|qvel| (whole): {maxv:.4g}")
    print(f"  final|qvel| max  : {float(np.abs(data.qvel).max()):.4g}")
    print(f"  contacts (final) : {data.ncon}")
    nz = [(i, int(n)) for i, n in enumerate(warns) if n]
    print(
        f"  solver warnings  : {nz if nz else 'none'}  "
        f"(indices: 0=INERTIA 4=CONTACTFULL 5=CNSTRFULL 8=BADQACC ...)"
    )
    if nan_step >= 0:
        print(
            "  >>> DIVERGED (non-finite). Try larger --joint_armature, smaller --timestep, "
            "or fewer/heavier links."
        )
    elif maxv < 50.0:
        print("  >>> STABLE.")
    else:
        print("  >>> high velocities — inspect interactively before trusting.")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--num_ropes", type=int, default=4)
    p.add_argument("--rope_length", type=float, default=1.0)
    p.add_argument("--payload_mass", type=float, default=2.0)
    p.add_argument("--payload_radius", type=float, default=0.15)
    p.add_argument("--elevation", type=float, default=40.0, help="rope elevation angle (deg)")
    p.add_argument("--base_height", type=float, default=1.5)
    p.add_argument("--timestep", type=float, default=0.002)
    p.add_argument("--joint_damping", type=float, default=0.01)
    p.add_argument("--joint_armature", type=float, default=1e-6)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--hold", dest="hold", action="store_true", help="drones held in formation (default)"
    )
    mode.add_argument("--drop", dest="hold", action="store_false", help="free-fall, no anchors")
    p.set_defaults(hold=True)
    p.add_argument("--headless", action="store_true", help="no GUI; step and print stability stats")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument(
        "--save_xml", type=str, default=str(Path(__file__).with_name("multilift_scene.xml"))
    )
    args = p.parse_args()

    import mujoco

    xml = build_mjcf(
        num_ropes=args.num_ropes,
        rope_length=args.rope_length,
        payload_mass=args.payload_mass,
        payload_radius=args.payload_radius,
        elevation_angle_deg=args.elevation,
        base_height=args.base_height,
        hold=args.hold,
        timestep=args.timestep,
        joint_damping=args.joint_damping,
        joint_armature=args.joint_armature,
    )
    if args.save_xml:
        Path(args.save_xml).write_text(xml)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    num_links = max(1, int(round(args.rope_length / LINK_TOTAL_LENGTH)))
    print(
        f"scene: {args.num_ropes} ropes × {num_links} links + {args.num_ropes} drones + payload "
        f"| mode={'HOLD' if args.hold else 'DROP'}"
    )
    print(
        f"model: nbody={model.nbody} nq={model.nq} nv={model.nv} njnt={model.njnt} "
        f"neq={model.neq}  (xml saved: {args.save_xml})"
    )

    if args.headless:
        report_stability(model, data, args.steps)
        return

    import mujoco.viewer

    print(
        "\nlaunching interactive viewer — double-click a body, Ctrl+drag to push;\n"
        "drag the red mocap spheres to move the drones (hold mode). Close the window to exit.\n"
    )
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
