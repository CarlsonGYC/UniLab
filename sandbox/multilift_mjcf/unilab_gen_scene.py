#!/usr/bin/env python
"""Generate the UniLab multilift scene.xml (ONE formation; the backend batches num_envs).

Unlike the sandbox (xfrc + grid), the UniLab task applies drone authority via ACTUATORS:
each drone has a COM <site> and 6 <motor> actuators with gear = the unit wrench directions
[Fx,Fy,Fz,Mx,My,Mz] in the body frame. The control stack's compute_wrench() output maps
straight to ctrl (validated equivalent to xfrc). Cables are the stable ball-jointed rigid
chains. The model default qpos IS the taut formation (reset reads get_default_qpos).

    uv run python sandbox/multilift_mjcf/unilab_gen_scene.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "control_test"))
from mj_drone import (  # noqa: E402
    ARM_LENGTH,
    BODY_BOTTOM_OFFSET,
    DRONE_INERTIA,
    DRONE_MASS,
    LINK_CYL_HEIGHT,
    LINK_MASS,
    LINK_RADIUS,
    LINK_TOTAL_LENGTH,
    PAYLOAD_HEIGHT_RATIO,
    quat_z_to,
)

OUT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "src",
    "unilab",
    "assets",
    "robots",
    "multilift",
    "scene.xml",
)
SIM_DT = 1.0 / 500.0


def _v(x):
    return " ".join(f"{float(c):.6g}" for c in x)


def build(
    num_ropes=4,
    rope_length=1.0,
    payload_mass=2.0,
    payload_radius=0.15,
    elevation_deg=40.0,
    base_height=1.5,
    joint_damping=0.01,
    joint_armature=1e-6,
):
    num_links = max(1, int(round(rope_length / LINK_TOTAL_LENGTH)))
    half_h = payload_radius * PAYLOAD_HEIGHT_RATIO / 2.0
    izz = 0.5 * payload_mass * payload_radius**2
    ixx = (
        (1.0 / 12.0)
        * payload_mass
        * (3.0 * payload_radius**2 + (payload_radius * PAYLOAD_HEIGHT_RATIO) ** 2)
    )
    beta = math.radians(elevation_deg)
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    payload_center = np.array([0.0, 0.0, base_height])
    a = ARM_LENGTH / math.sqrt(2.0)

    cap = (
        f'<geom type="capsule" fromto="0 0 0 0 0 {LINK_CYL_HEIGHT:.6g}" size="{LINK_RADIUS:.6g}" '
        f'mass="{LINK_MASS:.6g}" contype="0" conaffinity="0" rgba="0.85 0.85 0.85 1"/>'
    )
    jnt = f'<joint type="ball" damping="{joint_damping:.6g}" armature="{joint_armature:.6g}"/>'

    def drone_body(name, q_drone):
        arms = "".join(
            f'<geom type="capsule" fromto="0 0 0 {sx * a:.4g} {sy * a:.4g} 0" size="0.006" mass="0" '
            f'contype="0" conaffinity="0" rgba="0.15 0.15 0.18 1"/>'
            f'<geom type="cylinder" pos="{sx * a:.4g} {sy * a:.4g} 0.012" size="0.03 0.004" mass="0" '
            f'contype="0" conaffinity="0" rgba="0.2 0.45 0.8 0.6"/>'
            for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        )
        return (
            f'<body name="{name}" pos="0 0 {LINK_TOTAL_LENGTH + BODY_BOTTOM_OFFSET:.6g}" quat="{_v(q_drone)}">'
            f"{jnt}"
            f'<inertial pos="0 0 0" mass="{DRONE_MASS:.6g}" diaginertia="{_v(DRONE_INERTIA)}"/>'
            f'<geom type="box" size="0.05 0.05 0.02" mass="0" rgba="0.2 0.45 0.8 1"/>{arms}'
            f'<site name="{name}_site" pos="0 0 0" size="0.01" rgba="1 1 0 0.3"/>'
            f"</body>"
        )

    ropes = []
    drone_names = []
    for i in range(num_ropes):
        ang = 2.0 * math.pi * i / num_ropes
        rho = np.array([payload_radius * math.cos(ang), payload_radius * math.sin(ang), half_h])
        d = np.array([math.cos(ang) * cos_b, math.sin(ang) * cos_b, sin_b])
        q_link = quat_z_to(d)
        q_drone = np.array([q_link[0], -q_link[1], -q_link[2], -q_link[3]])
        dname = f"drone{i}"
        drone_names.append(dname)
        inner = drone_body(dname, q_drone)
        for j in range(num_links - 1, 0, -1):
            inner = f'<body name="rope{i}_link{j}" pos="0 0 {LINK_TOTAL_LENGTH:.6g}">{jnt}{cap}{inner}</body>'
        ropes.append(
            f'<body name="rope{i}_link0" pos="{_v(rho)}" quat="{_v(q_link)}">{jnt}{cap}{inner}</body>'
        )

    payload = (
        f'<body name="payload" pos="{_v(payload_center)}">'
        f"<freejoint/>"
        f'<geom type="cylinder" size="{payload_radius:.6g} {half_h:.6g}" rgba="0.22 0.43 0.55 1"/>'
        f'<inertial pos="0 0 0" mass="{payload_mass:.6g}" diaginertia="{ixx:.6g} {ixx:.6g} {izz:.6g}"/>'
        f"{''.join(ropes)}</body>"
    )

    # actuators: per drone, 6 motors (gear = unit wrench dirs in body/site frame), in drone order
    units = [
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0),
        (0, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 1),
    ]
    acts = "".join(
        f'<motor name="{dn}_w{k}" site="{dn}_site" gear="{_v(g)}" ctrlrange="-200 200"/>'
        for dn in drone_names
        for k, g in enumerate(units)
    )

    xml = f"""<mujoco model="multilift">
  <option timestep="{SIM_DT:.6g}" integrator="implicitfast" cone="elliptic"/>
  <compiler angle="radian" autolimits="true"/>
  <visual><global offwidth="1280" offheight="720"/><map force="0.003"/></visual>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="40 40 0.1" rgba="0.3 0.3 0.35 1"/>
    {payload}
  </worldbody>
  <actuator>{acts}</actuator>
</mujoco>
"""
    return xml, drone_names


if __name__ == "__main__":
    xml, names = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(os.path.abspath(OUT), "w") as f:
        f.write(xml)
    print(f"wrote {os.path.abspath(OUT)}")
    print("drone bodies:", names, "| actuators: 6 per drone")
    # sanity compile + check default qpos = taut formation
    import mujoco

    m = mujoco.MjModel.from_xml_string(xml)
    print(f"compiled: nbody={m.nbody} nq={m.nq} nv={m.nv} nu={m.nu} (nu expect {6 * len(names)})")
