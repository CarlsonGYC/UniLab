#!/usr/bin/env python
"""MuJoCo (mujoco-uni) scene builders + state/wrench helpers for the control tests.

This is the MuJoCo replacement for direct_rl's Isaac-Sim scene layer
(``assets/scene_builder.py`` + ``charpi_lift.py``). The CONTROLLER (``dynamics/``)
and trajectory are reused verbatim; only the simulator boundary changes:

* drone authority — Isaac applies the controller's body-frame net wrench via
  ``RigidObject.set_external_force_and_torque``. The MuJoCo equivalent is
  ``data.xfrc_applied[body]`` (force+torque at the body COM, in the WORLD frame),
  so we rotate the body-frame (F, M) into world with the body's orientation.
  This is force-equivalent to the Isaac path used by direct_rl/control_test.
* state — Isaac ``root_pos_w / root_quat_w / root_lin_vel_w / root_ang_vel_b`` map
  to MuJoCo ``xpos / xquat`` + ``mj_objectVelocity`` (world linvel, body angvel).

Frame conventions match direct_rl/Isaac: ENU world (z up), quaternion (w,x,y,z),
body frame FLU (x fwd, y left, z up = thrust axis). The PX4 controller converts to
NED/FRD internally, so the inputs here are identical to the Isaac drivers.

Drone physical params are direct_rl's validated sim values (charpi.usd "visual"):
mass 1.09 kg, diagonal inertia COMBINED_INERTIA — so the simulated plant matches
what direct_rl's control_test flew.
"""

from __future__ import annotations

import math
import os

import numpy as np

# ── geometry / mass constants (from direct_rl assets/charpi_lift.py + scene_builder.py) ──
LINK_TOTAL_LENGTH = 0.20
LINK_CYL_HEIGHT = 0.19
LINK_RADIUS = 0.005
LINK_MASS = 0.008
BODY_BOTTOM_OFFSET = 0.012
DRONE_MASS = 1.09  # BODY 1.05 + 4*PROP 0.01
DRONE_INERTIA = (0.004421, 0.006721, 0.010024)  # COMBINED_INERTIA (visual mode)
ARM_LENGTH = 0.14  # DroneControlCfg.arm_length (visual only here)
PAYLOAD_HEIGHT_RATIO = 1.0 / 8.0
G = 9.80665

# ── real charpi visual mesh (charpi_vision_description) ──
# Loaded directly by MuJoCo as the drone's VISUAL geom; the dynamics stay direct_rl's
# single rigid body (mass/inertia above), since direct_rl's control_test flew the folded
# "visual" body, not the full articulation. Override the path with $CHARPI_MESH.
CHARPI_MESH = os.environ.get(
    "CHARPI_MESH",
    "/home/carlson/rl_ws/charpi_vision_description/obj_files/charpi_vision.obj",
)
MESH_SCALE = 0.01  # mesh is in cm -> m
# MuJoCo re-aligns each mesh to its principal axes at compile, so the mesh orientation in
# the body frame is NOT the file's. We therefore orient + seat the drone mesh PROGRAMMATICALLY
# after compile (seat_drone_meshes) from the compiled vertices — robust to that re-alignment.


def mesh_available() -> bool:
    return os.path.isfile(CHARPI_MESH)


def _resolve_visual(visual: str) -> str:
    """'mesh' falls back to 'simple' if the charpi mesh file is not present."""
    if visual == "mesh" and not mesh_available():
        print(f"[mj_drone] charpi mesh not found at {CHARPI_MESH} -> using simple box visual")
        return "simple"
    return visual


def _merged_charpi_mesh() -> str:
    """Return a single-material merged copy of CHARPI_MESH.

    MuJoCo's OBJ loader splits a multi-material .obj by ``usemtl`` and keeps only ONE
    chunk, so the real charpi.obj (23 materials) loses ~99% of its faces — only the base
    plate renders. Merging all faces into one group (drop g/o/usemtl/mtllib/s) loads the
    whole drone. Cached next to the sandbox; regenerated if the source is newer."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    merged = os.path.abspath(os.path.join(cache_dir, "charpi_merged.obj"))
    if os.path.isfile(merged) and os.path.getmtime(merged) >= os.path.getmtime(CHARPI_MESH):
        return merged
    keep = ("v ", "vn ", "vt ", "f ")
    with open(CHARPI_MESH) as fi, open(merged, "w") as fo:
        for line in fi:
            if line.startswith(keep):
                fo.write(line)
    return merged


def _drone_assets_xml(visual: str) -> str:
    if visual == "mesh":
        mesh = _merged_charpi_mesh()
        return (
            f'<asset><mesh name="charpi" file="{mesh}" '
            f'scale="{MESH_SCALE} {MESH_SCALE} {MESH_SCALE}"/></asset>'
        )
    return ""


def quat_z_to(d: np.ndarray) -> np.ndarray:
    """Unit quaternion (w,x,y,z) rotating local +z onto world direction ``d``."""
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c > 1.0 - 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(z, d)
    axis /= np.linalg.norm(axis)
    ang = math.acos(max(-1.0, min(1.0, c)))
    s = math.sin(ang / 2.0)
    return np.array([math.cos(ang / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def _v(x) -> str:
    return " ".join(f"{float(c):.6g}" for c in x)


def _drone_visual_geom(visual: str) -> str:
    """The drone's VISUAL geom(s) only (mass=0, no collision). Dynamics come from <inertial>."""
    if visual == "mesh":
        # the real charpi mesh (orientation + seating fixed post-compile by seat_drone_meshes)
        return (
            '<geom type="mesh" mesh="charpi" mass="0" contype="0" conaffinity="0" '
            'rgba="0.78 0.78 0.82 1"/>'
        )
    a = ARM_LENGTH / math.sqrt(2.0)
    arms = "".join(
        f'<geom type="capsule" fromto="0 0 0 {sx * a:.4g} {sy * a:.4g} 0" size="0.006" '
        f'mass="0" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>'
        f'<geom type="cylinder" pos="{sx * a:.4g} {sy * a:.4g} 0.012" size="0.03 0.004" '
        f'mass="0" rgba="0.2 0.45 0.8 0.6" contype="0" conaffinity="0"/>'
        for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    )
    return f'<geom type="box" size="0.05 0.05 0.02" mass="0" rgba="0.2 0.45 0.8 1"/>{arms}'


def _drone_body_xml(name: str, pos, quat=(1.0, 0.0, 0.0, 0.0), visual: str = "mesh") -> str:
    """A free-floating charpi drone body: direct_rl mass/inertia + selectable visual.

    body +z is the thrust axis (FLU). COM at the body origin so xpos == COM (the
    controller / wrench reference point, matching Isaac's root frame). ``visual`` is
    'mesh' (real charpi_vision mesh) or 'simple' (light box+arms)."""
    return (
        f'<body name="{name}" pos="{_v(pos)}" quat="{_v(quat)}">'
        f"<freejoint/>"
        f'<inertial pos="0 0 0" mass="{DRONE_MASS:.6g}" diaginertia="{_v(DRONE_INERTIA)}"/>'
        f"{_drone_visual_geom(visual)}"
        f"</body>"
    )


_HEADER = """<mujoco model="{model}">
  <option timestep="{dt:.6g}" integrator="implicitfast" cone="elliptic"/>
  <compiler angle="radian" autolimits="true"/>
  <visual><global offwidth="1280" offheight="720"/><map force="0.003"/></visual>
  {assets}
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="40 40 0.1" rgba="0.3 0.3 0.35 1"/>
"""
_FOOTER = "  </worldbody>\n{equality}</mujoco>\n"


def build_single_scene(spawn_positions, sim_dt: float, visual: str = "mesh"):
    """N free drones (no payload) at ``spawn_positions`` [N,3]. Returns (xml, names)."""
    visual = _resolve_visual(visual)
    names = [f"drone{e}" for e in range(len(spawn_positions))]
    bodies = "".join(_drone_body_xml(n, p, visual=visual) for n, p in zip(names, spawn_positions))
    xml = (
        _HEADER.format(model="circle_single", dt=sim_dt, assets=_drone_assets_xml(visual))
        + bodies
        + _FOOTER.format(equality="")
    )
    return xml, names


def build_multilift_scene(
    sim_dt: float,
    num_ropes: int = 4,
    rope_length: float = 1.0,
    payload_mass: float = 2.0,
    payload_radius: float = 0.15,
    elevation_angle_deg: float = 40.0,
    base_height: float = 1.5,
    origin=(0.0, 0.0, 0.0),
    joint_damping: float = 0.01,
    joint_armature: float = 1e-6,
    visual: str = "mesh",
):
    """One cable-lift formation (payload + num_ropes explicit rope chains + drones),
    drones ACTIVE (no weld). Returns (xml, meta) where meta mirrors
    ``scene_builder.build_cable_lift_scene``: equilibrium [nd,3] world drone hover
    positions, payload_pos [3], plus the body names for state/wrench addressing."""
    visual = _resolve_visual(visual)
    body_xml, meta = _formation_xml(
        "",
        origin,
        num_ropes,
        rope_length,
        payload_mass,
        payload_radius,
        elevation_angle_deg,
        base_height,
        joint_damping,
        joint_armature,
        visual,
    )
    xml = (
        _HEADER.format(model="circle_multilift", dt=sim_dt, assets=_drone_assets_xml(visual))
        + body_xml
        + _FOOTER.format(equality="")
    )
    return xml, meta


def _formation_xml(
    prefix,
    origin,
    num_ropes,
    rope_length,
    payload_mass,
    payload_radius,
    elevation_angle_deg,
    base_height,
    joint_damping,
    joint_armature,
    visual,
):
    """Build ONE cable-lift formation's payload-body subtree (names prefixed for grids).
    Returns (payload_body_xml, meta) with world equilibrium + payload pos + body names."""
    ox, oy, oz = origin
    num_links = max(1, int(round(rope_length / LINK_TOTAL_LENGTH)))
    payload_height = payload_radius * PAYLOAD_HEIGHT_RATIO
    half_h = payload_height / 2.0
    izz = 0.5 * payload_mass * payload_radius**2
    ixx = (1.0 / 12.0) * payload_mass * (3.0 * payload_radius**2 + payload_height**2)
    beta = math.radians(elevation_angle_deg)
    cos_b, sin_b = math.cos(beta), math.sin(beta)
    payload_center = np.array([ox, oy, base_height + oz])

    cap = (
        f'<geom type="capsule" fromto="0 0 0 0 0 {LINK_CYL_HEIGHT:.6g}" size="{LINK_RADIUS:.6g}" '
        f'mass="{LINK_MASS:.6g}" contype="0" conaffinity="0" rgba="0.85 0.85 0.85 1"/>'
    )
    jnt = f'<joint type="ball" damping="{joint_damping:.6g}" armature="{joint_armature:.6g}"/>'

    ropes_xml, drone_names, link_names, equilibrium = [], [], [], []
    for i in range(num_ropes):
        ang = 2.0 * math.pi * i / num_ropes
        rho = np.array([payload_radius * math.cos(ang), payload_radius * math.sin(ang), half_h])
        d = np.array([math.cos(ang) * cos_b, math.sin(ang) * cos_b, sin_b])
        q_link = quat_z_to(d)
        q_drone = np.array([q_link[0], -q_link[1], -q_link[2], -q_link[3]])  # level the drone
        dname = f"{prefix}rope{i}_drone"
        drone_names.append(dname)
        equilibrium.append(
            payload_center + rho + d * (num_links * LINK_TOTAL_LENGTH + BODY_BOTTOM_OFFSET)
        )

        drone = _drone_body_xml(
            dname, (0.0, 0.0, LINK_TOTAL_LENGTH + BODY_BOTTOM_OFFSET), q_drone, visual=visual
        )
        drone = drone.replace("<freejoint/>", jnt, 1)  # ball joint to the cable, not a free body
        inner = drone
        for j in range(num_links - 1, 0, -1):
            lname = f"{prefix}rope{i}_link{j}"
            link_names.append(lname)
            inner = (
                f'<body name="{lname}" pos="0 0 {LINK_TOTAL_LENGTH:.6g}">{jnt}{cap}{inner}</body>'
            )
        l0 = f"{prefix}rope{i}_link0"
        link_names.append(l0)
        ropes_xml.append(
            f'<body name="{l0}" pos="{_v(rho)}" quat="{_v(q_link)}">{jnt}{cap}{inner}</body>'
        )

    pname = f"{prefix}payload"
    body_xml = (
        f'<body name="{pname}" pos="{_v(payload_center)}">'
        f"<freejoint/>"
        f'<geom type="cylinder" size="{payload_radius:.6g} {half_h:.6g}" rgba="0.22 0.43 0.55 1"/>'
        f'<inertial pos="0 0 0" mass="{payload_mass:.6g}" diaginertia="{ixx:.6g} {ixx:.6g} {izz:.6g}"/>'
        f"{''.join(ropes_xml)}"
        f"</body>"
    )
    meta = {
        "payload_name": pname,
        "drone_names": drone_names,
        "link_names": link_names,
        "equilibrium": np.array(equilibrium),
        "payload_pos": payload_center,
        "num_links": num_links,
    }
    return body_xml, meta


def build_multilift_grid(
    num_envs: int,
    env_spacing: float,
    sim_dt: float,
    *,
    num_ropes: int = 4,
    rope_length: float = 1.0,
    payload_mass: float = 2.0,
    payload_radius: float = 0.15,
    elevation_angle_deg: float = 40.0,
    base_height: float = 1.5,
    joint_damping: float = 0.01,
    joint_armature: float = 1e-6,
    visual: str = "simple",
):
    """``num_envs`` cable-lift formations on a square grid in ONE MuJoCo model (one mj_step
    advances all of them — the parallelism for RL). Returns (xml, metas) where metas[e] is the
    per-formation meta (env-prefixed body names, world equilibrium, payload pos)."""
    visual = _resolve_visual(visual)
    cols = max(1, int(math.ceil(math.sqrt(num_envs))))
    off = 0.5 * (cols - 1) * env_spacing
    bodies, metas = [], []
    for e in range(num_envs):
        r, c = divmod(e, cols)
        origin = (c * env_spacing - off, r * env_spacing - off, 0.0)
        body_xml, meta = _formation_xml(
            f"e{e}_",
            origin,
            num_ropes,
            rope_length,
            payload_mass,
            payload_radius,
            elevation_angle_deg,
            base_height,
            joint_damping,
            joint_armature,
            visual,
        )
        bodies.append(body_xml)
        metas.append(meta)
    xml = (
        _HEADER.format(model="multilift_grid", dt=sim_dt, assets=_drone_assets_xml(visual))
        + "".join(bodies)
        + _FOOTER.format(equality="")
    )
    return xml, metas


# ── state / wrench helpers ───────────────────────────────────────────────────
def body_ids(model, names):
    import mujoco

    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]


def read_drone_state(model, data, bids):
    """Returns (pos_w [n,3], quat_w [n,4] wxyz, linvel_w [n,3], angvel_b [n,3]).

    Mirrors Isaac root_pos_w / root_quat_w / root_lin_vel_w / root_ang_vel_b."""
    import mujoco

    n = len(bids)
    pos = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    linw = np.zeros((n, 3))
    omb = np.zeros((n, 3))
    tmp = np.zeros(6)
    for k, b in enumerate(bids):
        pos[k] = data.xpos[b]
        quat[k] = data.xquat[b]
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, b, tmp, 0)  # world frame
        linw[k] = tmp[3:6]
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, b, tmp, 1
        )  # local (body) frame
        omb[k] = tmp[0:3]
    return pos, quat, linw, omb


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


_UP_CACHE: dict = {}


def _charpi_compiled_up(model, mid: int, va: int, vn: int) -> np.ndarray:
    """Compiled-frame UP unit-vector for the charpi drone mesh.

    The mesh's true up is **raw +Y** (the xacro renders the drone bottom-on-ground; raw +Y → robot +Z).
    MuJoCo re-aligns the mesh to its principal axes AND recenters to the *volumetric* COM (which sits
    toward the solid cameras/body, opposite the dense-but-thin propeller vertices). A vertex-extent
    heuristic ("larger-|coord| side = up") therefore FLIPS this drone. We instead map raw +Y through
    MuJoCo's actual transform by correlating the raw up-coordinate with the compiled vertices
    (vertex order is preserved), cached in .cache/charpi_up.json."""
    if mid in _UP_CACHE:
        return _UP_CACHE[mid]
    import json

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache")
    cache_f = os.path.abspath(os.path.join(cache_dir, "charpi_up.json"))
    merged = _merged_charpi_mesh()
    up = None
    if os.path.isfile(cache_f) and os.path.getmtime(cache_f) >= os.path.getmtime(merged):
        up = np.array(json.load(open(cache_f)), dtype=float)
    else:
        raw_y = np.array([float(line.split()[2]) for line in open(merged) if line.startswith("v ")])
        Vc = np.array(model.mesh_vert[va : va + vn]).reshape(-1, 3)
        if raw_y.shape[0] == vn:
            corr = [np.corrcoef(raw_y, Vc[:, a])[0, 1] for a in range(3)]
            axis = int(np.argmax(np.abs(corr)))
            up = np.zeros(3)
            up[axis] = float(np.sign(corr[axis]))
            try:
                os.makedirs(cache_dir, exist_ok=True)
                json.dump(up.tolist(), open(cache_f, "w"))
            except OSError:
                pass
        else:
            up = np.array([0.0, 0.0, 1.0])
    _UP_CACHE[mid] = up
    return up


def seat_drone_meshes(model, *, center: bool = False, bottom_z: float = 0.0) -> None:
    """Orient + position each drone MESH geom AFTER compile.

    MuJoCo re-aligns meshes to their principal axes, so we rotate the drone's true UP axis
    (derived from the raw mesh via :func:`_charpi_compiled_up`, NOT a vertex-extent guess —
    that flips this drone) onto body +Z, and place it either centered on the body origin
    (``center=True``, free single drone) or with its BOTTOM at ``bottom_z`` (the cable attach
    point, so the cable meets the belly). Visual-only (the body keeps its explicit <inertial>)."""
    import mujoco

    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(model.geom_dataid[gid])
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        V = np.array(model.mesh_vert[va : va + vn]).reshape(-1, 3)
        src = _charpi_compiled_up(model, mid, va, vn)  # true drone UP in the compiled frame
        q = _quat_conj(quat_z_to(src))  # rotate drone UP -> body +Z
        model.geom_quat[gid] = q
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        Vw = V @ R.reshape(3, 3).T  # oriented verts in body frame
        gx = -0.5 * (Vw[:, 0].max() + Vw[:, 0].min())
        gy = -0.5 * (Vw[:, 1].max() + Vw[:, 1].min())
        gz = -0.5 * (Vw[:, 2].max() + Vw[:, 2].min()) if center else (bottom_z - Vw[:, 2].min())
        model.geom_pos[gid] = [gx, gy, gz]


def apply_body_wrench(model, data, bids, quat_w, F_body, M_body):
    """Apply body-frame (F, M) per drone as world-frame xfrc_applied at each COM.

    F_body / M_body: [n,3] numpy. quat_w: [n,4] wxyz (the bodies' world orientation)."""
    import mujoco

    R = np.zeros(9)
    for k, b in enumerate(bids):
        mujoco.mju_quat2Mat(R, quat_w[k])
        Rm = R.reshape(3, 3)
        data.xfrc_applied[b, 0:3] = Rm @ F_body[k]
        data.xfrc_applied[b, 3:6] = Rm @ M_body[k]
