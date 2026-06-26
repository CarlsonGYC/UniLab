# Copyright (c) 2026, The direct_rl Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Faithful, batched (multi-env) port of the PX4 v1.16 multicopter **position**
and **attitude** controllers.

The RL policy commands a per-drone **position / velocity / acceleration (PVA)**
setpoint (plus an optional yaw).  This module turns that PVA setpoint into the
**(collective-thrust, body-rate)** command consumed by the already-ported
:class:`~direct_rl.dynamics.control_stack.CTBRControlStack` rate loop (which is
itself PX4-equivalent).  The cascade is therefore complete and PX4-faithful::

    RL PVA setpoint
        │  PX4PositionControl  (mc_pos_control/PositionControl.cpp)
        │      P-position → PID-velocity → accelerationControl → thrust vector
        ▼
    thrust vector + yaw  ─ thrustToAttitude (ControlMath.cpp) ─▶ q_des , ‖thr‖
        │  PX4AttitudeControl  (mc_att_control/AttitudeControl.cpp, Brescianini 2013)
        ▼
    body-rate setpoint  +  normalized collective thrust
        │  CTBRControlStack  (BodyRateController + ControlAllocator + Motor + …)
        ▼
    body wrench → PhysX

Design notes
------------
* **Verbatim numerics.**  ``_positionControl`` / ``_velocityControl`` /
  ``_accelerationControl`` / ``thrustToAttitude`` and the quaternion attitude law
  reproduce the PX4 C++ line-for-line (same gains, limits, anti-windup, tilt
  limit, reduced/full-attitude yaw-weight blend).  Defaults are the real PX4
  parameter defaults (``MPC_*`` / ``MC_*``).
* **Frames.**  PX4 works in **NED world / FRD body**; IsaacLab works in
  **ENU world / FLU body**.  All state/setpoints are converted ENU/FLU→NED/FRD
  on the way in and the body-rate setpoint FRD→FLU on the way out by the small
  involutive adapters :func:`enu_ned` / :func:`flu_frd` and the constant
  quaternion sandwich :data:`_Q_ENU2NED` / :data:`_Q_FLU2FRD`.  The collective
  thrust is a magnitude → frame-independent.
* **Parallel / fast.**  Everything is pure batched ``torch`` over ``[N, …]`` with
  no per-env Python loops (the only loops are the constant 4-component quaternion
  canonicalization and the fixed-branch ``constrainXY``), so it vectorizes to
  thousands of envs on GPU and is differentiable-friendly.
* **Self-contained.**  No ``isaaclab`` import — the quaternion/rotation helpers
  are local, so the module imports on a bare Python install and is unit-testable
  on CPU without launching Isaac Sim.

Hover-thrust calibration
------------------------
PX4's ``_hover_thrust`` is the *normalized* collective thrust that holds the
vehicle at hover.  Because the controller hands its normalized thrust straight to
the CTBR mixer (whose thrust column gain ≈ 1), the hover thrust equals the mixer
actuator command that produces ``m_eff · g`` of total thrust.  For this drone
(``thrust_const`` 1.15e-6, ``max_motor_speed`` 3442 → ~54 N max) that is
≈ 0.25 for the bare 1.09 kg drone and ≈ 0.34 with a 0.5 kg payload share.  Use
:func:`hover_thrust_for_thrust` to compute it from the effective hover weight and
the stack config, or just set :attr:`PX4PositionControlCfg.hover_thrust`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

CONSTANTS_ONE_G = 9.80665  # PX4 standard gravity [m/s^2]
_FLT_EPSILON = 1.1920929e-7  # matches C++ FLT_EPSILON used by matrix::canonical()


# ════════════════════════════════════════════════════════════════════════════
# Frame adapters.  ENU/FLU ↔ NED/FRD are 180° rotations, hence involutions
# (applying the same map twice is the identity), so one function converts both
# directions.
# ════════════════════════════════════════════════════════════════════════════
def enu_ned(v: torch.Tensor) -> torch.Tensor:
    """World vector (E,N,U) ↔ (N,E,D):  ``(x, y, z) → (y, x, -z)``."""
    return torch.stack([v[..., 1], v[..., 0], -v[..., 2]], dim=-1)


def flu_frd(v: torch.Tensor) -> torch.Tensor:
    """Body vector (F,L,U) ↔ (F,R,D):  ``(x, y, z) → (x, -y, -z)``."""
    return torch.stack([v[..., 0], -v[..., 1], -v[..., 2]], dim=-1)


# Constant attitude quaternions for the world/body frame change (see module
# docstring derivation):  R_NEDfromFRD = S_world · R_ENUfromFLU · S_body, and
# because S_world, S_body are proper rotations the same holds for quaternions:
#   q_ned = q_enu2ned ⊗ q_enu ⊗ q_flu2frd
# S_world = 180° about (1,1,0)/√2 ; S_body = 180° about x.
_R2 = math.sqrt(0.5)
_Q_ENU2NED = (0.0, _R2, _R2, 0.0)  # (w,x,y,z)
_Q_FLU2FRD = (0.0, 1.0, 0.0, 0.0)  # (w,x,y,z)


# ════════════════════════════════════════════════════════════════════════════
# Quaternion utilities — (w, x, y, z) convention (same as PX4 matrix::Quatf and
# IsaacLab).  All batched over the leading dimension.
# ════════════════════════════════════════════════════════════════════════════
def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a ⊗ b`` for ``[N,4]`` quaternions."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    """Conjugate = inverse for a unit quaternion: ``(w, -x, -y, -z)``."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def quat_dcm_z(q: torch.Tensor) -> torch.Tensor:
    """Third column of the rotation matrix (body-z expressed in world).

    Reproduces ``matrix::Quaternion::dcm_z()``.
    """
    a, b, c, d = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack(
        [
            2.0 * (a * c + b * d),
            2.0 * (c * d - a * b),
            a * a - b * b - c * c + d * d,
        ],
        dim=-1,
    )


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """Reproduce ``matrix::Quaternion::canonical()``.

    Multiply the whole quaternion by ``sign(q_i)`` of its first component (order
    w,x,y,z) whose magnitude exceeds ``FLT_EPSILON``.  Resolves the ± double
    cover the same way PX4 does before extracting the attitude error.
    """
    sign = torch.ones(q.shape[:-1] + (1,), device=q.device, dtype=q.dtype)
    found = torch.zeros(q.shape[:-1] + (1,), device=q.device, dtype=torch.bool)
    for i in range(4):
        qi = q[..., i : i + 1]
        take = (~found) & (qi.abs() > _FLT_EPSILON)
        sign = torch.where(take, torch.sign(qi), sign)
        found = found | take
    return q * sign


def quat_from_two_vectors(src: torch.Tensor, dst: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Shortest-arc quaternion rotating ``src`` onto ``dst``.

    Reproduces the ``matrix::Quaternion(src, dst)`` half-way construction,
    including the 180° corner case (anti-parallel inputs) handled with an
    arbitrary perpendicular axis.
    """
    cr = torch.cross(src, dst, dim=-1)
    dt = (src * dst).sum(dim=-1, keepdim=True)
    crn = cr.norm(dim=-1, keepdim=True)

    # normal case: w = dot + sqrt(|src|^2 |dst|^2)
    w_normal = dt + torch.sqrt(
        (src * src).sum(dim=-1, keepdim=True) * (dst * dst).sum(dim=-1, keepdim=True)
    )

    # corner case (parallel & opposite): pick the axis of the smallest |src|
    # component and use the rejection cross(src, axis).
    a = src.abs()
    amin = a.argmin(dim=-1, keepdim=True)
    axis = torch.zeros_like(src).scatter_(-1, amin, 1.0)
    cr_corner = torch.cross(src, axis, dim=-1)

    corner = (crn < eps) & (dt < 0)
    w = torch.where(corner, torch.zeros_like(w_normal), w_normal)
    xyz = torch.where(corner, cr_corner, cr)
    return quat_normalize(torch.cat([w, xyz], dim=-1))


def quat_from_dcm_upright(R: torch.Tensor) -> torch.Tensor:
    """Quaternion from a rotation matrix, valid when ``trace(R) > 0``.

    The desired attitude produced by :func:`_bodyz_to_attitude` is tilt-limited
    (≤ ``MPC_TILTMAX_AIR`` = 45°, well under 90°), so ``trace`` is always
    positive and the numerically stable w-branch always applies.
    """
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = (m00 + m11 + m22).clamp(min=-0.999999)
    w = 0.5 * torch.sqrt((1.0 + trace).clamp(min=1e-12))
    inv = 0.25 / w
    x = (R[..., 2, 1] - R[..., 1, 2]) * inv
    y = (R[..., 0, 2] - R[..., 2, 0]) * inv
    z = (R[..., 1, 0] - R[..., 0, 1]) * inv
    return quat_normalize(torch.stack([w, x, y, z], dim=-1))


def quat_enu_flu_to_ned_frd(q_enu: torch.Tensor) -> torch.Tensor:
    """Convert an ENU-world / FLU-body attitude quaternion to NED / FRD."""
    qw = q_enu.new_tensor(_Q_ENU2NED).expand_as(q_enu)
    qb = q_enu.new_tensor(_Q_FLU2FRD).expand_as(q_enu)
    return quat_mul(quat_mul(qw, q_enu), qb)


# ════════════════════════════════════════════════════════════════════════════
# ControlMath helpers (vectorized) — NaN-aware setpoint combination + limits.
# ════════════════════════════════════════════════════════════════════════════
def _add_if_not_nan(sp: torch.Tensor, add: torch.Tensor) -> torch.Tensor:
    """``ControlMath::addIfNotNanVector3f`` — NaN-as-uncommitted addition."""
    sp_f = torch.isfinite(sp)
    add_f = torch.isfinite(add)
    out = torch.where(sp_f & add_f, sp + add, sp)  # both finite → add
    out = torch.where((~sp_f) & add_f, add, out)  # sp NaN, add finite → take add
    return out  # else keep sp (incl. both-NaN)


def _zero_if_nan(v: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(v), v, torch.zeros_like(v))


def _constrain_xy(v0: torch.Tensor, v1: torch.Tensor, vmax: float) -> torch.Tensor:
    """``ControlMath::constrainXY`` — sum two 2-D vectors, v0 has priority.

    v0, v1: ``[N,2]``.  Returns ``[N,2]`` with ‖·‖ ≤ vmax, prioritizing v0.
    The PX4 if/elif chain is replicated by applying the branches in reverse
    priority with ``torch.where`` (so the earliest true condition wins).
    """
    sumv = v0 + v1
    sum_norm = sumv.norm(dim=-1, keepdim=True)
    v0n = v0.norm(dim=-1, keepdim=True)
    diff_norm = (v1 - v0).norm(dim=-1, keepdim=True)
    u0 = v0 / v0n.clamp(min=1e-9)
    u1 = v1 / v1.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    # else branch: solve ‖v0 + s·u1‖ = vmax for s ≥ 0
    m = (u1 * v0).sum(dim=-1, keepdim=True)
    c = (v0 * v0).sum(dim=-1, keepdim=True) - vmax * vmax
    s = -m + torch.sqrt((m * m - c).clamp(min=0.0))
    res = v0 + u1 * s
    # elif v0.len < 0.001 → v1̂·vmax
    res = torch.where(v0n < 0.001, u1 * vmax, res)
    # elif |v1 - v0| < 0.001 → v0̂·vmax
    res = torch.where(diff_norm < 0.001, u0 * vmax, res)
    # elif v0.len >= vmax → v0̂·vmax
    res = torch.where(v0n >= vmax, u0 * vmax, res)
    # if ‖v0 + v1‖ <= vmax → v0 + v1   (highest priority, applied last)
    res = torch.where(sum_norm <= vmax, sumv, res)
    return res


def _limit_tilt(body_z: torch.Tensor, lim_tilt: float) -> torch.Tensor:
    """``ControlMath::limitTilt`` against world-up e3 = (0,0,1)."""
    e3 = body_z.new_tensor([0.0, 0.0, 1.0]).expand_as(body_z)
    dotp = (body_z * e3).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.acos(dotp).clamp(max=lim_tilt)
    rejection = body_z - dotp * e3
    rn2 = (rejection * rejection).sum(dim=-1, keepdim=True)
    e1 = body_z.new_tensor([1.0, 0.0, 0.0]).expand_as(body_z)
    rejection = torch.where(rn2 < _FLT_EPSILON, e1, rejection)
    rej_unit = rejection / rejection.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.cos(angle) * e3 + torch.sin(angle) * rej_unit


# ════════════════════════════════════════════════════════════════════════════
# Configs (defaults = real PX4 parameter defaults).
# ════════════════════════════════════════════════════════════════════════════
@dataclass
class PX4PositionControlCfg:
    """``mc_pos_control`` gains/limits (PX4 ``MPC_*`` defaults)."""

    # position P (MPC_XY_P, MPC_XY_P, MPC_Z_P)
    gain_pos_p: tuple = (0.95, 0.95, 1.0)
    # velocity PID (MPC_{XY,Z}_VEL_{P,I,D}_ACC)
    gain_vel_p: tuple = (1.8, 1.8, 4.0)
    gain_vel_i: tuple = (0.4, 0.4, 2.0)
    gain_vel_d: tuple = (0.2, 0.2, 0.0)
    # velocity limits (MPC_XY_VEL_MAX, MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DN)
    lim_vel_horizontal: float = 12.0
    lim_vel_up: float = 3.0
    lim_vel_down: float = 1.5
    # normalized thrust limits + horizontal margin (MPC_THR_MIN/MAX, MPC_THR_XY_MARG)
    lim_thr_min: float = 0.12
    lim_thr_max: float = 1.0
    lim_thr_xy_margin: float = 0.3
    # tilt limit (MPC_TILTMAX_AIR, degrees)
    tilt_max_deg: float = 45.0
    # normalized hover thrust (MPC_THR_HOVER default 0.5; calibrate per vehicle —
    # ~0.25 bare 1.09 kg drone, ~0.34 with a 0.5 kg payload share; see module doc)
    hover_thrust: float = 0.34
    # PX4 ignores the vertical accel setpoint when forming the tilt (default true)
    decouple_horizontal_and_vertical_accel: bool = True


@dataclass
class PX4AttitudeControlCfg:
    """``mc_att_control`` gains/limits (PX4 ``MC_*`` defaults)."""

    # attitude P (MC_ROLL_P, MC_PITCH_P, MC_YAW_P) and yaw de-prioritization
    roll_p: float = 6.5
    pitch_p: float = 6.5
    yaw_p: float = 2.8
    yaw_weight: float = 0.4  # MC_YAW_WEIGHT
    # rate-setpoint limits (MC_{ROLL,PITCH,YAW}RATE_MAX, degrees/s)
    rollrate_max_deg: float = 220.0
    pitchrate_max_deg: float = 220.0
    yawrate_max_deg: float = 200.0


@dataclass
class PX4ControlCfg:
    """Bundle of the position + attitude configs for the orchestrator."""

    position: PX4PositionControlCfg = field(default_factory=PX4PositionControlCfg)
    attitude: PX4AttitudeControlCfg = field(default_factory=PX4AttitudeControlCfg)
    yaw_des: float = 0.0  # default ENU yaw setpoint when the RL command omits it


def hover_thrust_for_thrust(
    hover_weight_n: float,
    thrust_const: float = 1.15e-6,
    min_motor_speed: float = 341.75,
    max_motor_speed: float = 3442.0,
    kl: float = 0.05,
) -> float:
    """Normalized hover thrust for a given hover weight, inverting the CTBR stack.

    Walks the stack's thrust path backwards (4 equal rotors):
    ``T = 4·thrust_const·ω²`` → ω → motor-speed ratio over ``[min,max]`` → invert
    the PX4 ``kl`` thrust-curve blend → normalized actuator/collective command.
    Defaults are the ``DroneControlCfg`` values (≈ 0.25 for the bare 1.09 kg drone,
    ≈ 0.34 with a 0.5 kg payload share).
    """
    omega = math.sqrt(max(0.0, hover_weight_n) / (4.0 * thrust_const))
    omega = min(max(omega, min_motor_speed), max_motor_speed)
    # ratio is the *output* of the thrust curve: ω = min + (max-min)·curve(actuator)
    ratio = (omega - min_motor_speed) / (max_motor_speed - min_motor_speed)
    curve = ratio * ratio
    # invert kl blend:  kl·y² + (1-kl)·y = curve  →  y = (a - 0.1)
    if kl > 0.0:
        y = (-(1 - kl) + math.sqrt((1 - kl) ** 2 + 4 * kl * curve)) / (2 * kl)
    else:
        y = curve
    return float(y + 0.1)


# ════════════════════════════════════════════════════════════════════════════
# Position controller — mc_pos_control/PositionControl.cpp (NED internally).
# ════════════════════════════════════════════════════════════════════════════
class PX4PositionControl:
    """Batched P-position + PID-velocity controller producing a thrust vector.

    Works entirely in **NED world**.  ``update`` returns the NED normalized
    thrust vector ``_thr_sp``; pair it with :func:`_bodyz_to_attitude` (i.e.
    ``thrustToAttitude``) to obtain the desired attitude quaternion.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        cfg: PX4PositionControlCfg | None = None,
        device="cpu",
        dtype=torch.float32,
    ):
        self.cfg = cfg or PX4PositionControlCfg()
        c = self.cfg
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.dtype = dtype

        def vec(t):
            return torch.tensor(t, device=device, dtype=dtype)

        self._kp = vec(c.gain_pos_p)
        self._kv_p = vec(c.gain_vel_p)
        self._kv_i = vec(c.gain_vel_i)
        self._kv_d = vec(c.gain_vel_d)
        self._lim_tilt = math.radians(c.tilt_max_deg)

        self._vel_int = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._thr_sp = torch.zeros(num_envs, 3, device=device, dtype=dtype)

    # ── PX4 _positionControl() ──────────────────────────────────────────────
    def _position_control(self, pos, vel_sp, pos_sp):
        vel_sp_position = (pos_sp - pos) * self._kp
        vel_sp = _add_if_not_nan(vel_sp, vel_sp_position)
        vel_sp_position = _zero_if_nan(vel_sp_position)
        c = self.cfg
        xy = _constrain_xy(
            vel_sp_position[:, :2], (vel_sp - vel_sp_position)[:, :2], c.lim_vel_horizontal
        )
        vz = vel_sp[:, 2:3].clamp(-c.lim_vel_up, c.lim_vel_down)
        return torch.cat([xy, vz], dim=-1)

    # ── PX4 _accelerationControl() ──────────────────────────────────────────
    def _acceleration_control(self, acc_sp):
        c = self.cfg
        z_specific_force = acc_sp.new_full((self.num_envs, 1), -CONSTANTS_ONE_G)
        if not c.decouple_horizontal_and_vertical_accel:
            z_specific_force = z_specific_force + acc_sp[:, 2:3]
        body_z = torch.cat([-acc_sp[:, 0:1], -acc_sp[:, 1:2], -z_specific_force], dim=-1)
        body_z = body_z / body_z.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        body_z = _limit_tilt(body_z, self._lim_tilt)
        thrust_ned_z = acc_sp[:, 2:3] * (c.hover_thrust / CONSTANTS_ONE_G) - c.hover_thrust
        cos_ned_body = body_z[:, 2:3]
        collective = torch.minimum(
            thrust_ned_z / cos_ned_body, acc_sp.new_full((self.num_envs, 1), -c.lim_thr_min)
        )
        return body_z * collective

    # ── PX4 _velocityControl() ──────────────────────────────────────────────
    def _velocity_control(self, vel, vel_dot, vel_sp, acc_sp):
        c = self.cfg
        G = CONSTANTS_ONE_G
        # constrain vertical integral
        vint = self._vel_int.clone()
        vint[:, 2] = vint[:, 2].clamp(-G, G)

        vel_error = vel_sp - vel
        acc_sp_velocity = vel_error * self._kv_p + vint - vel_dot * self._kv_d
        acc_sp = _add_if_not_nan(acc_sp, acc_sp_velocity)

        thr_sp = self._acceleration_control(acc_sp)

        # vertical integrator anti-windup
        cond_z = ((thr_sp[:, 2] >= -c.lim_thr_min) & (vel_error[:, 2] >= 0.0)) | (
            (thr_sp[:, 2] <= -c.lim_thr_max) & (vel_error[:, 2] <= 0.0)
        )
        vel_error = vel_error.clone()
        vel_error[:, 2] = torch.where(cond_z, torch.zeros_like(vel_error[:, 2]), vel_error[:, 2])

        # prioritize vertical thrust, keep horizontal margin
        thr_xy = thr_sp[:, :2]
        thr_xy_norm = thr_xy.norm(dim=-1, keepdim=True)
        thr_max_sq = c.lim_thr_max**2
        alloc_h = torch.minimum(
            thr_xy_norm, thr_xy_norm.new_full(thr_xy_norm.shape, c.lim_thr_xy_margin)
        )
        thr_z_max_sq = thr_max_sq - alloc_h**2
        thr_sp = thr_sp.clone()
        thr_sp[:, 2:3] = torch.maximum(thr_sp[:, 2:3], -torch.sqrt(thr_z_max_sq.clamp(min=0.0)))

        thr_max_xy_sq = (thr_max_sq - thr_sp[:, 2:3] ** 2).clamp(min=0.0)
        thr_max_xy = torch.sqrt(thr_max_xy_sq)
        scale = torch.where(
            thr_xy_norm > thr_max_xy,
            thr_max_xy / thr_xy_norm.clamp(min=1e-9),
            torch.ones_like(thr_xy_norm),
        )
        thr_sp[:, :2] = thr_xy * scale

        # horizontal tracking anti-reset-windup
        acc_xy_produced = thr_sp[:, :2] * (G / c.hover_thrust)
        acc_sp_xy = acc_sp[:, :2]
        saturated = acc_sp_xy.pow(2).sum(-1, keepdim=True) > acc_xy_produced.pow(2).sum(
            -1, keepdim=True
        )
        arw_gain = 2.0 / self._kv_p[0]
        ve_xy = vel_error[:, :2] - arw_gain * (acc_sp_xy - acc_xy_produced)
        vel_error[:, :2] = torch.where(saturated, ve_xy, vel_error[:, :2])

        vel_error = _zero_if_nan(vel_error)
        self._vel_int = vint + vel_error * self._kv_i * self.dt
        self._thr_sp = thr_sp
        return thr_sp

    def update(self, pos, vel, vel_dot, pos_sp, vel_sp, acc_sp):
        """One control step (all args ``[N,3]`` in **NED**).  Returns thr_sp ``[N,3]``."""
        vel_sp = self._position_control(pos, vel_sp.clone(), pos_sp)
        return self._velocity_control(vel, vel_dot, vel_sp, acc_sp.clone())

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._vel_int[env_ids] = 0.0
        self._thr_sp[env_ids] = 0.0


def _bodyz_to_attitude(thr_sp: torch.Tensor, yaw_sp: torch.Tensor) -> torch.Tensor:
    """``ControlMath::thrustToAttitude`` → ``bodyzToAttitude``.

    ``thr_sp`` ``[N,3]`` (NED), ``yaw_sp`` ``[N,1]`` (NED).  Returns the desired
    attitude quaternion ``[N,4]`` (NED/FRD).  The desired body-z is ``-thr_sp``.
    """
    body_z = -thr_sp
    n2 = (body_z * body_z).sum(dim=-1, keepdim=True)
    e3 = body_z.new_tensor([0.0, 0.0, 1.0]).expand_as(body_z)
    body_z = torch.where(n2 < _FLT_EPSILON, e3, body_z)
    body_z = body_z / body_z.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    y_C = torch.cat([-torch.sin(yaw_sp), torch.cos(yaw_sp), torch.zeros_like(yaw_sp)], dim=-1)
    body_x = torch.cross(y_C, body_z, dim=-1)
    # keep nose to front while inverted upside down
    body_x = torch.where(body_z[:, 2:3] < 0.0, -body_x, body_x)
    # thrust in XY plane corner case → set body_x = down
    e3z = body_z.new_tensor([0.0, 0.0, 1.0]).expand_as(body_z)
    body_x = torch.where(body_z[:, 2:3].abs() < 1e-6, e3z, body_x)
    body_x = body_x / body_x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    body_y = torch.cross(body_z, body_x, dim=-1)

    R = torch.stack([body_x, body_y, body_z], dim=-1)  # columns = body axes
    return quat_from_dcm_upright(R)


# ════════════════════════════════════════════════════════════════════════════
# Attitude controller — mc_att_control/AttitudeControl.cpp (Brescianini 2013).
# ════════════════════════════════════════════════════════════════════════════
class PX4AttitudeControl:
    """Batched quaternion attitude controller → body-rate setpoint (FRD)."""

    def __init__(
        self,
        num_envs: int,
        cfg: PX4AttitudeControlCfg | None = None,
        device="cpu",
        dtype=torch.float32,
    ):
        self.cfg = cfg or PX4AttitudeControlCfg()
        c = self.cfg
        self.num_envs = num_envs
        self.device = device
        self.dtype = dtype

        yaw_w = min(max(c.yaw_weight, 0.0), 1.0)
        # PX4 setProportionalGain rescales the yaw gain by 1/yaw_w (compensated
        # by the yaw_w blend inside update()).
        yaw_gain = c.yaw_p / yaw_w if yaw_w > 1e-4 else c.yaw_p
        self._yaw_w = yaw_w
        self._prop_gain = torch.tensor([c.roll_p, c.pitch_p, yaw_gain], device=device, dtype=dtype)
        self._rate_limit = torch.tensor(
            [
                math.radians(c.rollrate_max_deg),
                math.radians(c.pitchrate_max_deg),
                math.radians(c.yawrate_max_deg),
            ],
            device=device,
            dtype=dtype,
        )

    def update(
        self, q: torch.Tensor, qd: torch.Tensor, yawspeed_sp: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``q`` current, ``qd`` desired attitude ``[N,4]`` (NED/FRD).

        Returns the body-rate setpoint ``[N,3]`` (FRD).
        """
        e_z = quat_dcm_z(q)
        e_z_d = quat_dcm_z(qd)
        qd_red = quat_from_two_vectors(e_z, e_z_d)

        # corner case: vehicle & thrust nearly opposite → use full desired attitude
        cond = (qd_red[:, 1:2].abs() > (1.0 - 1e-5)) | (qd_red[:, 2:3].abs() > (1.0 - 1e-5))
        qd_red_full = quat_mul(qd_red, q)
        qd_red = torch.where(cond, qd, qd_red_full)

        # extract delta-yaw, blend it by yaw weight, recombine
        qd_dyaw = quat_canonical(quat_mul(quat_conj(qd_red), qd))
        q0 = qd_dyaw[:, 0:1].clamp(-1.0, 1.0)
        q3 = qd_dyaw[:, 3:4].clamp(-1.0, 1.0)
        zeros = torch.zeros_like(q0)
        dyaw = torch.cat(
            [
                torch.cos(self._yaw_w * torch.acos(q0)),
                zeros,
                zeros,
                torch.sin(self._yaw_w * torch.asin(q3)),
            ],
            dim=-1,
        )
        qd_full = quat_mul(qd_red, dyaw)

        # attitude error → rate setpoint (sin(α/2)-scaled axis, antipodal-safe)
        qe = quat_mul(quat_conj(q), qd_full)
        eq = 2.0 * quat_canonical(qe)[:, 1:4]
        rate_sp = eq * self._prop_gain

        # world-z yaw-rate feed-forward expressed in body frame
        if yawspeed_sp is not None:
            rate_sp = rate_sp + quat_dcm_z(quat_conj(q)) * yawspeed_sp

        return torch.maximum(torch.minimum(rate_sp, self._rate_limit), -self._rate_limit)


# ════════════════════════════════════════════════════════════════════════════
# Orchestrator — PVA (ENU/FLU) → (normalized collective thrust, FLU body rate).
# ════════════════════════════════════════════════════════════════════════════
class PX4PositionAttitudeController:
    """Full PX4 outer+mid cascade, ENU/FLU in, CTBR-stack command out.

    Typical use (cascaded rates, like PX4): call :meth:`update_position` once per
    *control* step (e.g. 100 Hz) to refresh the thrust + desired attitude, and
    :meth:`update_attitude` once per *sim* step (e.g. 500 Hz) to refresh the body
    rate from the latest attitude estimate.  :meth:`compute` does both at once.
    """

    def __init__(
        self,
        num_envs: int,
        dt: float,
        cfg: PX4ControlCfg | None = None,
        device="cpu",
        dtype=torch.float32,
    ):
        self.cfg = cfg or PX4ControlCfg()
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.dtype = dtype

        self._pos = PX4PositionControl(num_envs, dt, self.cfg.position, device, dtype)
        self._att = PX4AttitudeControl(num_envs, self.cfg.attitude, device, dtype)

        # NED yaw setpoint from the default ENU yaw (yaw_ned = π/2 − yaw_enu)
        self._yaw_sp_ned = torch.full(
            (num_envs, 1), math.pi / 2 - self.cfg.yaw_des, device=device, dtype=dtype
        )
        # cascade hand-off state
        self._q_d_ned = torch.zeros(num_envs, 4, device=device, dtype=dtype)
        self._q_d_ned[:, 0] = 1.0
        self._collective_thrust = torch.zeros(num_envs, 1, device=device, dtype=dtype)
        self._yawspeed_sp = torch.zeros(num_envs, 1, device=device, dtype=dtype)
        # measured-acceleration estimate (NED) via velocity finite difference
        self._prev_vel_ned = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._initialized = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)

    def update_position(
        self,
        pos_w,
        vel_w,
        quat_w,
        pos_sp_w,
        vel_sp_w,
        acc_sp_w,
        yaw_sp=None,
        yawspeed_sp=None,
        vel_dot_w=None,
    ):
        """Outer loop (ENU/FLU).  Stores thrust + desired attitude, returns thrust ``[N,1]``.

        All ``*_w`` are ENU world ``[N,3]`` (``quat_w`` is ``[N,4]`` w,x,y,z).
        ``yaw_sp`` ``[N,1]`` ENU (optional), ``yawspeed_sp`` ``[N,1]`` (optional),
        ``vel_dot_w`` measured ENU acceleration ``[N,3]`` (optional; finite-diff
        of the measured velocity if omitted).
        """
        # ENU → NED
        pos = enu_ned(pos_w)
        vel = enu_ned(vel_w)
        pos_sp = enu_ned(pos_sp_w)
        vel_sp = enu_ned(vel_sp_w)
        acc_sp = enu_ned(acc_sp_w)

        if vel_dot_w is not None:
            vel_dot = enu_ned(vel_dot_w)
        else:
            first = ~self._initialized
            vel_dot = torch.where(
                first, torch.zeros_like(vel), (vel - self._prev_vel_ned) / self.dt
            )
            self._prev_vel_ned = vel.clone()
            self._initialized[:] = True

        # yaw setpoint (ENU → NED). default held value if not provided.
        if yaw_sp is not None:
            self._yaw_sp_ned = math.pi / 2 - yaw_sp
        if yawspeed_sp is not None:
            # a yaw-rate about ENU-up is the opposite sign about NED-down
            self._yawspeed_sp = -yawspeed_sp

        thr_sp = self._pos.update(pos, vel, vel_dot, pos_sp, vel_sp, acc_sp)  # NED
        self._q_d_ned = _bodyz_to_attitude(thr_sp, self._yaw_sp_ned)
        self._collective_thrust = thr_sp.norm(dim=-1, keepdim=True)  # normalized
        return self._collective_thrust

    def update_attitude(self, quat_w: torch.Tensor) -> torch.Tensor:
        """Mid loop (ENU/FLU attitude in).  Returns FLU body-rate setpoint ``[N,3]``."""
        q_ned = quat_enu_flu_to_ned_frd(quat_w)
        rate_frd = self._att.update(q_ned, self._q_d_ned, self._yawspeed_sp)
        return flu_frd(rate_frd)  # FRD → FLU

    def compute(
        self,
        pos_w,
        vel_w,
        quat_w,
        pos_sp_w,
        vel_sp_w,
        acc_sp_w,
        yaw_sp=None,
        yawspeed_sp=None,
        vel_dot_w=None,
    ):
        """Run both loops; returns ``(collective_thrust [N,1], body_rate_flu [N,3])``."""
        thrust = self.update_position(
            pos_w, vel_w, quat_w, pos_sp_w, vel_sp_w, acc_sp_w, yaw_sp, yawspeed_sp, vel_dot_w
        )
        rate = self.update_attitude(quat_w)
        return thrust, rate

    @property
    def collective_thrust(self) -> torch.Tensor:
        return self._collective_thrust

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._pos.reset(env_ids)
        self._q_d_ned[env_ids] = 0.0
        self._q_d_ned[env_ids, 0] = 1.0
        self._collective_thrust[env_ids] = 0.0
        self._yawspeed_sp[env_ids] = 0.0
        self._prev_vel_ned[env_ids] = 0.0
        self._initialized[env_ids] = False
