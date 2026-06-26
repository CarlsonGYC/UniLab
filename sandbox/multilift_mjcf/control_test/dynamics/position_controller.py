# Copyright (c) 2026, The direct_rl Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Butterworth-based outer position controller.

Models the *closed-loop position-tracking response* of the outer loop as a
2nd-order Butterworth low-pass on the position setpoint (the lag/bandwidth you
would identify from real trajectory-tracking data: a position-tracking time
constant ``tau`` maps to a cutoff ``f_c = 1 / (2*pi*tau)`` Hz). The filtered
reference is then turned into a CTBR action for :class:`CTBRControlStack` via a
PID position/velocity law + a geometric (SO(3)) attitude controller:

    p_des --Butterworth(f_c)--> p_ref
    a_des = Kp (p_ref - p) + Ki ∫(p_ref - p) + Kv (v_ref - v)        (v_ref = d/dt p_ref)
    F_des = m_drone * a_des + m_support * g e3                       (world)
    T = F_des · (R e3);  z_b_des = F_des/||F_des||;  R_des from (z_b_des, yaw)
    omega_des = -K_att * vee(½(R_desᵀR - RᵀR_des))
    action = [ thrust→throttle(T), omega_des / max_body_rate ]  ∈ [-1,1]^4

The drone stays FORCE-DRIVEN (the action feeds the CTBR stack → body wrench →
PhysX), so cable tension still acts on it — unlike directly writing root
velocity. ``f_c`` (and the gains) are the natural DR knobs: randomise per env.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import quat_math as math_utils
from .filter.low_pass_filter import ButterworthFilter


@dataclass
class PositionControllerCfg:
    gravity: float = 9.81
    drone_mass: float = 1.09  # inertial term m·a_des
    support_mass: float = 1.09  # gravity feed-forward m·g (raise to include payload share)
    kp_pos: tuple = (6.0, 6.0, 8.0)
    ki_pos: tuple = (1.0, 1.0, 2.0)
    kv_pos: tuple = (4.0, 4.0, 5.0)
    i_clamp: float = 2.0  # anti-windup on the position integral [m·s]
    k_att: tuple = (12.0, 12.0, 4.0)  # attitude P → body-rate setpoint
    cutoff_hz: float = 2.0  # Butterworth cutoff = 1/(2*pi*tau_position)
    yaw_des: float = 0.0


class ButterworthPositionController:
    """Outer position loop for ``num_envs`` drones (each drone = one batch row)."""

    def __init__(
        self,
        num_envs: int,
        dt: float,
        stack,
        cfg: PositionControllerCfg | None = None,
        device="cpu",
        dtype=torch.float32,
    ):
        self.cfg = cfg or PositionControllerCfg()
        self.num_envs = num_envs
        self.dt = dt  # OUTER-loop dt (env-step dt = decimation * sim_dt)
        self.stack = stack  # CTBRControlStack: thrust inversion + body-rate scaling
        self.device = device
        self.dtype = dtype

        self._bw = ButterworthFilter(
            num_envs, dt=dt, cutoff_hz=self.cfg.cutoff_hz, device=device, dtype=dtype
        )
        self._prev_p_ref = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._int_p = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._initialized = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)

        self._g = torch.tensor([0.0, 0.0, self.cfg.gravity], device=device, dtype=dtype)
        self._kp = torch.tensor(self.cfg.kp_pos, device=device, dtype=dtype)
        self._ki = torch.tensor(self.cfg.ki_pos, device=device, dtype=dtype)
        self._kv = torch.tensor(self.cfg.kv_pos, device=device, dtype=dtype)
        self._katt = torch.tensor(self.cfg.k_att, device=device, dtype=dtype)
        yaw = self.cfg.yaw_des
        self._x_c = torch.tensor(
            [math.cos(yaw), math.sin(yaw), 0.0], device=device, dtype=dtype
        ).expand(num_envs, 3)

    @staticmethod
    def _vee(m: torch.Tensor) -> torch.Tensor:
        """Inverse-hat of a batch of skew matrices [N,3,3] -> [N,3]."""
        return torch.stack([m[:, 2, 1], m[:, 0, 2], m[:, 1, 0]], dim=-1)

    def compute(
        self, p_des: torch.Tensor, p: torch.Tensor, v: torch.Tensor, quat_w: torch.Tensor
    ) -> torch.Tensor:
        """p_des/p/v: [N,3] world; quat_w: [N,4] (w,x,y,z). Returns action [N,4] in [-1,1]."""
        c = self.cfg
        # Butterworth-filtered position reference (models outer-loop lag/bandwidth)
        p_ref = self._bw(p_des)
        # seed prev ref on first call so the FF velocity does not spike
        first = ~self._initialized.squeeze(-1)
        self._prev_p_ref[first] = p_ref[first]
        self._initialized[:] = True

        v_ref = (p_ref - self._prev_p_ref) / self.dt
        self._prev_p_ref = p_ref.clone()

        e_p = p_ref - p
        self._int_p = (self._int_p + e_p * self.dt).clamp(-c.i_clamp, c.i_clamp)
        e_v = v_ref - v
        a_des = self._kp * e_p + self._ki * self._int_p + self._kv * e_v  # [N,3] world

        # desired force (world): inertial term + gravity/support feed-forward
        F_des = c.drone_mass * a_des + c.support_mass * self._g  # [N,3]

        R = math_utils.matrix_from_quat(quat_w)  # [N,3,3], world_from_body (columns = body axes)
        z_b = R[:, :, 2]
        T = (F_des * z_b).sum(dim=-1, keepdim=True).clamp(min=0.1)  # collective thrust [N,1]

        Fn = F_des.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        z_b_des = F_des / Fn
        y_b_des = torch.cross(z_b_des, self._x_c, dim=-1)
        y_b_des = y_b_des / y_b_des.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        x_b_des = torch.cross(y_b_des, z_b_des, dim=-1)
        R_des = torch.stack([x_b_des, y_b_des, z_b_des], dim=-1)  # [N,3,3]

        err_mat = 0.5 * (torch.bmm(R_des.transpose(1, 2), R) - torch.bmm(R.transpose(1, 2), R_des))
        e_R = self._vee(err_mat)
        omega_des = -self._katt * e_R  # body-rate setpoint [N,3]

        thrust_action = self.stack.desired_thrust_to_action(T)  # [N,1]
        omega_action = torch.zeros(self.num_envs, 3, device=self.device, dtype=self.dtype)
        omega_action[:, 0:2] = omega_des[:, 0:2] / self.stack.cfg.max_body_rate_xy
        omega_action[:, 2] = omega_des[:, 2] / self.stack.cfg.max_body_rate_z
        return torch.cat([thrust_action, omega_action], dim=-1).clamp(-1.0, 1.0)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._bw.reset(env_ids)
        self._prev_p_ref[env_ids] = 0.0
        self._int_p[env_ids] = 0.0
        self._initialized[env_ids] = False
