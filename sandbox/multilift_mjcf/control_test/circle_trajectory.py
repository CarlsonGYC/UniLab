# Copyright (c) 2026, The direct_rl Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Trapezoidal-profile circle trajectory with analytic position/velocity/accel.

Ported from the real-flight test code
``realflight_ws/src/traj_test/src/traj_test_node.cpp``
(``calculate_theta/omega/alpha_at_time`` + ``generate_circular_trajectory``).

The angular speed follows a trapezoid — ramp up at constant angular acceleration,
hold at ``omega_max``, ramp down — so the whole motion starts and ends at rest
(smooth for the cable-lift payload). Position is the circle, velocity and
acceleration are its exact derivatives (tangential + centripetal), matching what
the real PX4 ``TrajectorySetpoint`` carried.

Conventions: ENU (z up); the circle lies in the x-y plane. Returns offsets
relative to the ``t = 0`` point so it can be added to any hover position /
per-drone formation equilibrium (the formation translates rigidly along the
circle, so velocity/acceleration are shared by every drone).
"""

from __future__ import annotations

import math

import torch


class CircleTrajectory:
    def __init__(
        self,
        radius: float,
        period: float,
        n_circles: int = 2,
        ramp_up: float = 4.0,
        ramp_down: float = 4.0,
        init_phase: float = 0.0,
        device="cpu",
        dtype=torch.float32,
    ):
        """
        Args:
            radius:     circle radius [m]               (realflight ``circle_radius``)
            period:     seconds per full circle [s]     (realflight ``circle_duration``)
            n_circles:  number of laps                  (realflight ``circle_times``)
            ramp_up/ramp_down: angular speed ramp times [s]
            init_phase: starting angle on the circle [rad] (realflight ``circle_init_phase``)
        """
        self.radius = radius
        self.init_phase = init_phase
        self.device = device
        self.dtype = dtype

        self.omega_max = 2.0 * math.pi / period
        self.t_up = ramp_up
        self.t_down = ramp_down
        self.alpha_up = self.omega_max / ramp_up if ramp_up > 0 else 0.0
        self.alpha_down = self.omega_max / ramp_down if ramp_down > 0 else 0.0

        theta_required = n_circles * 2.0 * math.pi
        theta_ramps = 0.5 * self.omega_max * ramp_up + 0.5 * self.omega_max * ramp_down
        theta_const = max(0.0, theta_required - theta_ramps)
        self.t_const = theta_const / self.omega_max if self.omega_max > 0 else 0.0
        self.total_time = self.t_up + self.t_const + self.t_down

        self._x0, self._y0 = self._circle_point(0.0)

    # ── trapezoidal angular profile (theta / omega / alpha at time t) ───────────
    def _theta(self, t: float) -> float:
        t_up, t_const, t_down = self.t_up, self.t_const, self.t_down
        a_up, a_dn, w = self.alpha_up, self.alpha_down, self.omega_max
        if t <= t_up:
            return 0.5 * a_up * t * t
        th_up = 0.5 * a_up * t_up * t_up
        if t <= t_up + t_const:
            return th_up + w * (t - t_up)
        th_down_start = th_up + w * t_const
        if t <= t_up + t_const + t_down:
            dt = t - (t_up + t_const)
            return th_down_start + w * dt - 0.5 * a_dn * dt * dt
        return th_down_start + w * t_down - 0.5 * a_dn * t_down * t_down

    def _omega(self, t: float) -> float:
        t_up, t_const, t_down = self.t_up, self.t_const, self.t_down
        if t <= t_up:
            return self.alpha_up * t
        if t <= t_up + t_const:
            return self.omega_max
        if t <= t_up + t_const + t_down:
            return max(0.0, self.omega_max - self.alpha_down * (t - (t_up + t_const)))
        return 0.0

    def _alpha(self, t: float) -> float:
        t_up, t_const, t_down = self.t_up, self.t_const, self.t_down
        if t <= t_up:
            return self.alpha_up
        if t <= t_up + t_const:
            return 0.0
        if t <= t_up + t_const + t_down:
            return -self.alpha_down
        return 0.0

    def _circle_point(self, theta_total: float):
        th = theta_total + self.init_phase
        return self.radius * math.cos(th), self.radius * math.sin(th)

    # ── public API ──────────────────────────────────────────────────────────────
    def offset_pva(self, t: float):
        """Returns ``(p_off [3], v [3], a [3])`` torch tensors on ``device``.

        ``p_off`` is the position **relative to the t=0 point** (so add it to the
        hover position / formation equilibrium). ``v`` and ``a`` are absolute.
        """
        theta, omega, alpha = self._theta(t), self._omega(t), self._alpha(t)
        th = theta + self.init_phase
        c, s = math.cos(th), math.sin(th)
        r = self.radius
        p_off = (r * c - self._x0, r * s - self._y0, 0.0)
        v_lin = omega * r
        v = (-v_lin * s, v_lin * c, 0.0)
        a = (-r * (alpha * s + omega * omega * c), r * (alpha * c - omega * omega * s), 0.0)

        def T(x):
            return torch.tensor(x, device=self.device, dtype=self.dtype)

        return T(p_off), T(v), T(a)

    def phase(self, t: float) -> str:
        if t <= self.t_up:
            return "RAMP-UP"
        if t <= self.t_up + self.t_const:
            return "CONSTANT"
        if t <= self.total_time:
            return "RAMP-DOWN"
        return "DONE"
