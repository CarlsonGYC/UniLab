# Copyright (c) 2026, The direct_rl Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Framework-agnostic CTBR (collective-thrust / body-rate) control stack.

This reproduces isaac_dreamer's ``BodyRateControlAction`` (process_actions +
apply_actions) WITHOUT the ManagerBasedRLEnv / Articulation / logger coupling,
so it can be called directly from a DirectRLEnv ``_apply_action`` or a standalone
script. The numerics and call order are identical to isaac_dreamer; only the
``log()`` calls and the ``set_external_force_and_torque`` write are removed (the
caller applies the returned wrench).

Pipeline (per the ported modules):
    action[-1,1]^4 --(per env step)--> IntegerTimeDelay --> scale to
        (throttle[min,max], body_rate)         = process_action()
    --(per sim step)--> BodyRateController(PID+Butterworth) --> norm_torque
    --> ControlAllocator(PX4 mixer) --> actuator_sp --> PX4 thrust curve --> omega_ref
    --> Motor(1st-order ZOH, accel clamp) --> omega_real
    --> AirDragEffect(thrust attenuation + drag + rolling moment)
    --> Allocation --> body wrench (F, M)      = compute_wrench()

The default config values are isaac_dreamer's ``FiveInDroneBodyRateControlActionCfg``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .actuators.air_drag_effect import AirDragEffect
from .actuators.allocation import Allocation
from .actuators.motor import Motor
from .base.time_delay import IntegerTimeDelay
from .low_level_control.body_rate_controller import BodyRateController
from .low_level_control.control_allocator import ControlAllocator


@dataclass
class DroneControlCfg:
    """CTBR stack parameters (isaac_dreamer FiveInDrone defaults)."""

    # ── allocation / motor (FiveInDroneControlActionCfg) ──
    arm_length: float = 0.14
    moment_const: tuple = (1.21e-2, 1.21e-2, 1.21e-2, 1.21e-2)
    thrust_const: tuple = (1.15e-6, 1.15e-6, 1.15e-6, 1.15e-6)
    max_motor_speed: float = 3442.0
    min_motor_speed: float = 341.75
    kl: float = 0.05
    taus_up: float = 0.03
    taus_down: float = 0.03
    init: tuple = (341.75, 341.75, 341.75, 341.75)
    max_rotor_acc: tuple = (50000.0, 50000.0, 50000.0, 50000.0)
    min_rotor_acc: tuple = (-50000.0, -50000.0, -50000.0, -50000.0)
    use_motor_model: bool = True

    # ── body-rate control + allocator + aero (FiveInDroneBodyRateControlActionCfg) ──
    rotor_pos_com: float = 0.06
    rolling_moment_const: float = 1e-6
    ctl_thrust_const: float = 1.0
    ctl_moment_const: float = 0.009
    rotor_drag_const: float = 8.065e-5
    max_v_para_z: float = 25.0
    max_body_rate_xy: float = 4.0
    max_body_rate_z: float = 3.0
    min_throttle: float = 0.15
    max_throttle: float = 0.4
    kp: tuple = (0.10, 0.15, 0.2)
    ki: tuple = (0.15, 0.15, 0.15)
    kd: tuple = (0.0018, 0.0024, 0.0024)
    kk: tuple = (1.0, 1.0, 1.0)
    k_ff: tuple = (0.0, 0.0, 0.0)
    cutoff_hz: tuple = (40.0, 30.0)
    action_delay_steps: int = 3


class CTBRControlStack:
    """CTBR inner loop for ``num_envs`` drones (treat each drone as a batch row)."""

    def __init__(
        self,
        num_envs: int,
        dt: float,
        cfg: DroneControlCfg | None = None,
        device="cpu",
        dtype=torch.float32,
    ):
        self.cfg = cfg or DroneControlCfg()
        c = self.cfg
        self.num_envs = num_envs
        self.dt = dt
        self.device = device
        self.dtype = dtype

        self._max_motor_speed = (
            torch.ones(num_envs, 1, device=device, dtype=dtype) * c.max_motor_speed
        )
        self._min_motor_speed = (
            torch.ones(num_envs, 1, device=device, dtype=dtype) * c.min_motor_speed
        )
        self._thrust_const = torch.cat(
            [torch.ones(num_envs, 1, device=device, dtype=dtype) * tc for tc in c.thrust_const],
            dim=1,
        )  # (N,4)

        self._allocation = Allocation(
            num_envs, c.arm_length, c.moment_const, device=device, dtype=dtype
        )
        self._motor = Motor(
            num_envs,
            c.taus_up,
            c.taus_down,
            c.init,
            c.max_rotor_acc,
            c.min_rotor_acc,
            dt=dt,
            use=c.use_motor_model,
            device=device,
            dtype=dtype,
        )
        self._body_rate_controller = BodyRateController(
            num_envs,
            dt=dt,
            kp=torch.tensor(c.kp, device=device, dtype=dtype),
            ki=torch.tensor(c.ki, device=device, dtype=dtype),
            kd=torch.tensor(c.kd, device=device, dtype=dtype),
            kk=torch.tensor(c.kk, device=device, dtype=dtype),
            k_ff=torch.tensor(c.k_ff, device=device, dtype=dtype),
            cutoff_hz=c.cutoff_hz,
            device=device,
            dtype=dtype,
        )
        self._ctl_allocator = ControlAllocator(
            num_envs,
            c.rotor_pos_com,
            c.ctl_thrust_const,
            c.ctl_moment_const,
            device=device,
            dtype=dtype,
        )
        self._air_drag = AirDragEffect(
            num_envs,
            c.rotor_drag_const,
            c.rolling_moment_const,
            c.max_v_para_z,
            device=device,
            dtype=dtype,
        )
        self._action_delay = IntegerTimeDelay(
            num_envs, input_dim=4, delay_steps=c.action_delay_steps, device=device, dtype=dtype
        )

        self._z_throttle = torch.zeros(num_envs, 1, device=device, dtype=dtype)
        self._body_rate_scaled = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        self._omega_real = torch.zeros(num_envs, 4, device=device, dtype=dtype)
        # thrust-column gain of the normalized PX4 allocator (≈1); used to invert
        # collective-thrust → norm-throttle in desired_thrust_to_action.
        self._thrust_col_gain = float(self._ctl_allocator._ctl_allocators[0, 0, 0].item())

        # nominal actuator values, for idempotent per-row domain randomization (scale off these).
        self._thrust_const_nom = self._thrust_const.clone()
        self._max_motor_speed_nom = self._max_motor_speed.clone()
        self._taus_up_nom = self._motor.taus_up.clone()
        self._taus_down_nom = self._motor.taus_down.clone()

    # ── per ENV step ──────────────────────────────────────────────────────────
    def process_action(self, action: torch.Tensor) -> None:
        """Delay + de-normalise the [-1,1]^4 CTBR action (== process_actions)."""
        c = self.cfg
        delayed = self._action_delay.step(action[:, :4].clamp(-1.0, 1.0))
        z = (delayed[:, 0:1] + 1.0) / 2.0
        self._z_throttle = c.min_throttle + z * (c.max_throttle - c.min_throttle)
        self._body_rate_scaled[:, 0:2] = c.max_body_rate_xy * delayed[:, 1:3]
        self._body_rate_scaled[:, 2] = c.max_body_rate_z * delayed[:, 3]

    # ── direct CTBR setpoint (PX4 pos/att path) ─────────────────────────────────
    def set_collective_thrust(self, norm_thrust: torch.Tensor) -> None:
        """Set the mixer's normalised collective thrust directly (N,1) in [0,1].

        Used by the PX4 position/attitude controller, which already outputs a
        physical normalised collective thrust — this bypasses the [-1,1] RL action
        remap done by :meth:`process_action`. Call once per CONTROL step.
        """
        self._z_throttle = norm_thrust.clamp(0.0, 1.0)

    def set_body_rate(self, body_rate: torch.Tensor) -> None:
        """Set the body-rate setpoint (N,3) in rad/s (FLU) feeding the rate PID.

        Used by the PX4 attitude controller. May be called per SIM step (faster
        than the position loop) for a cascaded-rate setup, mirroring PX4.
        """
        self._body_rate_scaled[:] = body_rate

    # ── per SIM step ──────────────────────────────────────────────────────────
    def compute_wrench(self, omega_b: torch.Tensor, lin_vel_w: torch.Tensor, quat_w: torch.Tensor):
        """Body-rate PID → mixer → motor → aero → wrench (== apply_actions).

        Returns (thrust [N,1,3], moment [N,1,3]) in the BODY frame, ready for
        ``RigidObject.set_external_force_and_torque(forces=thrust, torques=moment)``.
        """
        c = self.cfg
        norm_torque = self._body_rate_controller.compute(
            rate_ref=self._body_rate_scaled, rate=omega_b
        )
        actuator_sp = self._ctl_allocator.compute(
            norm_thrust=self._z_throttle, norm_torque=norm_torque
        )
        actuator_sp = actuator_sp.clamp(0.1, 1.0)
        omega_ref = (
            self._min_motor_speed
            + (self._max_motor_speed - self._min_motor_speed)
            * (c.kl * (actuator_sp - 0.1) ** 2 + (1 - c.kl) * (actuator_sp - 0.1)) ** 0.5
        )
        self._omega_real = self._motor.compute(omega_ref)

        self._air_drag.compute(
            linear_vel_w=lin_vel_w,
            root_quat_w=quat_w,
            omega=self._omega_real,
            wind_vel_w=torch.zeros_like(lin_vel_w),
            static_rotor_thrust=self._thrust_const * self._omega_real**2,
        )
        dyn_thrust = self._air_drag.dynamic_rotor_thrust
        drag_force_body = self._air_drag.air_drag_force_each_rotor.sum(
            dim=1, keepdim=True
        )  # (N,1,3)
        rolling = self._air_drag.rolling_moment  # (N,3)

        processed = self._allocation.compute(dyn_thrust)  # (N,4): [T, Mx, My, Mz]
        thrust = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=self.dtype)
        thrust[:, :, :] = drag_force_body
        thrust[:, 0, 2] = processed[:, 0]
        moment = torch.zeros(self.num_envs, 1, 3, device=self.device, dtype=self.dtype)
        moment[:, 0, :] = processed[:, 1:] + rolling
        return thrust, moment

    def desired_thrust_to_action(self, thrust_total: torch.Tensor) -> torch.Tensor:
        """Approximate inverse of the thrust path: desired total thrust [N] (N,1)
        → normalised action throttle channel in [-1,1].

        Inverts, in order: equal 4-rotor split → omega = sqrt(T/4/kt) → PX4 thrust
        curve → normalised allocator thrust column → throttle remap. The vertical
        position loop closes any residual error (thrust is open-loop here)."""
        c = self.cfg
        kt = self._thrust_const[:, 0:1]
        omega = torch.sqrt(torch.clamp(thrust_total / 4.0, min=0.0) / kt)
        omega = omega.clamp(self._min_motor_speed, self._max_motor_speed)
        r = (omega - self._min_motor_speed) / (
            self._max_motor_speed - self._min_motor_speed
        )  # [0,1]
        if c.kl > 0.0:
            y = (-(1 - c.kl) + torch.sqrt((1 - c.kl) ** 2 + 4 * c.kl * r**2)) / (2 * c.kl)
        else:
            y = r
        actuator_sp = y + 0.1
        norm_thrust = (actuator_sp / self._thrust_col_gain).clamp(0.0, 1.0)
        action = 2.0 * (norm_thrust - c.min_throttle) / (c.max_throttle - c.min_throttle) - 1.0
        return action.clamp(-1.0, 1.0)

    @property
    def omega_real(self) -> torch.Tensor:
        return self._omega_real

    def reset(self, env_ids=None) -> None:
        # Motor.reset / BodyRateController.reset index with len()/fancy indexing,
        # so pass an explicit index tensor (a slice would crash Motor.reset).
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._z_throttle[env_ids] = 0.0
        self._body_rate_scaled[env_ids] = 0.0
        self._motor.reset(env_ids)
        self._body_rate_controller.reset(env_ids)
        self._action_delay.reset(env_ids)

    def randomize_actuators(self, row_ids, thrust_range=0.0, max_speed_range=0.0, tau_range=0.0):
        """Per-row motor-model domain randomization (isaac_dreamer-style). Scales thrust_const,
        max_motor_speed, and motor time constants off their NOMINAL values by U(1-range, 1+range)
        for the selected rows (idempotent — repeated calls scale the nominal, not the live value)."""
        if row_ids is None:
            row_ids = torch.arange(self.num_envs, device=self.device)

        def _scale(rng):
            if rng <= 0.0:
                return 1.0
            return torch.empty(len(row_ids), 1, device=self.device, dtype=self.dtype).uniform_(
                1.0 - rng, 1.0 + rng
            )

        self._thrust_const[row_ids] = self._thrust_const_nom[row_ids] * _scale(thrust_range)
        self._max_motor_speed[row_ids] = self._max_motor_speed_nom[row_ids] * _scale(
            max_speed_range
        )
        ts = _scale(tau_range)
        self._motor.taus_up[row_ids] = self._taus_up_nom[row_ids] * ts
        self._motor.taus_down[row_ids] = self._taus_down_nom[row_ids] * ts
