# Copyright (c) 2026, The UniLab Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Cable-suspended payload multilift, fixed-point hover (UniLab NpEnv task).

A single centralized PPO agent holds a slung payload at a fixed world pose with
``num_ropes`` drones, keeping every cable taut. The agent emits a per-drone PVA
setpoint; the PX4 position loop (control rate, ``apply_action``) → collective thrust,
the PX4 attitude + CTBR rate loop (sim rate, ``set_pre_step_control``) → a body-frame
net wrench, applied through per-drone COM site actuators (gear = the 6 unit wrench
directions). Cables are explicit ball-jointed rigid chains; the backend resolves the
coupling. Ported from direct_rl's Isaac DirectRLEnv onto UniLab's MuJoCo backend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch

from unilab.assets import ASSETS_ROOT_PATH
from unilab.base import registry
from unilab.base.backend import create_backend
from unilab.base.base import EnvCfg
from unilab.base.np_env import NpEnv, NpEnvState
from unilab.base.scene import SceneCfg
from unilab.dr.provider import DomainRandomizationProvider
from unilab.dr.types import (
    DomainRandomizationCapabilities,
    IntervalRandomizationPlan,
    ResetPlan,
    ResetRandomizationPayload,
)
from unilab.dtype_config import get_global_dtype

from .control import (
    CTBRControlStack,
    DroneControlCfg,
    PX4ControlCfg,
    PX4PositionAttitudeController,
    hover_thrust_for_thrust,
)

_NUM_ROPES = 4
_DRONE_NAMES = [f"drone{i}" for i in range(_NUM_ROPES)]
_PAYLOAD_NAME = "payload"
_ACTION_DIM = 9 * _NUM_ROPES  # PVA per drone
_OBS_DIM = 18 + 27 * _NUM_ROPES + 1 + 2  # payload 18 + per-drone 27 + diag + mass/radius = 129
_G = 9.80665
_LINK_TOTAL_LENGTH = 0.20
_BODY_BOTTOM_OFFSET = 0.012
_DRONE_MASS = 1.09


@dataclass
class MultiliftRewardConfig:
    """Reward weights (``scales``) + shaping scalars. Hydra ``reward:`` overrides these."""

    scales: dict[str, float] = field(
        default_factory=lambda: {
            "pos": 2.0,
            "progress": 10.0,
            "att": 0.5,
            "pl_vel": 0.10,
            "pl_omega": 0.05,
            "taut": 2.0,
            "formation": 1.0,
            "drone_upright": 0.05,
            "drone_omega": 0.005,
            "action": 0.02,
            "action_sat": 0.5,
            "smooth": 0.02,
            "smooth2": 0.01,
            "proximity": 100.0,
            "time_bonus": 5.0,
            "alive": 0.1,
        }
    )
    crash_penalty: float = 10.0
    sigma_pos: float = 0.3
    pos_deadband: float = 0.05
    sigma_att: float = 0.5
    sigma_form: float = 0.4
    taut_margin: float = 0.05
    arrival_radius: float = 0.30
    arrival_speed: float = 0.5
    proximity_dist: float = 0.6
    collision_dist: float = 0.4


@registry.envcfg("MultiliftHover")
@dataclass
class MultiliftHoverCfg(EnvCfg):
    scene: SceneCfg = field(
        default_factory=lambda: SceneCfg(
            model_file=str(ASSETS_ROOT_PATH / "robots" / "multilift" / "scene.xml")
        )
    )
    sim_dt: float = 1.0 / 500.0
    ctrl_dt: float = 5.0 / 500.0  # decimation 5 -> 100 Hz outer loop
    max_episode_seconds: float = 15.0
    base_name: str = _PAYLOAD_NAME
    # scene geometry (must match scene.xml)
    num_ropes: int = _NUM_ROPES
    rope_length: float = 1.0
    payload_mass: float = 2.0
    payload_radius: float = 0.15
    elevation_angle: float = 40.0
    base_height: float = 1.5
    # action -> PVA setpoint scaling
    pos_range: float = 0.5
    v_max: float = 0.8
    a_max: float = 0.8
    zero_vel_acc_cmd: bool = True
    action_filter_tau: float = 0.5
    # formation command
    diag_dist_min: float = 0.85
    diag_dist_max: float = 2.0
    target_yaw: float = 0.0
    # curriculum (progress = step_counter / curriculum_steps)
    curriculum_enabled: bool = True
    curriculum_steps: float = 250000.0
    cur_target_start: float = 0.05
    cur_target_full: float = 0.50
    cur_formation_start: float = 0.05
    cur_formation_full: float = 0.35
    vel_acc_enable_frac: float = 0.40
    cur_perturb_start: float = 0.20
    cur_perturb_full: float = 0.65
    reset_pos_offset_max: float = 0.30
    reset_lin_vel_max: float = 0.30
    target_range_xy: float = 1.3
    target_range_z: float = 0.6
    # termination
    pos_max: float = 3.2
    payload_tilt_max_deg: float = 60.0
    drone_min_height: float = 0.2
    slack_terminate_steps: int = 25
    slack_terminate_margin: float = 0.25
    obs_param_noise: float = 0.4
    # ── domain randomization (robustness, like direct_rl) ──
    randomize_payload_mass: bool = True
    payload_mass_range: float = 0.4  # payload mass ~ nominal * U(1-r, 1+r), resampled per reset
    # payload inertia tracks mass (I ∝ m for fixed geometry) so mass/inertia stay self-consistent;
    # an extra independent jitter (direct_rl's randomize_payload_inertia) widens robustness.
    randomize_payload_inertia: bool = True
    payload_inertia_range: float = (
        0.4  # extra inertia jitter ~ U(1-r, 1+r) on top of the mass-scaled inertia
    )
    randomize_actuators: bool = (
        True  # per-drone motor-model DR, ramped late by [cur_dr_start, cur_dr_full]
    )
    actuator_thrust_range: float = 0.1  # thrust_const * U(1 +/- r)
    actuator_max_speed_range: float = 0.3  # max_motor_speed * U(1 +/- r)
    actuator_tau_range: float = 0.1  # motor time constants * U(1 +/- r)
    # mid-episode payload push: random world-frame force pulse (one control step, then cleared) on the
    # payload every push_interval control-steps, ramped in late by [cur_dr_start, cur_dr_full]. Keep this
    # small: direct_rl used a 3 N pulse; stronger pushes dominated early tracking loss in MuJoCo.
    push_payload: bool = True
    push_interval: int = 200  # control steps between pushes (~2 s at 100 Hz)
    push_max_force: tuple[float, float, float] = (3.0, 3.0, 3.0)  # +/- N per axis
    cur_dr_start: float = 0.60
    cur_dr_full: float = 0.90
    reward_config: MultiliftRewardConfig = field(default_factory=MultiliftRewardConfig)

    def validate(self) -> None:
        super().validate()
        if self.num_ropes % 2 != 0:
            raise ValueError(f"num_ropes must be even (got {self.num_ropes})")
        if self.num_ropes != _NUM_ROPES:
            raise ValueError(
                f"scene.xml is authored for {_NUM_ROPES} ropes; regenerate for {self.num_ropes}"
            )


def _t(x) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)


def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(q, -1)
    two = 2.0 / (q * q).sum(-1)
    o = torch.stack(
        [
            1 - two * (j * j + k * k),
            two * (i * j - k * r),
            two * (i * k + j * r),
            two * (i * j + k * r),
            1 - two * (i * i + k * k),
            two * (j * k - i * r),
            two * (i * k - j * r),
            two * (j * k + i * r),
            1 - two * (i * i + j * j),
        ],
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        -1,
    )


@registry.env("MultiliftHover", sim_backend="mujoco")
class MultiliftHoverEnv(NpEnv):
    _cfg: MultiliftHoverCfg

    def __init__(
        self,
        cfg: MultiliftHoverCfg,
        num_envs: int = 1,
        backend_type: str = "mujoco",
        dr_provider: DomainRandomizationProvider | None = None,
    ) -> None:
        backend = create_backend(
            backend_type,
            cfg.scene,
            num_envs,
            cfg.sim_dt,
            base_name=cfg.base_name,
            add_body_sensors=True,
        )
        super().__init__(cfg, backend, num_envs)
        self._np_dtype = get_global_dtype()
        self.nd = cfg.num_ropes
        self.rows = num_envs * self.nd
        self._action_space = gym.spaces.Box(-1.0, 1.0, (_ACTION_DIM,), dtype=np.float32)

        # body ids (cold path)
        self._drone_ids = self._backend.get_body_ids(_DRONE_NAMES)
        self._payload_ids = self._backend.get_body_ids([_PAYLOAD_NAME])

        # controllers (direct_rl params/rates), torch-cpu
        drone_cfg = DroneControlCfg()
        px4_cfg = PX4ControlCfg()
        px4_cfg.yaw_des = cfg.target_yaw
        support_mass = _DRONE_MASS + cfg.payload_mass / self.nd
        px4_cfg.position.hover_thrust = hover_thrust_for_thrust(support_mass * _G)
        self.stack = CTBRControlStack(self.rows, dt=cfg.sim_dt, cfg=drone_cfg, device="cpu")
        self.ctrl = PX4PositionAttitudeController(
            self.rows, dt=cfg.ctrl_dt, cfg=px4_cfg, device="cpu"
        )
        self.stack.reset()
        self.ctrl.reset()
        self._backend.set_pre_step_control(self._pre_step_wrench)

        # formation geometry
        nd = self.nd
        self._azimuths = np.arange(nd) * (2 * math.pi / nd)
        self._payload_half_h = cfg.payload_radius / 16.0
        self._cable_full_len = (
            max(1, round(cfg.rope_length / _LINK_TOTAL_LENGTH)) * _LINK_TOTAL_LENGTH
            + _BODY_BOTTOM_OFFSET
        )
        self._rho_local = _t(
            np.stack(
                [
                    cfg.payload_radius * np.cos(self._azimuths),
                    cfg.payload_radius * np.sin(self._azimuths),
                    np.full(nd, self._payload_half_h),
                ],
                axis=-1,
            )
        )  # [nd,3]
        self._target_base = np.array([0.0, 0.0, cfg.base_height], dtype=np.float64)
        self._spawn_diag = 2.0 * (
            cfg.payload_radius + self._cable_full_len * math.cos(math.radians(cfg.elevation_angle))
        )
        self._cos_tilt = math.cos(math.radians(cfg.payload_tilt_max_deg))
        self._pair_eye = torch.eye(nd, dtype=torch.bool)
        # substep collision latch: accumulates over the control step's physics substeps so a fast
        # approach that dips below collision_dist between control-rate checks is not missed.
        self._collided = np.zeros(num_envs, dtype=bool)

        self._init_domain_randomization(dr_provider or MultiliftDRProvider())

    # ── properties ────────────────────────────────────────────────────────────
    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def obs_groups_spec(self) -> dict[str, int]:
        return {"obs": _OBS_DIM}

    # ── geometry helper (shared with the DR provider) ─────────────────────────
    def formation_offset(self, diag: np.ndarray) -> np.ndarray:
        """Drone-vs-payload offsets [n,nd,3] for the symmetric taut trapezoid of diagonal ``diag`` [n]."""
        r, half_h, L = self._cfg.payload_radius, self._payload_half_h, self._cable_full_len
        R = 0.5 * np.asarray(diag)  # [n]
        h = np.sqrt(np.clip(L * L - (R - r) ** 2, 1e-4, None))  # [n]
        off = np.zeros((R.shape[0], self.nd, 3), dtype=np.float64)
        off[..., 0] = R[:, None] * np.cos(self._azimuths)[None, :]
        off[..., 1] = R[:, None] * np.sin(self._azimuths)[None, :]
        off[..., 2] = half_h + h[:, None]
        return off

    def _curriculum_progress(self) -> float:
        if not self._cfg.curriculum_enabled:
            return 0.0
        return min(1.0, self.step_counter / max(1.0, self._cfg.curriculum_steps))

    @staticmethod
    def _ramp(p, start, full):
        if full <= start:
            return 1.0 if p >= full else 0.0
        return min(1.0, max(0.0, (p - start) / (full - start)))

    def _reset_controllers(self, env_ids: np.ndarray) -> None:
        cfg = self._cfg
        rows = (np.asarray(env_ids)[:, None] * self.nd + np.arange(self.nd)[None, :]).reshape(-1)
        rows_t = torch.as_tensor(rows, dtype=torch.long)
        self.stack.reset(rows_t)
        self.ctrl.reset(rows_t)
        if cfg.randomize_actuators:  # per-drone motor-model DR, ramped late by the DR curriculum
            dr = (
                self._ramp(self._curriculum_progress(), cfg.cur_dr_start, cfg.cur_dr_full)
                if cfg.curriculum_enabled
                else 1.0
            )
            self.stack.randomize_actuators(
                rows_t,
                thrust_range=cfg.actuator_thrust_range * dr,
                max_speed_range=cfg.actuator_max_speed_range * dr,
                tau_range=cfg.actuator_tau_range * dr,
            )

    def _attach_points(self, p_l: torch.Tensor, q_l: torch.Tensor) -> torch.Tensor:
        E, nd = self._num_envs, self.nd
        q = q_l[:, None, :].expand(E, nd, 4).reshape(-1, 4)
        rho = self._rho_local[None].expand(E, nd, 3).reshape(-1, 3)
        R = _matrix_from_quat(q)
        return p_l[:, None, :] + torch.bmm(R, rho.unsqueeze(-1)).squeeze(-1).reshape(E, nd, 3)

    # ── control: position loop (control rate) ─────────────────────────────────
    def apply_action(self, actions: np.ndarray, state: NpEnvState) -> np.ndarray:
        cfg = self._cfg
        info = state.info
        self._collided[:] = False  # reset substep collision latch
        preclip = np.asarray(actions, dtype=np.float32).reshape(self._num_envs, _ACTION_DIM)
        clipped = np.clip(preclip, -1.0, 1.0)
        prev_effective = np.asarray(
            info.get("raw_action", np.zeros_like(clipped)), dtype=np.float32
        ).reshape(self._num_envs, _ACTION_DIM)
        tau = min(max(float(cfg.action_filter_tau), 0.0), 1.0)
        raw = tau * prev_effective + (1.0 - tau) * clipped if tau > 0.0 else clipped
        raw = np.clip(raw, -1.0, 1.0).astype(np.float32)
        info["preclip_action"] = preclip.astype(np.float32)
        info["raw_action"] = raw
        a = _t(raw).view(self._num_envs, self.nd, 9)
        eq_offset = _t(info["eq_offset"])  # [E,nd,3]
        target = _t(info["target_pos_w"])  # [E,3]
        eq_world = target[:, None, :] + eq_offset
        p_cmd = eq_world + a[..., 0:3] * cfg.pos_range
        if cfg.curriculum_enabled:
            va_enabled = self._curriculum_progress() >= cfg.vel_acc_enable_frac
        else:
            va_enabled = not cfg.zero_vel_acc_cmd
        info["va_enabled"] = np.full((self._num_envs,), float(va_enabled), dtype=np.float32)
        if va_enabled:
            v_cmd = a[..., 3:6] * cfg.v_max
            a_cmd = a[..., 6:9] * cfg.a_max
        else:
            v_cmd = torch.zeros_like(p_cmd)
            a_cmd = torch.zeros_like(p_cmd)
        p_sp = p_cmd.reshape(self.rows, 3)
        v_sp = v_cmd.reshape(self.rows, 3)
        a_sp = a_cmd.reshape(self.rows, 3)
        p = _t(self._backend.get_body_pos_w(self._drone_ids)).reshape(self.rows, 3)
        v = _t(self._backend.get_body_lin_vel_w(self._drone_ids)).reshape(self.rows, 3)
        q = _t(self._backend.get_body_quat_w(self._drone_ids)).reshape(self.rows, 4)
        thrust = self.ctrl.update_position(p, v, q, p_sp, v_sp, a_sp)
        self.stack.set_collective_thrust(thrust)
        return np.zeros((self._num_envs, self._backend.num_actuators), dtype=np.float32)

    # ── control: attitude + rate loop (sim rate, per substep) ─────────────────
    def _pre_step_wrench(self, backend, ctrl: np.ndarray) -> np.ndarray:
        q = _t(backend.get_body_quat_w(self._drone_ids)).reshape(self.rows, 4)
        omega_b = _t(backend.get_body_ang_vel_b(self._drone_ids)).reshape(self.rows, 3)
        lin = _t(backend.get_body_lin_vel_w(self._drone_ids)).reshape(self.rows, 3)
        # latch inter-drone collisions at sim rate (500 Hz) — a fast crossing can dip below
        # collision_dist between the 100 Hz control-rate checks (direct_rl substep latch).
        dp = _t(backend.get_body_pos_w(self._drone_ids)).view(self._num_envs, self.nd, 3)
        pd = (dp[:, :, None, :] - dp[:, None, :, :]).norm(dim=-1).masked_fill(self._pair_eye, 1e6)
        self._collided |= (pd < self._cfg.reward_config.collision_dist).any(-1).any(-1).numpy()
        self.stack.set_body_rate(self.ctrl.update_attitude(q))
        F, M = self.stack.compute_wrench(omega_b, lin, q)  # [rows,1,3] body frame
        F = torch.nan_to_num(F.view(self._num_envs, self.nd, 3))
        M = torch.nan_to_num(M.view(self._num_envs, self.nd, 3))
        wrench = torch.cat([F, M], dim=-1).reshape(
            self._num_envs, self.nd * 6
        )  # [E, 6*nd] = [Fx,Fy,Fz,Mx,My,Mz]*nd
        return wrench.numpy().astype(ctrl.dtype)

    # ── observation + reward + termination ────────────────────────────────────
    def update_state(self, state: NpEnvState) -> NpEnvState:
        cfg = self._cfg
        info = state.info
        E, nd = self._num_envs, self.nd
        pos, quat, linvel, angvel = self._backend.get_body_state_w(self._drone_ids)
        dp, dq, dl, do = (
            _t(pos),
            _t(quat),
            _t(linvel),
            _t(self._backend.get_body_ang_vel_b(self._drone_ids)),
        )
        pp_, pq_, pl_, pw_ = self._backend.get_body_state_w(self._payload_ids)
        pp, pq, pl, pw = _t(pp_)[:, 0], _t(pq_)[:, 0], _t(pl_)[:, 0], _t(pw_)[:, 0]

        target = _t(info["target_pos_w"])
        eq_offset = _t(info["eq_offset"])
        diag_cmd = _t(info["diag_cmd"])
        prev_action = _t(info["prev_action"])
        prev_prev_action = _t(info["prev_prev_action"])
        raw_action = _t(info["raw_action"])
        preclip_action = _t(info.get("preclip_action", info["raw_action"]))

        # ── observation ──
        R_l = _matrix_from_quat(pq)
        payload_obs = torch.cat([pp - target, R_l.reshape(E, 9), pl, pw], dim=-1)
        attach = self._attach_points(pp, pq)
        R_d = _matrix_from_quat(dq.reshape(-1, 4)).reshape(E, nd, 9)
        drone_obs = torch.cat([dp - attach, R_d, dl, do, prev_action.view(E, nd, 9)], dim=-1)
        mass_ratio = _t(info["mass_obs_factor"]).view(E, 1)
        obs = torch.cat(
            [
                payload_obs,
                drone_obs.reshape(E, nd * 27),
                diag_cmd.view(E, 1),
                mass_ratio,
                torch.ones(E, 1),
            ],
            dim=-1,
        )
        obs = torch.nan_to_num(obs).numpy().astype(self._np_dtype)

        # ── reward ──
        sc = cfg.reward_config.scales
        rc = cfg.reward_config
        pos_delta = pp - target
        pos_dist = pos_delta.norm(dim=-1)
        pos_reward_err = (pos_dist - rc.pos_deadband).clamp(min=0.0)
        r_pos = sc["pos"] * torch.exp(-pos_reward_err.pow(2) / rc.sigma_pos**2)
        prev_dist = _t(info["prev_dist"])
        curr_dist = pos_dist
        r_progress = sc["progress"] * (prev_dist - curr_dist)
        tq = torch.tensor([math.cos(cfg.target_yaw / 2), 0.0, 0.0, math.sin(cfg.target_yaw / 2)])
        q_err = _quat_mul(torch.stack([tq[0], -tq[1], -tq[2], -tq[3]]).expand(E, 4), pq)
        ang = 2.0 * torch.acos(q_err[:, 0].abs().clamp(max=1.0))
        r_att = sc["att"] * torch.exp(-ang.pow(2) / rc.sigma_att**2)
        r_pl_vel = -sc["pl_vel"] * pl.pow(2).sum(-1)
        r_pl_omega = -sc["pl_omega"] * pw.pow(2).sum(-1)
        d = (dp - attach).norm(dim=-1)
        slack = (self._cable_full_len - rc.taut_margin - d).clamp(min=0.0)
        r_taut = -sc["taut"] * slack.pow(2).sum(-1)
        rel = dp - pp[:, None, :]
        form_err2 = (rel - eq_offset).pow(2).sum(-1).mean(-1)
        r_formation = sc["formation"] * (torch.exp(-form_err2 / rc.sigma_form**2) - 1.0)
        r_drone_up = sc["drone_upright"] * R_d.reshape(E, nd, 3, 3)[..., 2, 2].sum(-1)
        r_drone_om = -sc["drone_omega"] * do.pow(2).sum(-1).sum(-1)
        r_action = -sc["action"] * raw_action.pow(2).sum(-1)
        action_overshoot = (preclip_action.abs() - 1.0).clamp(min=0.0).pow(2).sum(-1)
        r_action_sat = -sc.get("action_sat", 0.0) * action_overshoot
        r_smooth = -sc["smooth"] * (raw_action - prev_action).pow(2).sum(-1)
        # 2nd-order (jerk) smoothness: penalizes equal-and-opposite reversals (buzzing) the 1st
        # difference is blind to (direct_rl: r_smooth2).
        r_smooth2 = -sc.get("smooth2", 0.0) * (
            raw_action - 2.0 * prev_action + prev_prev_action
        ).pow(2).sum(-1)
        pdist = (
            (dp[:, :, None, :] - dp[:, None, :, :]).norm(dim=-1).masked_fill(self._pair_eye, 1e6)
        )
        band = rc.proximity_dist - rc.collision_dist
        deficit = (rc.proximity_dist - pdist).clamp(min=0.0, max=band)
        r_prox = -sc["proximity"] * 0.5 * deficit.pow(2).sum(dim=(-1, -2))
        reached = _t(info["reached"]).bool()
        arrived = (pos_dist < rc.arrival_radius) & (pl.norm(dim=-1) < rc.arrival_speed)
        newly = arrived & (~reached)
        steps = _t(info["steps"]).view(E) if "steps" in info else torch.zeros(E)
        time_frac = (1.0 - steps / max(1, self._max_episode_steps())).clamp(min=0.0)
        r_time = sc["time_bonus"] * time_frac * newly.float()
        reward_t = (
            r_pos
            + r_progress
            + r_att
            + r_pl_vel
            + r_pl_omega
            + r_taut
            + r_formation
            + r_drone_up
            + r_drone_om
            + r_action
            + r_action_sat
            + r_smooth
            + r_smooth2
            + r_prox
            + r_time
            + sc["alive"]
        )

        # ── termination ──
        finite = (
            torch.isfinite(pp).all(-1)
            & torch.isfinite(pq).all(-1)
            & torch.isfinite(dp).reshape(E, -1).all(-1)
        )
        diverged = pos_dist > cfg.pos_max
        tumbled = R_l[:, 2, 2] < self._cos_tilt
        R_d3 = R_d.reshape(E, nd, 3, 3)
        drone_low = (dp[..., 2] < cfg.drone_min_height).any(-1)
        drone_flip = (R_d3[..., 2, 2] < 0.0).any(-1)
        slack_cnt = _t(info["slack_counter"])
        any_slack = (d < (self._cable_full_len - cfg.slack_terminate_margin)).any(-1)
        slack_cnt = torch.where(any_slack, slack_cnt + 1, torch.zeros_like(slack_cnt))
        slack_term = slack_cnt >= cfg.slack_terminate_steps
        collided = _t(self._collided).bool() | (pdist < rc.collision_dist).any(-1).any(
            -1
        )  # substep latch | now
        terminated_t = (
            (~finite) | diverged | tumbled | drone_low | drone_flip | slack_term | collided
        )
        reward_t = reward_t - rc.crash_penalty * terminated_t.float()

        # ── persist episodic state ──
        info["prev_dist"] = curr_dist.numpy().astype(np.float32)
        info["reached"] = (reached | arrived).numpy()
        info["slack_counter"] = slack_cnt.numpy().astype(np.float32)
        info["prev_prev_action"] = prev_action.numpy().astype(np.float32)  # u_{t-2} <- u_{t-1}
        info["prev_action"] = raw_action.numpy().astype(np.float32)  # u_{t-1} <- u_t
        if nd >= 2:
            half = nd // 2
            d_diag = (dp[:, :half, :] - dp[:, half:, :]).norm(dim=-1)
            diag_err2 = (d_diag - diag_cmd[:, None]).pow(2).mean(-1)
        else:
            diag_err2 = torch.zeros(E)
        info["log"] = {
            "metrics/payload_pos_err_m": float(pos_dist.mean()),
            "metrics/target_offset_m": float((target - _t(self._target_base)).norm(dim=-1).mean()),
            "metrics/cable_d_mean_m": float(d.mean()),
            "metrics/formation_err_m": float(form_err2.sqrt().mean()),
            "metrics/diag_err_m": float(diag_err2.sqrt().mean()),
            "metrics/min_drone_dist_m": float(pdist.amin(dim=(-1, -2)).mean()),
            "metrics/reached_frac": float((reached | arrived).float().mean()),
            "metrics/action_jitter": float(
                (raw_action - 2.0 * prev_action + prev_prev_action).norm(dim=-1).mean()
            ),
            "metrics/action_saturation": float((preclip_action.abs() > 1.0).float().mean()),
            "metrics/action_clip_frac": float((raw_action.abs() >= 0.99).float().mean()),
            "metrics/action_overshoot": float(action_overshoot.mean()),
            "metrics/action_norm": float(raw_action.norm(dim=-1).mean()),
            "metrics/collision_frac": float(collided.float().mean()),  # envs colliding this step
            "curriculum/progress": float(self._curriculum_progress()),
            "curriculum/va_enabled": float(np.mean(info.get("va_enabled", 0.0))),
            "Episode_Reward/pos": float(r_pos.mean()),
            "Episode_Reward/action": float(r_action.mean()),
            "Episode_Reward/action_sat": float(r_action_sat.mean()),
            "Episode_Reward/smooth": float(r_smooth.mean()),
            "Episode_Reward/smooth2": float(r_smooth2.mean()),
        }
        reward = torch.nan_to_num(reward_t).numpy().astype(self._np_dtype)
        terminated = terminated_t.numpy()
        return state.replace(obs={"obs": obs}, reward=reward, terminated=terminated)

    def _max_episode_steps(self) -> int:
        return int(round(self._cfg.max_episode_seconds / self._cfg.ctrl_dt))

    # autoreset / explicit reset: also reset the controller integrators for the reset envs
    def reset(self, env_indices: np.ndarray):
        obs, info = super().reset(env_indices)
        self._reset_controllers(env_indices)
        self._collided[np.asarray(env_indices)] = False
        return obs, info


class MultiliftDRProvider(DomainRandomizationProvider):
    """Resets to the taut formation (= default qpos) + curriculum perturbation/target; seeds info."""

    def validate(self, env, capabilities: DomainRandomizationCapabilities) -> None:
        if env._cfg.push_payload and not capabilities.supports_interval_push:
            raise NotImplementedError(
                f"{env._backend.backend_type} backend does not support interval payload push; "
                "set env.push_payload=false"
            )

    def build_interval_randomization_plan(
        self, env: MultiliftHoverEnv, step_counter: int
    ) -> IntervalRandomizationPlan | None:
        # random force pulse on the payload base every push_interval control-steps, ramped late by DR
        cfg = env._cfg
        if not cfg.push_payload or step_counter <= 0 or step_counter % cfg.push_interval != 0:
            return None
        ramp = (
            env._ramp(env._curriculum_progress(), cfg.cur_dr_start, cfg.cur_dr_full)
            if cfg.curriculum_enabled
            else 1.0
        )
        if ramp <= 0.0:
            return None
        force = np.asarray(cfg.push_max_force, dtype=np.float64) * ramp
        return IntervalRandomizationPlan(push_perturbation_limit=force)

    def _build_episode_info(
        self,
        env: MultiliftHoverEnv,
        env_ids: np.ndarray,
        payload_xyz: np.ndarray,
        mass_scale: np.ndarray,
    ) -> dict:
        cfg = env._cfg
        n = int(env_ids.shape[0])
        p = env._curriculum_progress()
        form_s = env._ramp(p, cfg.cur_formation_start, cfg.cur_formation_full)
        target_s = env._ramp(p, cfg.cur_target_start, cfg.cur_target_full)
        # target point (curriculum-scaled) around the nominal hover
        rng = np.array([cfg.target_range_xy, cfg.target_range_xy, cfg.target_range_z]) * target_s
        toff = (np.random.rand(n, 3) * 2 - 1) * rng
        target = env._target_base[None, :] + toff  # [n,3]
        # commanded diagonal distance widens from the spawn diagonal to [min,max]
        lo = env._spawn_diag + (cfg.diag_dist_min - env._spawn_diag) * form_s
        hi = env._spawn_diag + (cfg.diag_dist_max - env._spawn_diag) * form_s
        diag = lo + (hi - lo) * np.random.rand(n)
        eq_offset = env.formation_offset(diag)  # [n,nd,3]
        init_err = np.linalg.norm(payload_xyz - target, axis=-1)
        no = cfg.obs_param_noise
        return {
            "target_pos_w": target.astype(np.float32),
            "diag_cmd": diag.astype(np.float32),
            "eq_offset": eq_offset.astype(np.float32),
            "prev_action": np.zeros((n, _ACTION_DIM), np.float32),
            "prev_prev_action": np.zeros((n, _ACTION_DIM), np.float32),
            "preclip_action": np.zeros((n, _ACTION_DIM), np.float32),
            "raw_action": np.zeros((n, _ACTION_DIM), np.float32),
            "va_enabled": np.zeros((n,), np.float32),
            "prev_dist": init_err.astype(np.float32),
            "reached": (init_err < cfg.reward_config.arrival_radius),
            "slack_counter": np.zeros((n,), np.float32),
            # observed payload-mass ratio = realized (DR) ratio * per-episode obs noise
            "mass_obs_factor": (mass_scale * (1.0 + (np.random.rand(n) * 2 - 1) * no)).astype(
                np.float32
            ),
        }

    def build_reset_plan(self, env: MultiliftHoverEnv, env_ids: np.ndarray) -> ResetPlan:
        cfg = env._cfg
        n = int(env_ids.shape[0])
        default_qpos = np.asarray(env._backend.get_default_qpos(), dtype=np.float64)
        qpos = np.broadcast_to(default_qpos, (n, default_qpos.shape[0])).copy()
        init_qvel = np.asarray(env._backend.get_init_qvel(), dtype=np.float64)
        qvel = np.broadcast_to(init_qvel, (n, init_qvel.shape[0])).copy()
        # curriculum: rigid offset+velocity of the whole (taut) formation (payload free joint = qpos[0:3])
        p = env._curriculum_progress()
        perturb = env._ramp(p, cfg.cur_perturb_start, cfg.cur_perturb_full)
        off = (np.random.rand(n, 3) * 2 - 1) * (cfg.reset_pos_offset_max * perturb)
        vel = (np.random.rand(n, 3) * 2 - 1) * (cfg.reset_lin_vel_max * perturb)
        qpos[:, 0:3] += off
        qvel[:, 0:3] = vel
        # payload MASS+INERTIA DR: scale ONLY the payload body. Inertia tracks mass (I ∝ m for a
        # fixed-geometry payload) so mass/inertia stay self-consistent; randomize_payload_inertia
        # adds an independent jitter on top for robustness (direct_rl parity).
        randomization = None
        mass_scale = np.ones(n)
        if cfg.randomize_payload_mass or cfg.randomize_payload_inertia:
            pbid = int(env._payload_ids[0])
            base_mass = np.asarray(env._backend.get_body_mass(), dtype=np.float64)  # (nbody,)
            base_inertia = np.asarray(
                env._backend.get_body_inertia(), dtype=np.float64
            )  # (nbody, 3)
            body_mass = np.broadcast_to(base_mass, (n, base_mass.shape[0])).copy()
            body_inertia = np.broadcast_to(
                base_inertia, (n, *base_inertia.shape)
            ).copy()  # (n, nbody, 3)
            if cfg.randomize_payload_mass:
                mass_scale = np.random.uniform(
                    1.0 - cfg.payload_mass_range, 1.0 + cfg.payload_mass_range, n
                )
                body_mass[:, pbid] = base_mass[pbid] * mass_scale
                body_inertia[:, pbid, :] = base_inertia[pbid] * mass_scale[:, None]  # I ∝ m
            if cfg.randomize_payload_inertia:
                jit = np.random.uniform(
                    1.0 - cfg.payload_inertia_range, 1.0 + cfg.payload_inertia_range, n
                )
                body_inertia[:, pbid, :] *= jit[:, None]
            randomization = ResetRandomizationPayload(
                body_mass=body_mass.astype(np.float32), body_inertia=body_inertia.astype(np.float32)
            )
        info_updates = self._build_episode_info(env, env_ids, qpos[:, 0:3].copy(), mass_scale)
        env._reset_controllers(env_ids)
        return ResetPlan(
            env_ids=env_ids,
            qpos=qpos,
            qvel=qvel,
            info_updates=info_updates,
            randomization=randomization,
        )

    def build_reset_observation(
        self, env: MultiliftHoverEnv, env_ids: np.ndarray, info_updates: dict
    ) -> dict:
        # build the SAME obs layout as update_state for the reset envs, from fresh body state
        E, nd = env._num_envs, env.nd
        pos, quat, linvel, _ = env._backend.get_body_state_w(env._drone_ids)
        do = _t(env._backend.get_body_ang_vel_b(env._drone_ids))
        dp, dq, dl = _t(pos), _t(quat), _t(linvel)
        pp_, pq_, pl_, pw_ = env._backend.get_body_state_w(env._payload_ids)
        pp, pq, pl, pw = _t(pp_)[:, 0], _t(pq_)[:, 0], _t(pl_)[:, 0], _t(pw_)[:, 0]
        target = torch.zeros(E, 3)
        target[env_ids] = _t(info_updates["target_pos_w"])
        eq = torch.zeros(E, nd, 3)
        eq[env_ids] = _t(info_updates["eq_offset"])
        diag = torch.zeros(E)
        diag[env_ids] = _t(info_updates["diag_cmd"])
        mass = torch.ones(E)
        mass[env_ids] = _t(info_updates["mass_obs_factor"])
        R_l = _matrix_from_quat(pq)
        payload_obs = torch.cat([pp - target, R_l.reshape(E, 9), pl, pw], dim=-1)
        attach = env._attach_points(pp, pq)
        R_d = _matrix_from_quat(dq.reshape(-1, 4)).reshape(E, nd, 9)
        drone_obs = torch.cat([dp - attach, R_d, dl, do, torch.zeros(E, nd, 9)], dim=-1)
        obs = torch.cat(
            [
                payload_obs,
                drone_obs.reshape(E, nd * 27),
                diag.view(E, 1),
                mass.view(E, 1),
                torch.ones(E, 1),
            ],
            dim=-1,
        )
        obs = torch.nan_to_num(obs).numpy().astype(get_global_dtype())
        return {"obs": obs[env_ids]}
