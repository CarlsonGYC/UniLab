#!/usr/bin/env python
"""Vectorised cable-lift multilift RL environment on MuJoCo (mujoco-uni) — sandbox port
of direct_rl ``multilift_env.py`` (Isaac DirectRLEnv) to UniLab's simulator.

``num_envs`` cable-lift formations live in ONE MuJoCo model (grid); a single
``mj_step`` advances them all (the parallelism). The controller stack + reward /
observation / termination / curriculum / payload mass-inertia DR are ported from
direct_rl; the simulator boundary uses ``xfrc_applied`` (body-frame net wrench, the
Isaac ``set_external_force_and_torque`` equivalent) and direct qpos/qvel resets.

Implements the rsl_rl ``VecEnv`` interface (get_observations -> TensorDict, step ->
(TensorDict, rew, done, extras)) so ``rl/train.py`` can train it with rsl_rl PPO.

Control / physics fidelity is direct_rl's (single rigid-body drones, charpi mass /
inertia, PX4 + CTBR params, decimation 5 @ 500 Hz). The cable is the stable
ball-jointed rigid chain verified in ``../passive_test.py``.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "control_test"))

import mujoco  # noqa: E402
from dynamics import (  # noqa: E402
    CTBRControlStack,
    DroneControlCfg,
    PX4ControlCfg,
    PX4PositionAttitudeController,
    hover_thrust_for_thrust,
)
from mj_drone import (  # noqa: E402
    BODY_BOTTOM_OFFSET,
    DRONE_MASS,
    LINK_TOTAL_LENGTH,
    G,
    build_multilift_grid,
    seat_drone_meshes,
)
from tensordict import TensorDict  # noqa: E402

SIM_DT = 1.0 / 500.0


@dataclass
class MultiliftCfg:
    # scene / time
    num_ropes: int = 4
    rope_length: float = 1.0
    payload_mass: float = 2.0
    payload_radius: float = 0.15
    elevation_angle: float = 40.0
    base_height: float = 1.5
    env_spacing: float = 6.0
    decimation: int = 5
    episode_length_s: float = 15.0
    visual: str = "simple"
    # action -> PVA setpoint scaling
    pos_range: float = 0.5
    v_max: float = 1.5
    a_max: float = 1.5  # jitter fix: accel cmd is direct thrust FF (was 3.0)
    action_filter_tau: float = 0.0
    # drone-side actuator DR (default OFF, ramped late by [cur_dr_start, cur_dr_full])
    randomize_actuators: bool = False
    actuator_thrust_range: float = 0.1
    actuator_max_speed_range: float = 0.3
    actuator_tau_range: float = 0.1
    cur_dr_start: float = 0.60
    cur_dr_full: float = 0.90
    # formation command
    diag_dist_min: float = 0.85
    diag_dist_max: float = 2.0
    w_formation: float = 1.0
    sigma_diag: float = 0.35  # diagnostics only
    sigma_form: float = 0.4  # per-drone symmetric-position match width (reward)
    target_yaw: float = 0.0
    # reward weights (current direct_rl source values)
    w_pos: float = 2.0
    sigma_pos: float = 0.3
    w_progress: float = 10.0  # potential-based: w*(prev_dist - curr_dist)
    w_att: float = 0.5
    sigma_att: float = 0.5
    k_pl_vel: float = 0.10
    k_pl_omega: float = 0.05
    k_taut: float = 2.0
    taut_margin: float = 0.05
    w_drone_upright: float = 0.05
    k_drone_omega: float = 0.005
    k_action: float = 0.002
    k_action_smooth: float = 0.02  # raised 4x to suppress jitter (was 0.005)
    k_action_smooth2: float = 0.01  # 2nd-order (jerk) smoothness: ||u_t - 2u_{t-1} + u_{t-2}||^2
    r_alive: float = 0.1
    crash_penalty: float = 10.0
    arrival_radius: float = 0.30
    arrival_speed: float = 0.5
    time_bonus_scale: float = 5.0
    proximity_dist: float = 0.6
    collision_dist: float = 0.4
    k_proximity: float = 100.0
    collision_terminate: bool = True
    # termination
    pos_max: float = 3.2
    payload_tilt_max_deg: float = 60.0
    drone_min_height: float = 0.2
    slack_terminate_steps: int = 25
    slack_terminate_margin: float = 0.25
    # curriculum (progress = step_counter / curriculum_steps)
    curriculum_enabled: bool = True
    curriculum_steps: float = 120000.0
    cur_target_start: float = 0.05
    cur_target_full: float = 0.50
    cur_formation_start: float = 0.05
    cur_formation_full: float = 0.35
    vel_acc_enable_frac: float = 0.40
    cur_perturb_start: float = 0.20
    cur_perturb_full: float = 0.65
    reset_pos_offset_max: float = 0.30
    reset_lin_vel_max: float = 0.30
    randomize_target: bool = True
    target_range_xy: float = 1.3
    target_range_z: float = 0.6
    zero_vel_acc_cmd: bool = True
    # payload mass / inertia DR (per-env, fixed once)
    randomize_payload_mass: bool = True
    payload_mass_range: float = 0.4
    randomize_payload_inertia: bool = True
    payload_inertia_range: float = 0.4
    obs_payload_params: bool = True
    obs_param_noise: float = 0.4
    hover_thrust: float = 0.0


def _matrix_from_quat(q: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
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


class MultiliftRLEnv:
    """rsl_rl-compatible vectorised cable-lift multilift env on MuJoCo."""

    def __init__(self, cfg: MultiliftCfg, num_envs: int = 64, device: str = "cpu"):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.nd = cfg.num_ropes
        self.rows = num_envs * self.nd
        self._sim_dt = SIM_DT
        self._ctrl_dt = SIM_DT * cfg.decimation
        self.max_episode_length = int(round(cfg.episode_length_s / self._ctrl_dt))

        # ── scene ──
        xml, metas = build_multilift_grid(
            num_envs,
            cfg.env_spacing,
            SIM_DT,
            num_ropes=cfg.num_ropes,
            rope_length=cfg.rope_length,
            payload_mass=cfg.payload_mass,
            payload_radius=cfg.payload_radius,
            elevation_angle_deg=cfg.elevation_angle,
            base_height=cfg.base_height,
            visual=cfg.visual,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        if cfg.visual == "mesh":
            seat_drone_meshes(self.model, bottom_z=0.0)
        self.data = mujoco.MjData(self.model)
        self._metas = metas

        def bid(name):
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

        self._payload_bids = np.array([bid(metas[e]["payload_name"]) for e in range(num_envs)])
        self._drone_bids = np.array(
            [[bid(n) for n in metas[e]["drone_names"]] for e in range(num_envs)]
        )
        self._drone_bids_flat = self._drone_bids.reshape(-1)
        self._all_bids = np.concatenate([self._drone_bids_flat, self._payload_bids])
        self._num_links = metas[0]["num_links"]

        # per-env qpos / qvel slices (bodies are added env-by-env, so each env is contiguous)
        self._qpos_slices, self._qvel_slices = self._compute_env_slices()
        mujoco.mj_forward(self.model, self.data)
        self._qpos0 = self.data.qpos.copy()
        self._payload_qadr = np.array(
            [self.model.jnt_qposadr[self.model.body_jntadr[b]] for b in self._payload_bids]
        )
        self._payload_vadr = np.array(
            [self.model.jnt_dofadr[self.model.body_jntadr[b]] for b in self._payload_bids]
        )

        # ── controllers (direct_rl params/rates) ──
        drone_cfg = DroneControlCfg()
        px4_cfg = PX4ControlCfg()
        px4_cfg.yaw_des = cfg.target_yaw
        support_mass = DRONE_MASS + cfg.payload_mass / self.nd
        self._hover_thrust = (
            cfg.hover_thrust if cfg.hover_thrust > 0 else hover_thrust_for_thrust(support_mass * G)
        )
        px4_cfg.position.hover_thrust = self._hover_thrust
        self.stack = CTBRControlStack(self.rows, dt=self._sim_dt, cfg=drone_cfg, device=device)
        self.ctrl = PX4PositionAttitudeController(
            self.rows, dt=self._ctrl_dt, cfg=px4_cfg, device=device
        )
        self.stack.reset()
        self.ctrl.reset()

        # ── geometry / command buffers ──
        nd = self.nd
        self._azimuths = torch.arange(nd, device=device, dtype=torch.float32) * (2 * math.pi / nd)
        self._payload_radius = torch.full((num_envs,), cfg.payload_radius, device=device)
        self._payload_half_h = self._payload_radius * (1.0 / 16.0)
        self._cable_full_len = self._num_links * LINK_TOTAL_LENGTH + BODY_BOTTOM_OFFSET
        cos_az, sin_az = torch.cos(self._azimuths)[None], torch.sin(self._azimuths)[None]
        self._rho_local = torch.stack(
            [
                self._payload_radius[:, None] * cos_az,
                self._payload_radius[:, None] * sin_az,
                self._payload_half_h[:, None].expand(num_envs, nd),
            ],
            dim=-1,
        )
        env_origins = torch.as_tensor(
            np.array([m["payload_pos"] for m in metas]), dtype=torch.float32, device=device
        )
        self._env_origin_xy = env_origins.clone()
        self._env_origin_xy[:, 2] = 0.0
        self._target_base = env_origins.clone()
        self._target_pos_w = self._target_base.clone()
        self._eq_local = (
            torch.tensor(
                np.stack([m["equilibrium"] for m in metas]), device=device, dtype=torch.float32
            )
            - env_origins[:, None, :]
        )  # local formation offsets
        self._spawn_diag = 2.0 * (
            self._payload_radius
            + self._cable_full_len * math.cos(math.radians(cfg.elevation_angle))
        )
        self._diag_cmd = self._spawn_diag.clone()
        self._eq_offset = self._formation_offset(
            self._diag_cmd, self._payload_radius, self._payload_half_h
        )
        zeros = torch.zeros(num_envs, device=device)
        self._target_quat = torch.stack(
            [
                torch.cos(torch.full_like(zeros, cfg.target_yaw) / 2),
                zeros,
                zeros,
                torch.sin(torch.full_like(zeros, cfg.target_yaw) / 2),
            ],
            -1,
        )
        self._cos_tilt = math.cos(math.radians(cfg.payload_tilt_max_deg))
        self._payload_default_mass = float(cfg.payload_mass)

        # ── runtime buffers ──
        self.num_actions = 9 * nd
        self.num_obs = 18 + 27 * nd + 1 + (2 if cfg.obs_payload_params else 0)
        self.obs_groups_spec = {"obs": self.num_obs}
        self._raw_action = torch.zeros(num_envs, self.num_actions, device=device)
        self._prev_action = torch.zeros_like(self._raw_action)
        self._prev_prev_action = torch.zeros_like(
            self._raw_action
        )  # u_{t-2} (2nd-order smoothness)
        self._reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._slack_counter = torch.zeros(num_envs, device=device)
        self._prev_dist = torch.zeros(
            num_envs, device=device
        )  # payload->target dist (progress reward)
        self._collided = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._pair_eye = torch.eye(nd, dtype=torch.bool, device=device)
        self._mass_obs_factor = torch.ones(num_envs, device=device)
        self._radius_obs_factor = torch.ones(num_envs, device=device)
        self._payload_mass_w = torch.full((num_envs,), cfg.payload_mass, device=device)
        self._raw_sat = torch.zeros((), device=device)
        self._va_enabled = not cfg.zero_vel_acc_cmd
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.step_counter = 0
        self.extras: dict = {}
        self._xfrc = self.data.xfrc_applied  # view

        self._randomize_payload(np.arange(num_envs))
        self.reset()

    # ── helpers ──────────────────────────────────────────────────────────────
    def _compute_env_slices(self):
        qslices, vslices = [], []
        nq, nv = self.model.nq, self.model.nv
        # env e owns qpos from its payload freejoint adr to the next env's payload adr
        padr = sorted(
            int(self.model.jnt_qposadr[self.model.body_jntadr[b]]) for b in self._payload_bids
        )
        vadr = sorted(
            int(self.model.jnt_dofadr[self.model.body_jntadr[b]]) for b in self._payload_bids
        )
        for e in range(self.num_envs):
            qs = padr[e]
            qe = padr[e + 1] if e + 1 < self.num_envs else nq
            vs = vadr[e]
            ve = vadr[e + 1] if e + 1 < self.num_envs else nv
            qslices.append((qs, qe))
            vslices.append((vs, ve))
        return qslices, vslices

    def _read_all(self):
        """Batched body states: returns drone (pos,quat,linw,omb)[N,nd,*] + payload (pos,quat,linw,omw)[N,*]."""
        pos = self.data.xpos[self._all_bids].copy()
        quat = self.data.xquat[self._all_bids].copy()
        n = len(self._all_bids)
        linw = np.empty((n, 3))
        ang = np.empty((n, 3))
        tmp = np.empty(6)
        for k, b in enumerate(self._all_bids):
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(b), tmp, 0
            )
            linw[k] = tmp[3:6]
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, int(b), tmp, 1
            )
            ang[k] = tmp[0:3]
        nd, N = self.nd, self.num_envs
        dp = torch.as_tensor(pos[: N * nd], dtype=torch.float32, device=self.device).view(N, nd, 3)
        dq = torch.as_tensor(quat[: N * nd], dtype=torch.float32, device=self.device).view(N, nd, 4)
        dl = torch.as_tensor(linw[: N * nd], dtype=torch.float32, device=self.device).view(N, nd, 3)
        do = torch.as_tensor(ang[: N * nd], dtype=torch.float32, device=self.device).view(N, nd, 3)
        pp = torch.as_tensor(pos[N * nd :], dtype=torch.float32, device=self.device)
        pq = torch.as_tensor(quat[N * nd :], dtype=torch.float32, device=self.device)
        pl = torch.as_tensor(linw[N * nd :], dtype=torch.float32, device=self.device)
        pw = torch.as_tensor(ang[N * nd :], dtype=torch.float32, device=self.device)
        return (dp, dq, dl, do), (pp, pq, pl, pw)

    def _attach_points(self, p_l, q_l):
        E, nd = self.num_envs, self.nd
        q = q_l[:, None, :].expand(E, nd, 4).reshape(-1, 4)
        rho = self._rho_local.reshape(-1, 3)
        R = _matrix_from_quat(q)
        return p_l[:, None, :] + torch.bmm(R, rho.unsqueeze(-1)).squeeze(-1).reshape(E, nd, 3)

    def _formation_offset(self, diag, r, half_h):
        R = 0.5 * diag
        span = R - r
        h = torch.sqrt((self._cable_full_len**2 - span**2).clamp(min=1e-4))
        off = torch.empty(diag.shape[0], self.nd, 3, device=self.device)
        off[..., 0] = R[:, None] * torch.cos(self._azimuths)[None, :]
        off[..., 1] = R[:, None] * torch.sin(self._azimuths)[None, :]
        off[..., 2] = half_h[:, None] + h[:, None]
        return off

    def _drone_pair_dist(self, p_d):
        dist = (p_d[:, :, None, :] - p_d[:, None, :, :]).norm(dim=-1)
        return dist.masked_fill(self._pair_eye, 1e6)

    def _progress(self):
        if not self.cfg.curriculum_enabled:
            return 1.0
        return min(1.0, self.step_counter / max(1.0, self.cfg.curriculum_steps))

    @staticmethod
    def _ramp(p, start, full):
        if full <= start:
            return 1.0 if p >= full else 0.0
        return min(1.0, max(0.0, (p - start) / (full - start)))

    def _randomize_payload(self, env_ids):
        c = self.cfg
        for e in env_ids:
            b = int(self._payload_bids[e])
            if c.randomize_payload_mass:
                s = float(np.random.uniform(1 - c.payload_mass_range, 1 + c.payload_mass_range))
                self.model.body_mass[b] = max(1e-6, self._payload_default_mass * s)
                self._payload_mass_w[e] = self.model.body_mass[b]
            if c.randomize_payload_inertia:
                s = float(
                    np.random.uniform(1 - c.payload_inertia_range, 1 + c.payload_inertia_range)
                )
                self.model.body_inertia[b] *= s

    # ── command hooks (overridden by the velocity-tracking subclass) ──────────
    def _reference_position(self) -> torch.Tensor:
        """Formation anchor (world). Hover: the static target. Velocity: the moving reference."""
        return self._target_pos_w

    def _velocity_feedforward(self):
        """Per-drone velocity feedforward [E,nd,3]. Hover: None. Velocity: the payload v_cmd."""
        return None

    # ── rsl_rl VecEnv API ─────────────────────────────────────────────────────
    def get_observations(self):
        return self._obs_td

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        env_ids = np.asarray(env_ids)
        c = self.cfg
        p = self._progress()
        perturb = self._ramp(p, c.cur_perturb_start, c.cur_perturb_full)
        target_s = self._ramp(p, c.cur_target_start, c.cur_target_full)
        form_s = self._ramp(p, c.cur_formation_start, c.cur_formation_full)

        for e in env_ids:
            qs, qe = self._qpos_slices[e]
            vs, ve = self._qvel_slices[e]
            self.data.qpos[qs:qe] = self._qpos0[qs:qe]
            self.data.qvel[vs:ve] = 0.0
            off = (np.random.rand(3) * 2 - 1) * (c.reset_pos_offset_max * perturb)
            vel = (np.random.rand(3) * 2 - 1) * (c.reset_lin_vel_max * perturb)
            pa = int(self._payload_qadr[e])
            va = int(self._payload_vadr[e])
            self.data.qpos[pa : pa + 3] += off
            self.data.qvel[va : va + 3] = vel
            # target point (curriculum-scaled), in world
            if c.randomize_target and target_s > 0:
                rng = np.array([c.target_range_xy, c.target_range_xy, c.target_range_z]) * target_s
                toff = (np.random.rand(3) * 2 - 1) * rng
                self._target_pos_w[e] = self._target_base[e] + torch.as_tensor(
                    toff, dtype=torch.float32, device=self.device
                )
            else:
                self._target_pos_w[e] = self._target_base[e]
        ei = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        # commanded diagonal distance (formation) widens with curriculum
        sd = self._spawn_diag[ei]
        lo = sd + (c.diag_dist_min - sd) * form_s
        hi = sd + (c.diag_dist_max - sd) * form_s
        diag = lo + (hi - lo) * torch.rand(len(env_ids), device=self.device)
        self._diag_cmd[ei] = diag
        self._eq_offset[ei] = self._formation_offset(
            diag, self._payload_radius[ei], self._payload_half_h[ei]
        )
        # reset controllers for these envs' rows
        row_ids = (
            ei.view(-1, 1) * self.nd + torch.arange(self.nd, device=self.device).view(1, -1)
        ).reshape(-1)
        self.stack.reset(row_ids)
        self.ctrl.reset(row_ids)
        if c.randomize_actuators:  # per-drone motor-model DR, ramped late by the DR curriculum
            dr = (
                self._ramp(self._progress(), c.cur_dr_start, c.cur_dr_full)
                if c.curriculum_enabled
                else 1.0
            )
            self.stack.randomize_actuators(
                row_ids,
                thrust_range=c.actuator_thrust_range * dr,
                max_speed_range=c.actuator_max_speed_range * dr,
                tau_range=c.actuator_tau_range * dr,
            )
        self._raw_action[ei] = 0.0
        self._prev_action[ei] = 0.0
        self._prev_prev_action[ei] = 0.0
        self._slack_counter[ei] = 0.0
        self._collided[ei] = False
        self.episode_length_buf[ei] = 0
        no = c.obs_param_noise
        if no > 0:
            self._mass_obs_factor[ei] = (
                1 + (torch.rand(len(env_ids), device=self.device) * 2 - 1) * no
            )
            self._radius_obs_factor[ei] = (
                1 + (torch.rand(len(env_ids), device=self.device) * 2 - 1) * no
            )
        # pre-latch arrival for envs reset already at target
        mujoco.mj_forward(self.model, self.data)
        (dp, dq, dl, do), (pp, pq, pl, pw) = self._read_all()
        init_err = (pp[ei] - self._target_pos_w[ei]).norm(dim=-1)
        self._reached[ei] = (init_err < c.arrival_radius) & (pl[ei].norm(dim=-1) < c.arrival_speed)
        self._prev_dist[ei] = init_err  # progress-reward baseline for the episode
        self._last_states = ((dp, dq, dl, do), (pp, pq, pl, pw))  # so subclasses can recompute obs
        self._obs_td = self._compute_obs((dp, dq, dl, do), (pp, pq, pl, pw))
        return self._obs_td, {}

    def step(self, actions: torch.Tensor):
        c = self.cfg
        actions = actions.to(self.device)
        self._collided[:] = False
        self._raw_sat = (actions.abs() > 1.0).float().mean()
        raw = actions.clamp(-1, 1)
        tau = c.action_filter_tau
        self._raw_action = tau * self._raw_action + (1 - tau) * raw if tau > 0 else raw
        a = self._raw_action.view(self.num_envs, self.nd, 9)
        # formation anchored to the reference position (hover: static target; velocity: moving ref)
        eq_world = self._reference_position()[:, None, :] + self._eq_offset
        p_cmd = eq_world + a[..., 0:3] * c.pos_range
        if c.curriculum_enabled:
            self._va_enabled = self._progress() >= c.vel_acc_enable_frac
        if self._va_enabled:
            v_cmd = a[..., 3:6] * c.v_max
            a_cmd = a[..., 6:9] * c.a_max
        else:
            v_cmd = torch.zeros_like(p_cmd)
            a_cmd = torch.zeros_like(p_cmd)
        ff = self._velocity_feedforward()  # hover: None; velocity task: the payload v_cmd
        if ff is not None:
            v_cmd = v_cmd + ff
        p_sp = p_cmd.reshape(self.rows, 3)
        v_sp = v_cmd.reshape(self.rows, 3)
        a_sp = a_cmd.reshape(self.rows, 3)

        (dp, dq, dl, do), _ = self._read_all()
        thrust = self.ctrl.update_position(
            dp.reshape(self.rows, 3),
            dl.reshape(self.rows, 3),
            dq.reshape(self.rows, 4),
            p_sp,
            v_sp,
            a_sp,
        )
        self.stack.set_collective_thrust(thrust)

        for _ in range(c.decimation):
            (dp, dq, dl, do), _ = self._read_all()
            q = dq.reshape(self.rows, 4)
            omb = do.reshape(self.rows, 3)
            lin = dl.reshape(self.rows, 3)
            self.stack.set_body_rate(self.ctrl.update_attitude(q))
            F, M = self.stack.compute_wrench(omb, lin, q)
            F = torch.nan_to_num(F.view(self.num_envs, self.nd, 3)).cpu().numpy()
            M = torch.nan_to_num(M.view(self.num_envs, self.nd, 3)).cpu().numpy()
            # body-frame wrench -> world xfrc at each drone COM
            quat_np = dq.cpu().numpy().reshape(-1, 4)
            R = np.zeros(9)
            for k, b in enumerate(self._drone_bids_flat):
                mujoco.mju_quat2Mat(R, quat_np[k])
                Rm = R.reshape(3, 3)
                self._xfrc[b, 0:3] = Rm @ F.reshape(-1, 3)[k]
                self._xfrc[b, 3:6] = Rm @ M.reshape(-1, 3)[k]
            mujoco.mj_step(self.model, self.data)
            if self.nd >= 2 and c.collision_terminate:
                p_d = torch.as_tensor(
                    self.data.xpos[self._drone_bids_flat], dtype=torch.float32, device=self.device
                ).view(self.num_envs, self.nd, 3)
                self._collided |= (self._drone_pair_dist(p_d) < c.collision_dist).any(-1).any(-1)

        self.step_counter += 1
        self.episode_length_buf += 1
        (dp, dq, dl, do), (pp, pq, pl, pw) = self._read_all()
        # dones BEFORE rewards (direct_rl order: crash penalty reflects the current step)
        terminated, timeout = self._compute_done((dp, dq, dl, do), (pp, pq, pl, pw))
        rew = self._compute_reward((dp, dq, dl, do), (pp, pq, pl, pw))
        dones = terminated | timeout
        self._prev_prev_action = self._prev_action.clone()
        self._prev_action = self._raw_action.clone()

        done_idx = torch.nonzero(dones, as_tuple=False).flatten().cpu().numpy()
        extras = {"time_outs": timeout, "log": self.extras.get("log", {})}
        if len(done_idx):
            self.reset(done_idx)  # autoreset; refreshes self._obs_td rows
        else:
            self._obs_td = self._compute_obs((dp, dq, dl, do), (pp, pq, pl, pw))
        return self._obs_td, rew, dones, extras

    # ── obs / reward / done (ported from direct_rl) ───────────────────────────
    def _compute_obs(self, drone, payload):
        E, nd, c = self.num_envs, self.nd, self.cfg
        dp, dq, dl, do = drone
        pp, pq, pl, pw = payload
        R_l = _matrix_from_quat(pq)
        payload_obs = torch.cat([pp - self._target_pos_w, R_l.reshape(E, 9), pl, pw], dim=-1)
        attach = self._attach_points(pp, pq)
        R_d = _matrix_from_quat(dq.reshape(-1, 4)).reshape(E, nd, 9)
        drone_obs = torch.cat([dp - attach, R_d, dl, do, self._prev_action.view(E, nd, 9)], dim=-1)
        extra = [self._diag_cmd[:, None]]
        if c.obs_payload_params:
            extra += [
                (self._payload_mass_w * self._mass_obs_factor / c.payload_mass)[:, None],
                (self._payload_radius * self._radius_obs_factor / c.payload_radius)[:, None],
            ]
        obs = torch.cat([payload_obs, drone_obs.reshape(E, nd * 27)] + extra, dim=-1)
        obs = torch.nan_to_num(obs)
        return TensorDict({"obs": obs}, batch_size=self.num_envs, device=self.device)

    def _compute_reward(self, drone, payload):
        c, E, nd = self.cfg, self.num_envs, self.nd
        dp, dq, dl, do = drone
        pp, pq, pl, pw = payload
        pos_err2 = (pp - self._target_pos_w).pow(2).sum(-1)
        r_pos = torch.exp(-pos_err2 / c.sigma_pos**2)
        # potential-based progress reward (telescoping; dense at all ranges)
        curr_dist = pos_err2.sqrt()
        r_progress = c.w_progress * (self._prev_dist - curr_dist)
        self._prev_dist = curr_dist
        q_err = _quat_mul(
            torch.stack(
                [
                    self._target_quat[:, 0],
                    -self._target_quat[:, 1],
                    -self._target_quat[:, 2],
                    -self._target_quat[:, 3],
                ],
                -1,
            ),
            pq,
        )
        ang = 2.0 * torch.acos(q_err[:, 0].abs().clamp(max=1.0))
        r_att = torch.exp(-ang.pow(2) / c.sigma_att**2)
        r_pl_vel = -c.k_pl_vel * pl.pow(2).sum(-1)
        r_pl_omega = -c.k_pl_omega * pw.pow(2).sum(-1)
        attach = self._attach_points(pp, pq)
        d = (dp - attach).norm(dim=-1)
        slack = (self._cable_full_len - c.taut_margin - d).clamp(min=0.0)
        r_taut = -c.k_taut * slack.pow(2).sum(-1)
        # symmetric per-drone formation (relative to payload): pins the WHOLE configuration,
        # so the policy can't satisfy it with a twisted/asymmetric formation.
        rel = dp - pp[:, None, :]
        form_err2 = (rel - self._eq_offset).pow(2).sum(-1).mean(-1)
        r_formation = c.w_formation * (torch.exp(-form_err2 / c.sigma_form**2) - 1.0)
        half = nd // 2  # diagnostics only (no longer used by the reward)
        d_diag = (dp[:, :half, :] - dp[:, half:, :]).norm(dim=-1)
        diag_err2 = (d_diag - self._diag_cmd[:, None]).pow(2).mean(-1)
        R_d = _matrix_from_quat(dq.reshape(-1, 4)).reshape(E, nd, 3, 3)
        r_drone_up = R_d[..., 2, 2].sum(-1)
        r_drone_om = -c.k_drone_omega * do.pow(2).sum(-1).sum(-1)
        r_action = -c.k_action * self._raw_action.pow(2).sum(-1)
        r_smooth = -c.k_action_smooth * (self._raw_action - self._prev_action).pow(2).sum(-1)
        r_smooth2 = -c.k_action_smooth2 * (
            self._raw_action - 2.0 * self._prev_action + self._prev_prev_action
        ).pow(2).sum(-1)
        if nd >= 2:
            pdist = self._drone_pair_dist(dp)
            band = c.proximity_dist - c.collision_dist
            deficit = (c.proximity_dist - pdist).clamp(min=0.0, max=band)
            r_prox = -c.k_proximity * 0.5 * deficit.pow(2).sum(dim=(-1, -2))
            min_dd = pdist.amin(dim=(-1, -2))
        else:
            r_prox = torch.zeros(E, device=self.device)
            min_dd = torch.full((E,), float("nan"), device=self.device)
        arrived = (pos_err2.sqrt() < c.arrival_radius) & (pl.norm(dim=-1) < c.arrival_speed)
        newly = arrived & (~self._reached)
        time_frac = (1.0 - self.episode_length_buf.float() / self.max_episode_length).clamp(min=0.0)
        r_time = c.time_bonus_scale * time_frac * newly.float()
        self._reached |= arrived
        reward = (
            c.w_pos * r_pos
            + r_progress
            + c.w_att * r_att
            + r_pl_vel
            + r_pl_omega
            + r_taut
            + r_formation
            + c.w_drone_upright * r_drone_up
            + r_drone_om
            + r_action
            + r_smooth
            + r_smooth2
            + r_prox
            + r_time
            + c.r_alive
        )
        reward = (
            reward - c.crash_penalty * self._last_terminated.float()
            if hasattr(self, "_last_terminated")
            else reward
        )
        self.extras["log"] = {
            "metrics/payload_pos_err_m": pos_err2.sqrt().mean(),
            "metrics/cable_d_mean_m": d.mean(),
            "metrics/diag_err_m": diag_err2.sqrt().mean(),
            "metrics/min_drone_dist_m": min_dd.mean(),
            "metrics/reached_frac": self._reached.float().mean(),
            "metrics/action_saturation": self._raw_sat,
            "curriculum/progress": torch.tensor(self._progress(), device=self.device),
        }
        return torch.nan_to_num(reward)

    def _compute_done(self, drone, payload):
        c, E, nd = self.cfg, self.num_envs, self.nd
        dp, dq, dl, do = drone
        pp, pq, pl, pw = payload
        finite = (
            torch.isfinite(pp).all(-1)
            & torch.isfinite(pq).all(-1)
            & torch.isfinite(dp).reshape(E, -1).all(-1)
        )
        diverged = (pp - self._target_pos_w).norm(dim=-1) > c.pos_max
        R_l = _matrix_from_quat(pq)
        tumbled = R_l[:, 2, 2] < self._cos_tilt
        R_d = _matrix_from_quat(dq.reshape(-1, 4)).reshape(E, nd, 3, 3)
        drone_low = (dp[..., 2] < c.drone_min_height).any(-1)
        drone_flip = (R_d[..., 2, 2] < 0.0).any(-1)
        d = (dp - self._attach_points(pp, pq)).norm(dim=-1)
        any_slack = (d < (self._cable_full_len - c.slack_terminate_margin)).any(-1)
        self._slack_counter = torch.where(
            any_slack, self._slack_counter + 1, torch.zeros_like(self._slack_counter)
        )
        slack_term = self._slack_counter >= c.slack_terminate_steps
        if nd >= 2 and c.collision_terminate:
            collided = self._collided | (self._drone_pair_dist(dp) < c.collision_dist).any(-1).any(
                -1
            )
        else:
            collided = torch.zeros(E, dtype=torch.bool, device=self.device)
        terminated = (~finite) | diverged | tumbled | drone_low | drone_flip | slack_term | collided
        self._last_terminated = terminated
        timeout = self.episode_length_buf >= self.max_episode_length
        return terminated, timeout
