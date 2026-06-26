#!/usr/bin/env python
"""Cable-lift multilift VELOCITY-COMMAND tracking — sandbox RL task.

Subclasses the hover env (``MultiliftRLEnv``) WITHOUT modifying it (only overrides the
command hooks + obs + reward), exactly like direct_rl's ``MultiliftVelocityEnv``.

Ported from direct_rl and then improved with quadruped-locomotion ideas
(``unilab.envs.locomotion`` go2/go1 joystick), since both tasks are "track a base/body
velocity command":

  * command is **[vx, vy, vz, vyaw]** — direct_rl tracked only linear velocity; here we
    add a **yaw-rate command** and a ``tracking_ang_vel``-style reward (locomotion).
  * the xy command is in the payload **heading (yaw) frame** (locomotion body-frame
    command): the yaw command integrates a yaw reference, the xy velocity rotates with
    it, and the payload's target attitude tracks that yaw -> curved trajectories.
  * a fraction of envs get a **zero (standing) command** -> hover, so one policy learns
    both station-keeping and tracking (locomotion ``rel_standing_envs``).
  * commands resample mid-episode and ramp with the curriculum.

Reward (added to the inherited pos-to-moving-reference + formation + attitude + tautness
+ collision terms; hover-only time-bonus and payload-velocity penalty are disabled in
the cfg):
    w_vel   * exp(-||v_l - v_world||^2 / sigma_vel^2)          (tracking_lin_vel)
    w_ang   * exp(-(w_yaw_l - vyaw)^2   / sigma_ang_vel^2)     (tracking_ang_vel)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from multilift_rl_env import MultiliftCfg, MultiliftRLEnv


@dataclass
class MultiliftVelocityCfg(MultiliftCfg):
    episode_length_s: float = 20.0  # longer for sustained trajectory tracking
    randomize_target: bool = False  # the "target" is the moving reference, not a sampled point
    w_progress: float = 0.0  # no progress-to-fixed-target (continuous tracking instead)
    # velocity command (payload), sampled per episode + resampled every vel_resample_s
    vel_cmd_range_xy: float = 1.0  # m/s, |vx|,|vy| at full curriculum
    vel_cmd_range_z: float = 0.4  # m/s, |vz|
    vel_cmd_range_yaw: float = 0.8  # rad/s, |vyaw|  (locomotion ang-vel command)
    vel_resample_s: float = 5.0  # resample period (>= episode -> per-episode)
    heading_frame_command: bool = (
        True  # xy command in the payload yaw frame (locomotion body-frame)
    )
    rel_standing_envs: float = 0.1  # fraction commanded to hover (locomotion standing envs)
    # tracking reward
    w_vel: float = 3.0  # PRIMARY linear-velocity tracking
    sigma_vel: float = 0.4  # m/s
    w_ang_vel: float = 1.5  # yaw-rate tracking
    sigma_ang_vel: float = 0.5  # rad/s
    # disable hover-only terms (the payload SHOULD move)
    time_bonus_scale: float = 0.0
    k_pl_vel: float = 0.0


class MultiliftVelocityRLEnv(MultiliftRLEnv):
    def __init__(self, cfg: MultiliftVelocityCfg, num_envs: int = 64, device: str = "cpu"):
        super().__init__(cfg, num_envs, device)  # obs computed with zero cmd (guarded below)
        self._cmd = torch.zeros(
            num_envs, 4, device=device
        )  # [vx, vy, vz, vyaw] (xy in heading frame)
        self._yaw_ref = torch.full((num_envs,), cfg.target_yaw, device=device)
        self._resample_steps = max(1, int(round(cfg.vel_resample_s / self._ctrl_dt)))
        self.num_obs += 4
        self.obs_groups_spec = {"obs": self.num_obs}
        self._resample_cmd(np.arange(num_envs))
        self.reset()

    # ── command sampling (curriculum-scaled, with standing envs) ──────────────
    def _resample_cmd(self, env_ids) -> None:
        c = self.cfg
        env_ids = np.asarray(env_ids)
        if env_ids.size == 0:
            return
        ei = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        scale = (
            self._ramp(self._progress(), c.cur_target_start, c.cur_target_full)
            if c.curriculum_enabled
            else 1.0
        )
        rng = (
            torch.tensor(
                [c.vel_cmd_range_xy, c.vel_cmd_range_xy, c.vel_cmd_range_z, c.vel_cmd_range_yaw],
                device=self.device,
            )
            * scale
        )
        cmd = (torch.rand(len(env_ids), 4, device=self.device) * 2 - 1) * rng
        standing = torch.rand(len(env_ids), device=self.device) < c.rel_standing_envs
        cmd[standing] = 0.0
        self._cmd[ei] = cmd

    def _world_lin_cmd(self) -> torch.Tensor:
        """Commanded payload linear velocity in WORLD frame [E,3] (xy rotated by yaw ref)."""
        vx, vy, vz = self._cmd[:, 0], self._cmd[:, 1], self._cmd[:, 2]
        if self.cfg.heading_frame_command:
            cs, sn = torch.cos(self._yaw_ref), torch.sin(self._yaw_ref)
            return torch.stack([cs * vx - sn * vy, sn * vx + cs * vy, vz], dim=-1)
        return torch.stack([vx, vy, vz], dim=-1)

    # ── command hooks (override the base) ─────────────────────────────────────
    def _reference_position(self) -> torch.Tensor:
        # mid-episode resample (locomotion-style)
        if self._resample_steps < self.max_episode_length:
            due = (self.episode_length_buf % self._resample_steps == 0) & (
                self.episode_length_buf > 0
            )
            if bool(due.any()):
                self._resample_cmd(due.nonzero(as_tuple=False).flatten().cpu().numpy())
        # integrate the command into the moving reference (trajectory) + yaw reference
        self._yaw_ref = self._yaw_ref + self._cmd[:, 3] * self._ctrl_dt
        self._target_pos_w = self._target_pos_w + self._world_lin_cmd() * self._ctrl_dt
        half = self._yaw_ref / 2
        self._target_quat = torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
            dim=-1,
        )
        return self._target_pos_w

    def _velocity_feedforward(self) -> torch.Tensor:
        return self._world_lin_cmd()[:, None, :].expand(self.num_envs, self.nd, 3)

    # ── observation: append the command ───────────────────────────────────────
    def _compute_obs(self, drone, payload):
        td = super()._compute_obs(drone, payload)
        cmd = getattr(self, "_cmd", torch.zeros(self.num_envs, 4, device=self.device))
        td["obs"] = torch.cat([td["obs"], cmd], dim=-1)
        return td

    # ── reward: add linear + angular velocity tracking ────────────────────────
    def _compute_reward(self, drone, payload):
        reward = super()._compute_reward(
            drone, payload
        )  # pos-to-moving-ref + formation + att + taut + ...
        c = self.cfg
        _, _, pl, pw = payload
        v_world = self._world_lin_cmd()
        v_err2 = (pl - v_world).pow(2).sum(-1)
        r_vel = c.w_vel * torch.exp(-v_err2 / c.sigma_vel**2)
        yaw_err2 = (pw[:, 2] - self._cmd[:, 3]).pow(2)
        r_ang = c.w_ang_vel * torch.exp(-yaw_err2 / c.sigma_ang_vel**2)
        self.extras["log"]["reward/vel_track"] = r_vel.mean()
        self.extras["log"]["reward/ang_track"] = r_ang.mean()
        self.extras["log"]["metrics/vel_err_mps"] = v_err2.sqrt().mean()
        self.extras["log"]["metrics/vel_cmd_mps"] = v_world.norm(dim=-1).mean()
        return torch.nan_to_num(reward + r_vel + r_ang)

    # ── reset: new command + reset the yaw reference ──────────────────────────
    def reset(self, env_ids=None):
        td, info = super().reset(env_ids)
        if not hasattr(self, "_cmd"):
            return td, info  # called during base __init__ (cmd not built yet)
        ids = np.arange(self.num_envs) if env_ids is None else np.asarray(env_ids)
        ei = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self._yaw_ref[ei] = self.cfg.target_yaw
        self._resample_cmd(ids)
        self._obs_td = self._compute_obs(*self._last_states)
        return self._obs_td, info
