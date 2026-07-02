# Copyright (c) 2026, The UniLab Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Multilift hover env invariants: obs layout, latency rings, held push, bounded penalties, DR."""

from __future__ import annotations

import numpy as np
import pytest

from unilab.base.registry import ensure_registries, make
from unilab.envs.manipulation.multilift.payload_lift import _OBS_DIM, MultiliftHoverEnv


@pytest.fixture(scope="module")
def env():
    ensure_registries()
    return make(
        "MultiliftHover",
        sim_backend="mujoco",
        num_envs=2,
        env_cfg_override={
            "curriculum_enabled": False,
            "zero_vel_acc_cmd": False,
            # deterministic sensors for the ring/obs tests; delay draws overridden per test
            "obs_noise_pos": 0.0,
            "obs_noise_vel": 0.0,
            "obs_noise_ang": 0.0,
            "obs_noise_att": 0.0,
            "obs_bias_pos": 0.0,
            "obs_bias_att": 0.0,
            "obs_bias_ang": 0.0,
        },
    )


def _fresh_state(env):
    state = env.init_state()
    state.info["steps"][:] = 0  # kill episode staggering so no truncation mid-test
    return state


def test_obs_dim_and_spec(env):
    state = _fresh_state(env)
    assert env.obs_groups_spec == {"obs": _OBS_DIM}
    assert state.obs["obs"].shape == (2, _OBS_DIM)
    assert np.isfinite(state.obs["obs"]).all()


def test_ring_read_math():
    ring = np.zeros((2, 4, 3), np.float32)
    for t in range(1, 7):  # writes at cursor (t % 4)
        ring[:, t % 4] = t
    # cursor now 6 % 4 = 2 (holds value 6); delay d must read value 6 - d
    out = MultiliftHoverEnv._ring_read(ring, 2, np.array([0, 3]))
    assert out[0, 0] == 6.0 and out[1, 0] == 3.0
    # delay is clamped to ring capacity - 1
    out = MultiliftHoverEnv._ring_read(ring, 2, np.array([9, 1]))
    assert out[0, 0] == 3.0 and out[1, 0] == 5.0


def test_action_transport_delay(env):
    state = _fresh_state(env)
    state.info["act_delay"] = np.array([0, 2], dtype=np.int64)
    sent = []
    for t in range(4):
        a = np.full((2, 36), 0.01 * (t + 1), np.float32)
        sent.append(a)
        state = env.step(a)
        preclip = state.info["preclip_action"]
        # env 0: no delay -> sees this step's action; env 1: 2-step delay (zeros until t=2)
        np.testing.assert_allclose(preclip[0], sent[t][0], atol=1e-6)
        expected1 = sent[t - 2][1] if t >= 2 else np.zeros(36, np.float32)
        np.testing.assert_allclose(preclip[1], expected1, atol=1e-6)


def test_va_scale_full_when_curriculum_off(env):
    state = _fresh_state(env)
    state = env.step(np.zeros((2, 36), np.float32))
    assert state.info["log"]["curriculum/va_scale"] == 1.0


def test_push_force_held_and_zeroed_on_reset(env):
    _fresh_state(env)
    for _ in range(30):
        env.step(np.zeros((2, 36), np.float32))
    force = env._push_force.copy()
    assert np.linalg.norm(force, axis=-1).max() > 0.0  # wind ramped up (curriculum off -> full)
    env.step(np.zeros((2, 36), np.float32))
    step_delta = np.abs(env._push_force - force).max()
    assert step_delta < 0.2  # low-passed: no per-step jumps toward a +/-1 N target
    env.reset(np.array([0]))
    assert np.all(env._push_force[0] == 0.0)
    assert np.all(env._push_countdown[0] == 0)


def test_action_sat_penalty_bounded(env):
    state = _fresh_state(env)
    state.info["act_delay"] = np.zeros(2, dtype=np.int64)
    state = env.step(np.full((2, 36), 100.0, np.float32))  # absurd raw output
    # unbounded quadratic would give ~ -0.5 * 36 * 99^2 ~= -1.8e5; the clip caps it near -10
    assert state.reward.min() > -100.0
    assert state.info["log"]["Episode_Reward/action_sat"] >= -0.5 * float(
        env._cfg.reward_config.sat_overshoot_clip
    )


def test_reset_plan_includes_com_offset(env):
    provider = env._dr_manager._provider
    plan = provider.build_reset_plan(env, np.array([0, 1]))
    payload = plan.randomization
    assert payload is not None and payload.body_ipos is not None
    base = env._backend.get_body_ipos()
    pbid = int(env._payload_ids[0])
    delta = payload.body_ipos - base[None]
    assert np.abs(delta[:, pbid, :]).max() > 0.0
    assert np.abs(delta[:, pbid, :]).max() <= env._cfg.payload_com_range + 1e-9
    other = np.delete(delta, pbid, axis=1)
    assert np.abs(other).max() < 1e-6  # float32 payload cast leaves ~1e-9 residue


def test_obs_corruption_channel_mask(env):
    # prev_action, diag/mass/const channels must never get noise or bias
    std = env._obs_channel_std(1.0, 2.0, 3.0, 4.0)
    assert std.shape == (_OBS_DIM,)
    nd = env.nd
    for i in range(nd):
        base = 18 + 30 * i
        assert np.all(std[base + 18 : base + 27] == 0.0)  # prev_action slice
        assert np.all(std[base + 27 : base + 30] == 1.0)  # slot error = position-type
    assert np.all(std[18 + 30 * nd :] == 0.0)  # diag_cmd, mass ratio, const
    assert np.all(std[0:3] == 1.0) and np.all(std[3:12] == 2.0)
    assert np.all(std[12:15] == 3.0) and np.all(std[15:18] == 4.0)
