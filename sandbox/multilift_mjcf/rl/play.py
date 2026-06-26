#!/usr/bin/env python
"""Play a trained MuJoCo cable-lift multilift policy in the interactive viewer.

Loads an rsl_rl checkpoint, runs the policy (deterministic), and draws the payload
target (red) + per-drone formation setpoints (blue) into the viewer.

    uv run python sandbox/multilift_mjcf/rl/play.py --checkpoint sandbox/.../logs/multilift_ppo/model_1500.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "control_test"))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
from multilift_rl_env import MultiliftCfg, MultiliftRLEnv  # noqa: E402
from multilift_velocity_rl_env import MultiliftVelocityCfg, MultiliftVelocityRLEnv  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from train import build_train_cfg  # noqa: E402
from viz import draw_overlay  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default="hover", choices=["hover", "velocity"])
    p.add_argument("--num_envs", type=int, default=1)
    p.add_argument("--visual", default="mesh", choices=["mesh", "simple"])
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.task == "velocity":
        cfg = MultiliftVelocityCfg(visual=args.visual)
        cfg.curriculum_enabled = False
        env = MultiliftVelocityRLEnv(cfg, num_envs=args.num_envs, device=args.device)
    else:
        cfg = MultiliftCfg(visual=args.visual)
        cfg.curriculum_enabled = False  # play at full difficulty
        env = MultiliftRLEnv(cfg, num_envs=args.num_envs, device=args.device)
    runner = OnPolicyRunner(env, build_train_cfg(24), log_dir=None, device=args.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=args.device)
    print(f"[play] loaded {args.checkpoint} | {args.num_envs} env(s); close the window to exit.")

    obs = env.get_observations()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        wall0 = time.time()
        steps = 0
        while viewer.is_running():
            with torch.no_grad():
                act = policy(obs)
            obs, _, _, _ = env.step(act)
            tgt = env._target_pos_w[0].cpu().numpy()
            drone_tgt = (env._target_pos_w[0] + env._eq_offset[0]).cpu().numpy()
            draw_overlay(
                viewer.user_scn,
                tgt,
                0.0,
                [(tgt, 0.06, (0.9, 0.2, 0.1, 0.9))]
                + [(d, 0.04, (0.2, 0.45, 0.9, 0.9)) for d in drone_tgt],
            )
            viewer.sync()
            steps += 1
            lag = steps * env._ctrl_dt - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
