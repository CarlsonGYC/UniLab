#!/usr/bin/env python
"""Train the MuJoCo cable-lift multilift policy with rsl_rl PPO (sandbox).

Uses the SAME rsl_rl 5.x stack UniLab trains with (via UniLab's
``normalize_ppo_train_cfg`` for schema compatibility). PPO hyper-parameters mirror
direct_rl's skrl config (MLP [256,256,128] elu, 24 rollouts, 5 epochs, 4 minibatches,
KL-adaptive lr, RunningStandardScaler == empirical_normalization).

    uv run python sandbox/multilift_mjcf/rl/train.py --num_envs 256 --iterations 1000
    uv run python sandbox/multilift_mjcf/rl/train.py --num_envs 8 --iterations 2 --smoke
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from multilift_rl_env import MultiliftCfg, MultiliftRLEnv  # noqa: E402
from multilift_velocity_rl_env import MultiliftVelocityCfg, MultiliftVelocityRLEnv  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from unilab.training.rsl_rl import normalize_ppo_train_cfg  # noqa: E402


def build_train_cfg(num_steps_per_env: int) -> dict:
    cfg = {
        "num_steps_per_env": num_steps_per_env,
        "save_interval": 200,
        "empirical_normalization": True,  # == skrl RunningStandardScaler
        "obs_groups": {"actor": ["obs"], "critic": ["obs"]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.37,  # exp(-1): skrl initial_log_std=-1
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms.PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 2.0,
            "entropy_coef": 0.005,
            "learning_rate": 1.0e-3,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
    }
    normalized = normalize_ppo_train_cfg(cfg)
    normalized.setdefault("multi_gpu", None)
    normalized.setdefault("check_for_nan", True)
    return normalized


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="hover", choices=["hover", "velocity"])
    p.add_argument("--num_envs", type=int, default=256)
    p.add_argument("--iterations", type=int, default=1500)
    p.add_argument("--num_steps_per_env", type=int, default=24)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--visual", default="simple", choices=["mesh", "simple"])
    p.add_argument("--log_dir", default=os.path.join(_HERE, "logs"))
    p.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = p.parse_args()
    if args.smoke:
        args.num_envs, args.iterations, args.num_steps_per_env = 8, 2, 8

    if args.task == "velocity":
        env = MultiliftVelocityRLEnv(
            MultiliftVelocityCfg(visual=args.visual), num_envs=args.num_envs, device=args.device
        )
    else:
        env = MultiliftRLEnv(
            MultiliftCfg(visual=args.visual), num_envs=args.num_envs, device=args.device
        )
    print(
        f"[train] num_envs={env.num_envs} obs={env.num_obs} act={env.num_actions} "
        f"device={args.device} | {args.iterations} iters x {args.num_steps_per_env} steps"
    )
    train_cfg = build_train_cfg(args.num_steps_per_env)
    log_dir = None if args.smoke else os.path.join(args.log_dir, f"multilift_{args.task}_ppo")
    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    print("[train] done")


if __name__ == "__main__":
    main()
