#!/usr/bin/env python
"""Track-evaluation + visualization for the UniLab MultiliftHover policy.

Port of direct_rl's ``eval_track.py``: flies the slung payload through a sequence of
random target points (holding each ``hold_s`` s), records tracking / formation / jitter
/ actions / payload velocity / drone tilt, and renders two PNGs (tracking + detail) + an
NPZ + a console summary. ``--video`` also records an MP4 of the SAME rollout (via
UniLab's working playback renderer). The target is driven by writing
``env._state.info["target_pos_w"]`` each step (like direct_rl writing
``base._target_pos_w``).

Examples:

    # latest/final model_*.pt from a run directory
    uv run python sandbox/multilift_mjcf/rl/eval_track.py logs/rsl_rl_ppo/MultiliftHover/<run>

    # explicit training checkpoint
    uv run python sandbox/multilift_mjcf/rl/eval_track.py logs/.../<run>/model_1500.pt

    # exported TorchScript policy
    uv run python sandbox/multilift_mjcf/rl/eval_track.py logs/.../<run>/policy.pt
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from unilab.base.registry import ensure_registries, make  # noqa: E402
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg  # noqa: E402
from unilab.training.sim2sim import (  # noqa: E402
    CrossBackendIncompatibleError,
    policy_load_dim_guard,
)


def _default_train_cfg() -> dict:
    """Fallback architecture for legacy checkpoints without a saved ``run_config.json``."""
    cfg = {
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": True,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.37,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "unilab.algos.torch.caps_ppo:CapsPPO",
            "caps_lambda_s": 0.0,
            "caps_sigma": 0.05,
            "learning_rate": 1.0e-3,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "value_loss_coef": 2.0,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "gamma": 0.99,
            "lam": 0.95,
        },
    }
    out = normalize_ppo_train_cfg(cfg)
    out.setdefault("multi_gpu", None)
    out.setdefault("check_for_nan", True)
    return out


def _train_cfg(run_dir: Path | None) -> dict:
    """Load the saved PPO architecture from ``run_config.json`` when available."""
    if run_dir is not None:
        run_config = run_dir / "run_config.json"
        if run_config.is_file():
            with run_config.open("r") as f:
                saved = json.load(f)
            algo_cfg = saved.get("config", {}).get("algo")
            if isinstance(algo_cfg, dict):
                out = normalize_ppo_train_cfg(deepcopy(algo_cfg))
                out.setdefault("multi_gpu", None)
                out.setdefault("check_for_nan", True)
                return out
    return _default_train_cfg()


def _model_iteration(path: Path) -> int:
    if not path.name.startswith("model_") or path.suffix != ".pt":
        return -1
    try:
        return int(path.stem.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def _latest_model_checkpoint(run_dir: Path) -> Path | None:
    ckpts = [p for p in run_dir.glob("model_*.pt") if p.is_file()]
    return max(ckpts, key=_model_iteration) if ckpts else None


def _summary_checkpoint(run_dir: Path) -> Path | None:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.is_file():
        return None
    with summary_path.open("r") as f:
        summary = json.load(f)
    last = summary.get("last_checkpoint")
    if not last:
        return None
    path = Path(last)
    if not path.is_absolute():
        path = run_dir / path
    return path if path.is_file() else None


def _best_checkpoint(run_dir: Path) -> Path | None:
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        with summary_path.open("r") as f:
            summary = json.load(f)
        for key in ("best_checkpoint", "best_model_path"):
            value = summary.get(key)
            if value:
                path = Path(value)
                if not path.is_absolute():
                    path = run_dir / path
                if path.is_file():
                    return path
    for name in ("model_best.pt", "best_model.pt", "best.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _resolve_run_dir(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
        if path.is_file():
            return path.parent.resolve()
        if path.is_dir():
            return path.resolve()
        raise SystemExit(f"run path does not exist: {value}")
    runs = sorted(Path("logs/rsl_rl_ppo/MultiliftHover").glob("*_mujoco"))
    if not runs:
        raise SystemExit("no logs/rsl_rl_ppo/MultiliftHover/*_mujoco run directories found")
    return runs[-1].resolve()


def _resolve_checkpoint(args) -> tuple[Path, Path]:
    """Resolve a ``model_*.pt`` training checkpoint or exported ``policy.pt``."""
    source = getattr(args, "source", None)
    run_value = args.run
    checkpoint_value = args.checkpoint

    if source:
        source_path = Path(source).expanduser()
        if source_path.is_file():
            if checkpoint_value:
                raise SystemExit("Use either positional .pt source or --checkpoint, not both.")
            ckpt = source_path.resolve()
            return ckpt, ckpt.parent
        if source_path.is_dir():
            if run_value:
                raise SystemExit("Use either positional run directory or --run, not both.")
            run_value = source
        else:
            raise SystemExit(f"source path does not exist: {source}")

    run_dir = _resolve_run_dir(run_value)
    selected = str(checkpoint_value or "final")
    selected_lower = selected.lower()

    if selected_lower in {"final", "latest", "-1"}:
        ckpt = _summary_checkpoint(run_dir) or _latest_model_checkpoint(run_dir)
    elif selected_lower == "best":
        ckpt = _best_checkpoint(run_dir)
        if ckpt is None:
            raise SystemExit(
                f"no best checkpoint metadata/file found in {run_dir}; use --checkpoint final "
                "or pass an explicit .pt file"
            )
    elif selected_lower in {"policy", "policy.pt", "jit"}:
        ckpt = run_dir / "policy.pt"
    elif selected.isdigit():
        ckpt = run_dir / f"model_{selected}.pt"
    else:
        candidate = Path(selected).expanduser()
        if not candidate.exists():
            candidate = run_dir / selected
        ckpt = candidate

    if ckpt is None or not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt} (run_dir={run_dir})")
    return ckpt.resolve(), run_dir


def _is_rsl_checkpoint(path: Path) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*TorchScript archive.*",
                category=UserWarning,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except RuntimeError as exc:
        if "TorchScript" in str(exc) or "constants.pkl" in str(exc):
            return False
        raise
    return isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint


def _load_policy(
    checkpoint: Path,
    run_dir: Path,
    wrapped: RslRlVecEnvWrapper,
    device: str,
) -> tuple[Callable[[Any], torch.Tensor], str]:
    """Load either a training ``model_*.pt`` or an exported TorchScript ``policy.pt``."""
    if _is_rsl_checkpoint(checkpoint):
        train_cfg = _train_cfg(run_dir)
        train_cfg.setdefault("runner", {})["logger"] = "none"
        train_cfg["logger"] = "none"
        runner = OnPolicyRunner(wrapped, train_cfg, log_dir=None, device=device)
        with policy_load_dim_guard(
            env_obs_dim=getattr(wrapped, "num_obs", None),
            env_action_dim=getattr(wrapped, "num_actions", None),
            algo_name="ppo",
        ):
            runner.load(str(checkpoint), map_location=device)
        return runner.get_inference_policy(device=device), "rsl-rl checkpoint"

    module = torch.jit.load(str(checkpoint), map_location=device)
    module.eval()

    def policy(obs) -> torch.Tensor:
        return module(obs["policy"])

    return policy, "torchscript policy"


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "source",
        nargs="?",
        help="Run directory, training checkpoint model_*.pt, or exported TorchScript policy.pt.",
    )
    p.add_argument(
        "--run", default=None, help="Run directory. Ignored when positional source is a file."
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint selector within --run: final/latest, best, policy, an iteration number, "
            "a filename, or a full .pt path. Default: final."
        ),
    )
    p.add_argument("--num_points", type=int, default=5)
    p.add_argument("--hold_s", type=float, default=8.0)
    p.add_argument("--range_xy", type=float, default=None)
    p.add_argument("--range_z", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--video", action="store_true", help="also record an MP4 of the rollout")
    p.add_argument(
        "--eval-mode",
        default="clean",
        choices=["clean", "full"],
        help=(
            "clean: no DR/push/reset perturbation and curriculum progress near zero. "
            "full: disable reset target curriculum but keep velocity/acceleration action channels on; "
            "DR/push/reset perturbation stay off unless enabled."
        ),
    )
    p.add_argument(
        "--enable-dr",
        action="store_true",
        help="Enable payload mass/inertia and actuator randomization during eval.",
    )
    p.add_argument(
        "--enable-push",
        action="store_true",
        help="Enable interval payload push disturbance during eval.",
    )
    p.add_argument(
        "--reset-perturb",
        action="store_true",
        help="Keep reset position/velocity perturbation enabled in full eval mode.",
    )
    p.add_argument(
        "--autoreset",
        action="store_true",
        help="Autoreset done episodes instead of leaving a failed rollout in-place.",
    )
    p.add_argument(
        "--cam_distance",
        type=float,
        default=None,
        help="camera distance (default: auto-frame the trajectory)",
    )
    p.add_argument("--cam_elevation", type=float, default=-20.0)
    p.add_argument("--cam_azimuth", type=float, default=120.0)
    args = p.parse_args()

    ensure_registries()
    ckpt, run_dir = _resolve_checkpoint(args)
    override = {
        "randomize_payload_mass": bool(args.enable_dr),
        "randomize_payload_inertia": bool(args.enable_dr),
        "randomize_payload_com": bool(args.enable_dr),
        "randomize_actuators": bool(args.enable_dr),
        "push_payload": bool(args.enable_push),
        "obs_param_noise": 0.0,
    }
    if not args.enable_dr:  # deterministic sensors/latency for a clean tracking measurement
        override.update(
            {
                "action_delay_steps_max": 0,
                "obs_delay_steps_max": 0,
                "ctrl_fb_delay_steps_max": 0,
                "obs_noise_pos": 0.0,
                "obs_noise_vel": 0.0,
                "obs_noise_ang": 0.0,
                "obs_noise_att": 0.0,
                "obs_bias_pos": 0.0,
                "obs_bias_att": 0.0,
                "obs_bias_ang": 0.0,
            }
        )
    if args.eval_mode == "clean":
        override.update(
            {
                "curriculum_enabled": True,
                "curriculum_steps": 1.0e12,
                "zero_vel_acc_cmd": True,
                "reset_pos_offset_max": 0.0,
                "reset_lin_vel_max": 0.0,
                "push_payload": False,
            }
        )
    else:
        override["curriculum_enabled"] = False
        override["zero_vel_acc_cmd"] = False
        if args.reset_perturb:
            override["curriculum_enabled"] = True
            override["curriculum_steps"] = 1.0
        if not args.reset_perturb:
            override["reset_pos_offset_max"] = 0.0
            override["reset_lin_vel_max"] = 0.0
    env = make("MultiliftHover", sim_backend="mujoco", num_envs=1, env_cfg_override=override)
    eval_seconds = float(args.num_points) * float(args.hold_s)
    env._cfg.max_episode_seconds = max(
        float(env._cfg.max_episode_seconds), eval_seconds + max(1.0, float(env._cfg.ctrl_dt))
    )
    wrapped = RslRlVecEnvWrapper(env, device=args.device)
    try:
        policy, policy_kind = _load_policy(ckpt, run_dir, wrapped, args.device)
    except CrossBackendIncompatibleError as exc:
        raise SystemExit(str(exc)) from None
    print(f"[eval_track] checkpoint: {ckpt}")
    print(f"[eval_track] policy kind: {policy_kind}")
    print(f"[eval_track] eval mode: {args.eval_mode} override={override}")

    cfg = env._cfg
    nd = env.nd
    ctrl_dt = cfg.ctrl_dt
    steps_per_pt = int(round(args.hold_s / ctrl_dt))
    total_steps = steps_per_pt * args.num_points
    range_xy = args.range_xy if args.range_xy is not None else cfg.target_range_xy
    range_z = args.range_z if args.range_z is not None else cfg.target_range_z
    rng = np.random.default_rng(args.seed)
    offsets = (rng.random((args.num_points, 3)) * 2 - 1) * np.array([range_xy, range_xy, range_z])
    target_base = env._target_base.astype(np.float64)
    arrival_r = cfg.reward_config.arrival_radius
    print(
        f"[eval_track] {args.num_points} pts x {args.hold_s:g}s = {total_steps} steps "
        f"(dt={ctrl_dt * 1e3:.0f}ms){' +video' if args.video else ''}"
    )

    keys = (
        "t",
        "tx",
        "ty",
        "tz",
        "px",
        "py",
        "pz",
        "err",
        "form",
        "jitter",
        "pvx",
        "pvy",
        "pvz",
        "pspeed",
        "max_tilt",
        "done",
    )
    rec: dict[str, list] = {k: [] for k in keys}
    rec_actions: list[np.ndarray] = []
    rec_actions_clipped: list[np.ndarray] = []
    state = {"i": 0}

    def init():
        obs, _ = wrapped.reset()
        env.set_autoreset(bool(args.autoreset))
        state["i"] = 0
        for v in rec.values():
            v.clear()
        rec_actions.clear()
        rec_actions_clipped.clear()
        return obs

    def do_step(obs):
        pt = min(state["i"] // steps_per_pt, args.num_points - 1)
        target = target_base + offsets[pt]
        state["target"] = target  # for the video marker
        env._state.info["target_pos_w"][:] = target.astype(np.float32)  # drive the target
        action = policy(obs)
        obs2, _, done, extras = wrapped.step(action)
        pl = env._backend.get_body_pos_w(env._payload_ids)[0, 0]
        pv = env._backend.get_body_lin_vel_w(env._payload_ids)[0, 0]
        dq = env._backend.get_body_quat_w(env._drone_ids)[0]  # (nd,4) wxyz
        tilt = np.degrees(
            np.arccos(np.clip(1.0 - 2.0 * (dq[:, 1] ** 2 + dq[:, 2] ** 2), -1.0, 1.0))
        )
        log = extras.get("log", {})
        t = state["i"] * ctrl_dt
        for k, val in zip(("t", "tx", "ty", "tz", "px", "py", "pz"), (t, *target, *pl)):
            rec[k].append(float(val))
        rec["err"].append(float(np.linalg.norm(pl - target)))
        rec["form"].append(float(log.get("metrics/formation_err_m", np.nan)))
        rec["jitter"].append(float(log.get("metrics/action_jitter", np.nan)))
        rec["pvx"].append(float(pv[0]))
        rec["pvy"].append(float(pv[1]))
        rec["pvz"].append(float(pv[2]))
        rec["pspeed"].append(float(np.linalg.norm(pv)))
        rec["max_tilt"].append(float(tilt.max()))
        rec["done"].append(float(done.detach().cpu().numpy().reshape(-1)[0]))
        rec_actions.append(action.detach().cpu().numpy().reshape(-1))
        clipped = env._state.info.get("raw_action")
        if clipped is not None:
            rec_actions_clipped.append(np.asarray(clipped, dtype=np.float32).reshape(-1))
        state["i"] += 1
        return obs2

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(ckpt), f"eval_track_{args.eval_mode}_{ckpt.stem}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # camera framed on the whole trajectory (targets span +/- range_xy / range_z about the base),
    # looking at the formation center; + a red sphere marker drawn at the live target each frame.
    cam_distance = (
        args.cam_distance
        if args.cam_distance is not None
        else 3.2 + 1.6 * max(range_xy, 1.5 * range_z)
    )  # frame the +/- range trajectory, not too far
    camera_kwargs = {
        "cam_distance": cam_distance,
        "cam_elevation": args.cam_elevation,
        "cam_azimuth": args.cam_azimuth,
        "cam_lookat": [0.0, 0.0, cfg.base_height + 0.4],
    }  # center the formation (payload + drones above)
    state["target"] = target_base + offsets[0]

    def target_marker():
        return np.asarray(state["target"], dtype=np.float32).reshape(1, 3)  # (num_envs, 3)

    with torch.inference_mode():
        if args.video:
            env.run_playback_mode(
                play_render_mode="record",
                play_steps=total_steps,
                output_video=os.path.join(out_dir, "track_video.mp4"),
                initialize=init,
                step=do_step,
                render_spacing=float(getattr(cfg, "render_spacing", 8.0)),
                camera_kwargs=camera_kwargs,
                extra_data_getter=target_marker,
            )
        else:
            obs = init()
            for _ in range(total_steps):
                obs = do_step(obs)

    R = {k: np.asarray(v) for k, v in rec.items()}
    R["actions"] = np.asarray(rec_actions)
    if rec_actions_clipped:
        R["actions_clipped"] = np.asarray(rec_actions_clipped)
    R["checkpoint"] = np.asarray(str(ckpt))
    R["policy_kind"] = np.asarray(policy_kind)
    R["eval_mode"] = np.asarray(args.eval_mode)
    np.savez(os.path.join(out_dir, "track_data.npz"), **R)
    switch_times = [i * args.hold_s for i in range(args.num_points)]
    _plot_tracking(R, switch_times, arrival_r, os.path.join(out_dir, "track_curves.png"))
    _plot_detail(R, nd, switch_times, os.path.join(out_dir, "track_detail.png"))
    _summary(R, steps_per_pt, args.num_points, arrival_r)
    extra = " , track_video.mp4" if args.video else ""
    print(
        f"[eval_track] saved -> {out_dir}/track_curves.png , track_detail.png , track_data.npz{extra}"
    )


def _plot_tracking(R, switch_times, arrival_r, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    t = R["t"]
    ax[0, 0].plot(R["px"], R["py"], lw=1.0, color="tab:blue", label="payload")
    ax[0, 0].plot(R["tx"], R["ty"], "x", color="tab:red", ms=8, label="targets")
    ax[0, 0].set_title("XY trajectory")
    ax[0, 0].set_xlabel("x [m]")
    ax[0, 0].set_ylabel("y [m]")
    ax[0, 0].axis("equal")
    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)
    ax[0, 1].plot(t, R["err"], color="tab:blue")
    ax[0, 1].axhline(arrival_r, color="g", ls="--", lw=1, label=f"arrival {arrival_r:g}m")
    for s in switch_times:
        ax[0, 1].axvline(s, color="0.7", lw=0.8)
    ax[0, 1].set_title("payload position error")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("err [m]")
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)
    for k, c in zip("xyz", ("tab:red", "tab:green", "tab:blue")):
        ax[1, 0].plot(t, R["p" + k], color=c, lw=1.0, label=k)
        ax[1, 0].plot(t, R["t" + k], color=c, ls="--", lw=1.0, alpha=0.7)
    ax[1, 0].set_title("position vs target (dashed)")
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("pos [m]")
    ax[1, 0].legend()
    ax[1, 0].grid(alpha=0.3)
    ax[1, 1].plot(t, R["form"], color="tab:orange", label="formation err [m]")
    axt = ax[1, 1].twinx()
    axt.plot(t, R["jitter"], color="tab:purple", lw=0.8, alpha=0.7)
    axt.set_ylabel("action jitter")
    ax[1, 1].set_title("formation error & action jitter")
    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("formation err [m]")
    ax[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def _plot_detail(R, nd, switch_times, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = R["t"]
    a = R["actions"].reshape(len(t), nd, 9)  # [T, nd, (p3,v3,a3)]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    # A: mean per-channel-type raw policy action magnitude over drones.
    for j, (lab, c) in enumerate(
        (("|pos raw|", "tab:red"), ("|vel raw|", "tab:green"), ("|acc raw|", "tab:blue"))
    ):
        mag = np.linalg.norm(a[:, :, j * 3 : (j + 1) * 3], axis=-1).mean(axis=1)
        ax[0, 0].plot(t, mag, color=c, label=lab)
    for s in switch_times:
        ax[0, 0].axvline(s, color="0.8", lw=0.7)
    ax[0, 0].set_title("raw policy action magnitude by channel")
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("|raw action|")
    ax[0, 0].legend()
    ax[0, 0].grid(alpha=0.3)
    # B: payload velocity per axis
    for k, c in zip("xyz", ("tab:red", "tab:green", "tab:blue")):
        ax[0, 1].plot(t, R["pv" + k], color=c, label="v" + k)
    ax[0, 1].set_title("payload velocity")
    ax[0, 1].set_xlabel("t [s]")
    ax[0, 1].set_ylabel("v [m/s]")
    ax[0, 1].legend()
    ax[0, 1].grid(alpha=0.3)
    # C: payload speed
    ax[1, 0].plot(t, R["pspeed"], color="tab:blue")
    for s in switch_times:
        ax[1, 0].axvline(s, color="0.8", lw=0.7)
    ax[1, 0].set_title("payload speed")
    ax[1, 0].set_xlabel("t [s]")
    ax[1, 0].set_ylabel("|v| [m/s]")
    ax[1, 0].grid(alpha=0.3)
    # D: max drone tilt
    ax[1, 1].plot(t, R["max_tilt"], color="tab:orange")
    ax[1, 1].set_title("max drone tilt")
    ax[1, 1].set_xlabel("t [s]")
    ax[1, 1].set_ylabel("tilt [deg]")
    ax[1, 1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def _summary(R, steps_per_pt, num_points, arrival_r):
    err = R["err"]
    done = R.get("done")
    print("\n[eval_track] summary")
    done_text = ""
    if done is not None and done.size:
        done_idx = np.flatnonzero(done > 0.5)
        done_text = f"  done_frac={float(np.mean(done > 0.5)):.3f}" + (
            f"  first_done_t={float(R['t'][done_idx[0]]):.2f}s" if done_idx.size else ""
        )
    print(
        f"  overall pos-err: RMS={math.sqrt(np.nanmean(err**2)):.3f}  max={np.nanmax(err):.3f}  "
        f"mean_jitter={np.nanmean(R['jitter']):.2f}  mean_form_err={np.nanmean(R['form']):.3f} m  "
        f"max_tilt={np.nanmax(R['max_tilt']):.1f}deg{done_text}"
    )
    reached = 0
    for i in range(num_points):
        seg = err[i * steps_per_pt : (i + 1) * steps_per_pt]
        if seg.size == 0:
            continue
        settle = seg[len(seg) // 2 :]
        ok = settle.mean() < arrival_r
        reached += int(ok)
        print(
            f"  pt {i}: final={seg[-1]:.3f}  settled_RMS={math.sqrt(np.mean(settle**2)):.3f} m  "
            f"{'REACHED' if ok else 'miss'}"
        )
    print(f"  reached {reached}/{num_points} (settled within {arrival_r:g} m)")


if __name__ == "__main__":
    main()
