#!/usr/bin/env python
"""Track-evaluation + visualization for the UniLab MultiliftHover policy.

Port of direct_rl's ``eval_track.py``: flies the slung payload through a sequence of
random target points (holding each ``hold_s`` s), records tracking / formation / jitter
/ actions / payload velocity / drone tilt, and renders two PNGs (tracking + detail) + an
NPZ + a console summary. ``--video`` also records an MP4 of the SAME rollout (via
UniLab's working playback renderer). The target is driven by writing
``env._state.info["target_pos_w"]`` each step (like direct_rl writing
``base._target_pos_w``); DR / curriculum / obs-noise are disabled for a clean read-out.

    uv run python sandbox/multilift_mjcf/rl/eval_track.py --num_points 5 --hold_s 8 --video
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from unilab.base.registry import ensure_registries, make  # noqa: E402
from unilab.training.rsl_rl import RslRlVecEnvWrapper, normalize_ppo_train_cfg  # noqa: E402


def _train_cfg() -> dict:
    """Matches conf/ppo/task/multilift_hover/mujoco.yaml (architecture must match the checkpoint)."""
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


def _resolve_checkpoint(args) -> str:
    if args.checkpoint:
        return os.path.abspath(args.checkpoint)
    run = args.run or sorted(glob.glob("logs/rsl_rl_ppo/MultiliftHover/*_mujoco"))[-1]
    ckpts = sorted(
        glob.glob(os.path.join(run, "model_*.pt")),
        key=lambda p: int(p.split("model_")[-1].split(".pt")[0]),
    )
    if not ckpts:
        raise SystemExit(f"no model_*.pt in {run}")
    return ckpts[-1]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--run", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--num_points", type=int, default=5)
    p.add_argument("--hold_s", type=float, default=8.0)
    p.add_argument("--range_xy", type=float, default=None)
    p.add_argument("--range_z", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--video", action="store_true", help="also record an MP4 of the rollout")
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
    override = {
        "curriculum_enabled": False,
        "randomize_payload_mass": False,
        "randomize_actuators": False,
        "obs_param_noise": 0.0,
    }
    env = make("MultiliftHover", sim_backend="mujoco", num_envs=1, env_cfg_override=override)
    wrapped = RslRlVecEnvWrapper(env, device=args.device)
    runner = OnPolicyRunner(wrapped, _train_cfg(), log_dir=None, device=args.device)
    ckpt = _resolve_checkpoint(args)
    runner.load(ckpt, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    print(f"[eval_track] checkpoint: {ckpt}")

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
    )
    rec: dict[str, list] = {k: [] for k in keys}
    rec_actions: list[np.ndarray] = []
    state = {"i": 0}

    def init():
        obs, _ = wrapped.reset()
        env.set_autoreset(False)
        state["i"] = 0
        for v in rec.values():
            v.clear()
        rec_actions.clear()
        return obs

    def do_step(obs):
        pt = min(state["i"] // steps_per_pt, args.num_points - 1)
        target = target_base + offsets[pt]
        state["target"] = target  # for the video marker
        env._state.info["target_pos_w"][:] = target.astype(np.float32)  # drive the target
        action = policy(obs)
        obs2, _, _, extras = wrapped.step(action)
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
        rec_actions.append(action.detach().cpu().numpy().reshape(-1))
        state["i"] += 1
        return obs2

    out_dir = args.out_dir or os.path.join(os.path.dirname(ckpt), "eval_track")
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
    # A: mean per-channel-type action magnitude over drones
    for j, (lab, c) in enumerate(
        (("|pos cmd|", "tab:red"), ("|vel cmd|", "tab:green"), ("|acc cmd|", "tab:blue"))
    ):
        mag = np.linalg.norm(a[:, :, j * 3 : (j + 1) * 3], axis=-1).mean(axis=1)
        ax[0, 0].plot(t, mag, color=c, label=lab)
    for s in switch_times:
        ax[0, 0].axvline(s, color="0.8", lw=0.7)
    ax[0, 0].set_title("action magnitude by channel (mean over drones)")
    ax[0, 0].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("|action| [-1,1]")
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
    print("\n[eval_track] summary")
    print(
        f"  overall pos-err: RMS={math.sqrt(np.nanmean(err**2)):.3f}  max={np.nanmax(err):.3f}  "
        f"mean_jitter={np.nanmean(R['jitter']):.2f}  mean_form_err={np.nanmean(R['form']):.3f} m  "
        f"max_tilt={np.nanmax(R['max_tilt']):.1f}deg"
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
