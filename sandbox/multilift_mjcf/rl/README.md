# `rl/` — cable-lift multilift RL on MuJoCo (sandbox port of direct_rl)

Vectorised RL environment + PPO training, ported from direct_rl's Isaac `multilift_env.py`
to UniLab's simulator (mujoco-uni), reusing the validated MuJoCo scene + control stack
from `../control_test/`.

## Tasks
- **hover** (`multilift_rl_env.py`) — fixed-point payload hover.
- **velocity** (`multilift_velocity_rl_env.py`) — payload velocity-command tracking. Subclasses
  hover WITHOUT modifying it (only overrides command hooks + obs + reward), like direct_rl's
  `MultiliftVelocityEnv`, then **improved with quadruped-locomotion ideas** (go2/go1 joystick):
  command is `[vx, vy, vz, vyaw]` (adds a **yaw-rate** command + `tracking_ang_vel` reward), the
  xy command is in the payload **heading frame** (curved trajectories), and a fraction of envs get
  a **zero/standing command** (one policy learns hover + tracking). Select with `--task`.

## Files
- `multilift_rl_env.py` — `MultiliftRLEnv`: `num_envs` cable-lift formations in ONE MuJoCo
  model (one `mj_step` advances all). Ports obs (129), action (9·nd=36), the full reward,
  termination, step-counter curriculum, and per-env payload mass/inertia DR. Implements the
  rsl_rl `VecEnv` interface (`get_observations`→TensorDict, `step`→(TensorDict, rew, done, extras)).
  Drone authority = controller body-frame wrench via `xfrc_applied`; cables = the stable
  ball-jointed rigid chain.
- `train.py` — rsl_rl PPO (same rsl_rl 5.x UniLab uses, via `normalize_ppo_train_cfg`).
  Hyper-parameters mirror direct_rl's skrl config (MLP [256,256,128] elu, 24 rollouts,
  5 epochs, 4 minibatches, KL-adaptive lr, empirical normalization).
- `play.py` — load a checkpoint, run the policy in the interactive viewer with the payload +
  drone setpoints drawn (uses the real charpi mesh by default).

## Run
```bash
cd /home/carlson/rl_ws/UniLab
uv run python sandbox/multilift_mjcf/rl/train.py --smoke                                  # pipeline check
uv run python sandbox/multilift_mjcf/rl/train.py --task hover    --num_envs 256 --iterations 1500
uv run python sandbox/multilift_mjcf/rl/train.py --task velocity --num_envs 256 --iterations 1500
uv run python sandbox/multilift_mjcf/rl/play.py  --task velocity --checkpoint sandbox/multilift_mjcf/rl/logs/multilift_velocity_ppo/model_1500.pt
```
`--device cuda` puts the policy + control/obs tensors on GPU (MuJoCo physics stays CPU —
the heterogeneous CPU-sim / GPU-learn split). `--visual mesh|simple` selects the drone look.

## Validated (headless)
- env: obs (N,129), action 36, hover_thrust 0.343 (incl. payload share); 60 random-action
  steps finite; reward responds to actions; metrics logged.
- training: full rsl_rl PPO runs (actor 129→256→256→128→36, critic→1, EmpiricalNormalization);
  ~0.5 s/iter at 32 envs (CPU); **episode length grows 19→36→61** (learning signal).

## Notes / next
- The mj_objectVelocity per-body velocity read is the CPU bottleneck; fine for moderate
  `num_envs`, optimise (batched cvel) for large-scale runs.
- Same modeling caveat as control_test: drones are ball-jointed to inextensible cables (not
  6-DOF free bodies as in Isaac) → expect a small payload-altitude bias; tune reward/cable
  if needed.
- This is the "sandbox first" half of the plan; folding into UniLab proper (NpEnv + registry +
  Hydra `conf/`) is the next step per `direct_rl/transfer_plan.md`.
```
