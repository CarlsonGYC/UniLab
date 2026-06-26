# Multilift 迁移代码 Review

审阅对象是当前 UniLab 工作树中从 `../direct_rl` 迁移来的 multilift 训练相关代码，重点包括：

- UniLab 正式路径：
  - `src/unilab/envs/manipulation/multilift/payload_lift.py`
  - `src/unilab/envs/manipulation/multilift/control/`
  - `src/unilab/assets/robots/multilift/scene.xml`
  - `conf/ppo/task/multilift_hover/mujoco.yaml`
  - `src/unilab/algos/torch/caps_ppo.py`
  - `scripts/train_rsl_rl.py`、`src/unilab/training/rsl_rl.py`
- 迁移来源/对照：
  - `../direct_rl/source/direct_rl/direct_rl/tasks/direct/multilift/multilift_env.py`
  - `../direct_rl/source/direct_rl/direct_rl/tasks/direct/multilift/multilift_env_cfg.py`
  - `../direct_rl/source/direct_rl/direct_rl/tasks/direct/multilift/agents/skrl_ppo_cfg.yaml`
  - `../direct_rl/source/direct_rl/direct_rl/assets/scene_builder.py`
  - `sandbox/multilift_mjcf/`

## 总结

当前 UniLab multilift 已经不是纯实验脚本：它通过 registry 暴露 `MultiliftHover`，能由统一 CLI compose，能实例化 MuJoCo backend，能接入 RSL-RL PPO/CapsPPO。一个 2 env、2 rollout steps、1 iteration 的 smoke 已经跑通，Actor/Critic 输入维度为 129，动作维度为 36，初始 `cable_d_mean_m ~= 1.012`，与 1.0 m cable + 0.012 m body-bottom offset 对齐。

物理模型的 nominal 结构总体合理：4 架 1.09 kg drone、2 kg cylinder payload、5 段刚体 rope link/rope、ball joint chain、500 Hz physics、100 Hz outer loop、PX4 position/attitude + CTBR/motor stack，和 direct_rl 的主要建模思路一致。MuJoCo site actuator 的 force axis 经最小模型验证会随 site/body 旋转，因此把 `compute_wrench()` 的 body-frame force 写入 site motor 这一路径静态上是合理的。

但目前还不能称为“物理真实性已经被证明”或“训练配置已经等价迁移”。关键风险是：

1. payload mass DR 只改质量、不改 inertia，物理上不自洽。
2. UniLab 正式任务缺少 direct_rl 已有的 substep collision latch、payload inertia/radius DR、payload push disturbance、action EMA、velocity task。
3. PPO 配置声称 mirror direct_rl，但实际上和当前 direct_rl/skrl tuned config 有明显差异：LR、entropy、action penalty、std cap、CAPS warmup 都不一致。
4. `algo.num_envs=2048` 对这个多刚体 cable 模型偏激进，且当前控制器在 CPU/Python pre-step callback 内每个 substep 做 torch/numpy 往返，没有 2048 env throughput 证据。
5. 正式测试树没有 multilift 专项 contract/smoke/physics regression test；验证仍主要依赖 sandbox 和手动 smoke。

建议合并前至少补上“小规模可运行 + 物理约束 + 配置 compose + actuator 坐标系”的自动化测试，并先把训练配置改到和 direct_rl 当前 tuned baseline 一致，再做长训。

## 已验证的事实

### CLI / Hydra / RSL-RL smoke

命令：

```bash
uv run train --algo ppo --task multilift_hover --sim mujoco \
  algo.num_envs=2 algo.num_steps_per_env=2 algo.max_iterations=1 \
  training.no_play=true training.play_render_mode=none training.logger=tensorboard
```

结果：

- 成功完成 1 个 learning iteration。
- RSL-RL 解析 observation sets：`default/actor/critic -> ['policy']`。
- Actor/Critic 都是 129 维输入；Actor 输出 36 维动作。
- 初始日志：
  - `metrics/payload_pos_err_m: 0.0010`
  - `metrics/cable_d_mean_m: 1.0120`
  - `metrics/formation_err_m: 0.0001`
  - `metrics/min_drone_dist_m: 1.3084`
  - `metrics/reached_frac: 1.0000`

这个只能证明接线和短路径可执行，不能证明长期训练稳定。

另一个命令：

```bash
uv run train --algo ppo --task multilift_hover --sim mujoco \
  algo.num_envs=2 algo.num_steps_per_env=2 algo.max_iterations=1 \
  training.no_play=true training.play_render_mode=none training.logger=none
```

失败：

```text
ValueError: Logger type not found. Please choose 'wandb', 'neptune', or 'tensorboard'.
```

默认 `conf/ppo/config.yaml` 是 `training.logger: tensorboard`，所以默认路径不受影响。但 `train_rsl_rl.py` 里把非 `tensorboard/wandb` 映射为 `"none"` 的逻辑和当前 RSL-RL logger 不兼容，这属于通用训练入口问题。

### MuJoCo site motor force 坐标系

我用一个最小 MuJoCo free body 模型验证了 `<motor site=... gear="0 0 1 0 0 0">` 的力方向：body 绕 y 轴转 90 度后，`qfrc_actuator[:3]` 是 `[1, 0, 0]`，说明 force gear 随 site/body frame 旋转。对 multilift 来说，`compute_wrench()` 的 body-frame force 直接写入 drone COM site actuator 是成立的。

力矩轴在 MuJoCo freejoint 的 rotational generalized coordinate 中不如 force 直观，仍建议加一个正式测试：施加 body torque，检查 body angular acceleration 是否沿期望 body axis。

## 物理模型真实性

### 成立的部分

1. Nominal mass/inertia 和 direct_rl 一致。
   - Drone mass = 1.09 kg，对应 `direct_rl/assets/scene_builder.py` 的 `BODY_MASS + 4 * PROP_MASS`。
   - Drone inertia = `(0.004421, 0.006721, 0.010024)`，对应 direct_rl visual-mode folded props inertia。
   - Payload 是 solid cylinder，`radius=0.15`、height ratio `1/8`、mass `2.0`，inertia 公式和 direct_rl/charpi_lift 一致。

2. Cable 结构是合理的显式刚体链近似。
   - 1.0 m rope 被离散成 5 段，每段 0.2 m。
   - 每段 link mass 0.008 kg，capsule visual/collision 关闭 contact。
   - link 和 payload/drone 之间用 ball joint；slack 可以通过多刚体链自然出现。

3. 控制抽象和 direct_rl 一致。
   - policy 输出每架 drone 的 PVA setpoint。
   - PX4 position loop 100 Hz。
   - PX4 attitude + CTBR/body-rate/motor stack 500 Hz。
   - 输出 net body-frame wrench，作用在 drone COM。

4. 热路径没有解析 asset/XML。
   - body ids 在 init path resolve。
   - MuJoCo body tracking sensors 在 backend cold path 注入。
   - env 热路径只调用 `SimBackend` 公共方法读取 body state，不直接探测 MuJoCo 私有结构。

### 物理真实性风险

#### 高：payload mass DR 不改 inertia，质量/转动惯量不自洽

`payload_lift.py` 的 reset DR 构造了 full `body_mass`，只缩放 payload body mass：

```python
body_mass[:, pbid] = base_mass[pbid] * mass_scale
randomization = ResetRandomizationPayload(body_mass=body_mass.astype(np.float32))
```

但是没有同步修改 payload inertia。当前默认 `payload_mass_range=0.4`，也就是 payload mass 可以变成 1.2 kg 到 2.8 kg，而 inertia 仍保持 nominal 2.0 kg cylinder inertia。这会改变 payload 的平动/转动比例，尤其影响 payload attitude reward、tumble termination、cable torque response。

direct_rl 里有 `randomize_payload_inertia`，并且注释明确 mass/inertia 都要随机。UniLab backend 也支持 `body_inertia` reset payload，但 env 目前没有构造它。

建议：

- 给 `SimBackend` 增加 `get_body_inertia()` contract，或在合法 backend contract 内提供 nominal inertia。
- mass_scale payload 时同步设置 `body_inertia[:, pbid] = nominal_inertia[pbid] * mass_scale`。
- 如果暂时不做 inertia DR，建议先关掉 `randomize_payload_mass`，避免训练在不物理的 mass/inertia 组合上学习。

#### 中高：正式 UniLab 版本缺少 payload inertia/radius DR

direct_rl 当前 cfg 包含：

- `randomize_payload_mass`
- `randomize_payload_inertia`
- `randomize_payload_radius`
- `obs_payload_params`，obs 包含 mass ratio 和 radius ratio

UniLab 版本只保留了 mass DR，obs 里最后两个 payload-param 位实际是：

- noisy mass ratio
- 常数 `1`

如果目标是先迁移 fixed-radius hover baseline，这可以接受；但它不是 direct_rl 当前 robust task 的等价迁移。报告/PR 中不要写“完整迁移了 direct_rl multilift robust DR”。

建议：

- 明确当前任务名/文档为 fixed-radius hover baseline。
- 后续若加 radius DR，要走 init/model variant 或 scene materialization，不要在 reset 热路径改 XML。
- radius DR 需要同步 obs、formation geometry、payload inertia 和 scene variant contract。

#### 中：cable 模型是 coarse rigid-chain，不是连续柔性绳

当前 1 m cable = 5 段刚体 capsule + ball joint。它能表达 slack 和摆动，但不是连续柔性绳：

- 没有 cable elasticity/stiffness。
- 没有 cable bending/friction/air drag。
- cable geoms `contype=0 conaffinity=0`，不会和 drone/payload/floor collision。
- joint damping/armature 是稳定性参数，不是经过实测拟合的 cable 参数。

这对 RL hover baseline 是合理近似，但不能直接称为 real cable physics。建议在 review/README 中使用“explicit rigid-link cable approximation”而不是“真实缆绳”。

#### 中：MuJoCo solver 参数缺少针对 cable-chain 的收敛证据

`scene.xml` 使用：

```xml
<option timestep="0.002" integrator="implicitfast" cone="elliptic"/>
```

没有显式设置 solver iterations/tolerance。多 ball-joint chain + payload + fast body wrench 对 solver 收敛敏感。direct_rl/PhysX 侧还有 `enable_external_forces_every_iteration` 的考虑；MuJoCo 侧至少需要被动稳定、hover 稳定和 large-force saturation 的 regression。

建议测试：

- zero-action/nominal hover setpoint 下 cable length drift。
- payload/drop passive energy 是否爆炸。
- 大 action clamp 下是否出现 NaN / qpos 爆炸。
- 2048 env 的 `finite`/reset rate/step throughput。

#### 中：body wrench at COM 是控制抽象，不是 rotor-resolved force

CTBR/motor stack 内部计算 rotor thrust、drag、rolling moment，但最终仍合成为 drone COM 的 6D wrench。这样和 direct_rl 的 Isaac implementation 等价，但不是完整 rotor-resolved aerodynamics：

- 没有 prop wash 对 payload/cable 的影响。
- 没有独立 rotor force application point 的结构柔性影响。
- drone visual arms/props 是 mass=0 visual/collision-disabled。

这对 policy-level cable-lift 是可接受抽象；如果目标是 sim2real motor/prop-level fidelity，需要另一个模型层级。

## 仿真合理性

### 成立的部分

1. Reset 从 MuJoCo `qpos0` 开始，`scene.xml` 的 default qpos 是 taut formation。
2. payload root freejoint 平移会带动整棵 rope/drone subtree，所以 reset 只改 `qpos[:, 0:3]` 能表达 direct_rl 中“刚性平移整个 formation”的核心效果。
3. Backend 每个 physics substep 都调用 `_pre_step_wrench()`，因此 inner attitude/rate loop 是 500 Hz，而 outer position loop 是 100 Hz。
4. `apply_action()` 返回零 actuator ctrl，真正 actuator ctrl 由 pre-step callback 生成；这符合当前 `SimBackend.set_pre_step_control()` contract。

### 仿真风险

#### 高：inter-drone collision 只在 control-rate 检查，丢失 direct_rl 的 substep latch

direct_rl 在 `_apply_action()` 每个 physics substep 更新：

```python
self._collided |= (self._drone_pair_dist() < self.cfg.collision_dist).any(...)
```

UniLab 版本只在 `update_state()` 里用当前 control-step 末尾的 `pdist` 判断 collision。对于 100 Hz control、500 Hz physics，快速接近的 drones 可能在中间 substep 小于 `collision_dist`，到 control-step 末尾又分开，从而漏掉 termination。

建议：

- 在 `_pre_step_wrench()` 中维护 `self._collided` latch。
- reset 时清零对应 env 的 latch。
- termination 用 `latched_collision | current_collision`。
- 加一个测试，构造两机高速穿越，确认中间 substep collision 会 terminate。

#### 中高：2048 env 配置缺少吞吐证据

`conf/ppo/task/multilift_hover/mujoco.yaml` 直接设：

```yaml
algo:
  num_envs: 2048
```

direct_rl cfg 注释里对这个 cable scene 的建议是先 512 env 量 throughput。UniLab 当前控制路径每个 control step 有 5 次 Python pre-step callback，每次：

- 从 backend sensor cache 拿 drone state。
- numpy -> torch CPU。
- 跑 PX4/CTBR torch CPU。
- torch -> numpy actuator ctrl。

这对 2048 env / 8192 drone rows 很可能是瓶颈。2-env smoke 的 steps/sec 没有参考意义；需要 256/512/1024/2048 throughput sweep。

建议：

- 默认先改为 `num_envs: 512`，除非已有 2048 benchmark。
- 增加 `scripts` 或测试里的 startup/throughput smoke。
- 记录 `env_step_total_ms`、`backend_physics_ms`、`apply_action_ms`、`backend_set_ctrl_ms`、`update_state_ms`。

#### 中：scene generator 在 sandbox，正式 asset 和 cfg 容易 drift

正式任务依赖 `src/unilab/assets/robots/multilift/scene.xml`，但生成器在 `sandbox/multilift_mjcf/unilab_gen_scene.py`。当前 cfg 常量也必须和 XML 对齐：

- `num_ropes=4`
- `rope_length=1.0`
- `payload_mass=2.0`
- `payload_radius=0.15`
- `base_height=1.5`
- `LINK_TOTAL_LENGTH=0.20`
- `BODY_BOTTOM_OFFSET=0.012`

如果以后改 XML 或 cfg，缺少测试会让 `formation_offset()`、tautness reward、reset geometry silently drift。

建议：

- 把 generator 移到正式 tooling 或在 asset README 中声明生成流程。
- 增加 XML contract test：
  - `nu == 24`
  - named bodies `payload`, `drone0..3` 存在
  - payload mass/radius/inertia 符合 cfg
  - default cable distance等于 `rope_length + BODY_BOTTOM_OFFSET`
  - drone pair distance等于 spawn diagonal/adjacent expected value

## 训练代码正确性

### 接线正确的部分

1. Registry 接入正确。
   - `src/unilab/envs/manipulation/__init__.py` 加入 `unilab.envs.manipulation.multilift`。
   - `payload_lift.py` 用 `@registry.envcfg("MultiliftHover")` 和 `@registry.env("MultiliftHover", sim_backend="mujoco")`。

2. Obs/action contract 正确。
   - `action_space.shape = (36,)`，即 4 drones * 9 PVA。
   - `obs_groups_spec = {"obs": 129}`。
   - RSL wrapper 把 `obs["obs"]` 映射为 TensorDict 的 `actor` 和 `policy`。
   - YAML 用 `obs_groups.actor/critic: [policy]`，smoke 中 RSL-RL 成功构建 129 输入网络。

3. Reward injection 正确。
   - `conf/ppo/task/multilift_hover/mujoco.yaml` 的 `reward:` 经 `BackendAdapter -> extract_reward_config -> reward_config` 注入 dataclass。
   - `reward_config.scales` 能覆盖 env default scales。

4. Time-limit bootstrap contract 沿用 UniLab NpEnv/RSL wrapper。
   - `max_episode_seconds / ctrl_dt = 1500` control steps。
   - `NpEnv` 在 time-out 时提供 final observation，`FinalObservationAwarePPO` 处理 timeout bootstrap。

### 训练配置迁移风险

#### 高：PPO YAML 不是 direct_rl 当前 tuned config 的等价迁移

`conf/ppo/task/multilift_hover/mujoco.yaml` 注释说“PPO hyper-params mirror direct_rl's skrl cfg”，但当前 direct_rl `agents/skrl_ppo_cfg.yaml` 和 UniLab YAML 有这些关键差异：

| 项 | direct_rl 当前值 | UniLab 当前值 | 风险 |
|---|---:|---:|---|
| learning rate | `3.0e-4` | `1.0e-3` | direct_rl 注释说 `1e-3` 早期不稳定 |
| entropy | `0.002` | `0.005` | 更鼓励 std 变大，动作更容易抖/饱和 |
| action penalty | `k_action=0.01` | `action=0.002` | direct_rl 已因 clamp saturation 把它调高 5x |
| policy std cap | `max_log_std=-0.5` | 未看到等价 cap | RSL Gaussian std 可能继续长大 |
| CAPS lambda | `0.5` with ramp | `0.3` from start | 无 warmup，早期可能压制探索/学习 |
| CAPS schedule | warmup 40%, full 80% | 无 schedule | 和 direct_rl 训练策略不一致 |

这不是小参数差异。direct_rl 当前配置的注释明确写了这些修改来自 eval 观察：高 action saturation、jitter、std 过大。如果 UniLab 目标是迁移“当前可训练 baseline”，应该先复刻这些 tuned 值，再用 UniLab 的差异做 ablation。

建议先改为：

- `learning_rate: 3.0e-4`
- `entropy_coef: 0.002`
- `reward.scales.action: 0.01`
- 给 RSL Gaussian distribution 增加 std upper clamp，或用 rsl_rl 支持的 distribution config 实现。
- CAPS 增加 progress schedule；至少在 env curriculum 完成前 `caps_lambda_s=0`。

#### 高：CAPS 当前从第 0 步全量生效

`CapsPPO` 本身实现可运行，但 direct_rl 的 CAPS 策略是：

- 任务先学会。
- env curriculum 大约 40% 时完成。
- CAPS 从 40% 到 80% ramp。
- 最后 20% 用 full lambda refine。

UniLab 当前 `caps_lambda_s: 0.3` 从第一个 minibatch 就加入 loss。对这种 PVA setpoint + cable payload 的探索任务，过早让 `mu(s) ~= mu(s+eps)` 可能降低早期策略对位置/姿态误差的敏感性。

建议：

- 在 `CapsPPO` 里支持 `caps_warmup_frac` / `caps_full_frac`，由 runner iteration 或 global step 更新 lambda。
- 或先把 YAML 默认 `caps_lambda_s: 0.0`，完成 baseline 后再打开。
- 日志里加 `loss/caps` 和 `caps/lambda`。

#### 中高：动作 std 没有 direct_rl 的显式上限

direct_rl/skrl policy 配置有：

```yaml
clip_log_std: True
max_log_std: -0.5
initial_log_std: -1.0
```

UniLab RSL config 只有：

```yaml
init_noise_std: 0.37
```

smoke 显示初始 std 是 0.37，但没有看到上限。对 `[-1, 1]` clamp 后的 36 维动作，std 过大不会增加有效探索，只会制造 saturation 和 high-frequency setpoint jitter。direct_rl 已经专门处理过这个问题。

建议：

- 查当前 rsl_rl `GaussianDistribution` 是否支持 `min_std/max_std` 或 log_std clamp。
- 若不支持，增加 UniLab distribution wrapper 或在 policy update 后 clamp scalar std param。
- 增加 `metrics/action_saturation` 日志，当前 UniLab 没有迁移 direct_rl 的 raw saturation metric。

#### 中：`max_iterations` 注释容易误导总样本预算

当前 YAML 注释写：

```yaml
# 500k control-step training (= 20834 iter x 24)
max_iterations: 20834
num_envs: 2048
num_steps_per_env: 24
```

这对 env 的 `step_counter` 来说确实是约 500k control steps。但 RSL-RL 总 transition 数是：

```text
20834 * 24 * 2048 ~= 1.024 billion transitions
```

如果这是有意的 per-env horizon budget，应在注释里明确“500k per-env control steps / about 1.0B aggregate samples”。如果不是有意的，训练预算会远超预期。

#### 中：`training.logger=none` 路径失败

`train_rsl_rl.py` 里：

```python
logger_type = cfg.training.logger if cfg.training.logger in ["tensorboard", "wandb"] else "none"
```

当前 RSL-RL 报错：只接受 `wandb/neptune/tensorboard`。默认是 tensorboard，所以不阻塞默认训练；但如果用户按 CLI 习惯设 `training.logger=none`，会失败。

建议：

- 不支持 no logger 就 fail fast，并给出明确报错。
- 或用 RSL-RL 支持的禁用方式，不要传 `"none"`。

## Reward / termination review

### 成立的部分

Reward 主要结构和 direct_rl hover task 对齐：

- payload position exponential reward
- progress potential reward
- payload attitude reward
- payload lin/ang velocity penalty
- cable slack penalty
- full 3D formation reward
- drone upright / omega regularization
- action / smooth / smooth2 penalty
- inter-drone proximity penalty
- one-time arrival time bonus
- alive bonus
- termination crash penalty

Termination 包括：

- non-finite state
- payload diverged
- payload tumbled
- drone too low
- drone flipped
- sustained slack
- inter-drone collision
- time truncation by base `NpEnv`

这些结构合理。

### Reward / termination 风险

1. `reward.scales.action=0.002` 和 direct_rl 当前 tuned value 不一致，可能重现 action saturation。
2. `metrics/action_saturation` 没迁移，长训时很难判断 policy 是否在 clamp 外发散。
3. `collision_dist` termination 只 control-rate 检查，见上面的 substep latch 问题。
4. `pdist.masked_fill(self._pair_eye, 1e6)` 依赖 `_pair_eye` 是 `[nd, nd]`，广播到 `[E, nd, nd]` 当前可行；建议用 explicit `self._pair_eye[None]` 增强可读性。
5. `r_time` 使用 `info["steps"]`，而 `NpEnv` 在 `update_state()` 后才 `steps += 1`。这意味着 time bonus 用的是当前 step 开始时的 episode length。影响很小，但如果要严格对齐 direct_rl `episode_length_buf`，可确认 off-by-one 是否符合预期。

## direct_rl 迁移缺口

当前 UniLab 正式任务只迁移了 hover 主体，不是 direct_rl multilift package 的完整功能集。

缺口列表：

1. 没有 `MultiliftVelocityEnv` / velocity-command tracking task。
2. 没有 payload inertia DR。
3. 没有 payload radius DR / model variant。
4. 没有 payload push disturbance。
5. 没有 action EMA filter。
6. 没有 substep collision latch。
7. 没有 target marker/debug visualization hook。
8. 没有 direct_rl 的 detailed reward component logs。
9. 没有 action saturation metric。
10. 没有 skrl policy std cap 的 RSL-RL 等价实现。
11. 没有 CAPS warmup/full schedule。
12. 没有 formal multilift tests。

其中 1、3、4、5、7 可以作为“暂未迁移功能”接受；2、6、9、10、11 更像训练/物理正确性风险。

## 建议的最小修复顺序

### P0：合并前应处理

1. 修正 payload mass/inertia DR。
   - mass scale 时同步 inertia scale。
   - 或先关闭 `randomize_payload_mass`。

2. 对齐 direct_rl 当前 tuned PPO 配置。
   - LR 3e-4。
   - entropy 0.002。
   - action penalty 0.01。
   - std cap。
   - CAPS warmup 或默认关闭 CAPS。

3. 加 substep collision latch。
   - 在 `_pre_step_wrench()` 中更新 latch。
   - reset 清零。
   - termination 读取 latch。

4. 加正式 smoke/contract tests。
   - Config compose includes `task=multilift_hover/mujoco`。
   - Env reset returns dict obs with shape `(n, 129)`。
   - One random step finite。
   - XML contains expected bodies/actuators and nominal cable length。
   - Site force axis follows body frame。

### P1：训练前应处理

1. 先把 default `num_envs` 降到 512，跑 throughput sweep 后再提高。
2. 增加 logging：
   - action saturation
   - action norm
   - reward components
   - collision fraction
   - target offset
   - diag command / diag error
   - reset reason fractions
3. 增加 256/512 env 的 100-1000 iteration smoke，观察：
   - reset rate
   - NaN rate
   - action std
   - action saturation
   - payload pos error
   - cable distance distribution
4. 明确训练预算注释：per-env steps vs aggregate samples。

### P2：功能完整性

1. 迁移 velocity-command task。
2. 加 payload push disturbance。
3. 加 radius DR 的 model variant/materialization 路径。
4. 把 sandbox generator 升级为正式 asset generation/check 工具。
5. 补 play/eval 脚本，输出 target tracking plots 和 videos。

## 建议测试清单

### Unit / contract

```bash
uv run pytest tests/config/test_config_system.py -q
uv run pytest tests/scripts/test_train_scripts.py -q
```

需要新增 multilift-specific tests，例如：

- `tests/envs/manipulation/test_multilift_contract.py`
  - registry contains `MultiliftHover`
  - reset obs shape is 129
  - action shape is 36
  - one random step finite
  - `reset(env_ids)` only changes selected info rows

- `tests/assets/test_multilift_scene.py`
  - compile `scene.xml`
  - `nu == 24`
  - `payload`, `drone0..3` exist
  - payload mass/inertia matches cfg
  - default cable distances match `rope_length + BODY_BOTTOM_OFFSET`

- `tests/backend/mujoco/test_site_wrench_frame.py`
  - tilted free body + site motor force follows site/body frame
  - torque axis produces expected angular acceleration

### Smoke

```bash
uv run train --algo ppo --task multilift_hover --sim mujoco \
  algo.num_envs=2 algo.num_steps_per_env=2 algo.max_iterations=1 \
  training.no_play=true training.play_render_mode=none
```

再做性能 smoke：

```bash
uv run train --algo ppo --task multilift_hover --sim mujoco \
  algo.num_envs=256 algo.num_steps_per_env=24 algo.max_iterations=10 \
  training.no_play=true training.play_render_mode=none
```

如果 256 稳定，再测 512/1024/2048，并记录 timing。

### Physics regression

- zero-action / nominal controller hover 10 s：
  - payload height drift
  - cable length min/max
  - payload tilt
  - drone tilt
- random action bounded stress：
  - no NaN
  - reset reason distribution sane
- mass DR consistency：
  - mass/inertia 同步缩放
  - payload angular response不随 DR 产生不物理极端

## 结论

当前 UniLab multilift 迁移已经具备可运行骨架，模型结构和 direct_rl 的 nominal cable-lift abstraction 基本一致，短 smoke 也证明了 CLI -> Hydra -> env -> MuJoCo -> RSL-RL 的主路径能跑通。

但它还不是一个可以直接长训/合并为“已验证物理真实 multilift baseline”的状态。最需要先处理的是 payload mass/inertia DR、自碰撞 substep latch、PPO tuned config 对齐、CAPS schedule/std cap，以及正式测试覆盖。处理完这些，再用 512 env 起步做长训和 eval，结论才有足够证据支撑。
