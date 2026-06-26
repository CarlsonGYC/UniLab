# Sim2Real 缩小仿真-真机差距 实施计划（charpi 缆绳吊运多机系统）

制定时间：2026-06-26
对象：charpi 四旋翼（PX4，vision-aided）+ 缆绳吊挂载荷的多机协同控制
仿真侧：`control_test`（圆轨迹对照工具）+ `dynamics/`（PX4 位置/姿态环 + CTBR 速率环 + 电机/气动模型）+ MuJoCo 多体缆绳 + RL 任务
真机侧：你有真机，可飞轨迹、可记录日志（PX4 ulog + offboard 命令）

> 核心思想：**把"sim 与 real 跑同一条轨迹、记同样的量、逐子系统归因差距、要么辨识对齐要么用 DR 覆盖"做成一个可迭代的闭环**。`control_test` 的圆轨迹 + `analyze_log.py`（RMS / 相位滞后 / 有效跟踪延迟）就是这个闭环的度量工具。

---

## 目录
1. [目标与范围](#1-目标与范围)
2. [系统级联结构与 sim2real 差距来源](#2-系统级联结构与-sim2real-差距来源)
3. [总体方法论：sim2real 迭代闭环](#3-总体方法论sim2real-迭代闭环)
4. [基础设施：统一日志与时间同步](#4-基础设施统一日志与时间同步)
5. [分阶段实验计划（P0–P6）](#5-分阶段实验计划p0p6)
6. [PX4 专项：需要对齐/改善的部分](#6-px4-专项需要对齐改善的部分)
7. [仿真侧需要补的改进清单](#7-仿真侧需要补的改进清单)
8. [参数辨识总表](#8-参数辨识总表)
9. [验证指标与通过标准](#9-验证指标与通过标准)
10. [安全与风险](#10-安全与风险)
11. [里程碑与优先级](#11-里程碑与优先级)

---

## 1. 目标与范围

**目标**：让仿真里训练/调好的控制（PX4 级联 + RL 策略）迁到真机后，轨迹跟踪行为（位置/速度/姿态/推力的时域响应、相位滞后、稳态误差、失效模式）与仿真一致到可接受范围，使 RL 策略零样本或少量微调即可在真机稳定运行。

**两条迁移路径都要覆盖**：
- **控制级联本身**（PX4 位置→姿态→速率→电机）的 sim2real：这是基础，缆绳/RL 都建立在它之上。先把**单机控制**对齐，再上缆绳与多机。
- **RL 策略**的 sim2real：策略输出 per-drone PVA setpoint → 真机各机 PX4 offboard。策略只在控制级联+物理足够一致时才迁得动。

**范围分层**（按依赖顺序，必须自下而上）：
1. 执行器/电机（推力曲线、时间常数、电池）
2. 单机刚体动力学 + PX4 级联（质量/惯量、增益、延迟、气动）
3. 状态估计（EKF2 vision 延迟/噪声）+ offboard 命令链路
4. 缆绳-载荷耦合
5. 多机（下洗、编队、通信）
6. RL 策略（DR 覆盖残差 + 部署）

---

## 2. 系统级联结构与 sim2real 差距来源

### 2.1 控制级联（sim 与 real 必须逐级对应）
```
策略 PVA setpoint
  → PX4 位置环 (MPC_*, 100Hz)  → 集体推力 + 姿态setpoint
  → PX4 姿态环 (MC_*_P)         → 机体角速率setpoint
  → CTBR 速率环 (MC_*RATE_*, PID + Butterworth 陀螺滤波, 500Hz)
  → 控制分配 (CA_ROTOR_* 混控)  → 每电机 actuator setpoint
  → 推力曲线 (THR_MDL / kt,kl)  → 目标转速 ω_ref
  → 电机一阶响应 (taus) + 加速度限幅 → 实际转速 ω
  → 气动 (轴向推力衰减 + 阻力 + 滚转力矩) → 净机体 wrench
  → 刚体 + 缆绳多体 → 载荷
```
sim 侧对应文件：`dynamics/px4_control.py`（位置/姿态环）、`dynamics/control_stack.py`（CTBR 速率环装配）、`low_level_control/{body_rate_controller,control_allocator}.py`、`actuators/{allocation,motor,air_drag_effect}.py`、`base/{first_order_response_zoh,time_delay}.py`、`filter/low_pass_filter.py`。

### 2.2 差距来源分类（按影响大小排序）

| 类别 | 具体差距 | sim 现状 | 影响 |
|---|---|---|---|
| **推力标定** | 悬停推力比、推力曲线 kt/kl、电池电压衰减 | `thrust_const=1.15e-6`,`kl=0.05`,`hover_thrust` 按名义质量算；**无电池模型** | 🔴 极大：直接决定升力/控制权限，电压一掉真机推力下降 |
| **状态估计** | EKF2 vision 融合延迟 + 噪声 + 丢帧 | sim 用 **ground-truth** 状态，零延迟零噪声 | 🔴 极大：真机 vision 延迟数十 ms + 抖动，是相位滞后/不稳的主因 |
| **姿态/速率回路** | 增益匹配、**rate 积分无 anti-windup**、K 倍率 | rate K 裸默认 `(1,1,1)`，真机 `0.6`；积分无限幅 | 🔴 大：决定姿态环带宽（≈1/MC_ROLL_P≈150ms 滞后）与抗扰 |
| **执行器动态** | 电机时间常数 taus、加速度限幅、ESC 延迟 | `taus=0.03`；**max_rotor_acc 已存但 compute() 未用**；无 ESC 延迟 | 🟠 中：影响快动作跟随、抖动 |
| **气动** | 转子阻力、诱导速度、下洗、地效 | `rotor_drag_const`,`rolling_moment`；**无下洗/地效** | 🟠 中：前飞/圆轨迹残差、多机互扰、近地推力变化 |
| **命令链路** | offboard 速率/延迟、setpoint 类型 | `IntegerTimeDelay(3步)` 但 **PVA 路径绕过它**；无 MAVLink 延迟 | 🟠 中：真机 offboard 有链路延迟，必须建模 |
| **物理参数** | 质量/惯量/COM、臂长、载荷、缆绳 | 名义值（1.09kg, COMBINED_INERTIA, arm 0.14） | 🟠 中：可直接辨识对齐 |
| **缆绳-载荷** | 缆绳弹性/阻尼/质量、附着点、摆动 | MuJoCo 球关节刚体链（不可拉伸） | 🟡 任务级允许差异，但摆频/张力要对 |
| **传感/振动** | 螺旋桨振动、IMU 噪声、notch 滤波 | 无噪声 | 🟡 真机需 notch（IMU_GYRO_NF*），sim 无 |

---

## 3. 总体方法论：sim2real 迭代闭环

每一轮迭代（针对一个子系统）：
```
①  在 sim 和 real 上跑【同一条轨迹/激励】，用【同一套日志格式】记录
②  用 analyze_log.py 对照：RMS 误差、相位滞后、有效跟踪延迟、姿态/推力时域曲线
③  归因：差距落在哪个子系统？(用 §2.2 + control_test/README 的滞后预算表定位)
④  二选一：
     (a) 辨识真机该参数 → 更新 sim 参数（首选，缩小系统性偏差）
     (b) 无法精确辨识/本身随机 → 把它纳入 Domain Randomization 范围（覆盖残差）
⑤  重跑 ① 验证差距缩小；不达标回 ③
```
**关键原则**：
- **自下而上**：先台架辨识电机（开环），再单机悬停/阶跃（去掉轨迹耦合），再单机圆轨迹（闭环），最后缆绳/多机。下层没对齐前不要上层调参，否则归因不了。
- **一次只改一个子系统**：避免多个偏差互相补偿造成"假对齐"。
- **先消系统性偏差（bias），再用 DR 覆盖随机残差（variance）**：DR 不是用来掩盖建模错误的。
- **相位滞后预算法归因**：`control_test/README.md` 已给出滞后预算（姿态 P 环 ≈150ms、电机 ≈30ms、速率环 ≈10–20ms、Butterworth ≈5ms、ZOH ≈5ms）。真机实测有效延迟 = `稳态误差/巡航速度`（`analyze_log.py` 输出）。把真机延迟拆进这张预算表，超出的部分就是 sim 缺的（estimator 延迟、offboard 延迟、电机额外延迟）。

---

## 4. 基础设施：统一日志与时间同步

### 4.1 两侧记同样的量（扩展 control_test CSV）
当前 `control_test` CSV：`t, *_sp, *(actual), err_pos, err_vel, tilt_deg, thrust`。**真机和 sim 都要再补记**：
- 每电机 actuator setpoint / 实际转速（真机从 ESC 遥测或 PX4 `actuator_outputs`）
- 姿态四元数 + 角速率（机体系）
- 集体推力指令 + **电池电压/电流**（真机 `battery_status`）
- offboard 命令时间戳 + 收到时间戳（量 offboard 链路延迟）
- EKF2 估计的位姿 vs 真值（真机用 mocap/RTK 当"真值"参照）
- 缆绳张力（若有测力计）/ 载荷位姿（mocap）

### 4.2 真机数据来源
- PX4 **ulog**（QGC 或 `logger`）：`vehicle_local_position`、`vehicle_attitude`、`vehicle_rates_setpoint`、`actuator_motors/outputs`、`battery_status`、`vehicle_local_position_setpoint`、`estimator_status`。
- offboard 侧：你发的 `TrajectorySetpoint`（PVA）带时间戳。
- 外部真值：动捕（OptiTrack/Vicon）或 RTK-GPS，做 ground-truth 对照（量 EKF2 延迟/误差）。

### 4.3 时间同步（关键）
- 把 offboard 主机与 PX4 时钟同步（PX4 `timesync` / 同一台机器发命令并打时间戳）。
- 估计 offboard→执行的端到端延迟：发一个已知阶跃，互相关命令时间戳与 PX4 响应时间戳。这个延迟要回填到 sim 的 `IntegerTimeDelay`/命令延迟（见 §6.7）。

### 4.4 sim 侧对照入口
直接用 `control_test/px4_circle_single.py`（单机）/ `px4_circle_multilift.py`（多机），配置走 `config/charpi.yaml`（真机 PX4 参数）。真机飞同一条 `circle_trajectory` 参数（同 radius/period/ramp），导出 CSV，用 `analyze_log.py --cruise <v>` 两边各跑一遍对照。

---

## 5. 分阶段实验计划（P0–P6）

### P0 — 准备与参数对齐（不飞，1–2 天）
**目的**：消除最容易、最系统性的偏差——参数本身没对上。
- [ ] **导出真机完整 PX4 参数**（QGC → Parameters → Save / `param save`），与 `config/charpi.yaml` + airframe `4022_gz_charpi_vision` **逐项核对**。重点：`MC_*_P`、`MC_*RATE_{P,I,D,K,FF}`、`MPC_*`、`MPC_THR_HOVER`、`CA_ROTOR*_{PX,PY,KM,CT,TC}`、`IMU_*_CUTOFF`、`THR_MDL_FAC`、`BAT_*`、`EKF2_*`。**任何不一致项，以真机为准更新 charpi.yaml。**
  - ⚠️ 已发现：`DroneControlCfg` 裸默认 rate `kk=(1,1,1)`，真机 `MC_*RATE_K=0.6`；务必让 sim 走 charpi.yaml（`apply_drone_config` 会覆盖），不要用裸默认。
- [ ] **称重 + 量惯量**：整机质量（含电池/相机/挂载）；惯量用 CAD 或双线摆法（bifilar pendulum）；量 COM、臂长、电机位置（对齐 `arm_length`、`CA_ROTOR*_PX/PY`）。
- [ ] **确认状态估计源**：真机是 vision-aided（`EKF2_EV_CTRL=11`，vision 主高度）。记录 vision 系统（VIO/mocap）的频率、延迟、噪声水平——这是后面 sim 要补的。
- [ ] **悬停推力比**：满电悬停时读 PX4 `MPC_THR_HOVER`（PX4 在线估计 `hover_thrust_estimate`），与 sim `hover_thrust_for_thrust` 算的对照。

### P1 — 电机/执行器台架辨识（开环，去掉飞行耦合）
**目的**：把推力链（最大 sim2real 杠杆）标定准。用拉力台/测力计。
- [ ] **静推力曲线**：扫 throttle/PWM → 测推力 T 与转速 ω（光电/ESC 遥测）。拟合 `T = kt·ω²`（对齐 `thrust_const`）与 throttle→ω 的非线性（对齐 `kl=0.05` 和 PX4 `THR_MDL_FAC`）。**记录满油门最大推力**（决定控制权限上限）。
- [ ] **力矩常数**：测反扭力矩 vs 推力，拟合 `moment_const`（对齐 `CA_ROTOR*_KM=0.05`）。
- [ ] **电机时间常数 taus**：给电机一个转速阶跃，测一阶上升时间常数（对齐 `taus_up/taus_down=0.03`），上升/下降可能不对称——分别测。
- [ ] **加速度限幅**：阶跃测最大 dω/dt（对齐 `max_rotor_acc`，目前 sim **未启用**，见 §7）。
- [ ] **电池电压效应**🔴：在 1S 不同电压（满电→低电）下重测推力曲线。量化"同 throttle 下推力随电压下降的比例"。这是 sim **完全没有**的（§7 电池模型）。
- [ ] **ESC/输出延迟**：命令到推力建立的纯延迟（除一阶外的传输延迟）。

### P2 — 单机系统辨识飞行（闭环、无缆绳、无载荷）
**目的**：辨识刚体+级联的频域特性（带宽、延迟、阻力），单机控制对齐。
- [ ] **悬停稳定性**：定点悬停 60s，记位置漂移、姿态抖动 RMS、推力均值。对照 sim 悬停（应几乎不漂）。差异 → estimator 噪声/偏置。
- [ ] **姿态阶跃**：commande roll/pitch 角度阶跃（小角度，安全），测上升时间、超调、稳态——辨识姿态 P 环时间常数（理论 ≈1/MC_ROLL_P≈150ms）与 rate 环阻尼。对照 sim。
- [ ] **频率扫描 / chirp**🔴：commande 正弦位置（或姿态）激励，频率从 0.1→3Hz 渐增，做 **Bode 图**（幅频/相频）→ 直接读出闭环带宽与**总相位延迟**。这是最干净的延迟/带宽辨识，把 sim 与 real 的 Bode 叠起来看差在哪。
- [ ] **前飞阻力辨识**：匀速直线/圆飞不同速度，测维持速度所需的额外倾角/推力 → 拟合 `rotor_drag_const`、`max_v_para_z`（轴向来流推力衰减）。
- [ ] **端到端延迟**：用 chirp 的相频或阶跃互相关，量"命令→响应"总延迟，拆进滞后预算表，超出 sim 已建模部分的 = estimator + offboard 延迟（回填 §6.7、§7）。

### P3 — 单机闭环轨迹对照（核心闭环，反复迭代）
**目的**：用圆轨迹做 sim↔real 的主对照度量，迭代收敛单机。
- [ ] 真机飞 `circle_trajectory` 同参数（先慢：r=1m，period=6s，cruise≈1.05m/s；再加速），导出 CSV。
- [ ] sim 跑 `px4_circle_single.py --drone_config config/charpi.yaml` 同参数。
- [ ] `analyze_log.py --cruise <v>` 两侧对照：位置 RMS、稳态 RMS、**有效跟踪延迟(ms)**、`plot_log.py` 四面板（XY 轨迹、误差、逐轴、倾角+推力）。
- [ ] **归因 + 迭代**：误差 ∝ 速度（相位滞后特征）→ 调带宽/延迟；稳态偏置 → 推力/重力补偿；高频抖动 → 滤波/anti-windup/电机。每改一项重跑。
- [ ] **通过门**：同一轨迹下 sim 与 real 的有效延迟差 < 30ms、稳态 RMS 差 < 30%（见 §9）。

### P4 — 缆绳-载荷辨识与对照
**目的**：标定缆绳-载荷耦合（任务级允许引擎差异，但摆频/张力/载荷分担要对）。
- [ ] **几何/质量**：量缆绳长度、载荷质量/惯量、附着点位置（对齐 `rope_length`、`payload_mass`、`payload_radius`、附着 `rho`）。
- [ ] **单机吊载摆动**：单机悬停吊已知载荷，给水平扰动，测**摆动频率/阻尼**（单摆 ≈ √(g/L)）。对照 sim（MuJoCo 球关节链）。若摆频差大 → 调缆绳建模（链节数/阻尼/或换 tendon）、附着点、有效摆长。
- [ ] **缆绳张力/弹性**：若有测力计，测悬停张力（= 载荷重/N 分担）与动态张力。真实缆绳有弹性/阻尼，sim 是刚性球关节链——评估是否需要加柔性（弹簧 tendon）。
- [ ] **载荷分担**：N 机吊载，测各机推力是否均衡，对照 sim 编队平衡。

### P5 — 多机协同（下洗、编队、通信）
**目的**：多机特有差距。
- [ ] **下洗互扰**🟠：两机不同水平/垂直间距悬停，测被吹机的推力/姿态扰动 → 量化下洗。sim **无下洗模型**（§7），近距编队会偏。决定：加下洗模型 或 用 DR/最小间距约束规避。
- [ ] **编队悬停**：N 机 + 载荷定点，测稳态误差、各机姿态、最小间距。对照 sim。
- [ ] **通信/同步**：N 机 offboard 命令的时间偏差、丢包；中心化策略下各机命令到达不同步 → 评估影响，必要时在 sim 加命令抖动 DR。

### P6 — DR 校准 + RL 策略部署
**目的**：用辨识出的真机不确定度设定 DR，部署策略。
- [ ] **DR 范围 = 残差不确定度**：把 P1–P5 辨识后**仍无法精确对齐或本身随机**的量，设成 DR 范围（而不是拍脑袋）。重点：载荷质量/惯量（已有）、推力/电池（新增）、电机 taus（已有 `randomize_actuators`）、estimator 延迟/噪声（新增）、下洗（新增）、命令延迟。
- [ ] **观测可实现性**：策略 obs 必须在真机能算出来——载荷位姿（mocap/估计）、各机位姿（EKF2 vision）、缆绳几何。privileged 量（mass ratio）部署时用估计值 + 训练时的 obs 噪声覆盖。
- [ ] **动作接口**：策略输出 per-drone PVA → 各机 PX4 offboard `TrajectorySetpoint`。确认 setpoint 类型、坐标系、频率（≥50Hz）与 sim 一致。
- [ ] **部署验证**：先在沙箱/系留下试，再小幅轨迹，逐步放开。用同样 metrics 对照训练时 sim 表现。
- [ ] **失败回灌**：真机暴露的失效模式（某姿态发散、某速度抖动）→ 回 sim 复现 → 加 DR/改建模/调奖励 → 重训。

---

## 6. PX4 专项：需要对齐/改善的部分

> 真机跑的是 PX4 固件；sim 的 `px4_control.py`+`control_stack.py` 是 PX4 级联的复刻。sim2real 的本质就是让这份复刻 + 物理 == 真机 PX4 + 物理。逐子系统：

### 6.1 参数全量对齐（最优先）
导出真机 `.params`，确保 sim（charpi.yaml）与真机**逐项一致**。任何不一致都是系统性偏差。已知风险项：rate `K` 倍率、`MPC_THR_HOVER`、`THR_MDL_FAC`。

### 6.2 控制分配 / 混控（CA_ROTOR_*）
- 真机混控由 `CA_ROTOR*_{PX,PY,KM,CT,TC}` 定义（位置 ±0.10、KM 0.05、CT 推力系数、TC 时间常数）。sim 的 `Allocation`（`arm_length`,`moment_const`）+ `ControlAllocator`（PX4 mixer 归一化）必须复现同一混控矩阵。
- **核对**：sim `arm_length=0.14` vs `CA_ROTOR_PX/PY=0.10`（√2·0.10≈0.141，自洽，但确认你的电机布局是 X 形对角 0.10 还是臂长 0.14）；`moment_const=1.21e-2` vs `CA_ROTOR_KM=0.05`（注意 airframe 注释提过 gz momentConstant 从 0.0121 提到 0.05 做 sim2real——**确认真机用哪个**，这是已知不一致点）。

### 6.3 推力模型（THR_MDL_FAC + 电池）🔴
- PX4 用 `THR_MDL_FAC` 在线性与二次推力曲线间插值。sim 的 throttle→ω→T（`kl=0.05` 非线性 + `kt·ω²`）必须与 PX4 的 `THR_MDL_FAC` + 实测电机曲线（P1）匹配。
- **电池补偿**：PX4 按电压缩放推力（`BAT_*`/thrust scaling）。sim **无电池模型** → 要么 sim 加电压衰减模型（§7），要么真机实验固定用满电/稳压，并在 obs/DR 里覆盖推力 ±范围。

### 6.4 速率环（mc_rate_control）
- 增益对齐：`MC_*RATE_{P,I,D,K,FF}`。sim `body_rate_controller.py` 的 kp/ki/kd/kk/k_ff 必须 == 真机（走 charpi.yaml）。
- **anti-windup**🔴：PX4 有积分限幅（`MC_*R_INT_LIM`，charpi 0.3）+ 输出饱和反算；sim 速率环**没有 anti-windup**（review.md P2-1）。大角速率误差/碰撞前/缆绳剧烈牵引时 sim 积分会与真机行为分叉。**sim 必须补 anti-windup**（§7）。
- 输出限幅：`MC_*RATE_MAX`（1200°/s）对齐 sim `max_body_rate_xy/z`。

### 6.5 姿态环（mc_att_control）
- `MC_ROLL_P/PITCH_P=6.5`、`MC_YAW_P=2.8`、`MC_YAW_WEIGHT=0.4`、tilt 优先级。sim `PX4AttitudeControlCfg` 对齐。这是相位滞后主项（≈1/6.5≈150ms），真机若 estimator 有额外延迟，总滞后更大。

### 6.6 位置环（mc_pos_control）
- `MPC_XY_P=0.95`、`MPC_*_VEL_P/I/D_ACC`、`MPC_TILTMAX_AIR=45°`、`MPC_THR_HOVER`、速度/推力限幅。sim `PX4PositionControlCfg` 对齐。
- 加速度前馈：sim 把 A 前馈进姿态 setpoint；确认真机 offboard 也发 acceleration（`TrajectorySetpoint.acceleration`）且 PX4 用它，否则前馈不一致。

### 6.7 状态估计（EKF2）+ offboard 链路🔴
- 真机 EKF2 vision-aided（`EV_CTRL=11`，vision 主高度，GPS/baro 关）。**估计延迟**（`EKF2_EV_DELAY` + 融合滞后，数十 ms）+ **噪声**（`EKF2_GYR_NOISE=0.015` 等）+ 可能丢帧。sim 用 ground-truth → 这是相位滞后/抖动的隐藏大头。
- **改善**：①真机侧调 EKF2（`EKF2_EV_DELAY` 设准、vision 频率够高）；②sim 侧在反馈状态上加 estimator 延迟（一阶滞后 + 纯延迟）+ 噪声模型（§7），让滞后预算补齐这部分。
- **offboard 延迟**：量化 offboard→执行端到端延迟（§4.3），回填 sim 命令延迟（`IntegerTimeDelay` 当前在 PVA 路径被绕过——若要建模链路延迟，在 PVA setpoint 上显式加延迟）。offboard 频率确保 ≥50Hz（PX4 要求 >2Hz，太低会失控）。

### 6.8 陀螺滤波 / 振动
- `IMU_GYRO_CUTOFF=40`、`IMU_DGYRO_CUTOFF=30` 已与 sim Butterworth `cutoff_hz=(40,30)` 对齐 ✓。
- **notch 滤波**：真机有螺旋桨振动，需 `IMU_GYRO_NF*`/动态 notch；sim 无噪声无需。真机不调好 notch，速率环会被振动污染——这是 sim 完全没有的真机问题，**真机侧务必先做好振动/notch 标定**。

### 6.9 安全/失效（不直接影响跟踪，但部署必须）
`COM_RCL_EXCEPT`、`COM_OBL_*`（offboard 丢失失效行为）、`BAT_LOW/CRIT_THR`、地理围栏、kill switch。

---

## 7. 仿真侧需要补的改进清单

按优先级（🔴必须 / 🟠重要 / 🟡可选），对应缩小上面识别的差距：

1. 🔴 **电池电压衰减模型**：推力随电压下降缩放（`thrust ∝ f(V)`），V 随时间/电流下降。位置：`motor.py`/`control_stack.py` 的推力计算前加电压缩放因子；DR 覆盖电压范围。→ 对应 §6.3、P1 电池实验。
2. 🔴 **速率环 anti-windup**：积分限幅（`i_limit`，对齐 `MC_*R_INT_LIM=0.3`）+ 饱和反算。位置：`body_rate_controller.py`（review.md P2-1 已标注）。→ §6.4。
3. 🔴 **estimator 延迟 + 噪声模型**：在喂给控制器的状态（pos/vel/quat/omega）上加一阶滞后 + 纯延迟 + 高斯噪声（按真机 EKF2 vision 实测）。这是补齐相位滞后预算的关键。位置：env 读 state 后、喂控制器前。→ §6.7、P2 延迟实验。
4. 🟠 **电机加速度限幅启用**：`max_rotor_acc` 当前保存但 `motor.compute()` 未用（review.md P2-2）；在一阶响应后加 `dω` clamp。→ §6.4、P1。
5. 🟠 **offboard/命令链路延迟**：在 PVA setpoint 上加显式延迟（实测端到端 ms）。位置：`apply_action` 前对 setpoint 做 ring-buffer 延迟。→ §6.7。
6. 🟠 **下洗（downwash）模型**：多机近距时上方机对下方机的诱导气流（推力损失/扰动）。位置：`air_drag_effect.py` 或新增；按 P5 实测。→ §6、P5。
7. 🟠 **气动精修**：诱导速度、`max_v_para_z` 轴向来流推力衰减、地效（近地推力变化）。按 P1/P2 前飞辨识。
8. 🟡 **缆绳柔性**：若 P4 摆频/张力对不上，刚性球关节链 → 加 limited-range spatial tendon（弹性+阻尼）。
9. 🟡 **IMU 振动注入**（高保真）：若要复现真机振动对速率环的影响，可注入带通噪声——一般不必，真机靠 notch 解决。

> 这些改进同时反哺 RL：把它们做成可开关 + DR 参数化，训练时随机覆盖残差（P6）。

---

## 8. 参数辨识总表

| 物理量 | 真机辨识方法 | sim 位置（默认值） | PX4 参数 | 优先级 |
|---|---|---|---|---|
| 整机质量 m | 称重 | 场景 mass (1.09) | — | 🔴 |
| 惯量 I | 双线摆/CAD | DRONE_INERTIA | — | 🟠 |
| 悬停推力比 | PX4 hover_thrust_estimate | `hover_thrust` | `MPC_THR_HOVER`(0.26) | 🔴 |
| 推力系数 kt | 拉力台 T-ω | `thrust_const`(1.15e-6) | `CA_ROTOR*_CT`,`THR_MDL_FAC` | 🔴 |
| throttle 非线性 | throttle-T 曲线 | `kl`(0.05) | `THR_MDL_FAC` | 🔴 |
| 电压-推力 | 不同电压重测 | **无（待加）** | `BAT_*` | 🔴 |
| 电机时间常数 τ | 转速阶跃 | `taus_up/down`(0.03) | `CA_ROTOR*_TC` | 🟠 |
| 最大转速/推力 | 满油门 | `max_motor_speed`(3442) | — | 🟠 |
| 加速度限幅 | dω/dt 阶跃 | `max_rotor_acc`(**未用**) | — | 🟠 |
| 臂长/电机位置 | 测量 | `arm_length`(0.14) | `CA_ROTOR*_PX/PY`(0.10) | 🟠 |
| 力矩常数 | 反扭测量 | `moment_const`(1.21e-2) | `CA_ROTOR*_KM`(0.05) | 🟠 |
| 转子阻力 | 前飞辨识 | `rotor_drag_const`(8.065e-5) | — | 🟠 |
| 姿态 P | 阶跃/chirp | roll_p/pitch_p(6.5) | `MC_ROLL_P/PITCH_P` | 🔴 |
| 速率 PID+K | chirp | kp/ki/kd/kk | `MC_*RATE_{P,I,D,K}` | 🔴 |
| 积分限幅 | — | **无（待加）** | `MC_*R_INT_LIM`(0.3) | 🔴 |
| 陀螺滤波 | — | `cutoff_hz`(40,30) | `IMU_GYRO/DGYRO_CUTOFF` | ✓已对齐 |
| 位置/速度增益 | 轨迹 | gain_pos/vel_* | `MPC_XY_P`,`MPC_*_VEL_*` | 🔴 |
| estimator 延迟/噪声 | mocap vs EKF2 对照 | **无（待加）** | `EKF2_EV_DELAY`,`EKF2_*_NOISE` | 🔴 |
| offboard 端到端延迟 | 命令-响应互相关 | `IntegerTimeDelay`(绕过) | — | 🟠 |
| 缆绳长度/质量/阻尼 | 测量 + 摆动 | `rope_length`,链节 | — | 🟠 |
| 载荷质量/惯量 | 称重 | `payload_mass`(2.0) | — | 🟠 |
| 下洗 | 双机间距实验 | **无（待加）** | — | 🟠 |

---

## 9. 验证指标与通过标准

**单机（P3）通过门**（同一圆轨迹，sim vs real）：
- 有效跟踪延迟差 `|delay_sim − delay_real|` < 30ms（`analyze_log.py` 的 steadyRMS/cruise）
- 稳态位置 RMS 差 < 30%
- 姿态倾角时域曲线形状一致（峰值/相位）、推力均值差 < 5%
- Bode 带宽差 < 0.5Hz、相频在 0–2Hz 内差 < 15°

**缆绳/载荷（P4）**：摆动频率差 < 15%；悬停张力（载荷分担）差 < 15%。

**多机（P5）**：编队稳态误差量级一致；下洗扰动被 sim 复现到 ±50% 以内或被 DR/间距覆盖。

**RL 部署（P6）**：策略在真机轨迹上的 metrics（payload_pos_err、reached_frac、min_drone_dist）与训练 sim 同量级，无发散/碰撞；零样本不行则少量真机微调（或扩 DR 重训）后达标。

---

## 10. 安全与风险

- **系留/限高优先**：缆绳-多机系统高耦合、失稳后果重，先在系留、室内限高、软网下试，逐步放开。
- **kill switch + 失效行为**：offboard 丢失（`COM_OBL_ACT`）、低电（`BAT_*`）、围栏必须配好。
- **推力裕度**：满电最大推力 / 悬停推力 ≥ 2，否则机动/抗扰无裕度（P1 必测）。
- **多机间距**：下洗未标定前，编队间距留足（>2× 桨径），避免互扰失稳。
- **先单机后多机**：单机控制没对齐前绝不上多机吊载。
- **每次只改一处 + 留版本**：参数变更记录（git + 飞行日志关联），避免"假对齐"。

---

## 11. 里程碑与优先级

| 里程碑 | 内容 | 依赖 |
|---|---|---|
| **M0** 参数对齐 | P0：真机 .params ↔ charpi.yaml 全量核对；称重测惯量 | — |
| **M1** 推力链标定 | P1 台架：推力曲线、taus、**电压-推力**；sim 加电池模型 | M0 |
| **M2** 单机辨识 | P2：悬停/阶跃/chirp → 带宽+延迟+阻力；sim 加 anti-windup + estimator 延迟模型 | M1 |
| **M3** 单机对照达标 | P3：圆轨迹 sim↔real 迭代到 §9 通过门 | M2 |
| **M4** 缆绳-载荷 | P4：摆频/张力对齐 | M3 |
| **M5** 多机 | P5：下洗标定/规避、编队对照 | M4 |
| **M6** RL 部署 | P6：DR 校准、策略上真机、失败回灌 | M5 |

**最高杠杆三件事（先做）**：① 全量 PX4 参数对齐（M0，零成本消系统偏差）；② 推力链+电池标定（M1，最大物理杠杆）；③ estimator 延迟 + anti-windup（M2，相位滞后/抗扰主因）。这三件做完，单机 sim2real 差距通常已缩小大半。

---

### 附：sim 侧改动落点速查
- 电池模型 / 推力曲线：`dynamics/actuators/motor.py`、`control_stack.py`
- anti-windup / 速率环：`dynamics/low_level_control/body_rate_controller.py`
- 电机加速度限幅：`dynamics/actuators/motor.py`
- estimator 延迟+噪声 / offboard 延迟：env 读 state 处、`apply_action` 前
- 下洗 / 气动：`dynamics/actuators/air_drag_effect.py`
- 混控/分配核对：`dynamics/actuators/allocation.py`、`low_level_control/control_allocator.py`
- 对照工具：`control_test/{px4_circle_single,px4_circle_multilift,analyze_log,plot_log}.py` + `config/charpi.yaml`
