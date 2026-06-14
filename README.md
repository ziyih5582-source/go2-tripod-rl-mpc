# RL+MPC Locomotion — Three-Legged (Tripod) Walking on Unitree Go2

> **A hierarchical RL+MPC control framework for quadruped robots, extended with three-legged (tripod) walking on the Unitree Go2 and full CPU training support.**
>
> **Based on [rl-mpc-locomotion](https://github.com/silvery107/rl-mpc-locomotion) by Yulun Zhuang, Wei Zhang.**

---

## What's New in This Fork

| Feature | Description |
|---------|-------------|
| **Three-Legged (Tripod) Walking** | FR leg retracted, walking on FL+RL+RR legs. RL controls COM offset for balance; MPC provides stable gait kinematics |
| **CPU-Only Training** | Full training pipeline runs on CPU (64 parallel Isaac Gym environments, `CUDA_VISIBLE_DEVICES=""`) |
| **Go2 Real-Robot Deployment** | Complete deployment pipeline via Unitree SDK with state machine (DAMP → QUAD_WALK → TRIPOD_STAND → WALK) |
| **Analytic IK Solver** | Custom `Go2IK` — geometric inverse kinematics for Go2, no external dependencies |
| **Reduced Action Space** | RL outputs 3-dim COM offset instead of 12-dim joint positions, drastically simplifying learning |

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                      RL+MPC 协同控制架构                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────┐          ┌─────────────────┐                   │
│  │  Isaac Gym 仿真  │          │    Go2 真机     │                   │
│  │  (64并行环境)    │          │  (Unitree SDK)  │                   │
│  └────────┬────────┘          └────────┬────────┘                   │
│           │                            │                            │
│           ▼                            ▼                            │
│  ┌─────────────────────────────────────────────────────┐            │
│  │               RLObserver (观测构建)                  │            │
│  │  base_pos + vel + gravity + cmd + dof_pos +         │            │
│  │  dof_vel + last_actions + height + contact          │            │
│  │                → obs [48维]                         │            │
│  └──────────────────────┬──────────────────────────────┘            │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────┐            │
│  │           Actor-Critic 神经网络 (PPO)                │            │
│  │                                                      │            │
│  │  Actor:  48 → 512 → 256 → 128 → 3 (COM偏移 mean)   │            │
│  │  Critic: 48 → 512 → 256 → 128 → 1 (Value 估计)     │            │
│  │                                                      │            │
│  │  动作范围: [-1, 1]³                                  │            │
│  └──────────────────────┬──────────────────────────────┘            │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────┐            │
│  │           动作解码 → COM 偏移                         │            │
│  │  com_x = action[0] × 0.1235 + 0.1465               │            │
│  │  com_y = action[1] × 0.002 + 0.065                 │            │
│  │  com_z = action[2] × 0.01 + 0.0                    │            │
│  └──────────────────────┬──────────────────────────────┘            │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────┐            │
│  │            MPCController (mpc_walk1.py)              │            │
│  │                                                      │            │
│  │  ① phase 推进 = (phase + dt/GAIT_PERIOD) % 1.0     │            │
│  │  ② _plan_feet(): 规划4足端轨迹                      │            │
│  │     - FR腿: fr_lift_alpha 线性插值 (接地↔收起)       │            │
│  │     - FL/RL/RR: 步态循环 (摆动相↔支撑相)            │            │
│  │  ③ Go2IK.solve(): 解析 IK → 关节角 [12维]          │            │
│  │  ④ jvel = (jpos_target - dof_pos) / dt             │            │
│  └──────────────────────┬──────────────────────────────┘            │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────┐            │
│  │              PD 力矩控制器                           │            │
│  │  torques = Kp × Δpos + Kd × Δvel                   │            │
│  │  torques = clamp(torques, -50, 50)                  │            │
│  └──────────────────────┬──────────────────────────────┘            │
│                         │                                           │
│                         ▼                                           │
│  ┌─────────────────────────────────────────────────────┐            │
│  │              机器人 (仿真/真机)                       │            │
│  │          12个关节电机执行力矩指令                      │            │
│  └─────────────────────────────────────────────────────┘            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**Core idea**: RL (high-level) outputs a 3-dim COM offset; MPC (low-level) computes 12 joint targets via analytic IK; PD controller converts to torque commands. This reduces the RL action space from 12-dim to 3-dim, significantly lowering learning difficulty.

---

## Dependencies

- *Python* - 3.8+
- [*PyTorch* - 1.10.0 with CUDA 11.3](https://pytorch.org/get-started/previous-versions/) (or CPU-only)
- [*Isaac Gym* - Preview 4](https://developer.nvidia.com/isaac-gym)

## Installation

1. Clone this repository
   ```bash
   git clone --recurse-submodules git@github.com:YOUR_USERNAME/YOUR_REPO.git
   ```

2. Initialize submodules (if not using `--recurse` above)
   ```bash
   git submodule update --init
   ```

3. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   ```

4. Install `rsl_rl`:
   ```bash
   cd extern/rsl_rl
   pip install -e .
   ```

5. Compile python binding of the MPC solver:
   ```bash
   pip install -e .
   ```

---

## Quick Start

### Training (Go2 Tripod Walking, CPU)

```bash
cd RL_Environment
CUDA_VISIBLE_DEVICES="" python train_go2_rl_mpc_simple.py task=Go2RLMPCSimple headless=True
```

Training progress can be monitored via TensorBoard:
```bash
tensorboard --logdir runs
```

Model checkpoints are saved every 100 iterations under `RL_Environment/runs/Go2RLMPCSimple/`.

### Training (GPU)

```bash
cd RL_Environment
python train_go2_rl_mpc_simple.py task=Go2RLMPCSimple headless=False
```

### Real-Robot Deployment (Go2)

```bash
# Pure MPC mode (BalanceController, no RL)
python deploy_walk1.py

# RL+MPC mode (uses trained neural network)
python deploy_walk1.py --rl --checkpoint=runs/Go2RLMPCSimple/May15_18-14-35/model_2000.pt

# Manual keyboard control
python deploy_walk1.py --manual

# Specify network interface
python deploy_walk1.py enp129s0
```

### Original MPC Controller (Aliengo/Go1/A1)

```bash
python RL_MPC_Locomotion.py --robot=Aliengo
```

Supported robot types: `Go1`, `A1`, `Aliengo`. Requires an Xbox-like gamepad, or pass `--disable-gamepad`.

---

## Training Data Flow

### Single-Step Training Pipeline

```
时间步 t (dt = 0.02s)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1 ─ 构建观测 (48维)
  ├─ 从仿真器读取: root_states(13) + dof_state(24) + contact_forces
  ├─ 计算:
  │   base_lin_vel = quat_rotate_inverse(quat, world_vel) × 2.0
  │   base_ang_vel = quat_rotate_inverse(quat, world_ang) × 0.25
  │   gravity_body = quat_rotate_inverse(quat, [0,0,-1])
  │   dof_pos_scaled = (dof_pos - default_pos) × 1.0
  │   dof_vel_scaled = dof_vel × 0.05
  │   commands_scaled = cmd × [2.0, 2.0, 0.25]
  │   base_height_norm = (height - 0.3) / 0.3
  └─ 拼接: [base_pos(3) + lin_vel(3) + ang_vel(3) + gravity(3)
           + cmd(3) + dof_pos(12) + dof_vel(12) + last_act(3)
           + height(1) + actions(3) + contact_proxy(2)]
     = 48维

Step 2 ─ RL 动作采样
  ├─ actor(obs) → mean [64, 3]
  ├─ action = mean + std × ε,  ε~N(0,1)
  └─ action ∈ [-1, 1]³

Step 3 ─ 解码 COM 偏移
  ├─ com_x = action[0] × 0.1235 + 0.1465 ∈ [0.023, 0.27]
  ├─ com_y = action[1] × 0.002 + 0.065   ∈ [0.063, 0.067]
  └─ com_z = action[2] × 0.01 + 0.0      ∈ [-0.01, 0.01]

Step 4 ─ MPC 运动学控制
  ├─ phase += 0.02/0.6 = 0.0333 (phase 推进)
  ├─ _plan_feet(vx_cmd, fr_lift_alpha=1.0):
  │   ├─ FR腿(leg=1): alpha=1 → 完全收起
  │   │   足端 = [nx-0.04, ny, -0.11] (不叠加 com_offset)
  │   └─ FL/RL/RR: 步态循环
  │       ├─ 摆动相(phase%1.0 < 0.15): z=抛物线, x=向前扫过
  │       └─ 支撑相: z=body_height, x=向后推
  ├─ Go2IK.solve() → 每腿3关节角 × 4腿 = 12维
  └─ jvel = (jpos_target - dof_pos) / 0.02

Step 5 ─ PD 力矩计算
  ├─ pos_err = jpos_target - dof_pos
  ├─ vel_err = jvel_target - dof_vel
  ├─ torques = 50.0 × pos_err + 2.5 × vel_err
  └─ torques = clamp(torques, -50, 50)

Step 6 ─ 仿真步进
  ├─ gym.set_dof_actuation_force_tensor(sim, torques)
  └─ gym.simulate(sim)  (PhysX 物理引擎)

Step 7 ─ 奖励计算
  └─ R = 0.02 × (
       + 2.0 × exp(-||cmd_xy-vel_xy||²/0.25)     速度跟踪
       + 0.5 × exp(-(cmd_yaw-vel_yaw)²/0.25)      偏航跟踪
       + 2.0 × exp(-||ang_vel_xy||²/0.5)          三足稳定
       - 2.0×vel_z² - 5.0×||ang_vel_xy||²        姿态惩罚
       - 0.00005×||τ||² - 1.0×collision           力/碰撞
       - 0.1×||a||² - 0.1×||Δa||²                平滑
       - 0.5×(height-0.3)²                         高度
     )

Step 8 ─ 存入 RolloutStorage
  └─ 累积 24 步(每个环境) → 1536 个 transitions 后执行 PPO Update
```

### Training Scale

| Metric | Value |
|------|------|
| Parallel Environments | 64 |
| Steps per Env / Iteration | 24 |
| Total Transitions per Iteration | 1536 |
| PPO Epochs per Iteration | 5 |
| Mini-batches per Epoch | 4 |
| Gradient Updates per Iteration | 20 |
| Mini-batch Size | 384 |
| Total Iterations | 5000 |
| Total Transitions | 7,680,000 |
| Model Save Interval | Every 100 iterations |

---

## Key Code Files

### Core File List

| File | Lines | Role |
|------|------|------|
| `RL_Environment/train_go2_rl_mpc_simple.py` | 133 | **Training entry point**: loads config, creates env, starts PPO training |
| `RL_Environment/tasks/go2_rl_mpc_simple.py` | 555 | **Training environment**: observation construction, COM decoding, MPC invocation, reward calculation |
| `Go2_Controller/mpc_walk1.py` | 475 | **MPC controller**: Go2IK inverse kinematics, MPCController gait, BalanceController |
| `deploy_walk1.py` | 915 | **Deployment entry point**: communication init, state machine, RL/MPC dual-mode switching |
| `RL_Environment/WeightPolicy.py` | 278 | **Model loader**: creates network, loads weights, infers COM |
| `extern/rsl_rl/rsl_rl/algorithms/ppo.py` | 188 | **PPO algorithm**: policy update, value function update, adaptive learning rate |
| `extern/rsl_rl/rsl_rl/modules/actor_critic.py` | 156 | **Actor-Critic network definition** |
| `extern/rsl_rl/rsl_rl/runners/on_policy_runner.py` | 234 | **Training Runner**: orchestrates sampling→training loop |
| `extern/rsl_rl/rsl_rl/storage/rollout_storage.py` | 235 | **Experience storage**: GAE computation, mini-batch generation |
| `RL_Environment/tasks/legged_config_ppo.py` | 41 | **PPO hyperparameter config class** |
| `RL_Environment/cfg/task/Go2RLMPCSimple.yaml` | 107 | **Training hyperparameter YAML config** |
| `RL_Environment/cfg/config_cpu.yaml` | 59 | **CPU training environment config** |

### Code Call Graph

```
train_go2_rl_mpc_simple.py          ← 训练入口（Hydra加载配置）
    │
    ├─→ Go2RLMPCSimple(env)         ← 训练环境（IsaacGym VecTask子类）
    │       │
    │       ├─→ MPCController       ← 简化版MPC步态控制器
    │       │       └─→ Go2IK       ← 解析逆运动学
    │       │
    │       ├─→ BalanceController   ← IMU闭环平衡
    │       │
    │       ├─→ pre_physics_step()  ← 核心链路：RL动作→COM→MPC→PD→力矩
    │       ├─→ compute_observations() ← 48维观测构建
    │       ├─→ compute_reward()    ← 奖励计算
    │       └─→ reset_idx()         ← 环境重置
    │
    └─→ OnPolicyRunner              ← PPO训练Runner
            │
            ├─→ ActorCritic         ← 策略网络 + 价值网络
            ├─→ PPO                 ← PPO算法核心
            └─→ RolloutStorage      ← 经验回放缓存

deploy_walk1.py                     ← 真机部署入口
    │
    ├─→ MPCDeployer                 ← 部署主控
    │       ├─→ MPCController       ← 同mpc_walk1.py
    │       ├─→ BalanceController   ← IMU平衡(纯MPC模式)
    │       ├─→ WeightPolicy        ← 加载训练好的RL模型
    │       └─→ RLObserver          ← 真机观测构建
    │
    └─→ Unitree SDK                 ← Go2真机通信
```

---

## Neural Network Training Details

### Network Architecture

**Actor (Policy Network)** — 3-layer MLP, ELU activation:

$$
\begin{aligned}
\mathbf{h}^{(0)} &= \mathbf{o}_t \in \mathbb{R}^{48}
\quad\text{(观测输入，48维)} \\[4pt]
\mathbf{h}^{(k)} &= \mathrm{ELU}\!\left(\mathbf{W}_a^{(k)}\mathbf{h}^{(k-1)} + \mathbf{b}_a^{(k)}\right),\quad k=1,2,3 \\[4pt]
\boldsymbol{\mu}_\theta(\mathbf{o}_t) &= \mathbf{W}_a^{(4)}\mathbf{h}^{(3)} + \mathbf{b}_a^{(4)} \in \mathbb{R}^{3}
\quad\text{(动作均值，COM偏移 [dx,dy,dz])} \\[4pt]
\boldsymbol{\sigma}_\theta &= \operatorname{diag}\!\big(e^{\phi_1},e^{\phi_2},e^{\phi_3}\big)
\quad\text{(可学习对角标准差，初始 } \phi_i = \ln(1.0)=0,\; \sigma_i^{\text{init}}=1.0\text{)}
\end{aligned}
$$

$$
\text{隐藏层维度: } \underbrace{512}_{h_1},\;\underbrace{256}_{h_2},\;\underbrace{128}_{h_3},\quad
\text{激活函数: ELU},\quad
\text{总参数量: } \approx 48\!\times\!512 + 512\!\times\!256 + 256\!\times\!128 + 128\!\times\!3 \approx 189\text{k}
$$

Training-time action sampling (Gaussian policy):

$$
\mathbf{a}_t \sim \mathcal{N}\!\left(\boldsymbol{\mu}_\theta(\mathbf{o}_t),\; \boldsymbol{\sigma}_\theta^2\right) \in \mathbb{R}^{3},
\qquad a_{t,i} \in [-1, 1]^{3}
$$

Inference-time action (deployment, deterministic):

$$
\mathbf{a}_t = \boldsymbol{\mu}_\theta(\mathbf{o}_t)
\quad\text{(关闭噪声，仅输出均值)}
$$

**Critic (Value Network)**:

$$
V_\phi(\mathbf{o}_t) = \mathrm{MLP}_c(\mathbf{o}_t) \in \mathbb{R}^{1}
$$

$$
\text{隐藏层维度: } [512,\;256,\;128],\quad
\text{激活函数: ELU},\quad
\text{输出维度: } 1\quad\text{(标量状态价值 } V(s) \text{)}
$$

### Training vs Inference

| Dimension | Training | Inference (Deployment) |
|------|----------|---------------|
| Actor Output | `action = mean + std × ε` (Gaussian sampling) | `action = mean` (deterministic) |
| Exploration Noise | `init_noise_std = 1.0` (learnable) | None |
| Action Space | [-1, 1]³ | [-1, 1]³ |
| Decoded COM | [0.023~0.27, 0.063~0.067, -0.01~0.01] | Same |
| Model State | `actor_critic.train()` | `actor_critic.eval()` |

### Observation Space (48-dim)

| Index | Component | Dim | Scale | Meaning |
|------|------|------|------|------|
| 0-2 | `base_pos` | 3 | raw (m) | Body world position |
| 3-5 | `base_lin_vel` | 3 | ×2.0 | Body-frame linear velocity |
| 6-8 | `base_ang_vel` | 3 | ×0.25 | Body-frame angular velocity |
| 9-11 | `gravity_body` | 3 | raw | Gravity direction in body frame |
| 12-14 | `commands_scaled` | 3 | ×[2.0, 2.0, 0.25] | Velocity commands |
| 15-26 | `dof_pos_scaled` | 12 | ×1.0 (relative to default) | Joint positions |
| 27-38 | `dof_vel_scaled` | 12 | ×0.05 | Joint velocities |
| 39-41 | `last_actions` | 3 | raw [-1,1] | Previous RL action |
| 42 | `base_height_norm` | 1 | (h-0.3)/0.3 | Normalized height |
| 43-45 | `actions` | 3 | raw [-1,1] | Current RL action |
| 46-47 | `contact_proxy` | 2 | contact proxy signal | Padding |

### Reward Function

Reward $R_t = \Delta t \cdot r_t$, where $\Delta t = 0.02\text{s}$ (control period), $r_t$ is instantaneous reward:

**Positive Incentives**:

$$
\begin{aligned}
r_t^{\text{vel\_xy}} &= \underbrace{2.0}_{w_{\text{vel\_xy}}} \cdot \exp\!\left(-\frac{\|\mathbf{v}_{\text{cmd},xy} - \mathbf{v}_{xy}\|^2}{0.25}\right)
\quad\text{(速度跟踪：指数衰减，温度 }\tau=0.25\text{)} \\[8pt]
r_t^{\text{yaw}} &= \underbrace{0.5}_{w_{\text{yaw}}} \cdot \exp\!\left(-\frac{(\omega_{\text{cmd},z} - \omega_z)^2}{0.25}\right)
\quad\text{(偏航跟踪)} \\[8pt]
r_t^{\text{stable}} &= \underbrace{2.0}_{w_{\text{stable}}} \cdot \exp\!\left(-\frac{\|\boldsymbol{\omega}_{xy}\|^2}{0.5}\right)
\quad\text{(三足稳定性：身体角速度越小越好，温度 }\tau=0.5\text{)}
\end{aligned}
$$

**Negative Penalties**:

$$
\begin{aligned}
r_t^{\text{vel\_z}} &= -\underbrace{2.0}_{w_{\text{vel\_z}}} \cdot v_z^2 
\quad\text{(垂直速度：防止跳跃)} \\[6pt]
r_t^{\text{ang\_xy}} &= -\underbrace{5.0}_{w_{\text{ang\_xy}}} \cdot \|\boldsymbol{\omega}_{xy}\|^2 
\quad\text{(横滚/俯仰角速度抑制)} \\[6pt]
r_t^{\text{torque}} &= -\underbrace{0.00005}_{w_{\tau}} \cdot \|\boldsymbol{\tau}\|^2 
\quad\text{(力矩：节能)} \\[6pt]
r_t^{\text{collision}} &= -\underbrace{1.0}_{w_{\text{col}}} \cdot \big(\mathbb{1}_{\text{knee}} + \mathbb{1}_{\text{hip}}\big) 
\quad\text{(碰撞：膝盖/胯部接地)} \\[6pt]
r_t^{\text{com}} &= -\underbrace{0.1}_{w_{\text{com}}} \cdot \|\mathbf{a}_t\|^2 
\quad\text{(COM幅度：鼓励小幅度精细调整)} \\[6pt]
r_t^{\text{smooth}} &= -\underbrace{0.1}_{w_{\text{smooth}}} \cdot \|\mathbf{a}_t - \mathbf{a}_{t-1}\|^2 
\quad\text{(动作平滑：抑制COM突变)} \\[6pt]
r_t^{\text{height}} &= -\underbrace{0.5}_{w_{\text{height}}} \cdot (h - 0.3)^2 
\quad\text{(高度：目标高度 }h^* = 0.3\text{m)}
\end{aligned}
$$

**Total instantaneous reward**:

$$
r_t = r_t^{\text{vel\_xy}} + r_t^{\text{yaw}} + r_t^{\text{stable}} + r_t^{\text{vel\_z}} + r_t^{\text{ang\_xy}} + r_t^{\text{torque}} + r_t^{\text{collision}} + r_t^{\text{com}} + r_t^{\text{smooth}} + r_t^{\text{height}}
$$

**Termination conditions** (episode ends when triggered):

$$
\begin{aligned}
\text{done} = \;&\mathbb{1}\big(\|\mathbf{F}_{\text{trunk}}\| > 1\text{N}\big) 
\;\lor\; \mathbb{1}\big(\|\mathbf{F}_{\text{knee}}\| > 1\text{N}\big) 
\;\lor\; \mathbb{1}\big(\|\mathbf{F}_{\text{hip}}\| > 1\text{N}\big) \\
\lor\;&\mathbb{1}(h < 0.1\text{m}) \;\lor\; \mathbb{1}(h > 0.6\text{m}) 
\;\lor\; \mathbb{1}(\text{step} > 750)
\end{aligned}
$$

### PPO Update Process

Main training loop: each rollout collects $\underbrace{64}_{N_{\text{envs}}} \times \underbrace{24}_{T} = 1536$ transitions, then performs PPO update.

**GAE Computation** (backward recurrence):

$$
\hat{A}_t = \delta_t + \gamma\lambda(1-d_t)\hat{A}_{t+1},\quad
\tilde{A}_t = \frac{\hat{A}_t - \mu_{\hat{A}}}{\sigma_{\hat{A}} + 10^{-8}}
$$

**PPO Update** ($E=5$ epochs $\times$ $B=4$ mini-batches $=20$ gradient updates):

$$
\begin{aligned}
\text{for epoch } e = 1,\dots,5:& \\
\text{for batch } b = 1,\dots,4:& \\
&\quad \rho_t = \frac{\pi_\theta(\mathbf{a}_t \mid \mathbf{o}_t)}{\pi_{\theta_{\text{old}}}(\mathbf{a}_t \mid \mathbf{o}_t)}
\quad\text{(重要性采样比率)} \\[4pt]
&\quad \mathcal{L}^{\text{CLIP}} = -\min(\rho_t \tilde{A}_t,\; \operatorname{clip}(\rho_t, 0.8, 1.2)\tilde{A}_t) \\[4pt]
&\quad \mathcal{L}^{\text{VF}} = \max((V_\phi - \hat{G})^2,\; (V_{\text{clipped}} - \hat{G})^2) \\[4pt]
&\quad \mathcal{L} = \mathcal{L}^{\text{CLIP}} + \underbrace{1.0}_{c_1}\mathcal{L}^{\text{VF}} - \underbrace{0.01}_{c_2}H(\pi_\theta) \\[4pt]
&\quad \theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L},\quad
\phi \leftarrow \phi - \eta \nabla_\phi \mathcal{L},\quad
\|\nabla\mathcal{L}\| \leq 1.0
\end{aligned}
$$

**Adaptive Learning Rate** (checked per mini-batch):

$$
\eta \leftarrow 
\begin{cases}
\eta / 1.5, & D_{\text{KL}} > 0.02 \\
\eta \times 1.5, & D_{\text{KL}} < 0.005
\end{cases}
$$

**Total training**: 5000 iterations $\times$ 1536 transitions/iter $= 7.68 \times 10^6$ transitions, $5000 \times 20 = 10^5$ gradient updates.

---

## MPC Low-Level Controller

### Go2IK — Analytic Inverse Kinematics

Each leg solved independently. Given foot position in body frame $\mathbf{p}_{\text{foot}} = [p_x, p_y, p_z]^\top$, solve for hip/thigh/calf joint angles via geometry.

**Hip joint positions** (Go2 measured values):

$$
\mathbf{h}_{\text{FL}} = 
\begin{bmatrix} 0.1934 \\ 0.0465 \\ 0 \end{bmatrix},\;
\mathbf{h}_{\text{FR}} = 
\begin{bmatrix} 0.1934 \\ -0.0465 \\ 0 \end{bmatrix},\;
\mathbf{h}_{\text{RL}} = 
\begin{bmatrix} -0.1934 \\ 0.0465 \\ 0 \end{bmatrix},\;
\mathbf{h}_{\text{RR}} = 
\begin{bmatrix} -0.1934 \\ -0.0465 \\ 0 \end{bmatrix}
$$

**Link lengths** (Go2 kinematic parameters):
$$
l_{\text{hip}} = 0.0955\text{m},\quad
l_{\text{thigh}} = 0.213\text{m},\quad
l_{\text{calf}} = 0.213\text{m}
$$

### MPCController — Tripod Gait Controller

**Gait phase design**:
```
腿       相位偏移    摆动相(0~0.15)    支撑相(0.15~1.0)
─────────────────────────────────────────────────────
FL(0)    0.000      [0,   0.15)        [0.15, 1.0)
FR(1)    (特殊)     由 fr_lift_alpha 控制
RL(2)    0.500      [0.5, 0.65)        [0.65, 0.5) 跨周期
RR(3)    0.750      [0.75, 0.90)       [0.90, 0.75) 跨周期
```

**FR leg special handling**:
```
fr_lift_alpha = 0 (grounded/quadruped):
  → foot pose = nominal pose, com_offset 100% active
  → used for QUAD_WALK

fr_lift_alpha = 1 (retracted/tripod):
  → foot pose = FR_RETRACT (retracted 4cm back, raised to -0.11m)
  → com_offset NOT applied to FR leg
  → used for tripod standing/walking

0 < fr_lift_alpha < 1 (transition):
  → foot XYZ linear interpolation
  → com_offset fades out at (1-alpha) weight
```

**COM offset superposition** (core: indirect center of mass control):

$$
\mathbf{p}_{\text{foot}}^{\text{final}} = 
\begin{bmatrix}
f_x(\phi_{\text{leg}}) \\ f_y \\ f_z(\phi_{\text{leg}})
\end{bmatrix} - 
\underbrace{\begin{bmatrix} c_x \\ c_y \\ c_z \end{bmatrix}}_{\text{com\_offset}}
$$

### BalanceController — IMU Closed-Loop Balancing (Pure MPC mode only)

PD control law (50Hz, $\Delta t = 0.02\text{s}$):

Lateral (Y-axis) balance — roll detection:

$$
\Delta c_y^{\text{raw}} = \underbrace{K_p^{\text{roll}}}_{\text{0.15 m/rad}} \cdot \theta_{\text{roll}} + \underbrace{K_d^{\text{roll}}}_{\text{0.08 m}\cdot\text{s/rad}} \cdot \omega_x
$$

Longitudinal (X-axis) balance — pitch detection:

$$
\Delta c_x^{\text{raw}} = \underbrace{K_p^{\text{pitch}}}_{\text{0.10 m/rad}} \cdot \theta_{\text{pitch}} + \underbrace{K_d^{\text{pitch}}}_{\text{0.06 m}\cdot\text{s/rad}} \cdot \omega_y
$$

> **Note**: In RL mode, the BalanceController is **not used**; COM offset is fully determined by the neural network.

---

## Hyperparameter Summary

### PPO Algorithm

| Parameter | Value | Meaning |
|------|-----|------|
| `clip_param` | **0.2** | PPO clipping range ε |
| `entropy_coef` | **0.01** | Entropy regularization coefficient |
| `num_learning_epochs` | **5** | Training epochs per batch |
| `num_mini_batches` | **4** | Number of mini-batches |
| `learning_rate` | **1e-3** (adaptive) | Initial learning rate |
| `schedule` | **adaptive** | Adaptive LR scheduling |
| `gamma` | **0.99** | Discount factor |
| `lam` | **0.95** | GAE λ parameter |
| `desired_kl` | **0.01** | Target KL divergence |
| `max_grad_norm` | **1.0** | Gradient clipping norm |
| `value_loss_coef` | **1.0** | Value loss weight |
| `use_clipped_value_loss` | **True** | Clip value loss |

### Network Architecture

| Parameter | Value |
|------|-----|
| Input dim | **48** (observation) |
| Output dim | **3** (COM offset) |
| Actor hidden layers | **[512, 256, 128]** |
| Critic hidden layers | **[512, 256, 128]** |
| Activation | **ELU** |
| Initial exploration std | **1.0** |
| Action space | **[-1, 1]³** |

### COM Output Ranges

| Parameter | Range [min, max] | scale (half-width) | bias (center) |
|------|-----------------|-------------|-------------|
| COM X | **[0.023, 0.27]** | 0.1235 | 0.1465 |
| COM Y | **[0.063, 0.067]** | 0.002 | 0.065 |
| COM Z | **[-0.01, 0.01]** | 0.01 | 0.0 |

> **Design intent**: X range biased forward (CoM shifts rearward onto hind legs), Y range extremely narrow (locked within support triangle), Z range minimal (maintain height stability).

### Reward Function Weights

| Reward Term | Weight | Formula |
|--------|------|------|
| XY Velocity Tracking | **+2.0** | `exp(-‖cmd_xy - vel_xy‖² / 0.25)` |
| Yaw Rate Tracking | **+0.5** | `exp(-(cmd_yaw - vel_yaw)² / 0.25)` |
| Tripod Stability | **+2.0** | `exp(-‖ang_vel_xy‖² / 0.5)` |
| Vertical Velocity Penalty | **-2.0** | `vel_z²` |
| Angular Velocity Penalty | **-5.0** | `‖ang_vel_xy‖²` |
| Torque Penalty | **-5e-5** | `‖τ‖²` |
| Collision Penalty | **-1.0** | `knee collision + hip collision` |
| COM Magnitude Penalty | **-0.1** | `‖action‖²` |
| Action Smoothness Penalty | **-0.1** | `‖action - last_action‖²` |
| Height Penalty | **-0.5** | `(height - 0.3)²` |

> **All reward terms multiplied by `dt = 0.02`**

---

## Real-Robot Deployment

### Deployment Architecture

```
deploy_walk1.py (MPCDeployer)
    │
    ├─ Communication Layer ──────────────────────────
    │   ChannelFactoryInitialize → Unitree SDK DDS
    │   ├─ rt/lowcmd  (ChannelPublisher): send joint commands
    │   └─ rt/lowstate (ChannelSubscriber): receive robot state
    │
    ├─ Control Modes ────────────────────────────────
    │   ├─ use_rl = False → BalanceController (IMU PD balance)
    │   └─ use_rl = True  → WeightPolicy (RL neural network)
    │
    ├─ State Machine ────────────────────────────────
    │   DAMP → INITIAL_STAND → QUAD_WALK → QUAD_STAND
    │   → PRE_SHIFT → LIFT_FR → TRIPOD_STAND
    │   → WALK_START → WALK
    │
    └─ Per-Frame Control Loop (50Hz) ────────────────
        ① Read imu_state + motor_state[0..11]
        ② Build observation (RLObserver, 48-dim)
        ③ RL inference / IMU balance → COM offset [3-dim]
        ④ MPCController.compute_control()
              → Go2IK.solve() → joint targets [12-dim]
        ⑤ PD: torques = Kp×Δpos + Kd×Δvel
        ⑥ Joint index remapping (sdk_to_ctrl / ctrl_to_sdk)
        ⑦ Send LowCmd to robot
```

### Joint Index Mapping

Unitree SDK and algorithm-side joint ordering differ:

```
SDK Index:    [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
               RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]

Ctrl Index:   [FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
               RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf]

Mapping: SDK_TO_CTRL = [3,4,5, 0,1,2, 9,10,11, 6,7,8]
```

### RL/MPC Hybrid Control in Deployment

A hybrid control strategy is used in real deployment:

```python
if use_rl:
    rl_com = rl_policy.inference(obs)  # NN output [dx, dy, dz]
    hybrid_com = [
        BalanceController.NOM_X,  # X: fixed empirical value 0.025
        rl_com[1],                # Y: use RL output
        rl_com[2],                # Z: use RL output
    ]
else:
    com_offset = BalanceController.update(roll, pitch, gyro_x, gyro_y)
```

> **Design intent**: The X-direction (fore-aft) has the greatest impact on stability; a safe fixed empirical value replaces RL output during deployment. Only Y (lateral balance) and Z (height regulation) are controlled by RL.

---

## Design Insights

### Why Hierarchical Architecture?

| Comparison | End-to-End RL (12-dim action) | Hierarchical RL+MPC (3-dim action) |
|----------|---------------------|----------------------|
| Action Space | 12-dim (direct joint positions) | 3-dim (COM offset) |
| Learning Difficulty | High (learn gait + balance simultaneously) | Low (MPC provides stable gait, RL only tunes balance) |
| Safety | Low (may output dangerous actions) | High (MPC clipping + COM range constraints) |
| Interpretability | Low (black box) | High (COM offset has clear physical meaning) |

### Key Technical Features

1. **CPU Training**: Entire pipeline runs on CPU (`CUDA_VISIBLE_DEVICES=""`), 64 parallel environments, lowering hardware requirements
2. **Go1 URDF approximates Go2**: Training uses Unitree Go1 URDF model approximating Go2; kinematic parameters are nearly identical
3. **Independent IK Solver**: Self-implemented analytic IK (`Go2IK`), no external library dependencies, lightweight and interpretable
4. **Progressive Deployment**: State machine transitions from quad→COM shift→FR lift→tripod stand→walk, avoiding abrupt state changes
5. **Joint Order Mapping**: Training and deployment use different joint orderings; explicit `sdk_to_ctrl` / `ctrl_to_sdk` mapping required

### Training vs Deployment Parameter Differences

| Parameter | Training | Deployment | Reason |
|------|------|------|------|
| Gait Period | 0.6s | 1.2s | Training: fast convergence; Deployment: stable & safe |
| Step Height | 0.07m | 0.025m | Training: large motions for exploration; Deployment: low lift reduces impact |
| Exploration Noise | std=1.0 | None | Training needs exploration; deployment needs stability |
| X-direction COM | RL free output | Fixed empirical value | X-direction safety prioritized in deployment |

### Potential Improvements

- **Domain Randomization**: Randomize friction, mass, motor delay during training to improve sim2real transfer
- **Terrain Generalization**: Currently trained on flat ground only; slopes, steps, etc. could be introduced
- **Expand COM_Y Range**: Current 0.004m extremely narrow range may limit disturbance recovery
- **RL Controls Gait Parameters**: Let RL also output gait period, step height, etc.
- **Teacher-Student Distillation**: Use trained MPC BalanceController as supervision to accelerate RL convergence

---

## File Index

| File | Path | Core Function |
|------|------|----------|
| Training Entry | `RL_Environment/train_go2_rl_mpc_simple.py` | Hydra config loading, PPO training launch |
| Training Env | `RL_Environment/tasks/go2_rl_mpc_simple.py` | Observation/reward/COM decode/MPC invocation |
| MPC Controller | `Go2_Controller/mpc_walk1.py` | Go2IK + MPCController + BalanceController |
| Real Deployment | `deploy_walk1.py` | MPCDeployer + RLObserver + state machine |
| Model Loader | `RL_Environment/WeightPolicy.py` | Create network, load weights, inference |
| PPO Algorithm | `extern/rsl_rl/rsl_rl/algorithms/ppo.py` | PPO update, adaptive LR |
| Actor-Critic | `extern/rsl_rl/rsl_rl/modules/actor_critic.py` | Network structure definition |
| Training Runner | `extern/rsl_rl/rsl_rl/runners/on_policy_runner.py` | Rollout→Update main loop |
| Experience Storage | `extern/rsl_rl/rsl_rl/storage/rollout_storage.py` | GAE computation, mini-batch generation |
| PPO Config | `RL_Environment/tasks/legged_config_ppo.py` | PPO hyperparameter config class |
| Training YAML | `RL_Environment/cfg/task/Go2RLMPCSimple.yaml` | Complete training hyperparameters |
| CPU Config | `RL_Environment/cfg/config_cpu.yaml` | CPU training environment config |

---

## Citation

If you use this work in your research, please cite both the original work and this extension:

**Original work**:
```bibtex
@software{Zhuang2022rlmpc,
  author = {Yulun Zhuang and Wei Zhang},
  title = {rl-mpc-locomotion},
  year = {2022},
  url = {https://github.com/silvery107/rl-mpc-locomotion}
}
```

## License

MIT License. Copyright (c) 2021 Yulun Zhuang.

This project is based on [rl-mpc-locomotion](https://github.com/silvery107/rl-mpc-locomotion). See [LICENSE](LICENSE) for details.
