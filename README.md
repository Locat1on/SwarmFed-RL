# SwarmFed-RL

强化学习版 P2P 联邦学习实验工程（SAC + 多机器人避障 + ROS2/Gazebo）。

## 目录

- [项目概览](#项目概览)
- [v2 重构要点](#v2-重构要点)
- [当前实现状态](#当前实现状态)
- [快速开始（本地模拟）](#快速开始本地模拟)
- [真实 ROS2/Gazebo 模式](#真实-ros2gazebo-模式)
- [统一 artifacts 输出结构](#统一-artifacts-输出结构)
- [架构与算法细节](#架构与算法细节)
- [实验流程与对照设置](#实验流程与对照设置)
- [参数参考](#参数参考)
- [常见问题](#常见问题)

## 项目概览

本项目目标是在动态拓扑下验证：

- 多机器人本地强化学习是否可通过 P2P 权重交换加速收敛。
- 与 `local` / `centralized(FedAvg)` 对照相比，`p2p` 的性能与通信开销差异。
- 在恶意节点注入时，防御策略（`cosine` / `trimmed_mean` / `krum`）的鲁棒性。

核心约束：

- 仅交换 **Actor** 权重。
- 触发条件为"距离阈值 + 冷却窗口"。
- 统一按 Total Timesteps 评估。

## v2 重构要点

v2 版本对核心训练逻辑进行了重构，解决了原版无法收敛的根本问题。**实测 3 机器人 P2P 模式在 10K 步内成功率达到 73%**。

### 修复的关键收敛问题

| 问题 | 原版行为 | v2 修复 |
|------|---------|---------|
| **训练严重不足** | 每步仅训练 1/N 个 Agent（轮询），加上 `update_every=2` 跳步，实际训练频率 1/(2N) | 每步训练**所有** Agent，无跳步 |
| **网络过度参数化** | 512-hidden, 3层残差 + 3层 Transformer，共 ~2M 参数 | 256-hidden, 2层 MLP，共 ~141K 参数 |
| **经验不共享** | 每个 Agent 仅从自己的 Buffer 学习 | 共享 Replay Buffer，所有 Agent 从集体经验学习 |
| **P2P 融合过于保守** | beta=0.7（70% 保留本地权重） | beta=0.5（50/50 平衡融合） |
| **异步交换竞态** | 异步线程 + 强制同步等待 + shadow agents | 同步交换，无竞态风险 |
| **NaN 静默失败** | alpha/actor loss NaN 后训练静默崩溃 | 所有 loss 增加 NaN 检测与跳过 |

### 默认参数变更

| 参数 | 原版 | v2 | 原因 |
|------|------|-----|------|
| `hidden_size` | 512 | 256 | 28维输入不需要大网络 |
| `hidden_layers` | 3 | 2 | 减少过拟合风险 |
| `actor_encoder` | attention | mlp | Transformer 对 28D 输入是杀鸡用牛刀 |
| `update_every` | 2 | 1 | 不再跳过训练步 |
| `actor_update_interval` | 2 | 1 | 不再延迟 Actor 更新 |
| `update_after` | 1000 | 256 | 更早开始训练 |
| `batch_size` | 512 | 256 | 更频繁的梯度更新 |
| `grad_clip_norm` | 5.0 | 1.0 | 更严格的梯度裁剪 |
| `beta` | 0.7 | 0.5 | 平衡的权重融合 |
| `comm_radius` | 2.0 | 3.0 | 更多交换机会 |
| `async_exchange` | True | False | 避免竞态 |
| `shared_replay` | - | True | 新增：共享经验池 |

### 收敛验证结果（3 机器人，P2P 模式，10K 步）

| 步数 | Episodes | 成功次数 | 碰撞次数 | Episode Return |
|------|----------|---------|---------|----------------|
| 2000 | 15 | 0 | 3 | -253.4 |
| 4000 | 32 | 4 | 5 | -211.5 |
| 6000 | 59 | 29 | 7 | -83.3 |
| 8000 | 91 | 60 | 7 | -16.3 |
| 10000 | 126 | 92 | 9 | +14.2 |

## 当前实现状态

已完成核心模块（`src/swarmfed_rl/`）：

- `config.py`：实验配置与默认参数。
- `env.py`：本地仿真环境（28维状态：24束雷达 + 速度/角速度/目标距离/朝向误差）。
- `sac.py`：SAC Agent（Actor/Critic/温度/回放缓冲区/共享缓冲区支持）。
- `p2p.py`：P2P 聚合（距离触发 + 冷却窗口 + 网格索引）与防御策略（cosine/trimmed_mean/krum）。
- `experiment.py`：三模式统一训练主循环（`local`/`centralized`/`p2p`）。
- `ros2_runtime.py`：ROS2 状态适配、权重消息分片与重组、Gazebo reset。
- `ros2_training.py`：真实 ROS2 训练回路。

脚本入口（`scripts/`）：

- `run_experiment.py`：本地模拟主入口（支持自动绘图与日志落盘）。
- `run_ros2_experiment.py`：真实 ROS2/Gazebo 入口。
- `run_warmup.py`：本地 warm-up 预训练。
- `plot_metrics.py`：绘图工具。
- `run_full_pipeline.py`：一键完整实验流水线。

## 快速开始（本地模拟）

```bash
python -m pip install -e .
```

### 单次实验

```bash
# P2P 模式（推荐起步配置）
python scripts/run_experiment.py --mode p2p --robots 3 --timesteps 20000 --seed 42

# Local 模式（无通信基线）
python scripts/run_experiment.py --mode local --robots 3 --timesteps 20000 --seed 42

# Centralized 模式（FedAvg 基线）
python scripts/run_experiment.py --mode centralized --robots 3 --timesteps 20000 --seed 42
```

### 三模式对照

```bash
for mode in local centralized p2p; do
  python scripts/run_experiment.py --mode $mode --robots 5 --timesteps 20000 --seed 42 --run-name "${mode}_5robots"
done
```

### 带 TensorBoard 可视化

```bash
python scripts/run_experiment.py --mode p2p --robots 3 --timesteps 20000 --seed 42 --enable-tensorboard
tensorboard --logdir artifacts/tb
```

### 攻击 + 防御

```bash
python scripts/run_experiment.py --mode p2p --robots 10 --timesteps 20000 --seed 42 \
  --defense --defense-strategy cosine \
  --malicious-nodes "8,9" --attack-type gaussian \
  --calibration-steps 500 --attack-start-step 500
```

### 禁用共享经验池（独立 Buffer）

```bash
python scripts/run_experiment.py --mode p2p --robots 5 --timesteps 20000 --seed 42 --no-shared-replay
```

## 真实 ROS2/Gazebo 模式

```bash
python scripts/run_ros2_experiment.py --robot-ids 0,1,2 --timesteps 1000 --seed 42 --topic-prefix /tb3_
```

常用参数：

- `--no-gazebo-reset`
- `--no-reset-on-done`
- `--collision-scan-threshold`
- `--retransmit-count`

无 ROS2 环境可先做自检：

```bash
python scripts/smoke_ros2_runtime.py
```

## 统一 artifacts 输出结构

默认根目录：`artifacts/`

```text
artifacts/
  logs/
    <mode>/
      <run_name>.csv          # 逐步 metrics
      <run_name>_epoch.csv    # Epoch 级 metrics
      <run_name>.log          # 终端输出镜像
  plots/
    <mode>/<run_name>/
  checkpoints/
    <mode>/<run_name>/
  tb/
    <mode>/<run_name>/
  configs/
    <mode>/<run_name>.json    # 配置快照
```

说明：

- 未指定 `--run-name` 时会自动生成时间戳名称。
- 终端输出会自动双写到 `.log` 文件。

## 架构与算法细节

### SAC（Soft Actor-Critic）

- **Actor**：MLP（28D → 256 → 256 → 2D），输出 tanh 压缩的动作均值和对数标准差
- **Critic（Q1, Q2）**：MLP（30D → 256 → 256 → 1），双 Q 网络减少过估计
- **温度 α**：自动调节，目标熵 = -action_dim
- **Target 网络**：Polyak 平均（τ=0.005）

### 环境（SimulatedROS2Env）

- **状态（28D）**：24 束雷达距离（0-4m） + 线速度 + 角速度 + 目标距离 + 朝向误差
- **动作（2D）**：线速度 [0, 0.22] m/s + 角速度 [-1.5, 1.5] rad/s
- **奖励**：progress × 10 + step_penalty(-0.5) - proximity_penalty - action_smoothness + goal_bonus(200) / collision_penalty(-100)

### P2P 聚合

- **触发条件**：每 `exchange_interval_steps`（默认 100）步，距离 < `comm_radius`（默认 3.0m），冷却 ≥ `cooldown_steps`（默认 100）步
- **融合公式**：`merged = beta × local + (1 - beta) × incoming_mean`，默认 beta=0.5
- **邻居查找**：网格空间哈希索引，O(n) 复杂度
- **防御策略**：cosine 相似度过滤 / trimmed mean / Krum

### 共享经验池（Shared Replay Buffer）

v2 默认启用共享经验池：所有 Agent 将经验推入同一个 Buffer，所有 Agent 从中采样训练。这使得每个 Agent 可以从其他机器人的经验中学习，大幅加速收敛。

- 开启：默认行为
- 关闭：`--no-shared-replay`

## 实验流程与对照设置

### 标准对比实验

| 实验组 | 核心验证点 | 启动指令 |
|--------|-----------|---------|
| **Local** | 无通信基准 | `python scripts/run_experiment.py --mode local --robots 5 --timesteps 40000 --run-name comp_local` |
| **FedAvg** | 全局同步效果 | `python scripts/run_experiment.py --mode centralized --robots 5 --timesteps 40000 --run-name comp_fedavg` |
| **P2P (constant β)** | 固定融合效果 | `python scripts/run_experiment.py --mode p2p --robots 5 --timesteps 40000 --run-name comp_p2p_const` |
| **P2P (linear β)** | 进度感知效果 | `python scripts/run_experiment.py --mode p2p --beta-schedule linear --robots 5 --timesteps 40000 --run-name comp_p2p_linear` |

### 关键对比指标

- **学习速度**：对比各组 `episode_return_mean` 上升趋势
- **避障安全性**：对比 `epoch_collisions` 下降速度
- **最终性能**：查看最终 epoch 成功率
- **通信负载**：P2P 字节数 vs Centralized 字节数

### 一键 Pipeline

```bash
python scripts/run_full_pipeline.py --robots 5 --warmup-timesteps 5000 --timesteps 20000 --seeds 42,43,44
```

## 参数参考

### 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | p2p | 训练模式：local / centralized / p2p |
| `--robots` | 3 | 机器人数量 |
| `--timesteps` | None | 总步数（优先于 epochs） |
| `--epochs` | 10 | 训练轮数 |
| `--steps-per-epoch` | 2000 | 每轮步数 |
| `--seed` | 42 | 随机种子 |
| `--progress-every` | 500 | 打印进度间隔 |

### SAC 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sac-batch-size` | 256 | 训练批次大小 |
| `--sac-gradient-updates` | 1 | 每次 train_step 的梯度更新次数 |
| `--actor-update-interval` | 1 | Actor 更新间隔（1=每次都更新） |
| `--gpu-replay-buffer` | False | 使用 GPU 驻留回放缓冲区 |
| `--no-shared-replay` | False | 禁用共享经验池 |

### P2P 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--comm-radius` | 3.0 | 通信半径（米） |
| `--cooldown-steps` | 100 | 同一对的交换冷却步数 |
| `--exchange-interval-steps` | 100 | 全局交换间隔步数 |
| `--beta-schedule` | constant | Beta 调度：constant / linear / exponential |
| `--grid-cell-size` | 3.5 | 网格索引单元大小 |

### 防御参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--defense` | False | 启用防御 |
| `--defense-strategy` | cosine | 防御策略：cosine / trimmed_mean / krum |
| `--malicious-nodes` | "" | 恶意节点 ID（逗号分隔） |
| `--attack-type` | zero | 攻击类型：zero / gaussian |
| `--calibration-steps` | 500 | 防御校准步数 |

## 常见问题

### Reward 一直是负的？

训练初期正常。每步惩罚 + 碰撞惩罚会压低总奖励。请关注 `episode_return_mean` 趋势（应逐渐上升）和成功率，而非累计奖励。

### 训练太慢？

- 减少机器人数量（3-5 即可验证算法）
- 使用 `--shared-agent` + `--mode local` 快速验证避障策略（所有机器人共享一个网络）
- 启用 `--gpu-replay-buffer`

### 如何看懂日志？

```
[p2p] epoch=3/5 step=6000/10000 | eps=59 rew=-5119.3 succ=29 coll=7 |
      ep_ret=-83.3 | exch=132 bytes=149.0MB | speed=23.6step/s |
      time: env=3% train=97% p2p=0%
```

- `eps`：完成的 episode 数
- `succ/coll`：累计成功/碰撞次数
- `ep_ret`：最近 episode 平均回报（核心收敛指标）
- `exch`：P2P 交换次数
- `speed`：训练速度
- `time`：时间分布（env=环境交互，train=SAC训练，p2p=通信聚合）
