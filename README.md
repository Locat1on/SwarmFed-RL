# SwarmFed-RL

强化学习版 P2P 联邦学习实验工程（SAC + 多机器人 + ROS2/Gazebo）。

## 目录

- [项目概览](#项目概览)
- [当前实现状态](#当前实现状态)
- [快速开始（本地模拟）](#快速开始本地模拟)
- [真实 ROS2/Gazebo 模式](#真实-ros2gazebo-模式)
- [统一 artifacts 输出结构](#统一-artifacts-输出结构)
- [关键修复与稳定性增强（最新）](#关键修复与稳定性增强最新)
- [实验流程与对照设置](#实验流程与对照设置)
- [一键 Pipeline](#一键-pipeline)
- [常见问题](#常见问题)

## 项目概览

本项目目标是在动态拓扑下验证：

- 多机器人本地强化学习是否可通过 P2P 权重交换加速收敛。
- 与 `local` / `centralized(FedAvg)` 对照相比，`p2p` 的性能与通信开销差异。
- 在恶意节点注入时，防御策略（`cosine` / `trimmed_mean` / `krum`）的鲁棒性。

核心约束：

- 仅交换 **Actor** 权重。
- 触发条件为“距离阈值 + 冷却窗口”。
- 统一按 Total Timesteps 评估。

## 当前实现状态

已完成核心模块（`src/swarmfed_rl/`）：

- `config.py`：实验配置与默认参数。
- `env.py`：本地仿真环境（28维状态）。
- `sac.py`：SAC（Actor/Critic、温度、回放、更新）。
- `p2p.py`：P2P 聚合与防御策略。
- `experiment.py`：三模式统一训练主循环（`local`/`centralized`/`p2p`）。
- `ros2_runtime.py`：ROS2 状态适配、权重消息分片与重组、Gazebo reset。
- `ros2_training.py`：真实 ROS2 训练回路。

脚本入口（`scripts/`）：

- `run_experiment.py`：本地模拟主入口（支持自动绘图与日志落盘）。
- `run_ros2_experiment.py`：真实 ROS2/Gazebo 入口。
- `run_warmup.py`：本地 warm-up 预训练。
- `plot_metrics.py` / `simple_plot.py`：绘图工具。
- `run_full_pipeline.py`：一键完整实验流水线。

## 快速开始（本地模拟）

```bash
python -m pip install -e .
python scripts/run_experiment.py --mode p2p --robots 30 --timesteps 5000 --run-name p2p_30_baseline
```

三组对照：

```bash
python scripts/run_experiment.py --mode local --robots 30 --timesteps 5000 --run-name local_30
python scripts/run_experiment.py --mode centralized --robots 30 --timesteps 5000 --run-name fedavg_30
python scripts/run_experiment.py --mode p2p --robots 30 --timesteps 5000 --run-name p2p_30
```

攻击 + 防御：

```bash
python scripts/run_experiment.py --mode p2p --robots 30 --timesteps 5000 --run-name p2p_30_def \
  --defense --defense-strategy krum --defense-krum-malicious 1 \
  --malicious-nodes 1 --attack-type gaussian --calibration-steps 500 --attack-start-step 500
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

默认根目录：`artifacts\`

```text
artifacts/
  logs/
    <mode>/
      <run_name>.csv
      <run_name>.log
  plots/
    <mode>/<run_name>/
  checkpoints/
    <mode>/<run_name>/
  tb/
    <mode>/<run_name>/
  configs/
    <mode>/<run_name>.json
```

说明：

- 未指定 `--run-name` 时会自动生成时间戳名称。
- 终端输出会自动双写到 `.log` 文件（详见下一节）。

## 关键修复与稳定性增强（最新）

### 1) SAC 动作二次归一化错误（已修复）

- 问题：Replay Buffer 中动作已归一化到 `[-1,1]`，训练时又二次归一化，导致 Critic 输入畸变。
- 修复：`sac.py` 中 Critic 训练直接使用采样动作，不再重复归一化。

### 1.1) SAC 骨干网络增强（已实现）

- 将原始浅层 MLP 升级为更深的网络骨干：
  - 默认 `hidden_layers=4`
  - 默认 `hidden_size=256`
  - 默认启用残差块与 LayerNorm（`residual=True`）
- 覆盖范围：
  - Actor backbone
  - Critic / Target Critic backbone
- 兼容性：
  - 训练流程、数据结构、脚本参数保持兼容；
  - 仅提升表达能力，不改变动作/奖励接口。

### 1.2) Actor 雷达编码器升级（已实现）

- Actor 输入支持 `1D-CNN` 雷达编码（默认开启）：
  - 前 24 维雷达使用卷积编码；
  - 其余尾部状态（速度、角速度、目标距离、朝向误差）走独立线性编码；
  - 二者融合后进入深层残差骨干。
- 配置项（`SACConfig`）：
  - `actor_encoder: str = "attention"`（`attention | cnn | mlp`）
  - `attention_dim: int = 64`
  - `attention_heads: int = 4`
  - `attention_layers: int = 1`
  - `actor_use_cnn` 保留为兼容字段（deprecated）
- Critic 仍保持 MLP 输入，保证训练稳定与兼容。

补充：已实现 Actor 自注意力编码器（Transformer Encoder），现在默认使用 `attention`。

### 2) ROS2 传感器 QoS 不匹配（已修复）

- 问题：`/scan`、`/odom` 订阅使用默认可靠 QoS，可能收不到 SensorData(BestEffort)。
- 修复：改用 `qos_profile_sensor_data`。

### 3) 分片损坏回退风险（已修复）

- 问题：分片解析失败后错误回退到 legacy 解包。
- 修复：检测到 chunk 魔数且解析失败时直接丢弃脏包，不再回退。

### 4) 本地雷达缺乏方向性（已修复）

- 问题：24束雷达同值，无法学习方向性避障。
- 修复：改为 24 束简化 ray-casting，按方向计算与边界交点距离。

### 5) 评估指标增强（已补齐）

- 新增 Episode Return 统计：
  - CSV: `episode_return_mean`
  - TensorBoard: `episode/return_mean`, `episode/return_latest`

### 6) 终端日志自动落盘（已补齐）

- `run_experiment.py`：
  - 终端输出同步写入 `artifacts\logs\<mode>\<run_name>.log`
- `run_ros2_experiment.py`：
  - 终端输出同步写入 `artifacts\logs\ros2\<run_name>.log`

## 实验流程与对照设置

推荐阶段：

1. Warm-up（禁用通信，本地训练）
2. 对照组：`local`、`centralized`
3. 实验组：`p2p`
4. 攻击注入 + 防御策略对照
5. 数据分析与绘图

建议关注指标：

- 成功率（Success Rate）
- 碰撞率（Collision Rate）
- `episode_return_mean`
- 通信字节数（Communication Bytes）

## 一键 Pipeline

```bash
python scripts/run_full_pipeline.py --robots 30 --warmup-timesteps 5000 --timesteps 5000 --seeds 42,43,44
```

默认输出在：`artifacts\pipeline\`

## 常见问题

- **Reward 一直是负的？**  
  训练初期常见。每步惩罚 + 碰撞惩罚会压低总奖励。请看 `episode_return_mean` 和成功率/碰撞率趋势，而不是只看累计和。

- **30 台机器人会不会太卡？**  
  真实 ROS2/Gazebo 常见瓶颈在 CPU 仿真与通信。建议先 headless、降低传感器频率、控制日志频率。

- **如何退出 ROS2 launch 卡住？**  
  若多次 `Ctrl+C` 不退出，另开终端按 PID 精确结束进程（避免按进程名批量杀）。
