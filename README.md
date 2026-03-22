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

### 1.3) 奖励塑形与通信优化（已实现）

- 奖励函数新增：
  - 危险区接近惩罚（`danger_zone_distance` + `proximity_penalty_coeff`）
  - 动作平滑惩罚（`action_smoothness_coeff`）
- 已在本地仿真与 ROS2 训练回路同时接入。
- 模型容量默认提升（适配高显存场景）：
  - `hidden_size=1024`
  - `attention_layers=3`
  - `attention_dim=128`
- P2P 通信统计新增“基于参数标准差阈值”的有效负载估计：
  - `P2PConfig.weight_std_threshold=0.01`
  - 优化通信开销评估口径，减少低变化参数对字节统计的影响。

### 1.4) 5090 训练吞吐优化（已实现）

- 训练侧新增（默认面向 CUDA 生效）：
  - AMP 混合精度：`use_amp=True`, `amp_dtype=bf16|fp16`
  - TF32：`enable_tf32=True`
  - 批量更新调度：`update_every=4`, `gradient_updates=4`
  - 可选 `torch.compile`：`enable_torch_compile`, `compile_mode`
- 在 `experiment.py` 与 `ros2_training.py` 统一接入 `configure_torch_runtime()`。
- CPU 回退兼容：在 CPU 上自动退化为每步单次更新，避免测试与行为回归。

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

## Phase 1 & 2 性能优化（最新）⚡

**综合加速约 2.5x**，包含以下优化：
- ✅ FP16 权重量化（通信字节减半）
- ✅ 异步 P2P 聚合（训练与通信并行）
- ✅ 延迟 Actor 更新（计算量减半 + 提升稳定性）
- ✅ 选择性层交换（跳过未变化层）

### 推荐命令（30机器人完整优化版）

```bash
python scripts/run_experiment.py \
    --mode p2p \
    --robots 30 \
    --timesteps 20000 \
    --env-step-workers 8 \
    --exchange-interval-steps 40 \
    --cooldown-steps 80 \
    --weight-std-threshold 0.05 \
    --grid-cell-size 2.0 \
    --frame-stack 4 \
    --gpu-replay-buffer \
    --actor-update-interval 2 \
    --layer-diff-threshold 0.001 \
    --run-name p2p_30r_optimized \
    --progress-every 2000
```

### 对照组（禁用新优化，用于性能对比）

```bash
python scripts/run_experiment.py \
    --mode p2p \
    --robots 30 \
    --timesteps 20000 \
    --env-step-workers 8 \
    --exchange-interval-steps 40 \
    --cooldown-steps 80 \
    --weight-std-threshold 0.05 \
    --disable-fp16-comm \
    --disable-async-exchange \
    --actor-update-interval 1 \
    --layer-diff-threshold 0 \
    --run-name p2p_30r_baseline \
    --progress-every 2000
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--actor-update-interval` | 2 | Actor延迟更新间隔（借鉴TD3） |
| `--disable-fp16-comm` | False | 禁用FP16量化（默认启用） |
| `--layer-diff-threshold` | 0.001 | 选择性层交换阈值（0=全交换） |
| `--disable-async-exchange` | False | 禁用异步聚合（默认启用） |

### 验证结果（8机器人 smoke test）

- FP32 baseline: 38,236,160 bytes
- FP16 优化: 19,118,080 bytes（**减少50%** ✅）

---

## 性能调优与故障排除

### P2P通信过于频繁（P2P占比>20%）

**症状**：日志显示 `time: p2p=22%` 或更高，`exch=4000+`/2000步

**诊断**：
```bash
# 查看最新日志
tail -1 artifacts/logs/p2p/your_run.log
# 如果 p2p > 15% 且 exch/step > 1.5，说明通信过于频繁
```

**方案A：推荐修复（降低通信频率）**

```bash
python scripts/run_experiment.py \
    --mode p2p \
    --robots 30 \
    --timesteps 20000 \
    --env-step-workers 8 \
    --exchange-interval-steps 100 \
    --cooldown-steps 200 \
    --weight-std-threshold 0.05 \
    --grid-cell-size 2.0 \
    --frame-stack 4 \
    --gpu-replay-buffer \
    --actor-update-interval 2 \
    --layer-diff-threshold 0.001 \
    --comm-radius 1.5 \
    --run-name p2p_30r_fixed \
    --progress-every 2000
```

**预期改善**：P2P从22%降至5-8%，速度提升2-3x

**方案B：激进优化（极速模式）**

```bash
python scripts/run_experiment.py \
    --mode p2p \
    --robots 30 \
    --timesteps 20000 \
    --env-step-workers 8 \
    --exchange-interval-steps 200 \
    --cooldown-steps 400 \
    --weight-std-threshold 0.1 \
    --grid-cell-size 3.0 \
    --frame-stack 4 \
    --gpu-replay-buffer \
    --actor-update-interval 2 \
    --layer-diff-threshold 0.005 \
    --comm-radius 1.2 \
    --run-name p2p_30r_aggressive \
    --progress-every 2000
```

**预期改善**：P2P降至2-5%，速度提升3-4x

---

### GPU利用率低（train占比<30%，env占比>60%）

**症状**：`time: env=70% train=15%`

**解决方案**：
```bash
# 增加并行环境worker充分利用CPU
python scripts/run_experiment.py \
    --mode p2p \
    --robots 30 \
    --timesteps 20000 \
    --env-step-workers 16 \
    --exchange-interval-steps 100 \
    --run-name p2p_more_workers
```

---

### 训练越来越慢

**诊断**：观察日志中 `speed=` 是否递减
```bash
grep "step=" artifacts/logs/p2p/your_run.log | grep -E "step=(2000|4000|6000|8000)"
```

**常见原因与解决**：
1. **P2P通信量增长** → 使用上面的方案A/B
2. **内存泄漏** → 更新到最新代码（已修复）
3. **Buffer采样变慢** → 启用 `--gpu-replay-buffer`

---

### 快速性能对比测试

```bash
# Baseline（禁用优化）
python scripts/run_experiment.py \
    --mode p2p --robots 30 --timesteps 2000 \
    --env-step-workers 0 \
    --disable-fp16-comm --disable-async-exchange \
    --actor-update-interval 1 --layer-diff-threshold 0 \
    --exchange-interval-steps 40 \
    --run-name baseline --progress-every 500

# 完全优化
python scripts/run_experiment.py \
    --mode p2p --robots 30 --timesteps 2000 \
    --env-step-workers 8 \
    --exchange-interval-steps 100 --cooldown-steps 200 \
    --comm-radius 1.5 --gpu-replay-buffer \
    --actor-update-interval 2 --layer-diff-threshold 0.001 \
    --run-name optimized --progress-every 500

# 对比速度
grep "step=2000" artifacts/logs/p2p/baseline.log
grep "step=2000" artifacts/logs/p2p/optimized.log
```

---

## 性能模式（5090推荐）

若你追求吞吐而非“严格 P2P 实验”：

```bash
python scripts/run_experiment.py --mode p2p --robots 30 --timesteps 20000 --shared-agent --progress-every 500
```

说明：

- `--shared-agent`：所有机器人共享同一套 Actor/Critic 与回放池；
- 每步仅执行一次集中更新，显著减少多Agent串行训练的CPU开销；
- 这是当前代码里最有效的提速模式之一。

注意：严格 P2P 评估应使用“独立 Agent + 交换”。当前脚本在 `--mode p2p` 下会自动禁用 `--shared-agent`，避免实验设定被破坏。

推荐严格 P2P命令：

```bash
python scripts/run_experiment.py --mode p2p --robots 30 --timesteps 20000 --progress-every 500
```

## 常见问题

### 性能相关

- **P2P通信占比过高（>20%）？**  
  参见上方"性能调优与故障排除"章节的方案A/B，大幅降低 `exchange-interval-steps` 和 `cooldown-steps`。

- **训练越来越慢？**  
  使用 `grep "step=" your_log.log` 查看速度趋势。如果持续下降，检查 P2P 时间占比或更新代码（已修复内存泄漏）。

- **GPU利用率低？**  
  增加 `--env-step-workers` 充分利用CPU多核，或启用 `--gpu-replay-buffer`。

### 训练相关

- **Reward 一直是负的？**  
  训练初期常见。每步惩罚 + 碰撞惩罚会压低总奖励。请看 `episode_return_mean` 和成功率/碰撞率趋势，而不是只看累计和。

- **30 台机器人会不会太卡？**  
  真实 ROS2/Gazebo 常见瓶颈在 CPU 仿真与通信。建议先 headless、降低传感器频率、控制日志频率。

### 日志解读

- **如何看懂新版日志？**  
  ```
  [p2p] step=2000/20000 | eps=125 rew=-1523.4 succ=12 coll=8 | 
        exch=145 bytes=38.2MB | speed=12.34step/s (162.0s) | buf=65% | 
        time: env=45% train=35% p2p=15%
  ```
  - `speed`: 当前训练速度（step/s）
  - `buf`: Replay Buffer填充率
  - `time`: 时间分布（env=环境交互，train=SAC训练，p2p=通信聚合）
  - 理想状态：`env 40-50%`, `train 30-40%`, `p2p <10%`

- **如何退出 ROS2 launch 卡住？**  
  若多次 `Ctrl+C` 不退出，另开终端按 PID 精确结束进程（避免按进程名批量杀）。
