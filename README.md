
---

## 当前代码实现状态（Phase-1 已启动）

已按本指南完成第一阶段核心闭环的首版代码骨架，位于 `src/swarmfed_rl/`：

- `config.py`：实验配置（SAC、奖励、P2P 距离与冷却时间、动作范围、总步数等）。
- `env.py`：ROS2 环境接口的本地模拟实现（24 维雷达 + 运动学状态，复合奖励函数）。
- `sac.py`：SAC 核心（Actor/Critic、Replay Buffer、温度参数、自适应更新）。
- `p2p.py`：仅 Actor 权重交换与距离+冷却双条件触发聚合，含通信字节统计。
- `train_phase1.py`：多机器人训练主循环（本地学习 + 动态 P2P 聚合）。
- `ros2_scaffold.py`：ROS2 节点对接脚手架（可在安装 `rclpy` 后接入真实 `/scan`、`/odom`、`/cmd_vel`）。
- `scripts/run_phase1.py`：命令行入口。

### 本地快速运行（无 ROS2）

```bash
python -m pip install -e .
python scripts/run_phase1.py --robots 3 --timesteps 5000 --seed 42
```

说明：上述运行使用本地仿真环境验证训练/P2P 逻辑闭环；后续可将 `ros2_scaffold.py` 与真实 Gazebo/ROS2 Topic 进行对接。

### Warm-up（第二阶段）

```bash
python scripts/run_warmup.py --robots 3 --timesteps 5000 --seed 42 --checkpoint-dir artifacts\checkpoints\warmup --log-csv artifacts\logs\warmup.csv
```

该模式会禁用通信，仅做本地独立训练，并导出每个机器人的 Actor checkpoint。

如需看到实时进度（避免“看起来没反应”），可加：

```bash
--progress-every 100
```

### 三组对照实验入口（第三阶段）

```bash
# 本地训练对照组
python scripts/run_experiment.py --mode local --robots 3 --timesteps 5000 --seed 42 --log-csv artifacts\logs\local.csv

# 中心化 FedAvg 对照组
python scripts/run_experiment.py --mode centralized --robots 3 --timesteps 5000 --seed 42 --log-csv artifacts\logs\centralized.csv

# P2P 实验组
python scripts/run_experiment.py --mode p2p --robots 3 --timesteps 5000 --seed 42 --log-csv artifacts\logs\p2p.csv
```

### 异常节点与防御（第四阶段）

```bash
python scripts/run_experiment.py --mode p2p --robots 3 --timesteps 5000 --seed 42 --defense --malicious-nodes 1 --attack-type gaussian --calibration-steps 500 --attack-start-step 500 --log-csv artifacts\logs\p2p_defense.csv
```

- `--malicious-nodes`：逗号分隔的恶意节点 ID。
- `--attack-type`：`zero`（全零）或 `gaussian`（高斯噪声）。
- `--defense`：开启余弦相似度动态阈值拒绝机制（标定后使用 `μ-3σ`）。

### 指标绘图（第五阶段）

```bash
python scripts/plot_metrics.py --csv artifacts\logs\p2p_defense.csv --out-dir artifacts\plots
```

---

## 真实 ROS2/Gazebo 运行模式（下一轮）

当前仓库已增加真实对接运行时模块（`src/swarmfed_rl/ros2_runtime.py`、`src/swarmfed_rl/ros2_training.py`）和入口脚本（`scripts/run_ros2_experiment.py`）。

### 环境准备

- Ubuntu 22.04 + ROS2 Humble + Gazebo Classic 11（建议）
- 安装并 `source` ROS2 与 Gazebo 环境后，再运行 Python 脚本
- 多机器人命名空间默认约定：`/tb3_0`、`/tb3_1`、`/tb3_2`
  - 每个机器人包含 `/scan`、`/odom`、`/cmd_vel`

### 启动真实实验

```bash
python scripts/run_ros2_experiment.py --robot-ids 0,1,2 --timesteps 1000 --seed 42 --control-hz 10 --topic-prefix /tb3_ --weights-topic /swarm/actor_weights
```

可选参数：
- `--no-gazebo-reset`：禁用 Gazebo reset 服务调用
- `--no-reset-on-done`：episode 结束后不触发重置
- `--collision-scan-threshold`：最小雷达距离判定碰撞阈值（默认 0.14）

### 本地无 ROS2 的最小自检

```bash
python scripts/smoke_ros2_runtime.py
```

该脚本验证：
- `/scan` 降采样 + 状态向量构建（28 维）
- Actor 权重序列化/反序列化与消息打包解包

### 常见问题排查

- 报错“ROS2 runtime unavailable”：未安装或未 `source` ROS2 环境。
- 报错“Robot X not ready: no /scan data within timeout”：对应机器人雷达 topic 未发布或命名空间不匹配。
- Gazebo reset 失败：确认 `gazebo_msgs` 已安装，且 `/gazebo/set_entity_state` 或 `/gazebo/set_model_state` 服务可用。

### 健壮性增强（已实现）

- **通信可靠性**：ROS2 Actor 权重消息支持分片重组、CRC32 校验与可选压缩，降低大包损坏风险。
- **重传开关**：`run_ros2_experiment.py` 支持 `--retransmit-count`，在弱网络下可提高可达率。
- **训练稳定性**：SAC 增加梯度裁剪与非有限损失检测（NaN/Inf 立即报错）。

### 自动化质量检查

```bash
python scripts/run_quality_checks.py
```

会执行：
- `compileall` 编译检查
- `tests/` 下 `unittest` 自动化测试

### 防御对照与复现增强（新增）

- `run_experiment.py` 新增防御策略参数：
  - `--defense-strategy cosine|trimmed_mean|krum`
  - `--defense-trim-ratio`（Trimmed-Mean）
  - `--defense-krum-malicious`（Krum 估计恶意节点数）
- 新增实验复现参数：
  - `--tensorboard-log-dir`：输出 TensorBoard 事件文件
  - `--config-snapshot`：保存本次运行配置快照 JSON

示例：

```bash
python scripts/run_experiment.py --mode p2p --defense --defense-strategy krum --defense-krum-malicious 1 --robots 3 --timesteps 500 --tensorboard-log-dir artifacts\tb\krum --config-snapshot artifacts\configs\run_krum.json --log-csv artifacts\logs\p2p_krum.csv
```

## 完整实验 Pipeline（一键）

如果你要一次跑完：
- 质量检查
- warm-up
- 三组对照（local/centralized/p2p）
- 三种防御对照（cosine/trimmed_mean/krum）
- 每组自动出图
- 配置快照与 TensorBoard 日志

可直接运行：

```bash
python scripts/run_full_pipeline.py --robots 3 --warmup-timesteps 5000 --timesteps 5000 --seeds 42,43,44 --progress-every 200
```

默认输出目录：
- 日志：`artifacts\pipeline\logs`
- 图像：`artifacts\pipeline\plots`
- checkpoint：`artifacts\pipeline\checkpoints\warmup`
- TensorBoard：`artifacts\pipeline\tb`
- 配置快照：`artifacts\pipeline\configs`
- 汇总索引：`artifacts\pipeline\summary.csv`

## 强化学习版 P2P 联邦学习实验流程

### 第一阶段：强化学习模型与 ROS 节点开发
在启动仿真前，完成底层算法层与通信层的代码编写。

1. **定义强化学习环境 (ROS2 侧)：**
   * **状态订阅 (State)：** 编写 ROS 节点，订阅 `/scan` (雷达数据) 并降采样为 24 维，订阅 `/odom` (里程计) 获取速度，并计算机器人与目标点的相对极坐标 $(d, \theta)$。
   * **动作发布 (Action)：** 将神经网络输出的连续值映射为 `/cmd_vel` 的线速度 $v$ 和角速度 $\omega$。
   * **奖励计算 (Reward)：** 植入复合奖励函数：到达目标 $+200$；撞墙 $-100$ 且重置；靠近目标 $+c \cdot \Delta d$；每步时间惩罚 $-0.5$。
2. **定义 DRL 模型 (PyTorch 侧)：**
   * 构建 Actor 网络（输出动作）和 Critic 网络（评估状态-动作价值）。
3. **P2P 通信与聚合模块：**
   * 编写通信节点，利用 ROS2 Topic 或 DDS 机制，仅广播和接收 **Actor 网络**的权重数组（大幅降低通信开销）。
   * 实现基于距离的发现机制：当两节点距离 $d \le R_{comm}$ 时建立连接。

### 第二阶段：本地独立预训练 (Warm-up)
在进行 P2P 交流前，让机器人先积累基础的避障“本能”，避免早期全网交换无价值的随机噪声。

1. **启动仿真环境：** 在 Gazebo 中加载包含多面墙壁和随机障碍物的迷宫地图，以及 3-5 个机器人模型（如 TurtleBot3）。
2. **独立探索：** 关闭 P2P 通信节点。所有机器人独立运行强化学习算法。
3. **自动化重置机制：** 当机器人触发碰撞或到达随机生成的目标点时，通过调用 Gazebo 的 `/gazebo/set_model_state` 服务，瞬间将机器人传送到新的安全起点。
4. **阶段目标：** 运行约 500-1000 个 Episode，直到机器人不再频繁原地打转或秒撞墙，具备初步的移动能力。

### 第三阶段：动态拓扑下的 P2P 联邦学习核心实验
验证去中心化网络下，模型能否通过共享经验加速收敛。

1. **开启 P2P 通信：** 激活所有机器人的通信节点，设定通信半径 $R_{comm}$。
2. **动态聚合过程：** * 机器人在迷宫中移动，网络拓扑随之动态变化（连接与断开）。
   * 当节点相遇时，触发 Actor 权重交换。
   * 接收节点执行加权平均算法：$W_{actor}^{(i)} = \beta W_{actor}^{(i)} + (1-\beta) \sum \alpha_j W_{actor}^{(j)}$。
3. **对比实验设置：** * **对照组 A：** 仅本地独立训练到底。
   * **对照组 B：** 传统中心化联邦学习（假设存在一个无视距离、能连接所有机器人的虚拟服务器）。
   * **实验组：** 本项目的 P2P 局部通信联邦学习。

### 第四阶段：异常节点注入与安全防御测试
针对项目创新点，验证系统在遭受恶意攻击时的鲁棒性。

1. **注入恶意节点 (Poisoning Attack)：** * 随机指定 1 个机器人为“异常节点”。
   * 篡改其通信逻辑，使其在相遇时向邻居广播**全零权重**或**高斯噪声权重**。
2. **激活防御机制：** * 在正常机器人的聚合逻辑中开启**异常检测算法**（例如：计算接收到的权重与本地权重的余弦相似度，若低于阈值则剔除，不参与聚合）。
3. **观察与记录：** 观察正常机器人在融合（或拒绝融合）恶意权重后，其避障成功率是否出现断崖式下跌。

### 第五阶段：数据记录与可视化分析
为撰写报告或论文收集硬核数据支撑。

1. **数据采集：** 实验全程使用 `ros2 bag record` 记录关键 Topic，或通过 Python 脚本将数据实时写入 CSV。
2. **核心评估指标与绘图：**
   * **学习曲线 (Episode Reward Curve)：** 绘制 Episode 数量与累计奖励的关系图，证明 P2P 组的收敛速度和最终奖励优于独立训练组。
   * **成功率与碰撞率 (Success/Collision Rate)：** 统计每 100 个 Episode 中的到达目标次数和撞墙次数。
   * **通信开销对比图 (Communication Overhead)：** 使用柱状图对比“交换整个模型”与“仅交换 Actor 网络”的总数据传输量。
   * **鲁棒性验证图：** 绘制在异常节点攻击下，开启与关闭防御机制的 Reward 跌幅对比。

---

## 1. 技术栈与版本锁定
为了保证社区支持率最高、遇到 Bug 能最快搜到解决方案，建议采用以下“黄金组合”：

* **操作系统**：Ubuntu 22.04 LTS (基于 WSL2)
* **ROS 版本**：ROS2 Humble Hawksbill (目前最稳定且长期支持的 ROS2 版本)
* **仿真器**：Gazebo Classic 11 (注意：虽然官方在推 Ignition/新版 Gazebo，但针对 TurtleBot3 的强化学习环境，Classic 11 的开源现成代码和资料最多，能帮你省去几周的底层踩坑时间)
* **机器人平台**：TurtleBot3 (Burger 型号)。它的底盘更小（圆柱形），在迷宫避障时对碰撞体积的计算更简单。
* **深度学习框架**：PyTorch 2.x

## 2. 核心算法选型：锁定 SAC (Soft Actor-Critic)
在连续动作空间（线速度和角速度控制）中，虽然 TD3 也很稳定，但强烈建议在这个项目中**锁定 SAC**。

* **选择理由**：SAC 的核心机制是“最大化熵 (Maximum Entropy)”。这意味着它在追求高回报的同时，会尽可能保持动作的随机性。在迷宫这种容易陷入局部最优（比如机器人原地打转或卡在死角）的环境中，SAC 的探索能力远强于 TD3。
* **通信优势**：在 P2P 联邦学习中，由于 SAC 鼓励探索，不同节点在本地收集到的轨迹 (Trajectories) 多样性更强。相遇时交换这些多样化的高质量 Actor 权重，能极大地加速全局收敛。

## 3. P2P 聚合触发规则：距离与冷却时间双重限制
“相遇即聚合”在理论上很好，但在实际工程中，如果两个机器人并排走，它们会每秒触发几十次聚合，瞬间挤爆网络。

* **规则设定**：采用 **条件触发 (Event-triggered)** 机制。
* **触发条件 A（空间）**：节点间欧氏距离 $d < R_{comm}$（例如设定通信半径为 2.0 米）。
* **触发条件 B（时间）**：距离上一次与该节点的聚合时间 $\Delta t > T_{cd}$（冷却时间，例如设定为 50 个控制步长或 5 秒）。
* **执行逻辑**：只有同时满足 A 和 B 时，才触发一次 Actor 权重的广播与接收融合。

## 4. 异常防御与阈值标定策略
不能一上来就拍脑袋定一个 0.5 的余弦相似度，这会缺乏说服力。防御机制的设计需要分为“标定阶段”和“执行阶段”。

* **标定阶段 (Warm-up)**：在实验的前 500 个 Episode 中，假定网络是纯净的。每个机器人记录下正常邻居权重的余弦相似度，计算出均值 $\mu$ 和标准差 $\sigma$。
* **执行阶段 (动态阈值)**：引入著名的统计学 $3\sigma$ 原则。设定动态拒绝阈值为 $\mu - 3\sigma$。
* **剔除策略**：当收到邻居 $j$ 的权重 $W_j$ 时，计算其与本地权重 $W_i$ 的余弦相似度。如果低于阈值，则直接丢弃该数据包，将其信任权重 $\alpha_j$ 设为 0，并将该节点标记为“潜在恶意节点”。

## 5. 对照组绝对公平性控制
为了让你的实验数据无可挑剔，必须在代码层面严格统一以下变量：

* **随机种子锁定**：在主程序的开头，强制统一所有库的随机种子，保证每次环境初始化的障碍物位置、网络初始权重完全一致。
* **评估基准**：不能用 Episode 数量来对比，因为有的机器人 10 步就撞墙（1 个 Episode 结束），有的走了 500 步。**必须统一使用环境交互总步数 (Total Timesteps)**（例如 500,000 步）作为 X 轴。
* **通信预算控制**：在对比中心化联邦学习（FedAvg）和 P2P 联邦学习时，必须记录“总传输字节数”。中心化模型通常要传输 Actor+Critic 全量参数，而你的项目只传输 Actor 参数，这是你证明“P2P 通信效率更高”的铁证。

---
