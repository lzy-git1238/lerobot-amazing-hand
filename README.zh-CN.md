# SO-ARM101 + AmazingHand 灵巧操作平台

[English](README.md) | **简体中文**

> 基于 [LeRobot](https://github.com/huggingface/lerobot) 框架，将 **AmazingHand 八自由度灵巧手** 集成到 **SO-ARM101 六自由度机械臂**，实现**遥操作 → 数据采集 → 模仿学习训练 → 自主执行**的完整闭环。

---

## 项目简介

本项目在 LeRobot 基础上完成了一套灵巧手机械臂的**硬件集成与软件适配**，主要解决三个工程问题：

| 问题 | 解决方案 |
|---|---|
| **协议冲突** —— 灵巧手舵机（SCS0009，协议 1）与臂体舵机（STS3215，协议 0）无法共用总线 | 三路物理隔离串口 + 独立供电 |
| **驱动栈污染** —— 同进程创建多个 Feetech 总线会互相干扰，导致读取数据损坏 | 灵巧手改用 `rustypot` 独立驱动栈，与臂体总线完全解耦 |
| **标定复杂** —— 8 舵机并联机构的开合标定困难 | 自研一体化标定工具，手指角度与夹爪方向一次标定、自动保存 |

## 核心特性

- **完整遥控闭环**：主动臂 5 关节 → 从动臂跟随；主动臂夹爪 → 灵巧手按比例开合
- **一体化标定**：`lerobot-calibrate-amazing-hand` 图形化标定手部开合角度与夹爪方向，结果自动落盘、启动自动加载，**无需修改任何代码**
- **数据采集与训练**：支持双路相机同步采集，产出 LeRobotDataset v3.0 格式数据集，可直接用于 ACT 等策略训练
- **容器化交付**：提供 CPU / CUDA 两套 Docker 镜像，导入即用，免装环境
- **文档完备**：六阶段教程，Windows / Linux 分平台（仓库内为英文版，中文版见 [wiki](https://wiki.juxitech.com/zh-hant/tutorials/robot-arms/so-arm-amazinghand/)）

## 硬件要求

| 部件 | 型号 / 说明 |
|---|---|
| 从动臂 | SO-ARM101，5 × `STS3215`（ID 1–5），**拆除原厂 6 号夹爪** |
| 灵巧手 | AmazingHand，8 × `SCS0009`（ID 1–8），**独立串口 + 独立供电** |
| 主动臂 | SO-ARM101，6 舵机（**保留 6 号夹爪**作为手部控制输入） |
| 相机 | 2 × USB 摄像头，640×480 @ 30 fps |
| 转接件 | 3D 打印，将灵巧手安装至从动臂腕部 |

> ⚠️ **三个设备必须各自独立串口、独立供电。** SCS0009（协议 1）与 STS3215（协议 0）不能共用总线。

## 快速开始

完整流程见 **[tutorials/](tutorials/)**，按阶段组织，Windows / Linux 分开：

| 阶段 | Windows | Linux |
|---|---|---|
| 1. 环境搭建 | [win](tutorials/01-environment/win.md) | [linux](tutorials/01-environment/linux.md) |
| 2. 标定 | [win](tutorials/02-calibration/win.md) | [linux](tutorials/02-calibration/linux.md) |
| 3. 遥操作 | [win](tutorials/03-teleoperation/win.md) | [linux](tutorials/03-teleoperation/linux.md) |
| 4. 数据采集 | [win](tutorials/04-data-collection/win.md) | [linux](tutorials/04-data-collection/linux.md) |
| 5. 模型训练 | [win](tutorials/05-training/win.md) | [linux](tutorials/05-training/linux.md) |
| 6. 部署评估 | [win](tutorials/06-deployment/win.md) | [linux](tutorials/06-deployment/linux.md) |

> 📖 **文档语言**
>
> | 语言 | 位置 |
> |---|---|
> | **English** | 本仓库 —— [`tutorials/`](tutorials/) |
> | **简体中文** | [wiki.juxitech.com](https://wiki.juxitech.com/zh-hant/tutorials/robot-arms/so-arm-amazinghand/) |

**环境安装（概要）**：

```bash
conda create -y -n lerobot python=3.12
conda activate lerobot
pip install -e ".[amazinghand,training]"
```

**常用命令**：

```bash
# 标定手部角度 + 夹爪方向（GUI）
lerobot-calibrate-amazing-hand --hand_port=<手串口> --leader_port=<主动臂串口>

# 遥操作
lerobot-teleoperate \
  --robot.type=so101_amazing_hand \
  --robot.port=<从动臂串口> --robot.hand_port=<手串口> \
  --robot.id=amazing_hand_follower \
  --teleop.type=so101_leader --teleop.port=<主动臂串口> \
  --teleop.id=amazing_hand_leader

# 部署训练好的策略
lerobot-rollout --strategy.type=base \
  --policy.path=<模型目录> \
  --robot.type=so101_amazing_hand \
  --robot.port=<从动臂串口> --robot.hand_port=<手串口> \
  --task="Pick up the cube with the dexterous hand"
```

## 前置摄像头遥操与联合训练

主动臂控制 SO-101 的 5 个关节，电脑前置摄像头跟踪拇指、食指、中指和无名指的屈伸。每根手指的两个舵机沿标定的张开到握拳路径协同运动。LeRobot 每帧记录 13 路实测角度和 13 路实际下发的目标角度，单位为度。手部字段为 `hand_servo_1.pos` 至 `hand_servo_8.pos`，角度以手部舵机中心位置为零点。使用新的数据集目录录制这一模式。

### 预览和标定

```bash
uv sync --extra amazinghand-camera --extra training
curl -fL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
  -o hand_landmarker.task

uv run --no-sync lerobot-preview-amazing-hand-camera \
  --model-path=hand_landmarker.task --camera=0 --handedness=Right
```

只连接灵巧手时，下面的命令启动实机跟随并自动打开浏览器。将串口替换为灵巧手串口。已标定时使用保存的舵机端点，首次试机使用启动姿态附近 ±5°。跟踪丢失约半秒后卸力，点击页面上的“停止跟随并卸力”结束测试。去掉 `--follow` 并加 `--dashboard` 可进入标定模式。用网页上的参考张开、回中位、参考握拳和逐指滑块驱动舵机，等实测角度到位后采集灵巧手端点。摄像头遥操作录制时会自动打开同一界面并显示实际下发的目标。

```bash
uv run --no-sync lerobot-preview-amazing-hand-camera --follow --hand-port=/dev/ttyACM0 \
  --camera=0 --model-path=hand_landmarker.task
```

预览可在机器人断开时运行，逐帧打印四根手指的屈伸比例。存在 `hand_angles.json` 时也会打印 8 路目标角度。用 `Left` 选择左手。

```bash
export LEADER_PORT=/dev/ttyACM0
export FOLLOWER_PORT=/dev/ttyACM1
export HAND_PORT=/dev/ttyUSB0
export DATASET_ID=local/amazinghand_camera
export DATASET_ROOT=outputs/datasets/amazinghand_camera

uv run --no-sync lerobot-calibrate-amazing-hand --hand_port="$HAND_PORT"
```

替换为实际串口和已标定的主动臂、从动臂 ID。在图形界面保存张开和握拳姿态。使用自定义手部标定目录时，给机器人传 `--robot.calibration_dir`，给遥操器传 `--teleop.hand_camera.calibration_path`，两者应指向同一份 `hand_angles.json`。

### 遥操、录制和训练

以下命令使用摄像头 0 跟踪人手，摄像头 1 拍摄工件和机械臂。录制前把所选的人手放进画面。短暂丢失时保持上次手部目标，持续丢失超过配置的保留时间会停止本次运行并保留已保存的回合。

```bash
robot_args=(
  --robot.type=so101_amazing_hand --robot.hand_control_mode=servo
  --robot.port="$FOLLOWER_PORT" --robot.hand_port="$HAND_PORT"
  --robot.id=amazing_hand_follower
)
teleop_args=(
  --teleop.type=so101_amazing_hand_camera --teleop.port="$LEADER_PORT"
  --teleop.id=amazing_hand_leader --teleop.hand_camera.model_path=hand_landmarker.task
  --teleop.hand_camera.camera.index_or_path=0 --teleop.hand_camera.handedness=Right
)
task_camera='--robot.cameras={task: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}'

uv run --no-sync lerobot-teleoperate "${robot_args[@]}" "${teleop_args[@]}" --fps=30
uv run --no-sync lerobot-record "${robot_args[@]}" "${teleop_args[@]}" "$task_camera" \
  --dataset.repo_id="$DATASET_ID" --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="Pick up the cube" --dataset.fps=30 --dataset.num_episodes=20 \
  --dataset.push_to_hub=false

uv run --no-sync lerobot-train --dataset.repo_id="$DATASET_ID" --dataset.root="$DATASET_ROOT" \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/train/amazinghand_camera_act

uv run --no-sync lerobot-rollout --strategy.type=base "${robot_args[@]}" "$task_camera" \
  --policy.path=outputs/train/amazinghand_camera_act/checkpoints/last/pretrained_model \
  --task="Pick up the cube"
```

只有前置摄像头时，去掉录制命令里的 `"$task_camera"` 可以先验证角度采集。有了任务画面后另建数据集，再用 ACT 联合训练。实机部署使用与训练时相同的任务相机配置。

### 分享工程并启动单手录制

分享整个工程文件夹时，请包含 `web_assets/` 下约 42 MB 的模型和 WASM 文件。接收者在工程根目录安装依赖后启动：

```bash
uv sync --locked --extra amazinghand-camera
uv run python examples/amazinghand/record_hand.py --hand-port /dev/ttyACM0 --seconds 5 --fps 15
```

串口替换为接收者的实际设备串口。网页默认打开 `http://127.0.0.1:8765`。三姿势标定完成后自动录制，数据保存到 `outputs/datasets/amazinghand_hand_only_live_<时间>/`，包含八路状态和八路目标角度。

回放时使用实际生成的数据目录：

```bash
uv run python examples/amazinghand/replay_hand.py outputs/datasets/<数据集目录> --hand-port /dev/ttyACM0
```

网页来源、许可证和导入文件的 SHA256 校验值放在 `web_assets/`。源码分发和 Python wheel 都包含本地网页资源。

### 使用 Juxi 原版跟踪

网页、配置、MediaPipe 模型和 WASM 运行时已收进本工程的 `src/lerobot/teleoperators/so_amazing_hand_camera/web_assets/`。启动默认使用这些内置资源。打开页面后启动摄像头，采集「伸直张开」「伸直合拢」「握拳」。浏览器识别姿态，LeRobot 管理串口。

单独测试右手 SCS0009（舵机 ID 1–8）：

```bash
uv run --no-sync lerobot-preview-amazing-hand-camera --follow --hand-port="$HAND_PORT"
```

联合录制时，将前文 `robot_args` 中的 `--robot.hand_control_mode=servo` 改为 `--robot.hand_control_mode=servo_juxi`。在 `teleop_args` 中省略 `--teleop.hand_camera.model_path=hand_landmarker.task`，会自动使用内置网页。保持相机页面可见，完成三个姿态后录制开始。

这个模式用舵机轴的绝对角度（度），包含屈伸和侧摆。录制桥接页启用四指屈伸和四指侧摆，共八个自由度。中指侧摆目标为 −20° 到 +20°，拇指为 0° 到 −30°。采集「伸直张开」和「伸直合拢」时让中指和拇指产生侧向位移，测量跨度不足 3° 时页面提示重采。侧摆仍随握拳程度衰减。数据字段维持八路舵机轴角度。为它新建数据集，回放或部署对应策略时使用 `--robot.hand_control_mode=servo_juxi`。

## 容器化

```bash
# 构建（在 Linux 上）
bash docker/build_and_export.sh cpu     # CPU 版，约 1 GB
bash docker/build_and_export.sh cuda    # CUDA 版，约 3.5 GB

# 客户侧导入即用
docker load -i dist/lerobot-amazinghand-cpu.tar
```

详见 **[docker/README_DEPLOY.md](docker/README_DEPLOY.md)**（含串口/相机/GUI 直通配置）。

## 本仓库相对原版 LeRobot 的改动

| 位置 | 改动 |
|---|---|
| `src/lerobot/robots/so_amazing_hand/` | **新增** —— SO-ARM101 + 灵巧手组合的机器人定义 |
| `src/lerobot/scripts/lerobot_calibrate_amazing_hand.py` | **新增** —— 灵巧手一体化标定 GUI |
| `src/lerobot/robots/so_follower/so_follower.py` | 总线写操作增加重试，提升偶发丢包容错 |
| `pyproject.toml` | 新增 `amazinghand` extra |
| `tutorials/` | **新增** —— 六阶段教程（英文版） |
| `docker/Dockerfile.amazinghand.*` | **新增** —— CPU / CUDA 镜像 |

其余目录（`policies/`、`datasets/`、`envs/` 等）与原版 LeRobot 一致。

## 目录结构

```
so-arm101-amazinghand/
├── src/lerobot/
│   ├── robots/so_amazing_hand/          # 灵巧手机械臂定义（本项目核心）
│   ├── scripts/lerobot_calibrate_amazing_hand.py  # 标定工具
│   └── ...                              # 其余为 LeRobot 原生代码
├── tutorials/                           # 六阶段教程（英文，Win / Linux）
├── docker/                              # 容器化方案
├── diagnose_feetech_bus.py              # Feetech 总线诊断工具
└── preview_cameras.py                   # 摄像头实时预览工具
```

## 故障排查

**总线找不到舵机**（`FeetechMotorsBus motor check failed`）：

```bash
python diagnose_feetech_bus.py /dev/ttyACM0    # Linux
python diagnose_feetech_bus.py COM58           # Windows
```

该脚本扫描全部 ID × 全部波特率，可区分「总线没通」与「ID 配置不对」。

**相机打不开**：

```bash
lerobot-find-cameras opencv       # 确认索引
python preview_cameras.py         # 实时预览
```

## 致谢与许可

本项目基于以下开源工作：

- **[LeRobot](https://github.com/huggingface/lerobot)** —— 训练与部署框架（Apache-2.0）
- **[AmazingHand](https://github.com/pollen-robotics/AmazingHand)** —— 开源灵巧手，Pollen Robotics 出品（软件 Apache-2.0 / 机械设计 CC BY 4.0）
- **SO-ARM101** —— 开源六自由度机械臂

本仓库代码以 **Apache-2.0** 协议开源，详见 [LICENSE](LICENSE)。硬件设计部分请遵循各上游项目协议。
