# UR5e + ShadowHand MuJoCo Trial

基于 MuJoCo 的 UR5e + Shadow Hand 仿真与抓取规划示例，便于同组同学直接复现和扩展。

## 获取代码（含子模组）
- 第一次克隆：`git clone --recursive https://github.com/EthanMaTam/Mujoco_trial.git`。
- 已经克隆过但没带 `--recursive`：在仓库内运行  
  ```bash
  git submodule sync --recursive
  git submodule update --init --recursive
  ```
- 说明：`assets/object_sim`  是子模组，需同步更新。

## 功能概览
- MuJoCo 场景：`ur5e_shadowhand_scene.xml`（仿真）与 `ur5e_shadowhand_scene_pin.xml`（Pinocchio/TSID 使用）。
- 规划与控制：OMPL 路径规划、toppra 时间参数化、ruckig 轨迹跟踪，配合 TSID/Pinocchio IK。
- 抓取生成：基于物体 AABB 的抓取候选生成，可通过 `config/grasp_config.toml` 调节掌面方向/离面距离/平面内采样。
- 手部协同：Shadow Hand synergy/gesture 解码与执行。

## 环境准备（推荐 Conda + Python 3.12）
1. 安装 Miniforge/Conda；在中国大陆可将 `.condarc` 渠道设为 `conda-forge`（USTC 镜像更快）。
2. 创建环境并激活：
   ```bash
   conda create -n mujoco_trial python=3.12
   conda activate mujoco_trial
   ```
3. 安装依赖（重度 C++ 依赖建议走 conda-forge，toppra/ruckig 不在 conda-forge，用 pip 安装）：
   ```bash
   mamba install -c conda-forge mujoco pinocchio tsid ompl numpy
   pip install ruckig toppra
   # requirements.txt 只包含纯 Python 依赖，不会重复覆盖上述 conda 包
   pip install -r requirements.txt
   ```
   - 服务器或无显示环境请设置 `MUJOCO_GL=egl`（或 `osmesa`）后再运行。
   - 若需要下载速度，加上 USTC 镜像或本地镜像源。

## 快速运行 demo
1. 确认根目录存在 `ur5e_shadowhand_scene.xml` 与 `ur5e_shadowhand_scene_pin.xml`。  
2. 运行：
   ```bash
   python main.py
   ```
   默认流程：加载场景 → TSID IK 生成手臂目标 → OMPL 规划主臂路径 → toppra/Ruckig 时间参数化 → Shadow Hand synergy 解码手势并执行。若不需要渲染，可将 `MjWorld(use_viewer=False)`。

## 目录与主要模块
- `main.py`：示例入口，串联 IK、抓取候选、路径规划与轨迹执行。
- `sim/`：`mj_world.py` 封装 MuJoCo 世界、关节/执行器接口、轨迹跟踪；`motion.py` 规划器；`grasping.py` AABB 抓取信息。
- `control/`：TSID/Pinocchio IK、Shadow Hand synergy（`hand_kin.py`）、toppra/ruckig 轨迹生成。
- `planning/`：抓取生成与 OMPL 配置。
- `config/grasp_config.toml`：抓取策略参数；`config/default_qpos.npy` 会在运行时保存默认姿态。
- `assets/`：MJCF/URDF/网格资源，附带一些 ROS/IGN 工具脚本与物体预览脚本。
- `utils/transform.py`：SE3/姿态变换工具。

## 配置与常用操作
- 修改抓取策略：编辑 `config/grasp_config.toml`（掌面轴、standoff、平面内 yaw 样本数、评分偏好）。
- 保存默认姿态：在 viewer 中调整后调用 `MjWorld.save_current_qpos()`，会写入 `config/default_qpos.npy`。
- 更换场景/末端：调整 `main.py` 顶部的 `SCENE_PATH`/`MJCF_PATH`/`HAND_EE`。
- 碰撞/规划：碰撞约束在 `sim/constraint.py`，路径约束通过 `MotionPlanner` 注入；需要放宽碰撞可修改 `allowed_collision_bodies()`。

## 常见问题
- `ompl`/`pinocchio`/`tsid` 在 pip 下可能需要编译，建议优先使用 conda-forge 预编译包。
- 远程运行渲染错误时，优先检查 `MUJOCO_GL` 和显卡驱动/EGL 依赖。
