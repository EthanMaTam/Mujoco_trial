'''
    Currently, we only use ruckig instead of curobo because of its 
    highly-binded robot specific config file. When it comes to our
    self-defined robot--ur5e + shadow, we may consider using curobo
    after mvp.
'''
from dataclasses import dataclass
from typing import Tuple, Sequence

import numpy as np
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo

@dataclass
class JointTrajectory:
    positions: np.ndarray  # (T, dof)
    velocities: np.ndarray  # (T, dof)
    accelerations: np.ndarray  # (T, dof)
    dt: float               # control_dt

def _generate_toppra_trajectory_core(
    waypoints,              # Sequence[np.ndarray] or (N, dof) array
    control_dt: float,
    v_max: np.ndarray | None = None,   # (dof,)
    a_max: np.ndarray | None = None,   # (dof,)
) -> JointTrajectory:
    """
    纯数值 toppra 核心：
    输入若干 waypoints（在同一个 joint / synergy 空间），返回离散的
    positions, velocities, accelerations 以及 dt（就是 control_dt）。
    不关心是 arm 还是 finger。
    """
    waypoints = [np.asarray(w, dtype=float).reshape(-1) for w in waypoints]
    dof = waypoints[0].shape[0]
    N = len(waypoints)

    # 默认速度 / 加速度上限
    if v_max is None:
        v_max = np.ones(dof)*1.0
    if a_max is None:
        a_max = np.ones(dof) * 2.0

    v_max = np.asarray(v_max, dtype=float).reshape(dof)
    a_max = np.asarray(a_max, dtype=float).reshape(dof)

    # 1) 几何路径：用参数 s ∈ [0, 1] 插值 waypoints
    ss = np.linspace(0.0, 1.0, N)
    path_positions = np.vstack(waypoints)  # (N, dof)
    path = ta.SplineInterpolator(ss, path_positions)

    # 2) 约束（逐维速度/加速度）
    pc_vel = constraint.JointVelocityConstraint(
        np.vstack([-v_max, v_max]).T  # shape (dof, 2)
    )
    pc_acc = constraint.JointAccelerationConstraint(
        np.vstack([-a_max, a_max]).T
    )
    constraints = [pc_vel, pc_acc]

    # 3) toppra 时间参数化
    algo_inst = algo.TOPPRA(constraints, path, solver_wrapper="seidel")
    jnt_traj = algo_inst.compute_trajectory()
    # ts: 时间节点，sdots: 对应的速度；x: 内部变量，这里可以不用
    if jnt_traj is None:
        raise RuntimeError("TOPPRA failed to compute a trajectory.")

    # 4) 根据 duration 按 control_dt 采样
    T_total = jnt_traj.duration
    T = int(np.ceil(T_total / control_dt)) + 1
    t_samples = np.linspace(0.0, T_total, T)

    positions = jnt_traj(t_samples)        # (T, dof)
    velocities = jnt_traj(t_samples, 1)    # 一阶导
    accelerations = jnt_traj(t_samples, 2) # 二阶导

    return positions, velocities, accelerations, control_dt

def generate_toppra_waypoints_trajectory(
    waypoints,              # Sequence[np.ndarray]
    control_dt: float,
    v_max: np.ndarray | None = None,   # (dof,)
    a_max: np.ndarray | None = None,   # (dof,)
) -> JointTrajectory:
    pos, vel, acc, dt = _generate_toppra_trajectory_core(
        waypoints=waypoints,
        control_dt=control_dt,
        v_max=v_max,
        a_max=a_max,
    )
    return JointTrajectory(
        positions=pos,
        velocities=vel,
        accelerations=acc,
        dt=dt,
    )


def generate_toppra_segment_trajectory(
    q_start: np.ndarray,
    q_goal: np.ndarray,
    control_dt: float,
    v_max: np.ndarray | None = None,
    a_max: np.ndarray | None = None,
) -> JointTrajectory:
    waypoints = [q_start, q_goal]
    return generate_toppra_waypoints_trajectory(
        waypoints=waypoints,
        control_dt=control_dt,
        v_max=v_max,
        a_max=a_max,
    )

