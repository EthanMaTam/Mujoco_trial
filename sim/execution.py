from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import mujoco
import numpy as np

# 默认臂部 actuator 名称（1:1 直驱）
DEFAULT_ARM_ACTUATORS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
    "rh_A_WRJ2",
    "rh_A_WRJ1",
]

# 手部直驱 actuator 名称（不含 J0 肌腱），顺序按 qpos 次序排列
DEFAULT_HAND_ACTUATORS: list[str] = [
    "rh_A_FFJ4",
    "rh_A_FFJ3",
    "rh_A_MFJ4",
    "rh_A_MFJ3",
    "rh_A_RFJ4",
    "rh_A_RFJ3",
    "rh_A_LFJ5",
    "rh_A_LFJ4",
    "rh_A_LFJ3",
    "rh_A_THJ5",
    "rh_A_THJ4",
    "rh_A_THJ3",
    "rh_A_THJ2",
    "rh_A_THJ1",
]
# 除拇指外 4 指的肌腱（J0）名称与对应的 actuator 名称
DEFAULT_HAND_TENDONS: list[str] = ["rh_FFJ0", "rh_MFJ0", "rh_RFJ0", "rh_LFJ0"]
DEFAULT_TENDON_ACTUATORS: list[str] = [
    "rh_A_FFJ0",
    "rh_A_MFJ0",
    "rh_A_RFJ0",
    "rh_A_LFJ0",
]


@dataclass
class HandCommand:
    """手部混合驱动的命令容器。字段可按需填充。"""

    q_direct: Optional[np.ndarray] = None  # 直接驱动关节目标（与 hand_actuators 对齐）
    dq_direct: Optional[np.ndarray] = None
    ddq_direct: Optional[np.ndarray] = None
    tendon_rest_length: Optional[np.ndarray] = None  # 欠驱动通道：目标松弛长度
    tendon_tension: Optional[np.ndarray] = None  # 或目标张力
    synergy: Optional[np.ndarray] = None  # 欠驱动弯曲量 s（与手部肌腱 actuator 对齐）


@dataclass
class HandTrajectory:
    """手部轨迹（可同时包含直驱与肌腱通道）。"""

    commands: list[HandCommand]
    dt: float


@dataclass
class CombinedTrajectory:
    """组合臂/手轨迹，支持不同通道共同执行。"""

    arm: Optional[np.ndarray]  # shape (T, nq_full) 或 (T, nq_arm) 视控制器而定
    arm_dq: Optional[np.ndarray] = None
    arm_ddq: Optional[np.ndarray] = None
    hand: Optional[HandTrajectory] = None
    dt: float = 0.0


class ActuatorPDController:
    """通用 PD+前馈控制器，适用于 1:1 直驱关节。"""

    def __init__(
        self,
        world,
        actuator_names: Iterable[str],
        qpos_adr: np.ndarray,
        qvel_adr: np.ndarray,
    ):
        self.world = world
        self.model = world.model
        self.data = world.data

        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                for n in actuator_names
            ],
            dtype=int,
        )
        if np.any(self.actuator_ids < 0):
            missing = [n for n, aid in zip(actuator_names, self.actuator_ids) if aid < 0]
            raise ValueError(f"Actuators not found: {missing}")

        self.qpos_adr = np.asarray(qpos_adr, dtype=int)
        self.qvel_adr = np.asarray(qvel_adr, dtype=int)
        self.kp = self.model.actuator_gainprm[self.actuator_ids, 0]
        self.kd = -self.model.actuator_biasprm[self.actuator_ids, 2]
        self.alpha = np.where(self.kp > 1e-9, self.kd / self.kp, 0.0)
        self.ctrl_range = self.model.actuator_ctrlrange[self.actuator_ids]
        self.data_ff = mujoco.MjData(self.model)

        # full_qpos_adr/full_qvel_adr -> group column mapping
        col_of_qpos = {int(qi): i for i, qi in enumerate(world.full_qpos_adr)}
        col_of_qvel = {int(vi): i for i, vi in enumerate(world.full_qvel_adr)}
        self.cols_q = np.array([col_of_qpos[int(qi)] for qi in self.qpos_adr], int)
        self.cols_v = np.array([col_of_qvel[int(vi)] for vi in self.qvel_adr], int)

    def apply_step(
        self,
        q_ref_full: np.ndarray,
        dq_ref_full: Optional[np.ndarray] = None,
        ddq_ref_full: Optional[np.ndarray] = None,
    ):
        d = self.data
        q_ref_full = np.asarray(q_ref_full, float).reshape(-1)
        # 允许传入完整状态或仅该控制组的切片
        if q_ref_full.shape[0] == self.cols_q.shape[0]:
            q_ref = q_ref_full
        else:
            q_ref = q_ref_full[self.cols_q]

        if dq_ref_full is not None:
            dq_ref_full = np.asarray(dq_ref_full, float).reshape(-1)
            ddq_ref_full = (
                np.asarray(ddq_ref_full, float).reshape(-1)
                if ddq_ref_full is not None
                else None
            )
            if dq_ref_full.shape[0] == self.cols_v.shape[0]:
                dq_ref = dq_ref_full
                ddq_ref = (
                    ddq_ref_full if ddq_ref_full is not None else None
                )
            else:
                dq_ref = dq_ref_full[self.cols_v]
                ddq_ref = ddq_ref_full[self.cols_v] if ddq_ref_full is not None else None

            self.data_ff.qpos[:] = d.qpos[:]
            self.data_ff.qvel[:] = d.qvel[:]
            self.data_ff.qacc[:] = d.qacc[:]

            self.data_ff.qpos[self.qpos_adr] = q_ref
            self.data_ff.qvel[self.qvel_adr] = dq_ref
            if ddq_ref is not None:
                self.data_ff.qacc[self.qvel_adr] = ddq_ref
            mujoco.mj_forward(self.model, self.data_ff)

            tau_ff = d.qfrc_bias[self.qvel_adr] + d.qfrc_passive[self.qvel_adr]
            ctrl = q_ref + self.alpha * dq_ref + tau_ff / self.kp
        else:
            ctrl = q_ref

        d.ctrl[self.actuator_ids] = np.clip(ctrl, self.ctrl_range[:, 0], self.ctrl_range[:, 1])


class TendonController:
    """肌腱/欠驱动通道控制器骨架。"""

    def __init__(
        self,
        world,
        tendon_names: Iterable[str],
        tendon_actuator_names: Iterable[str] | None = None,
    ):
        self.world = world
        self.model = world.model
        self.data = world.data
        self.tendon_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TENDON, n)
                for n in tendon_names
            ],
            dtype=int,
        )
        if np.any(self.tendon_ids < 0):
            missing = [n for n, tid in zip(tendon_names, self.tendon_ids) if tid < 0]
            raise ValueError(f"Tendons not found: {missing}")

        self.tendon_actuator_ids = None
        self.tendon_ctrl_range = None
        if tendon_actuator_names:
            actuator_ids = np.array(
                [
                    mujoco.mj_name2id(
                        self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n
                    )
                    for n in tendon_actuator_names
                ],
                dtype=int,
            )
            if np.any(actuator_ids < 0):
                missing = [
                    n for n, aid in zip(tendon_actuator_names, actuator_ids) if aid < 0
                ]
                raise ValueError(f"Tendon actuators not found: {missing}")

            self.tendon_actuator_ids = actuator_ids
            self.tendon_ctrl_range = self.model.actuator_ctrlrange[actuator_ids]

    def apply_rest_length(self, rest_length: np.ndarray):
        """示例：设置肌腱松弛长度。"""
        self.data.ten_length[self.tendon_ids] = rest_length

    def apply_tension(self, tension: np.ndarray):
        """示例：直接控制张力（若 XML actuator 支持）。"""
        self.data.ten_tension[self.tendon_ids] = tension

    def apply_bend_synergy(
        self,
        bend_s: np.ndarray,
        bend_ds: np.ndarray | None = None,
        bend_dds: np.ndarray | None = None,
    ):
        """
        将弯曲量 s 映射到 J0 肌腱 actuator。这里的肌腱控制同时约束 J1/J2，
        采用 u_J0 = q1_ref + q2_ref，且两关节比例相等 -> u_J0 = s。
        """
        if self.tendon_actuator_ids is None or self.tendon_ctrl_range is None:
            raise RuntimeError("Tendon actuator mapping not initialized.")

        bend_s = np.asarray(bend_s, dtype=float).reshape(-1)
        if bend_s.shape[0] != self.tendon_actuator_ids.shape[0]:
            raise ValueError(
                f"Expected {self.tendon_actuator_ids.shape[0]} bend values, "
                f"got {bend_s.shape[0]}"
            )

        # 计算 alpha = kd/kp，用于简单速度前馈
        kp = self.model.actuator_gainprm[self.tendon_actuator_ids, 0]
        kd = -self.model.actuator_biasprm[self.tendon_actuator_ids, 2]
        alpha = np.where(kp > 1e-9, kd / kp, 0.0)

        if bend_ds is not None:
            bend_ds = np.asarray(bend_ds, dtype=float).reshape(-1)
            if bend_ds.shape[0] != bend_s.shape[0]:
                raise ValueError("bend_ds length mismatch.")
            # 简单裁剪以抑制接触噪声
            bend_ds = np.clip(bend_ds, -10.0, 10.0)
            ctrl = bend_s + alpha * bend_ds
        else:
            ctrl = bend_s

        # 可选地用加速度做更完整的前馈（简单插值，避免复杂动力学计算）
        if bend_dds is not None:
            bend_dds = np.asarray(bend_dds, dtype=float).reshape(-1)
            if bend_dds.shape[0] != bend_s.shape[0]:
                raise ValueError("bend_dds length mismatch.")
            # 这里仅做小幅增益叠加，不额外估计惯性
            ctrl = ctrl + 0.0 * bend_dds

        ctrl = np.clip(ctrl, self.tendon_ctrl_range[:, 0], self.tendon_ctrl_range[:, 1])
        self.data.ctrl[self.tendon_actuator_ids] = ctrl


class TrajectoryExecutor:
    @staticmethod
    def _joint_adrs_from_actuators(world, actuator_names: Iterable[str]):
        """根据 actuator 名称取得对应关节的 qpos/qvel 下标。"""
        m = world.model
        actuator_ids = []
        qpos_adr = []
        qvel_adr = []

        for name in actuator_names:
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise ValueError(f"Actuator '{name}' not found in model.")

            trn_type = int(m.actuator_trntype[aid])
            if trn_type != int(mujoco.mjtTrn.mjTRN_JOINT):
                raise ValueError(f"Actuator '{name}' is not joint-driven.")

            jid = int(m.actuator_trnid[aid, 0])
            if jid < 0:
                raise ValueError(f"Actuator '{name}' has invalid joint id.")

            actuator_ids.append(aid)
            qpos_adr.append(int(m.jnt_qposadr[jid]))
            qvel_adr.append(int(m.jnt_dofadr[jid]))

        return (
            np.asarray(actuator_ids, dtype=int),
            np.asarray(qpos_adr, dtype=int),
            np.asarray(qvel_adr, dtype=int),
        )

    @staticmethod
    def _joint_qpos_adr(world, joint_name: str) -> int:
        jid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint '{joint_name}' not found in model.")
        return int(world.model.jnt_qposadr[jid])

    def __init__(
        self,
        world,
        arm_actuators: Iterable[str] = DEFAULT_ARM_ACTUATORS,
        hand_actuators: Iterable[str] = DEFAULT_HAND_ACTUATORS,
        hand_tendons: Iterable[str] | None = DEFAULT_HAND_TENDONS,
        tendon_actuators: Iterable[str] | None = DEFAULT_TENDON_ACTUATORS,
    ):
        self.world = world
        self.arm_controller = ActuatorPDController(
            world, arm_actuators, world.main_qpos_adr, world.main_qvel_adr
        )
        if hand_actuators:
            (
                self.hand_actuator_ids,
                hand_qpos_adr,
                hand_qvel_adr,
            ) = self._joint_adrs_from_actuators(world, hand_actuators)
            self.hand_controller = ActuatorPDController(
                world, hand_actuators, hand_qpos_adr, hand_qvel_adr
            )
        else:
            self.hand_controller = None
        # 欠驱动 J0 肌腱对应的 J2/J1 关节下标（顺序：FF/MF/RF/LF）
        self.synergy_qpos_pairs = [
            (
                self._joint_qpos_adr(world, "rh_FFJ2"),
                self._joint_qpos_adr(world, "rh_FFJ1"),
            ),
            (
                self._joint_qpos_adr(world, "rh_MFJ2"),
                self._joint_qpos_adr(world, "rh_MFJ1"),
            ),
            (
                self._joint_qpos_adr(world, "rh_RFJ2"),
                self._joint_qpos_adr(world, "rh_RFJ1"),
            ),
            (
                self._joint_qpos_adr(world, "rh_LFJ2"),
                self._joint_qpos_adr(world, "rh_LFJ1"),
            ),
        ]
        self.tendon_controller = (
            TendonController(world, hand_tendons, tendon_actuators)
            if hand_tendons
            else None
        )

    def _compute_synergy_from_q(
        self,
        q_ref_full: np.ndarray,
        dq_ref_full: np.ndarray | None = None,
        ddq_ref_full: np.ndarray | None = None,
    ):
        """
        从参考关节角/速度/加速度推导每指的弯曲量及其导数：
            s  = q1+q2
            ds = dq1+dq2
            dds = ddq1+ddq2
        """
        q_ref_full = np.asarray(q_ref_full, float).reshape(-1)
        s = []
        ds = [] if dq_ref_full is not None else None
        dds = [] if ddq_ref_full is not None else None

        dq_ref_full = (
            np.asarray(dq_ref_full, float).reshape(-1) if dq_ref_full is not None else None
        )
        ddq_ref_full = (
            np.asarray(ddq_ref_full, float).reshape(-1) if ddq_ref_full is not None else None
        )

        for q2_adr, q1_adr in self.synergy_qpos_pairs:
            s.append(q_ref_full[q2_adr] + q_ref_full[q1_adr])
            if dq_ref_full is not None:
                ds.append(dq_ref_full[q2_adr] + dq_ref_full[q1_adr])
            if ddq_ref_full is not None:
                dds.append(ddq_ref_full[q2_adr] + ddq_ref_full[q1_adr])

        s = np.asarray(s, float)
        ds_arr = np.asarray(ds, float) if ds is not None else None
        dds_arr = np.asarray(dds, float) if dds is not None else None
        return s, ds_arr, dds_arr

    def follow_trajectory_kinematic(self, traj):
        q_traj = np.asarray(traj.positions, dtype=float)
        dq_traj = (
            np.asarray(traj.velocities, dtype=float)
            if traj.velocities is not None
            else None
        )

        T, dof = q_traj.shape
        assert dof == self.world.full_qpos_adr.shape[0]

        if abs(traj.dt - self.world.dt) > 1e-6:
            print(
                f"[follow_full_trajectory_kinematic] WARNING: traj.dt={traj.dt} "
                f"doesn't match world.dt={self.world.dt} "
            )

        use_dq = dq_traj is not None and dq_traj.shape == q_traj.shape
        q_prev = q_traj[0].copy()

        for k in range(T):
            if not self.world.is_running():
                break

            q_des = q_traj[k]
            if use_dq:
                dq_des = dq_traj[k]
            else:
                dq_des = (
                    np.zeros_like(q_des)
                    if k == 0
                    else (q_des - q_prev) / self.world.dt
                )

            self.world.data.qpos[self.world.full_qpos_adr] = q_des
            self.world.data.qvel[self.world.full_qvel_adr] = dq_des

            self.world.step()
            q_prev = q_des

    def follow_trajectory_actuator(self, traj):
        q_traj = np.asarray(traj.positions, float)
        dq_traj = (
            np.asarray(traj.velocities, float) if traj.velocities is not None else None
        )
        ddq_traj = (
            np.asarray(traj.accelerations, float)
            if traj.accelerations is not None
            else None
        )

        expected_dof = self.world.full_qpos_adr.shape[0]
        if q_traj.shape[1] != expected_dof:
            raise ValueError(
                f"Trajectory dof={q_traj.shape[1]} does not match model dof={expected_dof}"
            )

        T = q_traj.shape[0]
        use_dq = dq_traj is not None

        for k in range(T):
            if not self.world.is_running():
                break

            dq_ref = dq_traj[k] if use_dq else None
            ddq_ref = ddq_traj[k] if ddq_traj is not None else None

            # 先发臂控制，后续插入手/肌腱控制
            self.arm_controller.apply_step(
                q_ref_full=q_traj[k], dq_ref_full=dq_ref, ddq_ref_full=ddq_ref
            )
            if self.hand_controller is not None:
                self.hand_controller.apply_step(
                    q_ref_full=q_traj[k], dq_ref_full=dq_ref, ddq_ref_full=ddq_ref
                )
            if self.tendon_controller is not None:
                synergy, synergy_ds, synergy_dds = self._compute_synergy_from_q(
                    q_traj[k], dq_ref, ddq_ref
                )
                self.tendon_controller.apply_bend_synergy(
                    synergy, synergy_ds, synergy_dds
                )

            self.world.step()

        last = self.world.data.ctrl[self.arm_controller.actuator_ids].copy()
        for _ in range(300):
            self.world.data.ctrl[self.arm_controller.actuator_ids] = last
            self.world.step()

    def follow_combined(
        self, combined: CombinedTrajectory, hold_steps: int = 300
    ):
        """示例：同时执行臂/手/肌腱轨迹。"""
        arm_traj = combined.arm
        dq = combined.arm_dq
        ddq = combined.arm_ddq
        hand_traj = combined.hand
        dt = combined.dt

        if dt and abs(dt - self.world.dt) > 1e-6:
            print(
                f"[TrajectoryExecutor] WARNING: traj.dt={dt} "
                f"doesn't match world.dt={self.world.dt}"
            )

        if arm_traj is None:
            raise ValueError("CombinedTrajectory.arm cannot be None")

        T = arm_traj.shape[0]

        for k in range(T):
            if not self.world.is_running():
                break

            dq_ref = dq[k] if dq is not None else None
            ddq_ref = ddq[k] if ddq is not None else None
            self.arm_controller.apply_step(
                q_ref_full=arm_traj[k], dq_ref_full=dq_ref, ddq_ref_full=ddq_ref
            )

            if self.hand_controller is not None and hand_traj is not None:
                cmd = hand_traj.commands[k]
                if cmd.q_direct is not None:
                    self.hand_controller.apply_step(
                        q_ref_full=cmd.q_direct,
                        dq_ref_full=cmd.dq_direct,
                        ddq_ref_full=cmd.ddq_direct,
                    )
                # 肌腱控制示例
                if self.tendon_controller is not None:
                    # 优先使用显式 synergy；否则根据参考 q/dq/ddq 反推 s, ds, dds
                    synergy = cmd.synergy
                    synergy_ds = None
                    synergy_dds = None
                    if synergy is None:
                        synergy, synergy_ds, synergy_dds = self._compute_synergy_from_q(
                            arm_traj[k], dq_ref, ddq_ref
                        )
                    self.tendon_controller.apply_bend_synergy(
                        synergy, synergy_ds, synergy_dds
                    )
                    if cmd.tendon_rest_length is not None:
                        self.tendon_controller.apply_rest_length(cmd.tendon_rest_length)
                    if cmd.tendon_tension is not None:
                        self.tendon_controller.apply_tension(cmd.tendon_tension)

            self.world.step()

        last = self.world.data.ctrl[self.arm_controller.actuator_ids].copy()
        for _ in range(hold_steps):
            self.world.data.ctrl[self.arm_controller.actuator_ids] = last
            self.world.step()
