"""
Quick tendon synergy test.

功能：
- 给定弯曲量 s（4 指 J0），调用 TendonController 的映射并打印 ctrl。
- 或者给定每指的 J1/J2 目标，先计算 s = q1 + q2，再驱动肌腱 actuator。
运行：
    python -m sim.test_tendon_synergy --s 0.5 0.4 0.3 0.2
    python -m sim.test_tendon_synergy --q1 0.2 0.2 0.2 0.2 --q2 0.3 0.2 0.1 0.0
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from sim.execution import CombinedTrajectory, HandCommand, HandTrajectory
from sim.mj_world import MjWorld
import mujoco


def _to_len4(vals: Sequence[float]) -> np.ndarray:
    """Broadcast 1 value to 4 fingers if needed."""
    arr = np.asarray(vals, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 4)
    if arr.size != 4:
        raise ValueError("Expected 1 or 4 values.")
    return arr


def compute_synergy_from_joints(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    q1 = _to_len4(q1)
    q2 = _to_len4(q2)
    return q1 + q2  # u_J0 = q1_ref + q2_ref，比例均等


def run_synergy_test(xml_path: str, s_vals: np.ndarray, steps: int, viewer: bool):
    with MjWorld(xml_path=xml_path, use_viewer=viewer) as world:
        world.hold_default_conf()

        # 构建重复的 hand commands
        cmds = [HandCommand(synergy=s_vals) for _ in range(steps)]
        hand_traj = HandTrajectory(commands=cmds, dt=world.dt)
        arm_traj = np.repeat(world.default_conf[None, :], steps, axis=0)
        combined = CombinedTrajectory(arm=arm_traj, hand=hand_traj, dt=world.dt)

        # 预期 ctrl 便于打印核对
        tc = world.executor.tendon_controller
        if tc is not None and tc.tendon_actuator_ids is not None:
            m = world.model
            names = [
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(aid))
                for aid in tc.tendon_actuator_ids
            ]
            print("Tendon actuators:", names)
            print("Tendon ctrlrange:", m.actuator_ctrlrange[tc.tendon_actuator_ids])
        expected_ctrl = None
        if tc is not None and tc.tendon_ctrl_range is not None:
            expected_ctrl = np.clip(
                s_vals, tc.tendon_ctrl_range[:, 0], tc.tendon_ctrl_range[:, 1]
            )

        print("Input synergy s:", s_vals)
        if expected_ctrl is not None:
            print("Expected tendon ctrl (clipped):", expected_ctrl)

        world.executor.follow_combined(combined, hold_steps=20)
        if tc is not None and tc.tendon_actuator_ids is not None:
            actual = world.data.ctrl[tc.tendon_actuator_ids]
            print("Actual tendon ctrl after execution:", actual)

        # 如果有 viewer，进入循环；否则自动退出
        while world.is_running() and viewer:
            world.step()


def main():
    parser = argparse.ArgumentParser(description="Test tendon synergy mapping.")
    parser.add_argument(
        "--xml",
        type=str,
        default="ur5e_shadowhand_scene.xml",
        help="Path to Mujoco XML.",
    )
    parser.add_argument(
        "--s",
        type=float,
        nargs="+",
        help="Synergy bend values for 4 fingers (1 or 4 numbers).",
    )
    parser.add_argument(
        "--q1",
        type=float,
        nargs="+",
        help="Joint J1 targets (1 or 4 numbers).",
    )
    parser.add_argument(
        "--q2",
        type=float,
        nargs="+",
        help="Joint J2 targets (1 or 4 numbers).",
    )
    parser.add_argument("--steps", type=int, default=200, help="Steps to run.")
    parser.add_argument(
        "--no-viewer", action="store_true", help="Disable Mujoco viewer."
    )
    args = parser.parse_args()

    if args.s is not None:
        s_vals = _to_len4(args.s)
    elif args.q1 is not None and args.q2 is not None:
        s_vals = compute_synergy_from_joints(args.q1, args.q2)
        print("Computed synergy s = q1 + q2:", s_vals)
    else:
        parser.error("Provide either --s or both --q1 and --q2.")

    run_synergy_test(
        xml_path=args.xml,
        s_vals=s_vals,
        steps=args.steps,
        viewer=not args.no_viewer,
    )


if __name__ == "__main__":
    main()
