from __future__ import annotations

import mujoco
import numpy as np

from control.main_trajectory_gen import (
    JointTrajectory,
    generate_toppra_segment_trajectory,
    generate_toppra_waypoints_trajectory,
)
from planning.path_planning import OmplPlanner, PlannerSpec
from sim.constraint import CollisionConstraint, Constraint


class MotionPlanner:
    def __init__(self, world):
        self.world = world

    @property
    def dt(self) -> float:
        return self.world.dt

    @property
    def vmax(self) -> np.ndarray:
        return self.world.vmax

    @property
    def amax(self) -> np.ndarray:
        return self.world.amax

    def bounds_from_qpos_adrs(self, qpos_adr: np.ndarray) -> np.ndarray:
        qpos_adr = np.asarray(qpos_adr, dtype=int)
        model = self.world.model

        qpos2jnt = {}
        for j in range(model.njnt):
            jtype = int(model.jnt_type[j])
            if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qadr = int(model.jnt_qposadr[j])
                qpos2jnt[qadr] = j

        bounds = np.zeros((qpos_adr.size, 2), dtype=float)
        for k, qi in enumerate(qpos_adr):
            if int(qi) not in qpos2jnt:
                raise ValueError(
                    f"qpos idx {qi} is not a 1DoF hinge/slide joint qposadr"
                )
            j = qpos2jnt[int(qi)]
            bounds[k, :] = model.jnt_range[j, :]
        return bounds

    def generate_segment_trajectory(
        self,
        q_goal_full: np.ndarray,
        v_max: np.ndarray | None = None,
        a_max: np.ndarray | None = None,
    ) -> JointTrajectory:
        if v_max is None:
            v_max = self.vmax
        if a_max is None:
            a_max = self.amax
        q_goal_full = np.asarray(q_goal_full, dtype=float).reshape(-1)
        dof_full = self.world.full_qpos_adr.shape[0]
        if q_goal_full.shape[0] != dof_full:
            raise ValueError(
                f"q_goal_full dof={q_goal_full.shape[0]}, expected {dof_full}"
            )

        q_start_full = self.world.full_qpos

        return generate_toppra_segment_trajectory(
            q_start=q_start_full,
            q_goal=q_goal_full,
            control_dt=self.dt,
            v_max=v_max,
            a_max=a_max,
        )

    def generate_waypoints_trajectory(
        self,
        waypoints: list[np.ndarray],
        v_max: np.ndarray | None = None,
        a_max: np.ndarray | None = None,
    ) -> JointTrajectory:
        if v_max is None:
            v_max = self.vmax
        if a_max is None:
            a_max = self.amax

        dof_full = self.world.full_qpos_adr.shape[0]
        if waypoints[0].shape[0] != dof_full:
            raise ValueError(
                f"waypoints dof={waypoints[0].shape[0]}, expected {dof_full}"
            )

        return generate_toppra_waypoints_trajectory(
            waypoints=waypoints,
            control_dt=self.dt,
            v_max=v_max,
            a_max=a_max,
        )

    @staticmethod
    def make_is_valid_full(constraints: list[Constraint]):
        def is_valid_full(q_full: np.ndarray) -> bool:
            for c in constraints:
                if not c.valid_config(q_full):
                    return False
            return True

        return is_valid_full

    @staticmethod
    def get_active_idx_from_full_ids(
        full_ids: np.ndarray,
        active_ids: np.ndarray,
        *,
        strict: bool = True,
        unique: bool = True,
    ) -> np.ndarray:
        full_ids = list(map(int, full_ids))
        active_ids = list(map(int, active_ids))

        pos = {}
        for i, jid in enumerate(full_ids):
            if jid in pos and unique:
                raise ValueError(f"full_ids has duplicate id: {jid}")
            pos[jid] = i

        out = []
        missing = []
        for jid in active_ids:
            if jid not in pos:
                missing.append(jid)
            else:
                out.append(pos[jid])

        if missing and strict:
            raise KeyError(f"active_ids not found in full_ids: {missing}")

        return np.asarray(out, dtype=int)

    def _expand_q_goal_main_to_full(
        self, q_goal_main: np.ndarray, q_init_full: np.ndarray
    ) -> np.ndarray:
        q_goal_main = np.asarray(q_goal_main, dtype=float).reshape(-1)
        if q_goal_main.shape[0] != self.world.main_qpos_adr.shape[0]:
            raise ValueError(
                f"q_goal_main dim={q_goal_main.shape[0]}, "
                f"expected {self.world.main_qpos_adr.shape[0]}"
            )

        q_goal_full = q_init_full.copy()
        q_goal_full[: len(self.world.main_qpos_adr)] = q_goal_main
        return q_goal_full

    def make_hand_adjacent_body_pairs(
        self, hand_prefix: str = "rh_"
    ) -> list[tuple[str, str]]:
        m = self.world.model
        hand_body_ids = []
        hand_body_names = []
        for bid in range(m.nbody):
            name = m.body(bid).name
            if name is not None and name.startswith(hand_prefix):
                hand_body_ids.append(bid)
                hand_body_names.append(name)

        hand_body_ids = np.asarray(hand_body_ids, dtype=int)
        hand_body_name_set = set(hand_body_names)

        allowed_pairs: list[tuple[str, str]] = []
        for bid in hand_body_ids:
            name = m.body(bid).name
            parent_id = m.body_parentid[bid]
            if parent_id < 0:
                continue
            parent_name = m.body(parent_id).name
            if parent_name in hand_body_name_set:
                allowed_pairs.append((parent_name, name))
        return allowed_pairs

    def main_motion_plan(
        self, q_goal_main: np.ndarray, max_planning_time: float = 5.0
    ):
        allowed_pairs = self.make_hand_adjacent_body_pairs()
        active_ids = self.get_active_idx_from_full_ids(
            self.world.full_joint_ids, self.world.main_joint_ids
        )
        collision_constraint = CollisionConstraint(
            self.world.model,
            robot_body_ids=self.world.full_body_ids,
            robot_qpos_adr=self.world.full_qpos_adr,
            qpos_template=self.world.data.qpos,
            allowed_collision_bodies=allowed_pairs,
            verbose=False,
            report_every=10,
        )
        planner_spec = PlannerSpec(
            ndof_full=self.world.full_ndof,
            active_idx=active_ids,
            bounds_active=self.world.main_joint_bounds,
        )

        planner = OmplPlanner(
            spec=planner_spec,
            is_valid_full=self.make_is_valid_full([collision_constraint]),
            simplify=True,
        )

        q_init_full = self.world.full_qpos
        q_goal_full = self._expand_q_goal_main_to_full(q_goal_main, q_init_full)
        return planner.plan(
            q_init_full=q_init_full, q_goal_full=q_goal_full, timeout=max_planning_time
        )
