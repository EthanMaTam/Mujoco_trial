from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Sequence

import logging
import numpy as np
import pinocchio as pin
import tsid

logger = logging.getLogger(__name__)


class TSIDIKSolver:
    """
    Lightweight TSID-based IK solver for a fixed-base 8-DoF arm+wrist model.

    Supports two model sources:
    - URDF: build full model from URDF and reduce to 8 DoF.
    - MJCF: build full model from MJCF and reduce to 8 DoF.

    The reduced model keeps:
    - 6 UR arm joints
    - 2 wrist joints (URDF: e.g. 'wrist_joint', 'forearm_joint';
                     MJCF: e.g. 'rh_WRJ1', 'rh_WRJ2').

    TSID tasks:
    - SE3 end-effector task (position + orientation)
    - Joint posture task as a soft preference (arm shape / wrist config)
    """

    def __init__(
        self,
        # Model source
        urdf_path: str | Path | None = None,
        mjcf_path: str | Path | None = None,
        use_mjcf: bool = False,
        ee_frame_name: Optional[str] = None,
        arm_joint_names: Optional[Sequence[str]] = None,
        wrist_joint_names: Optional[Sequence[str]] = None,
        # IK internal integration step
        dt: float = 0.002,
        # Default max iterations for IK
        max_iters_default: int = 300,
        # Task gains and weights
        kp_ee: float | np.ndarray = 100.0,
        kp_posture: float | np.ndarray = 10.0,
        w_ee: float = 1e3,
        w_posture: float = 1.0,
        # Optional default posture preference (if None: neutral posture)
        q_posture_default: Optional[np.ndarray] = None,
    ) -> None:
        """
        Parameters
        ----------
        urdf_path : str or Path, optional
            Path to the full URDF (UR5e + hand). Used when use_mjcf=False.
        mjcf_path : str or Path, optional
            Path to the MJCF XML. Used when use_mjcf=True.
        use_mjcf : bool
            If True, build model from MJCF; otherwise build from URDF.
        ee_frame_name : str, optional
            Name of the end-effector frame.
            If None:
                - URDF mode: "palm"
                - MJCF mode: "rh_palm"
        arm_joint_names : list of str, optional
            Names of the 6 arm joints. If None, default UR5e names are used.
        wrist_joint_names : list of str, optional
            Names of the 2 wrist joints.
            If None:
                - URDF mode: ["wrist_joint", "forearm_joint"]
                - MJCF mode: ["rh_WRJ1", "rh_WRJ2"]
        dt : float
            Internal integration step for the IK iterations (seconds).
        max_iters_default : int
            Default maximum iterations used by solve_ik when max_iters is None.
        kp_ee : float or array(6,)
            Proportional gains for SE3 task. If scalar, expanded to 6 dims.
        kp_posture : float or array(na,)
            Proportional gains for posture task. If scalar, expanded to `na`.
        w_ee : float
            Weight of end-effector task (higher = more important).
        w_posture : float
            Weight of posture task (lower = softer).
        q_posture_default : array(na,), optional
            Default preferred posture. If None, uses neutral(q) of the reduced model.
        """
        self.use_mjcf = bool(use_mjcf)
        self.dt = float(dt)
        self.max_iters_default = int(max_iters_default)

        # -------- 1) Build full model (URDF or MJCF) --------
        if self.use_mjcf:
            if mjcf_path is None:
                raise ValueError("mjcf_path must be provided when use_mjcf=True")
            self.model_path = str(mjcf_path)
            self.model_source = "mjcf"
            model_full: pin.Model = pin.buildModelFromMJCF(self.model_path)
        else:
            if urdf_path is None:
                raise ValueError("urdf_path must be provided when use_mjcf=False")
            self.model_path = str(urdf_path)
            self.model_source = "urdf"
            model_full = pin.buildModelFromUrdf(self.model_path)

        self.model_full: pin.Model = model_full

        # Default joint name lists if not provided
        if arm_joint_names is None:
            arm_joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]

        if wrist_joint_names is None:
            if self.use_mjcf:
                wrist_joint_names = [
                    "rh_WRJ1",
                    "rh_WRJ2",
                ]
            else:
                wrist_joint_names = [
                    "wrist_joint",
                    "forearm_joint",
                ]

        keep_joint_names = set(arm_joint_names + list(wrist_joint_names))

        lock_joint_ids: list[int] = []
        for jid, name in enumerate(self.model_full.names):
            # jid == 0 is the universe (pseudo-joint), never lock
            if jid == 0:
                continue
            if name not in keep_joint_names:
                lock_joint_ids.append(jid)

        q_ref_full = pin.neutral(self.model_full)
        self.model: pin.Model = pin.buildReducedModel(
            self.model_full, lock_joint_ids, q_ref_full
        )
        self.joint_names_tsid = [name for name in self.model.names[1:]]
        logger.info("[TSIDIKSolver] joint order (TSID): %s", self.joint_names_tsid)

        self.data_fk: pin.Data = self.model.createData()

        # End-effector frame name
        if ee_frame_name is None:
            self.ee_frame_name = "rh_palm" if self.use_mjcf else "palm"
        else:
            self.ee_frame_name = ee_frame_name

        # End-effector frame id in reduced model
        self.ee_fid: int = self.model.getFrameId(self.ee_frame_name)

        # -------- 2) Build TSID robot wrapper over the reduced model --------
        # Fixed-base system
        self.robot = tsid.RobotWrapper(self.model, tsid.FIXED_BASE_SYSTEM, False)
        assert self.robot.is_fixed_base, "TSID RobotWrapper is not fixed-base as expected."

        # TSID uses its own model instance; keep a reference
        self.model_tsid: pin.Model = self.robot.model()
        self.data_tsid: pin.Data = self.model_tsid.createData()

        logger.info(
            "[TSIDIKSolver] source=%s, nq=%d, nv=%d, na=%d",
            self.model_source,
            self.model_tsid.nq,
            self.model_tsid.nv,
            self.robot.na,
        )

        # -------- 3) Inverse dynamics formulation (used here for IK only) --------
        self.invdyn = tsid.InverseDynamicsFormulationAccForce("tsid-ik", self.robot, False)

        # Initial state
        self.q0 = pin.neutral(self.model_tsid)
        self.v0 = np.zeros(self.model_tsid.nv)

        # -------- Basic Joint Limits Bound Task --------
        self.bounds_task = tsid.TaskJointPosVelAccBounds(
            "joint-bounds", self.robot, self.dt
        )
        q_min = self.model_tsid.lowerPositionLimit.copy()
        q_max = self.model_tsid.upperPositionLimit.copy()

        eps = 1e-6
        q_min += eps
        q_max -= eps

        self.bounds_task.setPositionBounds(q_min, q_max)

        w_bounds = 1e3
        self.invdyn.addMotionTask(self.bounds_task, w_bounds, 0, 0.0)

        # -------- 4) SE3 end-effector task --------
        self.ee_task = tsid.TaskSE3Equality("task-ee", self.robot, self.ee_frame_name)

        if np.isscalar(kp_ee):
            kp_ee_arr = float(kp_ee) * np.ones(6)
        else:
            kp_ee_arr = np.asarray(kp_ee, dtype=float).reshape(6)
        kd_ee_arr = 2.0 * np.sqrt(kp_ee_arr)

        self.ee_task.setKp(kp_ee_arr)
        self.ee_task.setKd(kd_ee_arr)

        self.w_ee = float(w_ee)
        # Use only priority level 1 (multi-level support can be unstable in some builds)
        self.invdyn.addMotionTask(self.ee_task, self.w_ee, 1, 0.0)

        # A constant SE3 trajectory: we simply update its reference each step
        self.traj_ee = tsid.TrajectorySE3Constant("traj-ee", pin.SE3.Identity())

        # -------- 5) Joint posture task --------
        self.posture_task = tsid.TaskJointPosture("task-posture", self.robot)

        na = self.robot.na
        if np.isscalar(kp_posture):
            kp_post_arr = float(kp_posture) * np.ones(na)
        else:
            kp_post_arr = np.asarray(kp_posture, dtype=float).reshape(na)
        kd_post_arr = 2.0 * np.sqrt(kp_post_arr)

        self.posture_task.setKp(kp_post_arr)
        self.posture_task.setKd(kd_post_arr)

        self.w_posture = float(w_posture)
        self.invdyn.addMotionTask(self.posture_task, self.w_posture, 1, 0.0)

        # Default posture reference (if not provided, use neutral)
        if q_posture_default is None:
            self.q_posture_default = self.q0.copy()
        else:
            q_posture_default = np.asarray(q_posture_default, dtype=float).reshape(na)
            self.q_posture_default = q_posture_default

        self.traj_post = tsid.TrajectoryEuclidianConstant(
            "traj-posture", self.q_posture_default
        )

        # -------- 6) QP solver --------
        # Warm-up to get dimensions
        Hqp0 = self.invdyn.computeProblemData(0.0, self.q0, self.v0)
        self.solver = tsid.SolverHQuadProgFast("qpsolver")
        self.solver.resize(self.invdyn.nVar, self.invdyn.nEq, self.invdyn.nIn)
        logger.info(
            "[TSIDIKSolver] QP resized: nVar=%d, nEq=%d, nIn=%d",
            self.invdyn.nVar,
            self.invdyn.nEq,
            self.invdyn.nIn,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def solve_ik(
        self,
        pose6: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        q_posture: Optional[np.ndarray] = None,
        max_iters: Optional[int] = None,
        tol: float = 1e-4,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve a TSID-based IK problem for a given end-effector target pose.

        Parameters
        ----------
        pose6 : array(6,)
            Desired end-effector pose in base/world frame:
            [x, y, z, roll, pitch, yaw], angles in radians.
        q_init : array(nq,), optional
            Initial joint configuration. If None, uses neutral.
        q_posture : array(na,), optional
            Preferred posture. If None, uses the default posture
            specified at construction.
        max_iters : int, optional
            Maximum number of TSID iterations. If None, uses
            self.max_iters_default.
        tol : float
            Position error tolerance for early stopping.
        verbose : bool
            If True, logs progress every 40 iterations and convergence message.

        Returns
        -------
        q_goal : array(nq,)
            IK solution.
        success : bool
            True if position error fell below `tol` within `max_iters`.
        """
        if max_iters is None:
            max_iters = self.max_iters_default

        # Convert 6D pose vector -> SE3
        M_des = self.vec6_to_se3(pose6)

        q = self.q0.copy() if q_init is None else np.array(q_init, dtype=float).copy()
        v = self.v0.copy()

        if q_posture is None:
            q_post = self.q_posture_default.copy()
        else:
            q_post = np.asarray(q_posture, dtype=float).reshape(self.robot.na)

        t = 0.0
        success = False

        for it in range(max_iters):
            q, v = self._tsid_step(q, v, M_des, q_post, t)
            t += self.dt

            # Evaluate current end-effector position error for monitoring
            pin.forwardKinematics(self.model, self.data_fk, q, v)
            pin.updateFramePlacements(self.model, self.data_fk)
            M_ee = self.data_fk.oMf[self.ee_fid]

            err_pos = M_ee.translation - M_des.translation
            pos_norm = np.linalg.norm(err_pos)

            if verbose and it % 40 == 0:
                logger.info(
                    "[TSID IK] iter %3d, pos_error = %.6e",
                    it,
                    pos_norm,
                )

            if pos_norm < tol:
                if verbose:
                    logger.info(
                        "[TSID IK] Converged at iter %d, pos_error = %.6e",
                        it,
                        pos_norm,
                    )
                success = True
                break

        if not success and verbose:
            logger.warning(
                "[TSID IK] Did not converge within %d iterations (last pos_error = %.6e)",
                max_iters,
                pos_norm,
            )

        return q, success

    # -------------------------------------------------------------------------
    # Internal single-step TSID update
    # -------------------------------------------------------------------------

    def _tsid_step(
        self,
        q: np.ndarray,
        v: np.ndarray,
        M_des: pin.SE3,
        q_posture: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run one TSID step:
        - Update SE3 and posture trajectories.
        - Build and solve the HQP.
        - Integrate accelerations to get next q, v.
        """
        # 1) Update trajectory references
        self.traj_ee.setReference(M_des)
        sample_ee = self.traj_ee.computeNext()
        self.ee_task.setReference(sample_ee)

        self.traj_post.setReference(q_posture)
        sample_post = self.traj_post.computeNext()
        self.posture_task.setReference(sample_post)

        # 2) Build HQP and solve
        Hqp = self.invdyn.computeProblemData(t, q, v)
        sol = self.solver.solve(Hqp)
        if sol.status != 0:
            logger.error("TSID QP failed, status = %d", sol.status)
            raise RuntimeError(f"TSID QP failed, status = {sol.status}")

        ddq = self.invdyn.getAccelerations(sol)  # (nv,)

        # 3) Purely kinematic integration (no dynamics / torques used)
        v_next = v + self.dt * ddq
        dq = self.dt * v_next
        q_next = pin.integrate(self.model_tsid, q, dq)

        return q_next, v_next

    @staticmethod
    def vec6_to_se3(pose6: np.ndarray) -> pin.SE3:
        """
        Convert a 6D pose vector to SE3.

        pose6: [x, y, z, roll, pitch, yaw] (radians, base/world frame)
        """
        pose6 = np.asarray(pose6, dtype=float).reshape(6)
        xyz = pose6[:3]
        rpy = pose6[3:]
        R = pin.rpy.rpyToMatrix(rpy[0], rpy[1], rpy[2])
        return pin.SE3(R, xyz)

    @staticmethod
    def se3_to_vec6(M: pin.SE3) -> np.ndarray:
        """
        Convert an SE3 pose to a 6D vector [x, y, z, roll, pitch, yaw].
        """
        xyz = M.translation
        rpy = pin.rpy.matrixToRpy(M.rotation)
        return np.concatenate([xyz, rpy])


# -----------------------------------------------------------------------------
# Simple demo / test
# -----------------------------------------------------------------------------

def _demo_urdf() -> None:
    urdf_path = Path("assets/ur5e_shadow/ur5e_shadow.urdf")
    solver = TSIDIKSolver(
        urdf_path=urdf_path,
        use_mjcf=False,
        ee_frame_name="palm",
        max_iters_default=200,  # override default if you want
    )

    q_init = solver.q0.copy()

    pin.forwardKinematics(solver.model, solver.data_fk, q_init)
    pin.updateFramePlacements(solver.model, solver.data_fk)
    M_ee0 = solver.data_fk.oMf[solver.ee_fid]

    M_des = pin.SE3(M_ee0.rotation, M_ee0.translation.copy())
    M_des.translation[2] += 0.10

    # Convert SE3 -> pose6
    pose6_des = TSIDIKSolver.se3_to_vec6(M_des)

    q_goal, ok = solver.solve_ik(pose6_des, q_init=q_init, q_posture=None, verbose=True)
    logger.info("URDF IK success: %s", ok)
    logger.info("q_goal: %s", q_goal)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    )
    _demo_urdf()
