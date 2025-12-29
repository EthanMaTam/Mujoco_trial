from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
from enum import Enum, auto

import logging
import numpy as np
import pinocchio as pin
import tsid
from ruckig import Result, Ruckig, InputParameter, OutputParameter

# Type alias for readability
NDArray = np.ndarray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Basic hand model structures
# ---------------------------------------------------------------------------


@dataclass
class FingerInfo:
    """
    Simple container for per-finger metadata.

    Attributes
    ----------
    name:
        Logical name of the finger ("thumb", "index", etc.).
    joint_ids:
        List of Pinocchio joint IDs belonging to this finger, ordered from
        distal (tip) to proximal (base) or any consistent convention.
    tip_frame_id:
        Pinocchio frame ID of the fingertip frame.
    """

    name: str
    joint_ids: list[int]
    tip_frame_id: int


@dataclass
class SynergyParam:
    """
    High-level parameter describing how a gesture value maps into synergy space.

    Attributes
    ----------
    name:
        Name of the gesture parameter (e.g., "grasp", "thumb_opposition").
    index:
        Index into the underlying synergy vector s.
    ui_min, ui_max:
        Range of values coming from the UI / high-level interface.
    phys_min, phys_max:
        Range of the corresponding value in synergy space.
    curve:
        Optional non-linear shaping function defined on [0, 1]. If provided, it
        is applied after normalizing the UI value into [0, 1].
    """

    name: str
    index: int
    ui_min: float = 0.0
    ui_max: float = 1.0
    phys_min: float = 0.0
    phys_max: float = 1.0
    curve: Optional[Callable[[float], float]] = None

    def ui_to_s(self, ui_val: float) -> float:
        """Map a UI-level value into the synergy coordinate for this parameter."""
        # Normalize to [0, 1]
        denom = self.ui_max - self.ui_min
        if abs(denom) < 1e-12:
            u = 0.0
        else:
            u = (ui_val - self.ui_min) / denom
        u = float(np.clip(u, 0.0, 1.0))

        if self.curve is not None:
            u = float(self.curve(u))

        return self.phys_min + u * (self.phys_max - self.phys_min)


@dataclass
class HandSynergyModel:
    """
    Linear hand synergy model defined in joint space.

    The model assumes a linear relation between synergy coordinates s and
    a stack of hand joint angles q_hand:

        q_hand = q0_hand + S @ s

    where:
        - q_hand has dimension (n_h,)
        - S has shape (n_h, r)
        - s has dimension (r,)
    """

    hand_joint_ids: list[int]
    q0_hand: NDArray
    S: NDArray
    synergy_names: list[str] | None = None
    s_min: NDArray | None = None
    s_max: NDArray | None = None
    s_transform: Optional[Callable[[NDArray], NDArray]] = None

    def __post_init__(self) -> None:
        self.q0_hand = np.asarray(self.q0_hand, dtype=float).reshape(-1)
        self.S = np.asarray(self.S, dtype=float)

        n_h, r = self.S.shape
        if len(self.hand_joint_ids) != n_h:
            raise ValueError("hand_joint_ids length must match S row dimension.")
        if self.q0_hand.shape != (n_h,):
            raise ValueError("q0_hand must have shape (n_h,)")

        if self.synergy_names is None:
            self.synergy_names = [f"s_{i}" for i in range(r)]
        else:
            if len(self.synergy_names) != r:
                raise ValueError("synergy_names length must match S column dimension.")

        if self.s_min is None:
            self.s_min = -np.ones(r, dtype=float)
        if self.s_max is None:
            self.s_max = np.ones(r, dtype=float)

        self.s_min = np.asarray(self.s_min, dtype=float).reshape(r)
        self.s_max = np.asarray(self.s_max, dtype=float).reshape(r)

        # Pre-compute pseudo-inverse (S^T S)^{-1} S^T for encode()
        ST = self.S.T
        G = ST @ self.S
        self._S_pinv = np.linalg.inv(G) @ ST

        if self.s_transform is None:
            self.s_transform = lambda s: s

        logger.debug(
            "[HandSynergyModel] Initialized with n_h=%d, r=%d", n_h, r
        )

    @property
    def n_h(self) -> int:
        return len(self.hand_joint_ids)

    @property
    def dim_s(self) -> int:
        return self.S.shape[1]

    def extract_q_hand(self, model: pin.Model, q_full: NDArray) -> NDArray:
        """
        Extract the hand joint angles from a full configuration vector.

        Parameters
        ----------
        model:
            Pinocchio model of the full robot (arm + hand).
        q_full:
            Full configuration vector of length model.nq.

        Returns
        -------
        q_hand:
            Hand joint angles stacked in the same order as hand_joint_ids.
        """
        q_hand = np.empty(self.n_h, dtype=float)
        for k, jid in enumerate(self.hand_joint_ids):
            idx = model.idx_qs[jid]
            q_hand[k] = q_full[idx]
        return q_hand

    def decode(self, model: pin.Model, q_full: NDArray, s: NDArray) -> NDArray:
        """
        Decode synergy coordinates into a full configuration.

        Parameters
        ----------
        model:
            Pinocchio model of the full robot.
        q_full:
            Reference full configuration; non-hand joints are copied from here.
        s:
            Synergy vector of shape (r,).

        Returns
        -------
        q_new:
            New full configuration with hand joints replaced by synergy mapping.
        """
        s = np.asarray(s, dtype=float).reshape(self.dim_s)
        s_clamped = np.clip(s, self.s_min, self.s_max)
        s_eff = self.s_transform(s_clamped)

        q_new = q_full.copy()
        q_hand = self.q0_hand + self.S @ s_eff

        for k, jid in enumerate(self.hand_joint_ids):
            i = model.idx_qs[jid]
            q_new[i] = q_hand[k]

        return q_new

    def encode(self, model: pin.Model, q_full: NDArray) -> NDArray:
        """
        Recover synergy coordinates (in the linear sense) from a full configuration.

        This uses the relation:

            s_lin = (S^T S)^{-1} S^T (q_hand - q0_hand)
        """
        q_hand = self.extract_q_hand(model, q_full)
        delta = q_hand - self.q0_hand
        s_lin = self._S_pinv @ delta
        return s_lin

    def build_s_from_named(self, **named_values: float) -> NDArray:
        """
        Construct a synergy vector from a mapping of name -> value.

        Any synergy not present in named_values is set to zero.
        """
        s = np.zeros(self.dim_s, dtype=float)
        name_to_idx = {n: i for i, n in enumerate(self.synergy_names)}
        for name, val in named_values.items():
            if name not in name_to_idx:
                raise KeyError(f"Unknown synergy name: {name}")
            s[name_to_idx[name]] = float(val)
        return s

    def decode_named(self, model: pin.Model, q_full: NDArray, **named_values: float) -> NDArray:
        """Decode a configuration using named synergy values."""
        s = self.build_s_from_named(**named_values)
        return self.decode(model, q_full, s)

    def encode_named(self, model: pin.Model, q_full: NDArray) -> Dict[str, float]:
        """Encode the current configuration into a dict of synergy_name -> value."""
        s_vec = self.encode(model, q_full)
        return {name: float(s_vec[i]) for i, name in enumerate(self.synergy_names)}


@dataclass
class HandGestureSpace:
    """
    High-level gesture interface built on top of a HandSynergyModel.

    Gesture parameters are defined in UI space (e.g., [0, 1]) and then mapped
    to synergy coordinates using a set of SynergyParam objects.
    """

    synergy_model: HandSynergyModel
    params: Dict[str, SynergyParam]

    def gesture_to_s(self, gesture: Dict[str, float]) -> NDArray:
        """
        Map a dict of gesture_name -> UI value to a synergy vector s.

        Any synergy dimension that does not appear in params is set to zero.
        """
        s = np.zeros(self.synergy_model.dim_s, dtype=float)

        for gesture_name, param in self.params.items():
            ui_val = gesture.get(gesture_name, 0.0)
            s_val = param.ui_to_s(ui_val)
            s[param.index] = s_val

        return s

    def decode_gesture(
        self,
        model: pin.Model,
        q_full_ref: NDArray,
        gesture: Dict[str, float],
    ) -> NDArray:
        """
        Decode a full configuration from a high-level gesture.

        Parameters
        ----------
        model:
            Pinocchio model of the full robot.
        q_full_ref:
            Reference configuration used for the non-hand joints.
        gesture:
            Mapping of gesture_name -> UI value.
        """
        s = self.gesture_to_s(gesture)
        logger.debug("[HandGestureSpace] gesture=%s -> s=%s", gesture, s)
        return self.synergy_model.decode(model, q_full_ref, s)


# ---------------------------------------------------------------------------
# Kinematics / IK
# ---------------------------------------------------------------------------


class HandKinematics:
    """
    Kinematic interface for a hand attached to a full robot model.

    This class wraps Pinocchio to provide:
    - forward kinematics for the full model,
    - fingertip pose queries,
    - fingertip Jacobians,
    - single-step IK updates in joint space or synergy space.
    """

    def __init__(
        self,
        mjcf_path: str,
        palm_frame_name: str,
        finger_joint_name_map: Dict[str, list[str]] | None = None,
        finger_tip_frame_map: Dict[str, str] | None = None,
        q_ref: np.ndarray | None = None
    ) -> None:
        """
        Parameters
        ----------
        mjcf_path:
            Path to the MJCF file describing the full robot (arm + hand).
        palm_frame_name:
            Name of the palm frame in the Pinocchio model.
        finger_joint_name_map:
            Optional mapping from logical finger names to lists of joint names.
            If None, a ShadowHand-like default is used.
        finger_tip_frame_map:
            Optional mapping from logical finger names to fingertip frame names.
            If None, a ShadowHand-like default is used.
        """
        self.model_full: pin.Model = pin.buildModelFromMJCF(mjcf_path)
        self.data_full: pin.Data = self.model_full.createData()

        self.palm_frame_id: int = self.model_full.getFrameId(palm_frame_name)

        if q_ref is None:
            self.q_ref = pin.neutral(self.model_full)
        else:
            self.q_ref = np.asarray(q_ref, dtype=float).reshape(self.model_full.nq)

        if finger_joint_name_map is None or finger_tip_frame_map is None:
            finger_joint_name_map, finger_tip_frame_map = self._default_shadowhand_finger_config()

        self.fingers: Dict[str, FingerInfo] = {}
        self._init_fingers(finger_joint_name_map, finger_tip_frame_map)

        logger.info(
            "[HandKinematics] Loaded model from %s with %d DOF",
            mjcf_path,
            self.model_full.nq,
        )

    @staticmethod
    def _default_shadowhand_finger_config() -> tuple[Dict[str, list[str]], Dict[str, str]]:
        """
        Default finger configuration for a ShadowHand-like model.

        You should adapt these names to match your own MJCF / URDF if needed.
        """
        finger_joint_name_map: Dict[str, list[str]] = {
            "index": ["rh_FFJ1", "rh_FFJ2", "rh_FFJ3", "rh_FFJ4"],
            "middle": ["rh_MFJ1", "rh_MFJ2", "rh_MFJ3", "rh_MFJ4"],
            "ring": ["rh_RFJ1", "rh_RFJ2", "rh_RFJ3", "rh_RFJ4"],
            "little": ["rh_LFJ1", "rh_LFJ2", "rh_LFJ3", "rh_LFJ4", "rh_LFJ5"],
            "thumb": ["rh_THJ1", "rh_THJ2", "rh_THJ3", "rh_THJ4", "rh_THJ5"],
        }

        finger_tip_frame_map: Dict[str, str] = {
            "index": "rh_ffdistal",
            "middle": "rh_mfdistal",
            "ring": "rh_rfdistal",
            "little": "rh_lfdistal",
            "thumb": "rh_thdistal",
        }

        return finger_joint_name_map, finger_tip_frame_map

    def _init_fingers(
        self,
        finger_joint_name_map: Dict[str, list[str]],
        finger_tip_frame_map: Dict[str, str],
    ) -> None:
        """Populate the self.fingers dictionary from name maps."""
        m = self.model_full

        for finger_name, joint_names in finger_joint_name_map.items():
            if finger_name not in finger_tip_frame_map:
                raise KeyError(f"No tip frame defined for finger '{finger_name}'")

            joint_ids = [m.getJointId(jn) for jn in joint_names]
            tip_id = m.getFrameId(finger_tip_frame_map[finger_name])

            self.fingers[finger_name] = FingerInfo(
                name=finger_name,
                joint_ids=joint_ids,
                tip_frame_id=tip_id,
            )

        logger.info(
            "[HandKinematics] Initialized fingers: %s", list(self.fingers.keys())
        )

    def forward_kinematics(self, q: NDArray) -> None:
        """Run forward kinematics and update all frame placements."""
        if q.shape[0] != self.model_full.nq:
            raise ValueError(
                f"Configuration size mismatch: got {q.shape[0]}, "
                f"expected {self.model_full.nq}"
            )
        pin.forwardKinematics(self.model_full, self.data_full, q)
        pin.updateFramePlacements(self.model_full, self.data_full)

    def get_tip_pose(self, finger_name: str, q: NDArray) -> pin.SE3:
        """Return the fingertip pose in the world frame."""
        self.forward_kinematics(q)
        info = self.fingers[finger_name]
        return self.data_full.oMf[info.tip_frame_id]

    def get_tip_jacobian(self, finger_name: str, q: NDArray) -> NDArray:
        """Return the 6xn Jacobian of the fingertip in its LOCAL frame."""
        info = self.fingers[finger_name]
        self.forward_kinematics(q)
        J = pin.computeFrameJacobian(
            self.model_full,
            self.data_full,
            q,
            info.tip_frame_id,
            pin.ReferenceFrame.LOCAL,
        )
        return J

    # ------------------------------------------------------------------
    # Single-step IK in joint space (q)
    # ------------------------------------------------------------------

    def finger_ik_step_q_palm(
        self,
        finger_name: str,
        q: NDArray,
        M_des_palm: pin.SE3,
        alpha: float = 0.5,
        damping: float = 1e-4,
        position_only: bool = True,
    ) -> tuple[NDArray, float]:
        """
        Perform one damped least-squares IK step for a single finger in joint space.

        The target pose is expressed in the palm frame. Internally it is converted
        to a world frame target using the current palm pose.

        Parameters
        ----------
        finger_name:
            Name of the finger to move.
        q:
            Current full configuration (will not be modified in-place).
        M_des_palm:
            Desired fingertip pose expressed in the palm frame.
        alpha:
            Step length in joint space.
        damping:
            Damping factor for DLS.
        position_only:
            If True, only the translational error is used (orientation error
            is set to zero).

        Returns
        -------
        q_new:
            Updated configuration.
        err_norm:
            Norm of the task-space error used for the update.
        """
        model, data = self.model_full, self.data_full
        info = self.fingers[finger_name]

        # Forward kinematics
        self.forward_kinematics(q)

        # Current tip and palm poses in world frame
        M_world_tip = data.oMf[info.tip_frame_id]
        M_world_palm = data.oMf[self.palm_frame_id]

        # Desired tip pose in world frame: palm * desired_in_palm
        M_world_tip_des = M_world_palm * M_des_palm

        # SE(3) error in LOCAL frame of the tip
        err = pin.log(M_world_tip.inverse() * M_world_tip_des).vector  # (6,)
        if position_only:
            # Zero out rotational error (first three components)
            err[:3] = 0.0

        # Full 6 x nv Jacobian at the tip (LOCAL frame)
        J_full = pin.computeFrameJacobian(
            model,
            data,
            q,
            info.tip_frame_id,
            pin.ReferenceFrame.LOCAL,
        )

        # Restrict to this finger's joints using velocity indices
        finger_joint_ids = info.joint_ids
        idxs = [model.idx_vs[jid] for jid in finger_joint_ids]
        J_f = J_full[:, idxs]  # 6 x n_finger

        # Damped least-squares: dq = J^T (J J^T + λ^2 I)^(-1) err
        m_dim = J_f.shape[0]
        H = J_f @ J_f.T + (damping**2) * np.eye(m_dim)
        dq_finger = J_f.T @ np.linalg.solve(H, err)

        # Apply update to a copy of q
        q_new = q.copy()
        q_min = model.lowerPositionLimit
        q_max = model.upperPositionLimit

        for col, jid in enumerate(finger_joint_ids):
            i = model.idx_qs[jid]
            val = q_new[i] + alpha * dq_finger[col]
            q_new[i] = np.clip(val, q_min[i], q_max[i])

        err_norm = float(np.linalg.norm(err))
        return q_new, err_norm

    # ------------------------------------------------------------------
    # Single-step IK in synergy space (s)
    # ------------------------------------------------------------------

    def finger_ik_step_s_palm(
        self,
        finger_name: str,
        s: NDArray,
        synergy: HandSynergyModel,
        q_full_ref: NDArray,
        M_des_palm: pin.SE3,
        alpha: float = 0.5,
        damping: float = 1e-4,
        position_only: bool = True,
    ) -> tuple[NDArray, NDArray, float]:
        """
        Perform one IK step in synergy space for a single finger.

        The arm and wrist joints are taken from q_full_ref. The hand joints are
        controlled through synergy coordinates s.

        Parameters
        ----------
        finger_name:
            Name of the finger to move.
        s:
            Current synergy vector.
        synergy:
            HandSynergyModel instance.
        q_full_ref:
            Reference configuration for non-hand joints.
        M_des_palm:
            Desired fingertip pose expressed in the palm frame.
        alpha:
            Step length in synergy space.
        damping:
            Damping factor for DLS.
        position_only:
            If True, only the translational error is used.

        Returns
        -------
        s_new:
            Updated synergy vector.
        q_new:
            Updated full configuration.
        err_norm:
            Norm of the task-space error.
        """
        model, data = self.model_full, self.data_full
        info = self.fingers[finger_name]

        # Decode current configuration from synergy
        q = synergy.decode(model, q_full_ref, s)

        # Forward kinematics
        self.forward_kinematics(q)

        # Current tip and palm poses
        M_world_tip = data.oMf[info.tip_frame_id]
        M_world_palm = data.oMf[self.palm_frame_id]

        # Desired tip pose in world frame
        M_world_tip_des = M_world_palm * M_des_palm

        # Task-space error
        err = pin.log(M_world_tip.inverse() * M_world_tip_des).vector
        if position_only:
            err[:3] = 0.0

        # Full Jacobian
        J_full = pin.computeFrameJacobian(
            model,
            data,
            q,
            info.tip_frame_id,
            pin.ReferenceFrame.LOCAL,
        )

        # Restrict to hand joints
        hand_idx_vs = [model.idx_vs[jid] for jid in synergy.hand_joint_ids]
        J_hand = J_full[:, hand_idx_vs]  # 6 x n_h

        # Map to synergy space: J_s = J_hand * S
        J_s = J_hand @ synergy.S  # 6 x r

        # DLS in synergy space
        m_dim = J_s.shape[0]
        H = J_s @ J_s.T + (damping**2) * np.eye(m_dim)
        ds = J_s.T @ np.linalg.solve(H, err)

        s_new = s + alpha * ds
        q_new = synergy.decode(model, q_full_ref, s_new)

        err_norm = float(np.linalg.norm(err))
        return s_new, q_new, err_norm


class HandIKSolver:
    """
    Convenience wrapper around HandKinematics for iterative finger IK solving.

    Supports both:
    - joint-space IK (directly updating q),
    - synergy-space IK (updating s and decoding to q).
    """

    def __init__(
        self,
        kin: HandKinematics,
        alpha: float = 0.5,
        damping: float = 1e-4,
        max_iters: int = 50,
        tol: float = 1e-4,
        position_only: bool = True,
    ) -> None:
        self.kin = kin
        self.alpha = alpha
        self.damping = damping
        self.max_iters = max_iters
        self.tol = tol
        self.position_only = position_only

    def solve_finger_ik_q_palm(
        self,
        finger_name: str,
        q_init: NDArray,
        M_des_palm: pin.SE3,
    ) -> tuple[NDArray, bool]:
        """
        Iteratively solve IK for a single finger in joint space.

        Returns
        -------
        q_sol:
            Final configuration (whether converged or not).
        converged:
            True if the error norm fell below tol within max_iters.
        """
        q = q_init.copy()
        for it in range(self.max_iters):
            q, err = self.kin.finger_ik_step_q_palm(
                finger_name=finger_name,
                q=q,
                M_des_palm=M_des_palm,
                alpha=self.alpha,
                damping=self.damping,
                position_only=self.position_only,
            )
            logger.debug(
                "[HandIKSolver:q] iter=%d finger=%s err=%.3e",
                it,
                finger_name,
                err,
            )
            if err < self.tol:
                logger.info(
                    "[HandIKSolver:q] Converged in %d iterations (err=%.3e)",
                    it + 1,
                    err,
                )
                return q, True

        logger.warning(
            "[HandIKSolver:q] Did not converge in %d iterations (last err=%.3e)",
            self.max_iters,
            err,
        )
        return q, False

    def solve_finger_ik_s_palm(
        self,
        finger_name: str,
        s_init: NDArray,
        synergy: HandSynergyModel,
        q_full_ref: NDArray,
        M_des_palm: pin.SE3,
    ) -> tuple[NDArray, NDArray, bool]:
        """
        Iteratively solve IK for a single finger in synergy space.

        Returns
        -------
        s_sol:
            Final synergy vector.
        q_sol:
            Final full configuration.
        converged:
            True if the error norm fell below tol within max_iters.
        """
        s = s_init.copy()
        q = synergy.decode(self.kin.model_full, q_full_ref, s)

        for it in range(self.max_iters):
            s, q, err = self.kin.finger_ik_step_s_palm(
                finger_name=finger_name,
                s=s,
                synergy=synergy,
                q_full_ref=q_full_ref,
                M_des_palm=M_des_palm,
                alpha=self.alpha,
                damping=self.damping,
                position_only=self.position_only,
            )
            logger.debug(
                "[HandIKSolver:s] iter=%d finger=%s err=%.3e",
                it,
                finger_name,
                err,
            )
            if err < self.tol:
                logger.info(
                    "[HandIKSolver:s] Converged in %d iterations (err=%.3e)",
                    it + 1,
                    err,
                )
                return s, q, True

        logger.warning(
            "[HandIKSolver:s] Did not converge in %d iterations (last err=%.3e)",
            self.max_iters,
            err,
        )
        return s, q, False


# ---------------------------------------------------------------------------
# Helpers to build a default synergy model and gesture space
# ---------------------------------------------------------------------------


def build_default_hand_synergy(
    kin: HandKinematics,
    base_q_full: np.ndarray | None = None,
) -> HandSynergyModel:
    """
    Construct a simple hand-crafted synergy model.

    The following synergies are used (if the corresponding fingers exist):
    - s[0]: common flexion for index, middle, ring, little (global grasp)
    - s[1]: thumb opposition (flex + move toward palm)
    - s[2]: index-only flexion
    """
    model = kin.model_full

    hand_joint_ids: list[int] = []
    finger_order = ["thumb", "index", "middle", "ring", "little"]
    offsets: Dict[str, int] = {}

    offset = 0
    for name in finger_order:
        if name in kin.fingers:
            jids = kin.fingers[name].joint_ids
            hand_joint_ids.extend(jids)
            offsets[name] = offset
            offset += len(jids)

    n_h = len(hand_joint_ids)
    r = 3  # number of synergies
    S = np.zeros((n_h, r), dtype=float)

    def add_finger_flexion(
        finger: str,
        col: int,
        scale: float = 1.0,
        w_tip: float = 0.5,
        w_mid: float = 1.0,
        w_base: float = 1.0,
    ) -> None:
        """Helper to fill S for flexion of a single finger."""
        if finger not in offsets:
            return
        base = offsets[finger]
        # local indices: 0,1,2,3 = J1 (distal), J2 (middle), J3 (base), J4 (abduction)
        S[base + 0, col] += w_tip * scale
        if base + 1 < n_h:
            S[base + 1, col] += w_mid * scale
        if base + 2 < n_h:
            S[base + 2, col] += w_base * scale
        # J4 (abduction) is intentionally left out.

    # s[0]: global grasp (four fingers)
    for finger in ["index", "middle", "ring", "little"]:
        add_finger_flexion(finger, col=0, scale=1.0)

    # s[2]: index-only flexion
    add_finger_flexion("index", col=2, scale=1.0)

    # s[1]: thumb opposition
    if "thumb" in offsets:
        base = offsets["thumb"]
        n_th = len(kin.fingers["thumb"].joint_ids)

        # Assume order [distal, ..., proximal, maybe rotation]
        if n_th >= 2:
            S[base + 1, 1] += 1.0  # main flexion
        S[base + 0, 1] += 1.0      # distal support flexion
        if n_th >= 3:
            S[base + 2, 1] += 1.0  # toward palm
        if n_th >= 4:
            S[base + 3, 1] += 1.0  # further abduction toward palm
        if n_th >= 5:
            S[base + 4, 1] += 0.6  # base rotation

    # base configuration: neutral pose (can be overridden)
    if base_q_full is None:
        q0_full = pin.neutral(model)
    else:
        q0_full = np.asarray(base_q_full, dtype=float).reshape(model.nq)
    q0_hand = np.empty(n_h, dtype=float)
    for k, jid in enumerate(hand_joint_ids):
        idx = model.idx_qs[jid]
        q0_hand[k] = q0_full[idx]

    synergy_names = ["grasp_all", "thumb_opposition", "index_only"]
    s_min = np.array([0.0, 0.0, 0.0], dtype=float)
    s_max = np.array([1.0, 1.0, 1.0], dtype=float)

    synergy_model = HandSynergyModel(
        hand_joint_ids=hand_joint_ids,
        q0_hand=q0_hand,
        S=S,
        synergy_names=synergy_names,
        s_min=s_min,
        s_max=s_max,
    )

    logger.info(
        "[build_default_hand_synergy] hand DOF=%d, dim_s=%d",
        synergy_model.n_h,
        synergy_model.dim_s,
    )

    return synergy_model


def build_default_gesture_space(synergy: HandSynergyModel) -> HandGestureSpace:
    """
    Build a default HandGestureSpace that exposes the synergy dimensions as
    gesture parameters.

    By default:
    - "grasp" controls synergy "grasp_all" with a slightly convex curve.
    - "thumb_opposition" controls synergy "thumb_opposition".
    - "index_iso" controls synergy "index_only".
    """
    idx = {name: i for i, name in enumerate(synergy.synergy_names)}

    params: Dict[str, SynergyParam] = {
        "grasp": SynergyParam(
            name="grasp",
            index=idx["grasp_all"],
            ui_min=0.0,
            ui_max=1.0,
            phys_min=0.0,
            phys_max=1.0,
            curve=lambda x: x**1.5,
        ),
        "thumb_opposition": SynergyParam(
            name="thumb_opposition",
            index=idx["thumb_opposition"],
            ui_min=0.0,
            ui_max=1.0,
            phys_min=0.0,
            phys_max=1.0,
        ),
        "index_iso": SynergyParam(
            name="index_iso",
            index=idx["index_only"],
            ui_min=0.0,
            ui_max=1.0,
            phys_min=0.0,
            phys_max=1.0,
        ),
    }

    gesture_space = HandGestureSpace(synergy_model=synergy, params=params)

    logger.info(
        "[build_default_gesture_space] gesture params=%s",
        list(params.keys()),
    )

    return gesture_space
#------------------------------------------------------------------------
# Hand grasp control
#------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Example usage (can be removed when importing as a library)
# ---------------------------------------------------------------------------


def _demo() -> None:
    """
    Minimal demo showing how to:

    - load the model,
    - build the synergy model and gesture space,
    - generate a hand posture from a high-level gesture,
    - run a simple index finger IK in joint space.
    """
    # NOTE: adapt these names/paths to your own setup.
    mjcf_path = "ur5e_shadowhand_scene_pin.xml"
    palm_frame_name = "rh_palm"

    kin = HandKinematics(
        mjcf_path=mjcf_path,
        palm_frame_name=palm_frame_name,
    )

    synergy = build_default_hand_synergy(kin)
    gesture_space = build_default_gesture_space(synergy)

    model = kin.model_full
    q_ref = pin.neutral(model)

    # 1) Use gesture space to generate a posture
    gesture = {"grasp": 0.7, "thumb_opposition": 0.5}
    q_hand_posture = gesture_space.decode_gesture(model, q_ref, gesture)

    kin.forward_kinematics(q_hand_posture)
    M_tip_index = kin.get_tip_pose("index", q_hand_posture)
    logger.info(
        "[demo] index fingertip position (world): %s",
        M_tip_index.translation,
    )

    # 2) Run a simple finger IK in joint space: move index fingertip 2 cm along palm x
    # Compute current tip pose in palm frame
    data = kin.data_full
    M_world_palm = data.oMf[kin.palm_frame_id]
    M_world_tip = data.oMf[kin.fingers["index"].tip_frame_id]
    M_palm_tip = M_world_palm.inverse() * M_world_tip

    offset = np.array([0.02, 0.0, 0.0])  # 2 cm along +x in palm frame
    M_palm_tip_des = pin.SE3(
        M_palm_tip.rotation.copy(),
        M_palm_tip.translation + offset,
    )

    ik_solver = HandIKSolver(kin, alpha=0.5, damping=1e-4, max_iters=30, tol=1e-4)
    q_sol, converged = ik_solver.solve_finger_ik_q_palm(
        finger_name="index",
        q_init=q_hand_posture,
        M_des_palm=M_palm_tip_des,
    )

    kin.forward_kinematics(q_sol)
    M_tip_index_final = kin.get_tip_pose("index", q_sol)
    logger.info(
        "[demo] IK converged=%s, index fingertip position (world) after IK: %s",
        converged,
        M_tip_index_final.translation,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    )
    _demo()
