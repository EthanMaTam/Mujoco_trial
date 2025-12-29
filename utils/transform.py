from __future__ import annotations
from dataclasses import dataclass, fields
import numpy as np

@dataclass(frozen=True)
class SE3:
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3,:3] = self.R
        T[:3,3]  = self.t
        return T

def inv_se3(T: SE3) -> SE3:
    Rt = T.R.T
    return SE3(R=Rt, t=-(Rt @ T.t))

def mul_se3(A: SE3, B: SE3) -> SE3:
    return SE3(R=A.R @ B.R, t=A.R @ B.t + A.t)

def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    p = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ p
    return Ti

def SE3_from_T(T: np.ndarray) -> SE3:
    return SE3(R=T[:3,:3].copy(), t=T[:3,3].copy())

def T_from_SE3(X: SE3) -> np.ndarray:
    T = np.eye(4)
    T[:3,:3] = X.R
    T[:3,3]  = X.t
    return T

def mat_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
    """Return [roll, pitch, yaw] with R = Rz(yaw) Ry(pitch) Rx(roll)."""
    R = np.asarray(R, float).reshape(3, 3)
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-9

    if not singular:
        roll  = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = np.arctan2(R[1, 0], R[0, 0])
    else:
        # gimbal lock: yaw set to 0
        roll  = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw   = 0.0

    return np.array([roll, pitch, yaw], dtype=float)

def T_to_xyzrpy(T: np.ndarray) -> np.ndarray:
    """Return (6,) [x,y,z, roll,pitch,yaw] in radians."""
    T = np.asarray(T, float).reshape(4, 4)
    xyz = T[:3, 3]
    rpy = mat_to_rpy_zyx(T[:3, :3])
    return np.concatenate([xyz, rpy], axis=0)

def SE3_to_xyzrpy(X: SE3) -> np.ndarray:
    """Return (6,) [x,y,z, roll,pitch,yaw] (radians)."""
    xyz = np.asarray(X.t, float).reshape(3)
    rpy = mat_to_rpy_zyx(np.asarray(X.R, float).reshape(3, 3))
    return np.concatenate([xyz, rpy], axis=0)