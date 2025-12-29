# pinocchio_ik.py
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin

urdf_path = Path("assets/ur5e_shadow/ur5e_shadow.urdf")
mjcf_path = Path("ur5e_shadowhand_scene_pin.xml")
# model_full: pin.Model = pin.buildModelFromMJCF(str(mjcf_path))
model_full: pin.Model = pin.buildModelFromUrdf(str(urdf_path))

arm_joint_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# -----------------MJCF mode---------------
# wrist_joint_names = [
#     "rh_WRJ1",
#     "rh_WRJ2",
# ]

#-------------------URDF mode------------------
wrist_joint_names = [
    "wrist_joint",
    "forearm_joint",
]

keep_joint_names = set(arm_joint_names + wrist_joint_names)

lock_joint_ids = []
for jid, name in enumerate(model_full.names):
    if jid == 0:
        continue  
    if name not in keep_joint_names:
        lock_joint_ids.append(jid)

q_ref_full = pin.neutral(model_full)
model_8dof: pin.Model = pin.buildReducedModel(model_full, lock_joint_ids, q_ref_full)
data_8dof: pin.Data = model_8dof.createData()

# ee_frame_name = "rh_palm" 
ee_frame_name = "palm" 
ee_fid = model_8dof.getFrameId(ee_frame_name)

def solve_ik_se3(
    target: pin.SE3,
    q_init: Optional[np.ndarray] = None,
    max_iters: int = 80,
    tol: float = 1e-4,
    step: float = 0.2,
    damp: float = 1e-4,
    pos_only: bool = False,
) -> Tuple[np.ndarray, bool]:
    """
    simple damp ik, for testing pipeline

    参数
    ----
    target : pin.SE3 (base_frame)
    q_init : np.ndarray, use pin.neutral(model) when None 
    max_iters : int
    tol : float  Confirm when e is less than tol
    step : float
    damp : float
        阻尼系数 λ，用于伪逆 (J J^T + λ^2 I)^-1
    pos_only : bool

    return:
    ----
    q : np.ndarray (*8)
    success : bool
        是否在 max_iters 内收敛
    """
    model = model_8dof
    data = data_8dof
    print(model.joints)

    if q_init is None:
        q = pin.neutral(model).copy()
    else:
        q = np.array(q_init, dtype=float).copy()

    for it in range(max_iters):
        # 前向运动学 & frame 位姿
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M = data.oMf[ee_fid]

        # 误差 SE3 -> se(3)
        dM = M.inverse() * target
        err6 = pin.log(dM).vector  # [ω, v] 或 [vx, vy, vz, wx, wy, wz]，看你版本

        if pos_only:
            e = err6[3:] if err6.shape[0] == 6 else err6[:3]  # 你可以根据 log 的定义调整
        else:
            e = err6

        if np.linalg.norm(e) < tol:
            return q, True

        # Jacobian
        J6 = pin.computeFrameJacobian(
            model, data, q, ee_fid, pin.ReferenceFrame.LOCAL
        )
        # Jlog = pin.Jlog6(dM)
        # J    = Jlog @ J6
        if pos_only:
            # 只用位置的 3 行（根据 log 约定选 3 行）
            J = J6[3:, :] if err6.shape[0] == 6 else J6[:3, :]
        else:
            J = J6

        # 阻尼最小二乘 J^T (J J^T + λ^2 I)^-1 e
        m = J.shape[0]
        H = J @ J.T + (damp ** 2) * np.eye(m)
        dq = J.T @ np.linalg.solve(H, e)

        # 更新关节角
        q = pin.integrate(model, q, step * dq)

    # 没收敛也返回最后的 q
    return q, False



def ik_from_xyz_rpy(
    xyz: np.ndarray,
    rpy: np.ndarray,
    q_init: Optional[np.ndarray] = None,
    **ik_kwargs,
) -> Tuple[np.ndarray, bool]:
    R = pin.rpy.rpyToMatrix(*rpy)
    target = pin.SE3(R, np.asarray(xyz, dtype=float))
    return solve_ik_se3(target, q_init=q_init, **ik_kwargs)


if __name__ == "__main__":
    # 小测试：把手掌送到某个位置 + 姿态
    xyz = np.array([0.5, 0.0, 0.3])
    rpy = np.array([0.0, np.pi / 2, 0.0])

    q_sol, ok = ik_from_xyz_rpy(xyz, rpy, pos_only=False, max_iters=80)
    print("success:", ok)
    print("q_sol:", q_sol)
