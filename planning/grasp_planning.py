from __future__ import annotations
from dataclasses import dataclass, fields
import numpy as np
from pathlib import Path
import tomllib
from utils.transform import SE3, inv_se3, mul_se3

@dataclass(frozen=True)
class OBB:
    # box frame: 原点在中心，三轴是盒子的主轴（右手系）
    center: np.ndarray      # (3,)
    R: np.ndarray           # (3,3) box->world 或 world->box 二选一，下面我用 box->world
    half: np.ndarray        # (3,) half extents along box x,y,z

@dataclass(frozen=True)
class PalmAxes:
    approach: np.ndarray  # (3,)
    closing: np.ndarray   # (3,)

@dataclass(frozen=True)
class GraspConfig:
    standoff: float = 0.04
    surface_clearance: float = 0.0
    inplane_yaw_samples: int = 4
    strategy: str = "faces"

@dataclass(frozen=True)
class ScoringConfig:
    prefer_world_up: bool = True

@dataclass(frozen=True)
class GraspConfigBundle:
    palm_axes: PalmAxes
    grasp: GraspConfig
    scoring: ScoringConfig
    schema_version: int = 1

@dataclass(frozen=True)
class GraspCandidate:
    T_box_palm: SE3
    score: float
    note: str

def _normalize(v):
    v = np.asarray(v, float).reshape(3)
    n = np.linalg.norm(v)
    return v / max(n, 1e-12)

def _orthonormal_from_az(approach_world: np.ndarray, x_hint_world: np.ndarray) -> np.ndarray:
    # 构造一个右手系：z=approach, x 尽量贴 x_hint（再正交化）
    z = _normalize(approach_world)
    x = x_hint_world - (x_hint_world @ z) * z
    x = _normalize(x)
    y = np.cross(z, x)
    y = _normalize(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])

def _as_vec3(x, key: str) -> np.ndarray:
    v = np.asarray(x, dtype=float).reshape(3)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{key} contains NaN/Inf")
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"{key} is near-zero vector")
    return v / n

def _validate_axes(palm: PalmAxes) -> PalmAxes:
    a = _as_vec3(palm.approach, "palm_axes.approach")
    c = _as_vec3(palm.closing,  "palm_axes.closing")
    # closing 不能和 approach 平行，否则无法构造面内 yaw
    if abs(float(a @ c)) > 0.95:
        raise ValueError("palm_axes.closing is too parallel to palm_axes.approach")
    return PalmAxes(approach=a, closing=c)

def _filter_kwargs(raw: dict, cls) -> dict:
    allowed = {f.name for f in fields(cls)}
    return {k: raw[k] for k in raw.keys() if k in allowed}

def load_grasp_config(path: str | Path) -> GraspConfigBundle:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    ver = int(raw.get("schema_version", 1))
    if ver != 1:
        raise ValueError(f"Unsupported schema_version={ver}")

    palm_raw    = raw.get("palm_axes", {})
    grasp_raw   = raw.get("grasp", {})
    scoring_raw = raw.get("scoring", {})

    # palm_axes：我建议仍然显式处理，因为需要归一化+正交检查
    palm = PalmAxes(
        approach=_as_vec3(palm_raw.get("approach", [0,0,1]), "palm_axes.approach"),
        closing=_as_vec3(palm_raw.get("closing",  [1,0,0]), "palm_axes.closing"),
    )
    palm = _validate_axes(palm)

    # grasp：仅覆盖配置里出现的字段，其余用 dataclass 默认
    gkw = _filter_kwargs(grasp_raw, GraspConfig)
    grasp = GraspConfig(**gkw)

    # 基本校验
    if grasp.inplane_yaw_samples <= 0:
        raise ValueError("grasp.inplane_yaw_samples must be > 0")
    if grasp.standoff < 0:
        raise ValueError("grasp.standoff must be >= 0")

    # scoring：同理
    skw = _filter_kwargs(scoring_raw, ScoringConfig)
    scoring = ScoringConfig(**skw)

    return GraspConfigBundle(palm_axes=palm, grasp=grasp, scoring=scoring, schema_version=ver)

def generate_grasps_from_obb(
    obb: OBB,
    palm_axes: PalmAxes,
    cfg: GraspConfig,
) -> list[GraspCandidate]:
    # box-frame 中的 3 个轴
    ex, ey, ez = np.eye(3)
    face_normals = [
        (+ex, " +X"), (-ex, " -X"),
        (+ey, " +Y"), (-ey, " -Y"),
        (+ez, " +Z"), (-ez, " -Z"),
    ]

    # palm 内部轴（单位化）
    a_p = _normalize(palm_axes.approach)
    c_p = _normalize(palm_axes.closing)

    cands: list[GraspCandidate] = []
    for n_box, tag in face_normals:
        # 让 palm 的 approach 轴指向“从外到表面”的方向：即 -n（靠近面）
        approach_box = -n_box

        # 选一个面内方向作为“闭合轴目标”（closing 的投影）
        # 简单起步：用与法向不平行的 box 轴作为 x_hint
        if abs(approach_box @ ex) < 0.9:
            x_hint = ex
        else:
            x_hint = ey

        # 在 box-frame 里构造 grasp frame（先得到 R_box_grasp）
        R_box_grasp0 = _orthonormal_from_az(approach_box, x_hint)

        half_on_normal = float((obb.half * np.abs(n_box)).sum())  # 对 ±ex/±ey/±ez 就是 half_x/half_y/half_z
        p_face = obb.center + n_box * (half_on_normal + cfg.surface_clearance)
        p_palm = p_face - approach_box * cfg.standoff


        # 面内 yaw 旋转
        for k in range(cfg.inplane_yaw_samples):
            yaw = (2*np.pi) * k / cfg.inplane_yaw_samples
            R_yaw = np.array([
                [ np.cos(yaw), -np.sin(yaw), 0],
                [ np.sin(yaw),  np.cos(yaw), 0],
                [          0,            0, 1],
            ], float)
            R_box_grasp = R_box_grasp0 @ R_yaw

            # 现在要把 grasp-frame 映射到 palm-frame：让 palm_axes 对齐 grasp axes
            # 做法：构造一个 R_palm_align，使 palm 的 a_p -> grasp z，c_p -> grasp x
            # 先在 palm 内部构造 “palm 语义基”
            z_p = a_p
            x_p = c_p - (c_p @ z_p) * z_p
            x_p = _normalize(x_p)
            y_p = _normalize(np.cross(z_p, x_p))
            x_p = _normalize(np.cross(y_p, z_p))
            R_p_sem = np.column_stack([x_p, y_p, z_p])  # palm semantic basis

            # grasp basis：x=closing, z=approach
            # 我们这里让 grasp z 就是 approach_box，x 是 R_box_grasp[:,0]
            R_g = R_box_grasp

            # palm 在 box 下的旋转：R_box_palm = R_box_grasp * inv(R_p_sem)
            R_box_palm = R_g @ R_p_sem.T

            T_box_palm = SE3(R=R_box_palm, t=p_palm)

            score = 1.0  # 先不打分，或者按“抓最短边/最长边”再扩展
            cands.append(GraspCandidate(T_box_palm=T_box_palm, score=score, note=f"face{tag}, yaw{k}"))
            dist = (T_box_palm.t - obb.center) @ n_box
            # assert dist > half_on_normal, "palm 跑到盒子里面/表面内侧了"
    return cands

