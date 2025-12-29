from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np

from planning.grasp_planning import generate_grasps_from_obb, load_grasp_config, OBB
from utils.transform import SE3_from_T, SE3_to_xyzrpy, T_from_SE3, inv_T, inv_se3, mul_se3


@dataclass(slots=True)
class AABB:
    mn: np.ndarray  # (3,)
    mx: np.ndarray  # (3,)

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.mn + self.mx)

    @property
    def half(self) -> np.ndarray:
        return 0.5 * (self.mx - self.mn)


def _corners_local(he: np.ndarray) -> np.ndarray:
    he = he.reshape(3)
    s = np.array([-1, 1], float)
    corners = np.array([[i, j, k] for i in s for j in s for k in s], float)
    return corners * he[None, :]


def _build_body_children(model: mujoco.MjModel) -> list[list[int]]:
    children: list[list[int]] = [[] for _ in range(model.nbody)]
    for bid in range(1, model.nbody):
        pid = int(model.body_parentid[bid])
        if pid >= 0:
            children[pid].append(bid)
    return children


@dataclass(slots=True)
class GraspCandidateInfo:
    score: float
    note: str
    T_obj_palm: np.ndarray  # (4,4)
    T_palm_obj: np.ndarray  # (4,4)
    T_w_palm: np.ndarray  # (4,4)
    xyzrpy_w_palm: np.ndarray
    bbox_half: np.ndarray  # (3,)


@dataclass(slots=True)
class GraspInfo:
    body_name: str
    bbox_frame: str
    T_w_obj: np.ndarray  # (4,4)
    T_w_box: np.ndarray  # (4,4)
    candidates: list[GraspCandidateInfo]

    @property
    def best(self) -> GraspCandidateInfo | None:
        return self.candidates[0] if self.candidates else None


class Grasping:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._body_children = _build_body_children(model)

    def _collect_subtree_bodies(self, root_bid: int) -> set[int]:
        stack = [root_bid]
        subtree: set[int] = set()
        while stack:
            b = stack.pop()
            if b in subtree:
                continue
            subtree.add(b)
            stack.extend(self._body_children[b])
        return subtree

    def _geom_world_aabb(self, gid: int, exact_mesh: bool = False) -> AABB:
        m, d = self.model, self.data

        c = d.geom_xpos[gid].copy()
        R = d.geom_xmat[gid].reshape(3, 3).copy()
        gtype = int(m.geom_type[gid])
        size = m.geom_size[gid].copy()

        if gtype == mujoco.mjtGeom.mjGEOM_MESH and exact_mesh:
            mid = int(m.geom_dataid[gid])
            vadr = int(m.mesh_vertadr[mid])
            vnum = int(m.mesh_vertnum[mid])
            verts = m.mesh_vert[vadr : vadr + vnum].reshape(-1, 3)

            s = size.copy()
            if np.any(s != 0):
                verts = verts * s

            pts_w = (verts @ R.T) + c
            return AABB(pts_w.min(axis=0), pts_w.max(axis=0))

        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            he_local = size
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            he_local = np.array([size[0], size[0], size[0]], float)
        elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            r = float(size[0])
            h = float(size[1])
            he_local = np.array([r, r, h], float)
        elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            he_local = size
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            s = size.copy()
            if np.all(s == 0):
                s[:] = 0.01
            he_local = s
        else:
            r = float(size[0]) if size[0] > 0 else 0.01
            he_local = np.array([r, r, r], float)

        he_world = np.abs(R) @ he_local
        return AABB(c - he_world, c + he_world)

    def get_body_world_aabb(
        self,
        body_name: str,
        include_subtree: bool = True,
        exact_mesh: bool = False,
        geom_predicate=None,
    ) -> AABB:
        m = self.model

        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found.")

        body_ids = self._collect_subtree_bodies(bid) if include_subtree else {bid}

        mn = np.array([np.inf, np.inf, np.inf], float)
        mx = np.array([-np.inf, -np.inf, -np.inf], float)

        for gid in range(m.ngeom):
            gbid = int(m.geom_bodyid[gid])
            if gbid not in body_ids:
                continue
            if geom_predicate is not None and not geom_predicate(gid, m):
                continue

            aabb = self._geom_world_aabb(gid, exact_mesh=exact_mesh)
            mn = np.minimum(mn, aabb.mn)
            mx = np.maximum(mx, aabb.mx)

        if not np.all(np.isfinite(mn)):
            raise RuntimeError(
                f"No geoms found under body '{body_name}' (include_subtree={include_subtree})."
            )

        return AABB(mn, mx)

    def get_body_aabb_in_body_frame(
        self,
        body_name: str,
        include_subtree: bool = True,
        exact_mesh: bool = False,
        geom_predicate=None,
    ) -> AABB:
        m, d = self.model, self.data

        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found.")

        p_w_obj = d.xpos[bid].copy()
        R_w_obj = d.xmat[bid].reshape(3, 3).copy()

        body_ids = self._collect_subtree_bodies(bid) if include_subtree else {bid}

        mn = np.array([np.inf, np.inf, np.inf], float)
        mx = np.array([-np.inf, -np.inf, -np.inf], float)

        for gid in range(m.ngeom):
            if int(m.geom_bodyid[gid]) not in body_ids:
                continue
            if geom_predicate is not None and not geom_predicate(gid, m):
                continue

            gtype = int(m.geom_type[gid])
            size = m.geom_size[gid].copy()

            c_w = d.geom_xpos[gid].copy()
            R_w = d.geom_xmat[gid].reshape(3, 3).copy()

            if gtype == mujoco.mjtGeom.mjGEOM_MESH and exact_mesh:
                mid = int(m.geom_dataid[gid])
                vadr = int(m.mesh_vertadr[mid])
                vnum = int(m.mesh_vertnum[mid])
                verts = m.mesh_vert[vadr : vadr + vnum].reshape(-1, 3)

                s = size.copy()
                if np.any(s != 0):
                    verts = verts * s

                pts_w = verts @ R_w.T + c_w
                pts_obj = (pts_w - p_w_obj) @ R_w_obj
                mn = np.minimum(mn, pts_obj.min(axis=0))
                mx = np.maximum(mx, pts_obj.max(axis=0))
                continue

            if gtype == mujoco.mjtGeom.mjGEOM_BOX:
                he = size
            elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
                he = np.array([size[0]] * 3, float)
            elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
                r, h = float(size[0]), float(size[1])
                he = np.array([r, r, h], float)
            elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r, h = float(size[0]), float(size[1])
                he = np.array([r, r, h + r], float)
            elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                he = size
            else:
                r = float(size[0]) if size[0] > 0 else 0.01
                he = np.array([r, r, r], float)

            corners_w = _corners_local(he) @ R_w.T + c_w
            corners_obj = (corners_w - p_w_obj) @ R_w_obj

            mn = np.minimum(mn, corners_obj.min(axis=0))
            mx = np.maximum(mx, corners_obj.max(axis=0))

        if not np.all(np.isfinite(mn)):
            raise RuntimeError(
                f"No geoms under body '{body_name}' to build bbox."
            )

        return AABB(mn, mx)

    def propose_grasps_from_bbox(
        self,
        body_name: str,
        bbox_frame: str = "body",
        include_subtree: bool = True,
        exact_mesh: bool = False,
        topk: int = 10,
        prefer_world_up: bool = True,
        world_up_cos_thresh: float = 0.7,
        require_above: bool = True,
        world_up_weight: float = 2.0,
        grasp_config_path: str = "config/grasp_config.toml",
    ) -> GraspInfo:
        config_bundle = load_grasp_config(grasp_config_path)
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)

        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Body '{body_name}' not found.")

        p_w_obj = d.xpos[bid].copy()
        R_w_obj = d.xmat[bid].reshape(3, 3).copy()

        T_w_obj = np.eye(4)
        T_w_obj[:3, :3] = R_w_obj
        T_w_obj[:3, 3] = p_w_obj
        T_obj_w = inv_T(T_w_obj)

        if bbox_frame == "world":
            aabb_w = self.get_body_world_aabb(
                body_name, include_subtree=include_subtree, exact_mesh=exact_mesh
            )
            c_w = aabb_w.center
            half = aabb_w.half

            T_w_box = np.eye(4)
            T_w_box[:3, 3] = c_w
            T_obj_box = T_obj_w @ T_w_box
        elif bbox_frame == "body":
            aabb_obj = self.get_body_aabb_in_body_frame(
                body_name,
                include_subtree=include_subtree,
                exact_mesh=exact_mesh,
            )
            c_obj = aabb_obj.center
            half = aabb_obj.half

            T_obj_box = np.eye(4)
            T_obj_box[:3, 3] = c_obj

            T_w_box = T_w_obj @ T_obj_box
            c_w = T_w_box[:3, 3].copy()
        else:
            raise ValueError("bbox_frame must be 'world' or 'body'.")

        obb = OBB(center=np.zeros(3), R=np.eye(3), half=half)
        cands = generate_grasps_from_obb(
            obb, palm_axes=config_bundle.palm_axes, cfg=config_bundle.grasp
        )

        g = np.asarray(m.opt.gravity, dtype=float).reshape(3)
        gn = float(np.linalg.norm(g))
        if gn > 1e-9:
            down_w = g / gn
            up_w = -down_w
        else:
            up_w = np.array([0.0, 0.0, 1.0], float)
            down_w = -up_w

        a_p = np.asarray(config_bundle.palm_axes.approach, dtype=float).reshape(3)
        a_p /= (np.linalg.norm(a_p) + 1e-12)

        X_obj_box = SE3_from_T(T_obj_box)
        X_w_box = SE3_from_T(T_w_box)

        scored_infos: list[tuple[float, GraspCandidateInfo]] = []

        for cand in cands:
            X_box_palm = cand.T_box_palm

            X_obj_palm = mul_se3(X_obj_box, X_box_palm)
            X_palm_obj = inv_se3(X_obj_palm)
            X_w_palm = mul_se3(X_w_box, X_box_palm)

            score = float(cand.score)

            if prefer_world_up:
                R_w_palm = X_w_palm.R
                p_w_palm = X_w_palm.t

                a_w = R_w_palm @ a_p
                align = float(a_w @ down_w)

                if align < world_up_cos_thresh:
                    continue

                if require_above:
                    above = float((p_w_palm - c_w) @ up_w)
                    if above < 0.0:
                        continue

                score = score + world_up_weight * align

            info = GraspCandidateInfo(
                score=score,
                note=cand.note,
                T_obj_palm=T_from_SE3(X_obj_palm),
                T_palm_obj=T_from_SE3(X_palm_obj),
                T_w_palm=T_from_SE3(X_w_palm),
                xyzrpy_w_palm=SE3_to_xyzrpy(X_w_palm),
                bbox_half=half.copy(),
            )
            scored_infos.append((score, info))

        scored_infos.sort(key=lambda x: x[0], reverse=True)
        out = [info for _, info in scored_infos[:topk]]

        return GraspInfo(
            body_name=body_name,
            bbox_frame=bbox_frame,
            T_w_obj=T_w_obj,
            T_w_box=T_w_box,
            candidates=out,
        )
