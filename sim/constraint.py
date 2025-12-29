import mujoco
import numpy as np
from abc import ABC, abstractmethod

class Constraint(ABC):
    """Abstract base class for a constraint."""

    @abstractmethod
    def valid_config(self, q: np.ndarray) -> bool:
        """Check if a configuration obeys a constraint.

        Args:
            q: The configuration.

        Returns:
            True if `q` obeys the constraint. False otherwise.
        """
        pass

    @abstractmethod
    def apply(self, q_old: np.ndarray, q: np.ndarray) -> np.ndarray | None:
        """Apply a constraint to a configuration.

        Args:
            q_old: An older configuration w.r.t `q`.
            q: The configuration to apply a constraint to.

        Returns:
            A configuration derived from `q` that obeys the constraint, or None if
            deriving a configuration that obeys the constraint is not possible.
        """
        pass

class CollisionRuleset:
    def __init__(self, model: mujoco.MjModel,
                 allowed_collision_bodies: list[tuple[str, str]] = []) -> None:
        self.model = model

        # 允许碰撞的 body 对（白名单）；为空则表示“任何碰撞都不允许”
        self.allowed_set: set[tuple[int, int]] | None = None
        if allowed_collision_bodies:
            s = set()
            for a, b in allowed_collision_bodies:
                ida = int(self.model.body(a).id)
                idb = int(self.model.body(b).id)
                s.add((ida, idb) if ida < idb else (idb, ida))
            self.allowed_set = s

    def allowed(self, body1: int, body2: int) -> bool:
        if self.allowed_set is None:
            return False
        a, b = (body1, body2) if body1 < body2 else (body2, body1)
        return (a, b) in self.allowed_set

    def obeys_ruleset(self, collision_geometries: np.ndarray) -> bool:
        if collision_geometries.ndim != 2 or collision_geometries.shape[1] != 2:
            raise ValueError("`collision_geometries` must be a nx2 matrix.")

        if collision_geometries.shape[0] == 0:
            return True
        elif self.allowed_set is None:
            return False

        collision_bodies = np.sort(
            self.model.geom_bodyid[collision_geometries], axis=1
        )
        matches = (collision_bodies[:, None, :] == self.allowed_set).all(axis=2)
        return np.all(matches.any(axis=1))

    
class CollisionConstraint(Constraint):
    def __init__(
        self,
        model: mujoco.MjModel,
        robot_body_ids: np.ndarray,
        robot_qpos_adr: np.ndarray,      # len = robot_dof (例如 30)
        qpos_template: np.ndarray,       # len = model.nq（世界快照）
        allowed_collision_bodies: list[tuple[str,str]] = [],
        dist_threshold: float = 0.0,
        *,
        verbose: bool = False,
        max_report: int = 10,            # 每次最多打印多少条违规 contact
        report_every: int = 200,         # 每多少次 valid_config 打印一次（防刷屏）
    ):
        self.model = model
        self.data  = mujoco.MjData(model)

        self.robot_qpos_adr = np.asarray(robot_qpos_adr, dtype=int)
        self.qpos_template  = np.asarray(qpos_template, dtype=float).copy()

        self.robot_body_ids = set(int(x) for x in np.asarray(robot_body_ids, dtype=int))
        self.robot_geom_ids = set(np.nonzero(np.isin(self.model.geom_bodyid, list(self.robot_body_ids)))[0].tolist())

        self.cr = CollisionRuleset(model, allowed_collision_bodies)
        self.dist_threshold = float(dist_threshold)

        self.verbose = bool(verbose)
        self.max_report = int(max_report)
        self.report_every = max(1, int(report_every))
        self._call_count = 0

    def update_template(self, qpos_template: np.ndarray):
        self.qpos_template[:] = np.asarray(qpos_template, dtype=float)

    def _name(self, objtype, objid: int) -> str:
        n = mujoco.mj_id2name(self.model, objtype, int(objid))
        return n if n is not None else f"<id={int(objid)}>"

    def valid_config(self, q: np.ndarray) -> bool:
        self._call_count += 1
        q = np.asarray(q, dtype=float)

        # 1) 写入世界快照 + 覆盖机器人关节
        self.data.qpos[:] = self.qpos_template
        self.data.qpos[self.robot_qpos_adr] = q

        # 2) 碰撞（位姿更新用 fwdPosition 更稳）
        mujoco.mj_fwdPosition(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

        ncon = int(self.data.ncon)
        if ncon == 0:
            return True

        # 3) 找“违规 contact”（dist<=threshold 且不在允许列表）
        violations = []
        for k in range(ncon):
            con = self.data.contact[k]
            dist = float(con.dist)
            if dist > self.dist_threshold:
                continue

            g1, g2 = int(con.geom1), int(con.geom2)

            # ✅ 关键：不涉及机器人就跳过（环境-环境碰撞不影响规划）
            if (g1 not in self.robot_geom_ids) and (g2 not in self.robot_geom_ids):
                continue

            b1 = int(self.model.geom_bodyid[g1])
            b2 = int(self.model.geom_bodyid[g2])

            if not self.cr.allowed(b1, b2):
                violations.append((k, g1, g2, b1, b2, dist))

        if not violations:
            return True

        # 4) verbose 打印（限频）
        if self.verbose and (self._call_count % self.report_every == 0):
            print(f"[CollisionConstraint] collision detected: {len(violations)} violating contacts "
                  f"(showing up to {self.max_report}), dist_threshold={self.dist_threshold}")
            for (k, g1, g2, b1, b2, dist) in violations[: self.max_report]:
                geom1_name = self._name(mujoco.mjtObj.mjOBJ_GEOM, g1)
                geom2_name = self._name(mujoco.mjtObj.mjOBJ_GEOM, g2)
                body1_name = self._name(mujoco.mjtObj.mjOBJ_BODY, b1)
                body2_name = self._name(mujoco.mjtObj.mjOBJ_BODY, b2)
                print(f"  k={k:3d}  geom{g1}({geom1_name}) body{b1}({body1_name})"
                      f"  <->  geom{g2}({geom2_name}) body{b2}({body2_name})  dist={dist:.6f}")

        return False

    def apply(self, q_old, q):
        return q if self.valid_config(q) else None
    
