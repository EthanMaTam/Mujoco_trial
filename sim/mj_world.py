# sim/mj_world.py
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import mujoco
import mujoco.viewer
import numpy as np

from control.main_trajectory_gen import JointTrajectory
from sim.execution import TrajectoryExecutor
from sim.grasping import AABB, GraspCandidateInfo, GraspInfo, Grasping
from sim.motion import MotionPlanner

'''
    Fuck mujoco, jid -> qpos -> jpos
'''

MAIN_VMAX_DEG = np.array([60, 60, 60, 90, 90, 90, 120, 120], dtype=float)
MAIN_AMAX_DEG = np.array([120, 120, 120, 180, 180, 180, 240, 240], dtype=float)
MAIN_VMAX = np.deg2rad(MAIN_VMAX_DEG)
MAIN_AMAX = np.deg2rad(MAIN_AMAX_DEG)

FINGER_VMAX = float(1.0) * np.ones(22, dtype=float)
FINGER_AMAX = float(2.0) * np.ones(22, dtype=float)

FULL_VMAX = np.append(MAIN_VMAX, FINGER_VMAX)
FULL_AMAX = np.append(MAIN_AMAX, FINGER_AMAX)

DEFAULT_QPOS_PATH = Path(__file__).resolve().parent.parent / "config" / "default_qpos.npy"

MAIN_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "rh_WRJ1",
    "rh_WRJ2",
]

DEFAULT_MAIN_CONF = [
    0.0,
    -1.57,
    1.57,
    0.0,
    1.57,
    1.57,
    0.0,
    0.0,
]

_DEFAULT_MAIN_CONF_DICT = dict(zip(MAIN_JOINT_NAMES, DEFAULT_MAIN_CONF))


@dataclass
class ActuatorInfo:
    id: int
    name: str
    trn_type: int
    trn_type_name: str
    trn_ids: tuple[int, int]
    joint_name: str | None
    qpos_adr: int | None
    qvel_adr: int | None
    ctrl_range: np.ndarray
    gainprm: np.ndarray
    biasprm: np.ndarray
    dynprm: np.ndarray
    gear: np.ndarray

@dataclass
class BodyGroup:
    ids: np.ndarray     
    names: list[str]

    @property
    def map(self):
        return dict(zip(self.ids, self.names))

    def from_name(self, name: str) -> int:
        '''
            From body_name to body_id
        '''
        assert name in self.names, "Invalid body name. Not included."
        temp_dic = dict(zip(self.names, self.ids))
        return temp_dic.get(name)
    
    def from_id(self, id: int) -> str:
        '''
            From body_id to name
        '''
        assert id in self.ids, "Invalid body id."
        temp_dic = dict(zip(self.ids, self.names))
        return temp_dic.get(id)

class MjWorld:
    def __init__(self, xml_path: str, use_viewer: bool = True):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self._viewer_ctx = None
        self.viewer = None
        self.use_viewer = use_viewer
        (
            self.main_joint_names,
            self.main_joint_ids,
            self.main_qpos_adr,
            self.main_qvel_adr,
        ) = self._init_main_joints(MAIN_JOINT_NAMES)
        (
            self.finger_joint_ids,
            self.finger_joint_names,
            self.finger_qpos_adr,
            self.finger_qvel_adr,
        ) = self._init_finger_joints_from_prefix(
            hand_prefix="rh_",
            exclude_names=MAIN_JOINT_NAMES,
        )
        self.body_groups = self._init_body_groups_from_qpos()

        self._body_children = [[] for _ in range(self.model.nbody)]
        for bid in range(1, self.model.nbody):
            pid = int(self.model.body_parentid[bid])
            if pid >= 0:
                self._body_children[pid].append(bid)

        self.grasping = Grasping(self.model, self.data)
        self.motion = MotionPlanner(self)
        self.executor = TrajectoryExecutor(self)

    def __enter__(self):
        if self.use_viewer:
            self._viewer_ctx = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer = self._viewer_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._viewer_ctx is not None:
            self._viewer_ctx.__exit__(exc_type, exc, tb)

    def get_actuator_infos(self) -> list[ActuatorInfo]:
        """Collect actuator properties to help map trajectories to every actuator."""
        infos: list[ActuatorInfo] = []
        m = self.model

        for aid in range(m.nu):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            name = name or f"actuator_{aid}"

            trn_type = int(m.actuator_trntype[aid])
            try:
                trn_type_name = mujoco.mjtTrn(trn_type).name
            except ValueError:
                trn_type_name = str(trn_type)
            trn_ids = (int(m.actuator_trnid[aid, 0]), int(m.actuator_trnid[aid, 1]))

            joint_name = None
            qpos_adr = None
            qvel_adr = None
            if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT):
                jid = trn_ids[0]
                if jid >= 0:
                    joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
                    qpos_adr = int(m.jnt_qposadr[jid])
                    qvel_adr = int(m.jnt_dofadr[jid])

            infos.append(
                ActuatorInfo(
                    id=aid,
                    name=name,
                    trn_type=trn_type,
                    trn_type_name=trn_type_name,
                    trn_ids=trn_ids,
                    joint_name=joint_name,
                    qpos_adr=qpos_adr,
                    qvel_adr=qvel_adr,
                    ctrl_range=m.actuator_ctrlrange[aid].copy(),
                    gainprm=m.actuator_gainprm[aid].copy(),
                    biasprm=m.actuator_biasprm[aid].copy(),
                    dynprm=m.actuator_dynprm[aid].copy(),
                    gear=m.actuator_gear[aid].copy(),
                )
            )

        return infos

    def print_actuator_infos(self) -> list[ActuatorInfo]:
        infos = self.get_actuator_infos()

        print("=== Actuator summary ===")
        for info in infos:
            joint_field = info.joint_name if info.joint_name is not None else "-"
            qpos_field = info.qpos_adr if info.qpos_adr is not None else "-"
            qvel_field = info.qvel_adr if info.qvel_adr is not None else "-"
            print(
                f"[{info.id:2d}] act={info.name:22s} "
                f"trn={info.trn_type_name:10s} trnid={info.trn_ids} "
                f"joint={joint_field:20s} qpos={qpos_field} qvel={qvel_field} "
                f"ctrlrange={info.ctrl_range} gainprm={info.gainprm} "
                f"biasprm={info.biasprm} dynprm={info.dynprm} gear={info.gear}"
            )

        return infos
    
    def _init_main_joints(self, joint_names: list[str]):
        jids = []
        qpos_ids = []
        qvel_ids = []
        found_names = []

        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint name '{name}' not found in model.")
            jids.append(jid)
            found_names.append(name)

            # 对 hinge / slide 关节来说，都是 1 DoF
            qpos_adr = self.model.jnt_qposadr[jid]
            dof_adr  = self.model.jnt_dofadr[jid]
            qpos_ids.append(qpos_adr)
            qvel_ids.append(dof_adr)

        return (
            np.asarray(found_names, dtype=str),
            np.asarray(jids, dtype=int),
            np.asarray(qpos_ids, dtype=int),
            np.asarray(qvel_ids, dtype=int),
        )


    def _init_finger_joints_from_prefix(
        self,
        hand_prefix: str = "rh_",
        exclude_names: list[str] | None = None,
    ):
        if exclude_names is None:
            exclude_names = []
        
        finger_jids = []
        finger_names = []
        finger_qpos = []
        finger_qvel = []

        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name is None:
                continue

            # 过滤掉主关节 / 非手的关节
            if name in exclude_names:
                continue
            if not name.startswith(hand_prefix):
                continue

            # 只要 hinge 关节（Shadow 手指基本都是 hinge）
            if self.model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue

            finger_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

            qpos_adr = self.model.jnt_qposadr[jid]
            dof_adr  = self.model.jnt_dofadr[jid]

            finger_jids.append(finger_jid)
            finger_names.append(name)
            finger_qpos.append(qpos_adr)
            finger_qvel.append(dof_adr)

        if not finger_names:
            print("[MjWorld] WARNING: 没有通过前缀推断出任何手指关节，请检查 hand_prefix。")

        # 按 qpos 下标排序，和状态向量顺序对齐
        order = np.argsort(finger_qpos)
        finger_names = [finger_names[i] for i in order]
        finger_qpos  = np.asarray(finger_qpos, dtype=int)[order]
        finger_qvel  = np.asarray(finger_qvel, dtype=int)[order]

        # print("[MjWorld] inferred finger joints:")
        # for n, qp in zip(finger_names, finger_qpos):
        #     print(f"  {n:<15} qpos[{qp}]")

        return (np.asarray(finger_jids, dtype=int), 
                np.asarray(finger_names, dtype=str), 
                finger_qpos, 
                finger_qvel)
    
    def _init_body_groups_from_qpos(self) -> dict[str, BodyGroup]:
        m = self.model

        main_qpos_set = set(int(i) for i in self.main_qpos_adr)
        hand_qpos_set = set(int(i) for i in self.finger_qpos_adr)

        # ---- 0) 做个 sanity check，防止 parent 索引异常 ----
        parent = m.body_parentid
        # parent 应该都在 [-1, nbody-1] 之间
        assert np.all((parent >= -1) & (parent < m.nbody)), \
            f"Invalid body_parentid detected: {parent}"

        # ---- 1) joint -> body 粗分 arm / hand ----
        arm_body_ids  = set()
        hand_body_ids = set()

        for j in range(m.njnt):
            qadr    = int(m.jnt_qposadr[j])
            body_id = int(m.jnt_bodyid[j])

            if qadr in main_qpos_set:
                arm_body_ids.add(body_id)
            if qadr in hand_qpos_set:
                hand_body_ids.add(body_id)

        # ---- 2) 预先构建 parent -> children 邻接表，方便做 subtree ----
        children: list[list[int]] = [[] for _ in range(m.nbody)]
        for bid in range(1, m.nbody):  # 0 通常是 world / root
            p = int(parent[bid])
            if p >= 0:
                children[p].append(bid)

        def add_subtree(root_id: int, dst_set: set[int]):
            """从 root_id 往下 DFS，把整棵子树加入 dst_set。"""
            if root_id < 0 or root_id >= m.nbody:
                return

            stack = [root_id]
            visited = set()

            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                dst_set.add(cur)
                for ch in children[cur]:
                    if ch not in visited:
                        stack.append(ch)

        # ---- 3) 如果有手掌 body，把整个子树都视为 hand ----
        palm_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rh_palm")
        if palm_id != -1:
            add_subtree(palm_id, hand_body_ids)

        # ---- 4) 机器人整体：以 base_link 为根的子树 ----
        all_robot_body_ids: set[int] = set()
        base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_id != -1:
            add_subtree(base_id, all_robot_body_ids)
        else:
            # 如果没有 base_link，就默认整个模型都是机器人
            all_robot_body_ids = set(range(m.nbody))

        # arm = 机器人 body - hand body
        arm_body_ids = all_robot_body_ids - hand_body_ids

        # ---- 5) 转成排序好的数组，并建立 name 列表 ----
        arm_ids  = np.array(sorted(arm_body_ids), dtype=int)
        hand_ids = np.array(sorted(hand_body_ids), dtype=int)

        arm_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body_{int(bid)}"
            for bid in arm_ids
        ]
        hand_names = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body_{int(bid)}"
            for bid in hand_ids
        ]

        body_groups = {
            "arm":  BodyGroup(ids=arm_ids,  names=arm_names),
            "hand": BodyGroup(ids=hand_ids, names=hand_names),
        }

        # 顺手建一个 name -> id 的 map
        self.body_name_to_id = {
            name: bid
            for g in body_groups.values()
            for name, bid in zip(g.names, g.ids)
        }

        return body_groups
    
    def _combine_conf(self) -> np.ndarray:
        pass


    @property
    def dt(self) -> float:
        return self.model.opt.timestep
    
    @property
    def default_main_conf(self) -> np.ndarray:
        return np.array(
            [_DEFAULT_MAIN_CONF_DICT[name] for name in self.main_joint_names],
            dtype=float,
        )
    
    @property
    def default_conf(self) -> np.ndarray:
        """Prefer a saved qpos snapshot if it matches the current model."""
        saved_conf = self._load_saved_default_qpos()
        if saved_conf is not None:
            return saved_conf

        default_conf = np.zeros(self.full_joint_ids.shape[0])
        default_conf[:self.default_main_conf.shape[0]] = self.default_main_conf
        return default_conf

    @property
    def main_qpos(self):
        return self.data.qpos[self.main_qpos_adr].copy()
    
    @property
    def full_joint_ids(self):
        return np.append(self.main_joint_ids, self.finger_joint_ids)

    @property
    def full_qpos(self):
        return self.data.qpos[self.full_qpos_adr].copy()
    
    @property
    def full_joint_names(self):
        return np.append(self.main_joint_names, self.finger_joint_names)
    
    @property
    def full_qpos_adr(self):
        return np.append(np.array(self.main_qpos_adr), np.array(self.finger_qpos_adr))
    
    @property
    def full_qvel_adr(self):
        return np.append(np.array(self.main_qvel_adr), np.array(self.finger_qvel_adr))
    
    @property
    def full_body_ids(self):
        return np.append(self.hand_body_groups.ids, self.arm_body_groups.ids)
    
    @property
    def hand_body_groups(self):
        return self.body_groups["hand"]
    
    @property
    def arm_body_groups(self):
        return self.body_groups["arm"]
    
    @property
    def hand_cfrc_ext(self):
        return self.data.cfrc_ext[self.hand_body_groups.ids].copy()
    
    @property
    def main_cfrc_ext(self):
        return self.data.cfrc_ext[self.main_body_groups.ids].copy()
    
    @property
    def full_ndof(self):
        return len(self.full_qpos_adr)
    
    @property
    def main_joint_bounds(self):
        return self.motion.bounds_from_qpos_adrs(self.main_qpos_adr)
    
    @property
    def palm_axes(self):
        return 
    
    @property
    def vmax(self):
        return FULL_VMAX
    
    @property
    def amax(self):
        return FULL_AMAX
    
    def set_default(self):
        self.set_conf(self.full_qpos_adr, conf=self.default_conf)

    def set_main_conf(self, main_conf: Sequence[float]):
        self.set_conf(self.main_qpos_adr, conf=main_conf)

    def set_conf(self, active_qpos_adr, conf):
        active_qpos_adr = np.asarray(active_qpos_adr, dtype=int)
        conf = np.asarray(conf, dtype=float)
        assert conf.shape[0] == active_qpos_adr.shape[0]

        self.data.qpos[active_qpos_adr] = conf
        self.data.qvel[:] = 0.0  # 可选：teleport 后清速度更稳
        mujoco.mj_forward(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def hold_main_conf(self, main_conf):
        self.set_main_conf(main_conf)

        # 让 actuator 目标也等于当前关节角（常见 position servo 用法）
        # 需要你知道 main_conf 对应哪些 actuator；这里给一个“按 joint transmission 找 joint id”的通用映射
        m = self.model
        jids = self.main_joint_ids  # 你的“主关节 joint ids”列表（不是 qposadr）

        # joint id -> actuator id
        j2a = {}
        for aid in range(m.nu):
            if int(m.actuator_trntype[aid]) == int(mujoco.mjtTrn.mjTRN_JOINT):
                jid = int(m.actuator_trnid[aid, 0])
                j2a[jid] = aid

        act_ids = [j2a[j] for j in jids]
        self.data.ctrl[act_ids] = np.asarray(main_conf, float)

    def hold_default_conf(self):
        self.set_default()
        m, d = self.model, self.data

        qpos_ref = np.asarray(self.default_conf, float)

        # 先不动其它 actuator，只覆盖能映射到 joint 的那部分
        ctrl = d.ctrl.copy()

        for aid in range(m.nu):
            if int(m.actuator_trntype[aid]) == int(mujoco.mjtTrn.mjTRN_JOINT):
                jid  = int(m.actuator_trnid[aid, 0])
                qadr = int(m.jnt_qposadr[jid])   # 该 joint 在 qpos 里的起始地址（hinge/slide 正好 1 个）
                ctrl[aid] = qpos_ref[qadr]

        d.ctrl[:] = ctrl

    
    def bounds_from_qpos_adrs(self, qpos_adr: np.ndarray) -> np.ndarray:
        return self.motion.bounds_from_qpos_adrs(qpos_adr)
    
    def allowed_collision_bodies(self) -> list:
        return []

    def is_running(self) -> bool:
        return (self.viewer is None) or self.viewer.is_running()

    def get_state(self):
        q = self.data.qpos.copy()
        dq = self.data.qvel.copy()
        t = self.data.time
        return q, dq, t

    def save_current_qpos(self, path: str | Path | None = None) -> Path:
        """
        Save the current full qpos as the new default.

        Args:
            path: Optional override for the save path. Defaults to config/default_qpos.npy.
        Returns:
            Path to the saved file.
        """
        dst = Path(path) if path is not None else DEFAULT_QPOS_PATH
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, self.full_qpos)
        print(f"[MjWorld] Saved current qpos ({len(self.full_qpos)} values) to {dst}")
        return dst

    def _load_saved_default_qpos(self) -> np.ndarray | None:
        """Load a saved default qpos if present and compatible."""
        if not DEFAULT_QPOS_PATH.exists():
            return None

        try:
            qpos = np.load(DEFAULT_QPOS_PATH)
        except Exception as exc:  # noqa: BLE001
            print(f"[MjWorld] Failed to load saved qpos from {DEFAULT_QPOS_PATH}: {exc}")
            return None

        if qpos.shape[0] != self.full_ndof:
            print(
                f"[MjWorld] Ignoring saved qpos (len={qpos.shape[0]}), "
                f"model expects {self.full_ndof}"
            )
            return None

        return np.asarray(qpos, dtype=float)

    def set_ctrl(self, tau: np.ndarray):
        # 假设 actuator 数量和 tau 长度一致
        self.data.ctrl[:len(tau)] = tau

    def step(self):
        t0 = time.time()
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        dt = self.dt - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)

    def generate_segment_trajectory(
        self,
        q_goal_full: np.ndarray,              
        v_max: np.ndarray | None = None,
        a_max: np.ndarray | None = None,
    ) -> JointTrajectory:
        return self.motion.generate_segment_trajectory(
            q_goal_full=q_goal_full, v_max=v_max, a_max=a_max
        )
    
    def generate_waypoints_trajectory(
        self,
        waypoints: Sequence[np.ndarray],              
        v_max: np.ndarray | None = None,
        a_max: np.ndarray | None = None,
    ) -> JointTrajectory:
        return self.motion.generate_waypoints_trajectory(
            waypoints=list(waypoints), v_max=v_max, a_max=a_max
        )
    
    def follow_trajectory_kinematic(self, traj: JointTrajectory):
        self.executor.follow_trajectory_kinematic(traj)

    def follow_trajectory_actuator(self, traj: JointTrajectory):
        self.executor.follow_trajectory_actuator(traj)

    def main_motion_plan(self, 
                         q_goal_main: Sequence[float],
                         max_planning_time=5.0,
                         ):
        return self.motion.main_motion_plan(
            q_goal_main=q_goal_main, max_planning_time=max_planning_time
        )
    
    def get_body_world_aabb(
        self,
        body_name: str,
        include_subtree: bool = True,
        exact_mesh: bool = False,
        geom_predicate=None,   # 可选：过滤某些 geoms，比如只用碰撞 geom
    ) -> AABB:
        return self.grasping.get_body_world_aabb(
            body_name=body_name,
            include_subtree=include_subtree,
            exact_mesh=exact_mesh,
            geom_predicate=geom_predicate,
        )
    
    def get_body_aabb_in_body_frame(
        self,
        body_name: str,
        include_subtree: bool = True,
        exact_mesh: bool = False,
        geom_predicate=None,
    ) -> AABB:
        return self.grasping.get_body_aabb_in_body_frame(
            body_name=body_name,
            include_subtree=include_subtree,
            exact_mesh=exact_mesh,
            geom_predicate=geom_predicate,
        )

    
    def propose_grasps_from_bbox(
        self,
        body_name: str,
        bbox_frame: str = "body",
        include_subtree: bool = True,
        exact_mesh: bool = False,
        topk: int = 10,

        # NEW: prefer from above (world up)
        prefer_world_up: bool = True,
        world_up_cos_thresh: float = 0.7,   # cos(angle) threshold, 0.7≈45deg
        require_above: bool = True,         # palm position must be above bbox center
        world_up_weight: float = 2.0,       # score += weight * alignment
    ) -> GraspInfo:
        return self.grasping.propose_grasps_from_bbox(
            body_name=body_name,
            bbox_frame=bbox_frame,
            include_subtree=include_subtree,
            exact_mesh=exact_mesh,
            topk=topk,
            prefer_world_up=prefer_world_up,
            world_up_cos_thresh=world_up_cos_thresh,
            require_above=require_above,
            world_up_weight=world_up_weight,
        )

def debug_print_actuators(world):
    world.print_actuator_infos()
