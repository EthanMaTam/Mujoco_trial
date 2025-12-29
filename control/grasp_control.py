from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
from enum import Enum, auto

import numpy as np
from ruckig import Result, Ruckig, InputParameter, OutputParameter

class GraspPhase(Enum):
    IDLE = auto()
    APPROACH = auto()
    PRE_CLOSE = auto()
    FORCE_TUNE = auto()
    HOLD = auto()

class MujocoWorldAdapter:
    from sim.mj_world import MjWorld
    def __init__(self, world: MjWorld):
        self.model = world.model
        self.data  = world.data
        self.body_groups = world.body_groups
        self.q_idx = world.full_qpos_adr
        self.v_idx = world.full_qvel_adr
        self.dt    = world.dt

    # ---- 被控制器调用的几个“窄接口” ----
    def get_kin_state(self):
        q   = self.data.qpos[self.q_idx].copy()
        dq  = self.data.qvel[self.v_idx].copy()
        ddq = self.data.qacc[self.v_idx].copy()
        return q, dq, ddq

    def get_palm_wrench(self):
        hand_body_groups = self.body_groups["hand"]
        return hand_body_groups, self.data.cfrc_ext[hand_body_groups.ids].copy()  # 6D
    
    def get_hand_cfrc_map(self):
        hbg, cfrc_ext = self.get_palm_wrench()
        body_names = hbg.names
        name_cfrc_map = dict(zip(body_names, cfrc_ext))
        return name_cfrc_map

    def set_ctrl(self, u, act_idx):
        # u 可以是 torque 或 target position
        self.data.ctrl[act_idx] = u