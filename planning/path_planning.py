from __future__ import annotations
import numpy as np
from sim.constraint import Constraint
from typing import List, Sequence, Optional, Callable
from dataclasses import dataclass
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
ou.setLogLevel(ou.LOG_DEBUG) 


'''
    Conf space path planning. RRTConnect
'''




@dataclass(frozen=True)
class PlannerSpec:
    ndof_full: int                 # full q 的维度（建议是“机器人关节向量”的维度，不要用 model.nq）
    active_idx: np.ndarray         # 在 full q 里的索引（shape=(ndof_active,)）
    bounds_active: np.ndarray      # (ndof_active, 2)

class OmplPlanner:
    def __init__(
        self,
        spec: PlannerSpec,
        is_valid_full: Callable[[np.ndarray], bool],
        simplify: bool = True,
    ):
        self.spec = spec
        self.is_valid_full = is_valid_full
        self.simplify = simplify

        self.ndof = int(spec.active_idx.shape[0])
        self.space = ob.RealVectorStateSpace(self.ndof)

        b = ob.RealVectorBounds(self.ndof)
        for i in range(self.ndof):
            lo, hi = float(spec.bounds_active[i, 0]), float(spec.bounds_active[i, 1])
            b.setLow(i, lo)
            b.setHigh(i, hi)
        self.space.setBounds(b)

        self.ss = og.SimpleSetup(self.space)

    def _state_to_qred(self, state: ob.State) -> np.ndarray:
        q = np.empty(self.ndof, dtype=float)
        for i in range(self.ndof):
            q[i] = float(state[i])
        return q

    def _qred_to_state(self, qred: np.ndarray) -> ob.State:
        qred = np.asarray(qred, dtype=float)
        s = ob.State(self.space)
        for i in range(self.ndof):
            s[i] = float(qred[i])
        return s

    def _embed(self, qred: np.ndarray, q_template_full: np.ndarray) -> np.ndarray:
        q = q_template_full.copy()
        q[self.spec.active_idx] = qred
        return q

    def plan(self, q_init_full: np.ndarray, q_goal_full: np.ndarray, timeout: float = 5.0) -> List[np.ndarray]:
        q_init_full = np.asarray(q_init_full, dtype=float).copy()
        q_goal_full = np.asarray(q_goal_full, dtype=float).copy()

        if q_init_full.shape[0] != self.spec.ndof_full or q_goal_full.shape[0] != self.spec.ndof_full:
            raise ValueError("q_init_full/q_goal_full dim mismatch with spec.ndof_full")

        q_template = q_init_full.copy()

        def _valid(state: ob.State) -> bool:
            qred = self._state_to_qred(state)
            qfull = self._embed(qred, q_template)
            return bool(self.is_valid_full(qfull))

        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(_valid))

        q_init_red = q_init_full[self.spec.active_idx]
        q_goal_red = q_goal_full[self.spec.active_idx]
        self.ss.setStartAndGoalStates(self._qred_to_state(q_init_red), self._qred_to_state(q_goal_red))

        self.ss.setPlanner(og.RRTConnect(self.ss.getSpaceInformation()))
        solved = self.ss.solve(timeout)
        if not solved:
            return []

        if self.simplify:
            self.ss.simplifySolution()

        path = self.ss.getSolutionPath()
        out: List[np.ndarray] = []
        for i in range(path.getStateCount()):
            qred = self._state_to_qred(path.getState(i))
            out.append(self._embed(qred, q_template))
        return out

