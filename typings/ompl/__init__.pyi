from __future__ import annotations
import typing
__all__: list[str] = ['PlanningAlgorithms', 'dll_loader', 'initializePlannerLists']
class PlanningAlgorithms:
    BOOL: typing.ClassVar[int] = 1
    DOUBLE: typing.ClassVar[int] = 4
    ENUM: typing.ClassVar[int] = 2
    INT: typing.ClassVar[int] = 3
    UNKNOWN: typing.ClassVar[int] = 0
    def __init__(self, module):
        ...
    def addPlanner(self, planner):
        ...
    def getPlanners(self):
        ...
def dll_loader(lib, path):
    ...
def initializePlannerLists():
    ...
