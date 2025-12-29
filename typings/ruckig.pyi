"""
Instantaneous Motion Generation for Robots and Machines. Real-time and time-optimal trajectory calculation given a target waypoint with position, velocity, and acceleration, starting from any initial state limited by velocity, acceleration, and jerk constraints.
"""
from __future__ import annotations
import collections.abc
import enum
import typing
__all__: list[str] = ['Bound', 'BrakeProfile', 'Continuous', 'ControlInterface', 'Discrete', 'DurationDiscretization', 'Error', 'ErrorExecutionTimeCalculation', 'ErrorInvalidInput', 'ErrorPositionalLimits', 'ErrorSynchronizationCalculation', 'Finished', 'InputParameter', 'No', 'OutputParameter', 'Phase', 'Position', 'Profile', 'Result', 'Ruckig', 'RuckigError', 'Synchronization', 'Time', 'TimeIfNecessary', 'Trajectory', 'Velocity', 'Working']
class Bound:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self) -> str:
        ...
    @property
    def max(self) -> float:
        ...
    @property
    def min(self) -> float:
        ...
    @property
    def t_max(self) -> float:
        ...
    @property
    def t_min(self) -> float:
        ...
class BrakeProfile:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    @property
    def a(self) -> list[float]:
        ...
    @property
    def duration(self) -> float:
        ...
    @property
    def j(self) -> list[float]:
        ...
    @property
    def p(self) -> list[float]:
        ...
    @property
    def t(self) -> list[float]:
        ...
    @property
    def v(self) -> list[float]:
        ...
class ControlInterface(enum.Enum):
    Position: typing.ClassVar[ControlInterface]  # value = ControlInterface.Position
    Velocity: typing.ClassVar[ControlInterface]  # value = ControlInterface.Velocity
class DurationDiscretization(enum.Enum):
    Continuous: typing.ClassVar[DurationDiscretization]  # value = DurationDiscretization.Continuous
    Discrete: typing.ClassVar[DurationDiscretization]  # value = DurationDiscretization.Discrete
class InputParameter:
    control_interface: ControlInterface
    duration_discretization: DurationDiscretization
    interrupt_calculation_duration: float | None
    minimum_duration: float | None
    synchronization: Synchronization
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, dofs: int) -> None:
        """
        __init__(self, dofs: int, max_number_of_waypoints: int) -> None
        """
    def __ne__(self, arg: InputParameter) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def validate(self, check_current_state_within_limits: bool = False, check_target_state_within_limits: bool = True) -> bool:
        ...
    @property
    def current_acceleration(self) -> list[float]:
        ...
    @current_acceleration.setter
    def current_acceleration(self, arg: collections.abc.Sequence[float]) -> None:
        ...
    @property
    def current_position(self) -> list[float]:
        ...
    @current_position.setter
    def current_position(self, arg: collections.abc.Sequence[float]) -> None:
        ...
    @property
    def current_velocity(self) -> list[float]:
        ...
    @current_velocity.setter
    def current_velocity(self, arg: collections.abc.Sequence[float]) -> None:
        ...
    @property
    def degrees_of_freedom(self) -> int:
        ...
    @property
    def enabled(self) -> list[bool]:
        ...
    @enabled.setter
    def enabled(self, arg: collections.abc.Sequence[bool]) -> None:
        ...
    @property
    def intermediate_positions(self) -> list[list[float]]:
        ...
    @intermediate_positions.setter
    def intermediate_positions(self, arg: collections.abc.Sequence[collections.abc.Sequence[float]]) -> None:
        ...
    @property
    def max_acceleration(self) -> list[float]:
        ...
    @max_acceleration.setter
    def max_acceleration(self, arg: collections.abc.Sequence[float]) -> None:
        ...
    @property
    def max_jerk(self) -> list[float]:
        ...
    @max_jerk.setter
    def max_jerk(self, arg: collections.abc.Sequence[float]) -> None:
        ...
    @property
    def max_position(self) -> list[float] | None:
        ...
    @max_position.setter
    def max_position(self, arg: collections.abc.Sequence[float] | None) -> None:
        ...
    @property
    def max_velocity(self) -> list[float]:
        ...
    @max_velocity.setter
    def max_velocity(self, arg: collections.abc.Sequence[float]) -> None:
        ...
    @property
    def min_acceleration(self) -> list[float] | None:
        ...
    @min_acceleration.setter
    def min_acceleration(self, arg: collections.abc.Sequence[float] | None) -> None:
        ...
    @property
    def min_position(self) -> list[float] | None:
        ...
    @min_position.setter
    def min_position(self, arg: collections.abc.Sequence[float] | None) -> None:
        ...
    @property
    def min_velocity(self) -> list[float] | None:
        ...
    @min_velocity.setter
    def min_velocity(self, arg: collections.abc.Sequence[float] | None) -> None:
        ...
    @property
    def per_dof_control_interface(self) -> list[ControlInterface] | None:
        ...
    @per_dof_control_interface.setter
    def per_dof_control_interface(self, arg: collections.abc.Sequence[ControlInterface] | None) -> None:
        ...
    @property
    def per_dof_synchronization(self) -> list[Synchronization] | None:
        ...
    @per_dof_synchronization.setter
    def per_dof_synchronization(self, arg: collections.abc.Sequence[Synchronization] | None) -> None:
        ...
    @property
    def per_section_max_acceleration(self) -> list[list[float]] | None:
        ...
    @per_section_max_acceleration.setter
    def per_section_max_acceleration(self, arg: collections.abc.Sequence[collections.abc.Sequence[float]] | None) -> None:
        ...
    @property
    def per_section_max_jerk(self) -> list[list[float]] | None:
        ...
    @per_section_max_jerk.setter
    def per_section_max_jerk(self, arg: collections.abc.Sequence[collections.abc.Sequence[float]] | None) -> None:
        ...
    @property
    def per_section_max_position(self) -> list[list[float]] | None:
        ...
    @per_section_max_position.setter
    def per_section_max_position(self, arg: collections.abc.Sequence[collections.abc.Sequence[float]] | None) -> None:
        ...
    @property
    def per_section_max_velocity(self) -> list[list[float]] | None:
        ...
    @per_section_max_velocity.setter
    def per_section_max_velocity(self, arg: collections.abc.Sequence[collections.abc.Sequence[float]] | None) -> None:
        ...
    @property
    def per_section_min_acceleration(self) -> list[list[float]] | None:
        ...
    @per_section_min_acceleration.setter
    def per_section_min_acceleration(self, arg: collections.abc.Sequence[collections.abc.Sequence[float]] | None) -> None:
        ...
    @property
    def per_section_min_position(self) -> list[list[float]] | None:
        ...
    @per_section_min_position.setter
    def per_section_min_position(self, arg: collections.abc.Sequence[collections.abc.Sequence[float]] | None) -> None:
        ...
    @property
    def per_section_min_velocity(self) -> list[list[float]] | None:
        ...
    @per_section_min_velocity.setter
    def per_section_min_velocity(self, arg: collections.abc.Sequence[collections.abc.Sequence[float]] | None) -> None:
        ...
    @property
    def per_section_minimum_duration(self) -> list[float] | None:
        ...
    @per_section_minimum_duration.setter
    def per_section_minimum_duration(self, arg: collections.abc.Sequence[float] | None) -> None:
        ...
    @property
    def target_acceleration(self) -> list[float]:
        ...
    @target_acceleration.setter
    def target_acceleration(self, arg: collections.abc.Sequence[float]) -> None:
        ...
    @property
    def target_position(self) -> list[float]:
        ...
    @target_position.setter
    def target_position(self, arg: collections.abc.Sequence[float]) -> None:
        ...
    @property
    def target_velocity(self) -> list[float]:
        ...
    @target_velocity.setter
    def target_velocity(self, arg: collections.abc.Sequence[float]) -> None:
        ...
class OutputParameter:
    time: float
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __copy__(self) -> OutputParameter:
        ...
    def __init__(self, dofs: int) -> None:
        """
        __init__(self, dofs: int, max_number_of_waypoints: int) -> None
        """
    def __repr__(self) -> str:
        ...
    def pass_to_input(self, input: InputParameter) -> None:
        ...
    @property
    def calculation_duration(self) -> float:
        ...
    @property
    def degrees_of_freedom(self) -> int:
        ...
    @property
    def did_section_change(self) -> bool:
        ...
    @property
    def new_acceleration(self) -> list[float]:
        ...
    @property
    def new_calculation(self) -> bool:
        ...
    @property
    def new_position(self) -> list[float]:
        ...
    @property
    def new_section(self) -> int:
        ...
    @property
    def new_velocity(self) -> list[float]:
        ...
    @property
    def trajectory(self) -> Trajectory:
        ...
    @property
    def was_calculation_interrupted(self) -> bool:
        ...
class Profile:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __repr__(self) -> str:
        ...
    @property
    def a(self) -> list[float]:
        ...
    @property
    def accel(self) -> BrakeProfile:
        ...
    @property
    def af(self) -> float:
        ...
    @property
    def brake(self) -> BrakeProfile:
        ...
    @property
    def control_signs(self) -> ...:
        ...
    @property
    def direction(self) -> ...:
        ...
    @property
    def j(self) -> list[float]:
        ...
    @property
    def limits(self) -> ...:
        ...
    @property
    def p(self) -> list[float]:
        ...
    @property
    def pf(self) -> float:
        ...
    @property
    def t(self) -> list[float]:
        ...
    @property
    def v(self) -> list[float]:
        ...
    @property
    def vf(self) -> float:
        ...
class Result(enum.IntEnum):
    Error: typing.ClassVar[Result]  # value = Result.Error
    ErrorExecutionTimeCalculation: typing.ClassVar[Result]  # value = Result.ErrorExecutionTimeCalculation
    ErrorInvalidInput: typing.ClassVar[Result]  # value = Result.ErrorInvalidInput
    ErrorPositionalLimits: typing.ClassVar[Result]  # value = Result.ErrorPositionalLimits
    ErrorSynchronizationCalculation: typing.ClassVar[Result]  # value = Result.ErrorSynchronizationCalculation
    Finished: typing.ClassVar[Result]  # value = Result.Finished
    Working: typing.ClassVar[Result]  # value = Result.Working
    @classmethod
    def __new__(cls, value):
        ...
    def __format__(self, format_spec):
        """
        Convert to a string according to format_spec.
        """
    def __repr__(self):
        ...
    def __str__(self):
        ...
class Ruckig:
    delta_time: float
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, dofs: int) -> None:
        """
        __init__(self, dofs: int, delta_time: float) -> None
        __init__(self, dofs: int, delta_time: float, max_number_of_waypoints: int = 0) -> None
        """
    def calculate(self, input: InputParameter, trajectory: Trajectory) -> Result:
        """
        calculate(self, input: ruckig.InputParameter, trajectory: ruckig.Trajectory, was_interrupted: bool) -> ruckig.Result
        """
    def filter_intermediate_positions(self, input: InputParameter, threshold_distance: collections.abc.Sequence[float]) -> list[list[float]]:
        ...
    def reset(self) -> None:
        ...
    def update(self, input: InputParameter, output: OutputParameter) -> Result:
        ...
    def validate_input(self, input: InputParameter, check_current_state_within_limits: bool = False, check_target_state_within_limits: bool = True) -> bool:
        ...
    @property
    def degrees_of_freedom(self) -> int:
        ...
    @property
    def max_number_of_waypoints(self) -> int:
        ...
class RuckigError(Exception):
    pass
class Synchronization(enum.Enum):
    No: typing.ClassVar[Synchronization]  # value = Synchronization.No
    Phase: typing.ClassVar[Synchronization]  # value = Synchronization.Phase
    Time: typing.ClassVar[Synchronization]  # value = Synchronization.Time
    TimeIfNecessary: typing.ClassVar[Synchronization]  # value = Synchronization.TimeIfNecessary
class Trajectory:
    @staticmethod
    def __new__(type, *args, **kwargs):
        """
        Create and return a new object.  See help(type) for accurate signature.
        """
    def __init__(self, dofs: int) -> None:
        """
        __init__(self, dofs: int, max_number_of_waypoints: int) -> None
        """
    def at_time(self, time: float, return_section: bool = False) -> tuple:
        ...
    def get_first_time_at_position(self, dof: int, position: float, time_after: float = 0.0) -> float | None:
        ...
    @property
    def degrees_of_freedom(self) -> int:
        ...
    @property
    def duration(self) -> float:
        ...
    @property
    def independent_min_durations(self) -> list[float]:
        ...
    @property
    def intermediate_durations(self) -> list[float]:
        ...
    @property
    def position_extrema(self) -> list[Bound]:
        ...
    @property
    def profiles(self) -> list[list[Profile]]:
        """
        (self) -> list[list[ruckig::Profile]]
        """
Continuous: DurationDiscretization  # value = DurationDiscretization.Continuous
Discrete: DurationDiscretization  # value = DurationDiscretization.Discrete
Error: Result  # value = Result.Error
ErrorExecutionTimeCalculation: Result  # value = Result.ErrorExecutionTimeCalculation
ErrorInvalidInput: Result  # value = Result.ErrorInvalidInput
ErrorPositionalLimits: Result  # value = Result.ErrorPositionalLimits
ErrorSynchronizationCalculation: Result  # value = Result.ErrorSynchronizationCalculation
Finished: Result  # value = Result.Finished
No: Synchronization  # value = Synchronization.No
Phase: Synchronization  # value = Synchronization.Phase
Position: ControlInterface  # value = ControlInterface.Position
Time: Synchronization  # value = Synchronization.Time
TimeIfNecessary: Synchronization  # value = Synchronization.TimeIfNecessary
Velocity: ControlInterface  # value = ControlInterface.Velocity
Working: Result  # value = Result.Working
__version__: str = '0.15.3'
