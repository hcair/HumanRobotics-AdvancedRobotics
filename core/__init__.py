"""Core package for the human/advanced robotics course: MuJoCo simulation, trajectories, GUI, logging, and real-time utilities."""

from . import math_utils, mujoco_robotics, viewer_utils
from .gui import GUIParameter, UnifiedGUI
from .keyboard_handler import KeyboardHandler
from .mujoco_gui_helpers import make_qpos_parameter, make_qpos_parameter_from_model
from .mujoco_robotics import load_model
from .realtime import RateLimiter, RealtimeConfig, RealtimeSync
from .signal import Signal
from .trajectory import FifthOrderTrajectory, MultiSegmentTrajectory, Trajectory

robotics = mujoco_robotics

__all__ = [
    "FifthOrderTrajectory",
    "GUIParameter",
    "KeyboardHandler",
    "MultiSegmentTrajectory",
    "RateLimiter",
    "RealtimeConfig",
    "RealtimeSync",
    "Signal",
    "Trajectory",
    "UnifiedGUI",
    "load_model",
    "make_qpos_parameter",
    "make_qpos_parameter_from_model",
    "math_utils",
    "robotics",
    "viewer_utils",
]
