"""Core package for the human/advanced robotics course: MuJoCo simulation, GUI, keyboard, and realtime utilities."""

from . import mujoco_robotics
from .gui import GUIParameter, UnifiedGUI
from .keyboard_handler import KeyboardHandler
from .mujoco_gui_helpers import make_qpos_parameter_from_model
from .mujoco_robotics import load_model
from .realtime import RateLimiter, RealtimeConfig, RealtimeSync

robotics = mujoco_robotics

__all__ = [
    "GUIParameter",
    "KeyboardHandler",
    "RateLimiter",
    "RealtimeConfig",
    "RealtimeSync",
    "UnifiedGUI",
    "load_model",
    "make_qpos_parameter_from_model",
    "robotics",
]
