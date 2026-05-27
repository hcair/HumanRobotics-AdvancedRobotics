from __future__ import annotations

import logging
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import mujoco
import numpy as np

_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "core").is_dir():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
PROJECT_ROOT = _p

from core import DataLogger, Signal, robotics  # noqa: E402
from core.math_utils import rot_to_rpy  # noqa: E402
from core.viewer_utils import draw_polyline  # noqa: E402
from homeworks.helpers import HWAppBase  # noqa: E402

logging.basicConfig(level=logging.INFO)
logging.getLogger("homeworks.helpers").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ==============================
# Constants
# ==============================
ARM_DOFS = 7
Q_HOME = np.array(
    [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
    dtype=float,
)


# ==============================
# Controller
# ==============================
# Ball release (call inside compute_torque):
#   self.ball.release(data)       — release ball from gripper
#   self.ball.is_attached(data)   — returns True if ball is still held
class Controller:
    def __init__(
        self,
        robot: robotics.RobotWrapper,
        model: mujoco.MjModel,
        ball: BallReleaseSystem,
    ) -> None:
        self.robot = robot
        self.model = model
        self.ball = ball

    def compute_torque(self, data: mujoco.MjData, t: float) -> np.ndarray:
        g = self.robot.get_gravity_vector(data)[:ARM_DOFS]

        tau = g
        return tau


# ==============================
# Ball release system
# ==============================
class BallReleaseSystem:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.weld_eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_weld")
        self.hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        ball_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
        self.ball_qpos_adr = int(model.jnt_qposadr[ball_jnt_id])
        self.ball_qvel_adr = int(model.jnt_dofadr[ball_jnt_id])
        self.ball_qpos0 = model.qpos0[self.ball_qpos_adr : self.ball_qpos_adr + 7].copy()

    def is_attached(self, data: mujoco.MjData) -> bool:
        return bool(data.eq_active[self.weld_eq_id])

    def release(self, data: mujoco.MjData) -> None:
        data.eq_active[self.weld_eq_id] = 0

    def attach(self, data: mujoco.MjData) -> None:
        hand_pos = np.array(data.xpos[self.hand_body_id])
        hand_quat = np.array(data.xquat[self.hand_body_id])
        ball_pos = np.array(data.xpos[self.ball_body_id])
        ball_quat = np.array(data.xquat[self.ball_body_id])

        hand_mat = np.zeros(9)
        mujoco.mju_quat2Mat(hand_mat, hand_quat)
        rel_pos = hand_mat.reshape(3, 3).T @ (ball_pos - hand_pos)

        hand_quat_conj = np.array(
            [hand_quat[0], -hand_quat[1], -hand_quat[2], -hand_quat[3]]
        )
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, hand_quat_conj, ball_quat)

        self.model.eq_data[self.weld_eq_id, 3:6] = rel_pos
        self.model.eq_data[self.weld_eq_id, 6:10] = rel_quat
        data.eq_active[self.weld_eq_id] = 1

    def reset(self, data: mujoco.MjData) -> None:
        data.eq_active[self.weld_eq_id] = 0
        data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 7] = self.ball_qpos0
        data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 6] = 0.0


# ==============================
# Ball trail visualizer
# ==============================
class BallTrailVisualizer:
    def __init__(self, ball_body_id: int) -> None:
        self.ball_body_id = ball_body_id
        self._trail: list[np.ndarray] = []

    def record(self, data: mujoco.MjData) -> None:
        self._trail.append(data.xpos[self.ball_body_id].copy())

    def draw(self, viewer) -> None:
        viewer.user_scn.ngeom = 0
        if len(self._trail) >= 2:
            step = max(1, len(self._trail) // 500)
            draw_polyline(
                viewer, self._trail[::step],
                line_radius=0.004,
                rgba=np.array([1.0, 0.3, 0.1, 0.8], dtype=np.float32),
            )

    def clear(self) -> None:
        self._trail.clear()


# ==============================
# App wiring
# ==============================
class FinalProjectApp(HWAppBase):
    model_path = PROJECT_ROOT / "project" / "models" / "panda" / "project2.xml"
    title = "Final Project Task 2 — Throwing"
    site_name = "ee_site"
    camera = {
        "lookat": [0.5, 0.0, 0.3],
        "distance": 2.0,
        "azimuth": -180.0,
        "elevation": -25.0,
    }

    def __init__(
        self,
        plot_enabled: bool,
        duration: float | None,
        start_recording: bool,
    ) -> None:
        super().__init__()
        self.q_home = Q_HOME
        self.arm_dofs = ARM_DOFS
        self.finger_open_position = 0.04
        self.duration = duration
        self.plot_enabled = plot_enabled
        self.plot_api = None
        self._log_button: ttk.Button | None = None

        self.ball = BallReleaseSystem(self.model)
        self.trail = BallTrailVisualizer(self.ball.ball_body_id)
        self.controller = Controller(self.robot, self.model, self.ball)

        self.data_dir = PROJECT_ROOT / "project" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sig_q = Signal("q", lambda d: d.qpos[:ARM_DOFS].copy())
        self.sig_dq = Signal("dq", lambda d: d.qvel[:ARM_DOFS].copy())
        self.sig_ddq = Signal("ddq", lambda d: d.qacc[:ARM_DOFS].copy())
        self.sig_tau = Signal("tau", lambda d: d.ctrl[:ARM_DOFS].copy())
        self.sig_power = Signal(
            "power", lambda d: d.qvel[:ARM_DOFS] * d.ctrl[:ARM_DOFS]
        )
        self.sig_ee_pos = Signal("ee_pos", lambda d: self.robot.get_ee_pose(d)[0].copy())
        self.sig_ee_rpy = Signal("ee_rpy", lambda d: rot_to_rpy(self.robot.get_ee_pose(d)[1]))
        self._all_signals = [
            self.sig_q,
            self.sig_dq,
            self.sig_ddq,
            self.sig_tau,
            self.sig_power,
            self.sig_ee_pos,
            self.sig_ee_rpy,
        ]

        self.data_logger = DataLogger(
            signals=self._all_signals,
            output_dir=self.data_dir,
            filename_prefix="p2_log",
            enabled=start_recording,
        )

    def sim_step(self) -> bool:
        if self.duration is not None and self.sim_time > self.duration:
            return False
        mujoco.mj_forward(self.model, self.data)
        tau = self.controller.compute_torque(self.data, self.sim_time)
        self.robot.set_torque(self.data, tau)

        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt

        self.trail.record(self.data)

        self.data_logger.log(self.data.time, self.data)
        if self.plot_api is not None:
            self.plot_api.push(self.data.time, self.data)
        return True

    def pre_render(self) -> None:
        if self.plot_api is not None:
            self.plot_api.update()

    def draw_overlay(self, viewer) -> None:
        self.trail.draw(viewer)

    def on_reset(self) -> None:
        self.trail.clear()
        self.ball.reset(self.data)
        mujoco.mj_forward(self.model, self.data)
        self.ball.attach(self.data)
        if self.plot_api is not None:
            self.plot_api.clear()

    def build_gui_content(self, parent: tk.Widget, api) -> None:
        if self.plot_enabled:
            plot_frame, self.plot_api = api.add_plot_area(
                parent,
                signals=[self.sig_tau, self.sig_dq, self.sig_power],
                dt=self.dt,
                window_sec=10.0,
                figsize=(7, 6),
            )
            plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        btns = ttk.Frame(parent)
        self._log_button = ttk.Button(
            btns, text=self._log_button_text(), command=self._toggle_logging
        )
        self._log_button.pack(side=tk.LEFT, padx=4, pady=4)
        ttk.Button(btns, text="Release Ball", command=self._release_ball).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        ttk.Button(btns, text="Reset", command=self.base_reset).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        ttk.Button(btns, text="Exit", command=self.keyboard.request_exit).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        btns.pack(side=tk.TOP, fill=tk.X)

    def _release_ball(self) -> None:
        if self.ball.is_attached(self.data):
            self.ball.release(self.data)
            logger.info("Ball released at t=%.3f.", self.data.time)

    def _log_button_text(self) -> str:
        return (
            "Stop & Save  (REC ●)" if self.data_logger.enabled else "Start Recording"
        )

    def _toggle_logging(self) -> None:
        was_enabled = self.data_logger.enabled
        self.data_logger.set_enabled(not was_enabled)
        if self._log_button is not None:
            self._log_button.config(text=self._log_button_text())
        if not was_enabled:
            logger.info("Recording started.")
        else:
            logger.info("Recording stopped and saved.")


def main() -> None:
    app = FinalProjectApp(
        plot_enabled=True,
        duration=None,
        start_recording=False,
    )
    try:
        app.run()
    finally:
        if app.data_logger.enabled:
            saved = app.data_logger.save_and_clear()
            if saved is not None:
                logger.info("Final log saved on exit: %s", saved)


if __name__ == "__main__":
    main()
