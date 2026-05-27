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
class Controller:
    def __init__(self, robot: robotics.RobotWrapper, model: mujoco.MjModel) -> None:
        self.robot = robot
        self.model = model

    def compute_torque(self, data: mujoco.MjData, t: float) -> np.ndarray:
        g = self.robot.get_gravity_vector(data)[:ARM_DOFS]

        tau = g
        return tau


# ==============================
# App wiring
# ==============================
class FinalProjectApp(HWAppBase):
    model_path = PROJECT_ROOT / "project" / "models" / "panda" / "project_free.xml"
    title = "Final Project — Free"
    site_name = "ee_site"
    camera = {
        "lookat": [0.3, 0.0, 0.4],
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

        self.controller = Controller(self.robot, self.model)

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
            filename_prefix="p_free_log",
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

        self.data_logger.log(self.data.time, self.data)
        if self.plot_api is not None:
            self.plot_api.push(self.data.time, self.data)
        return True

    def pre_render(self) -> None:
        if self.plot_api is not None:
            self.plot_api.update()

    def on_reset(self) -> None:
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
        ttk.Button(btns, text="Reset", command=self.base_reset).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        ttk.Button(btns, text="Exit", command=self.keyboard.request_exit).pack(
            side=tk.LEFT, padx=4, pady=4
        )
        btns.pack(side=tk.TOP, fill=tk.X)

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
