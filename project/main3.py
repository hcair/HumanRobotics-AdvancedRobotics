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
from core.math_utils import orientation_error, rot_to_rpy  # noqa: E402
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
# Spot tracking:
#   self.spots.spot_pose(data, idx)  — (pos, R, normal) of spot `idx`
#   self.spots.completed[idx]        — True once SpotTracker auto-marks it done
#   self.spots.next_pending()        — index of the next pending spot, or None
class Controller:
    def __init__(
        self,
        robot: robotics.RobotWrapper,
        model: mujoco.MjModel,
        spots: SpotTracker,
    ) -> None:
        self.robot = robot
        self.model = model
        self.spots = spots

    def reset(self) -> None:
        pass

    def compute_torque(self, data: mujoco.MjData, t: float) -> np.ndarray:
        g = self.robot.get_gravity_vector(data)[:ARM_DOFS]

        tau = g
        return tau


# ==============================
# Spot tracker
# ==============================
SPOT_NAMES = (
    "spot_b1", "spot_b2", "spot_b3",
    "spot_l1", "spot_l2", "spot_l3",
    "spot_r1", "spot_r2", "spot_r3",
)
SPOT_COLOR_DONE = np.array([0.10, 0.95, 0.30, 1.0], dtype=np.float32)
SPOT_DIST_TOL = 0.03           # m   — EE within 3 cm of a spot to count as "on it"
SPOT_FORCE_TARGET = 20.0       # N   — desired normal-direction press force
SPOT_FORCE_TOL = 1.0           # N   — ±5% tolerance band around target (19~21 N)
SPOT_HOLD_TIME = 2.0           # s   — band must be held continuously


class SpotTracker:
    def __init__(self, model: mujoco.MjModel, names: tuple[str, ...] = SPOT_NAMES) -> None:
        self.model = model
        self.names = names
        self.spot_ids: list[int] = []
        for name in names:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"site '{name}' not found")
            self.spot_ids.append(sid)
        self._orig_rgba = [np.array(model.site_rgba[sid]).copy() for sid in self.spot_ids]
        self.completed: list[bool] = [False] * len(names)
        self._press_t: list[float | None] = [None] * len(names)

        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        sid_f = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force")
        if self.ee_site_id < 0 or sid_f < 0:
            raise ValueError("ee_site or ee_force sensor not found")
        self.ee_force_adr = int(model.sensor_adr[sid_f])

    def reset(self) -> None:
        for sid, orig in zip(self.spot_ids, self._orig_rgba):
            self.model.site_rgba[sid] = orig
        self.completed = [False] * len(self.names)
        self._press_t = [None] * len(self.names)

    def spot_pose(self, data: mujoco.MjData, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sid = self.spot_ids[idx]
        pos = np.array(data.site_xpos[sid])
        mat = np.array(data.site_xmat[sid]).reshape(3, 3)
        normal = mat[:, 2].copy()
        return pos, mat, normal

    def mark_completed(self, idx: int) -> None:
        self.completed[idx] = True
        self.model.site_rgba[self.spot_ids[idx]] = SPOT_COLOR_DONE

    def next_pending(self) -> int | None:
        for i, done in enumerate(self.completed):
            if not done:
                return i
        return None

    def update(self, data: mujoco.MjData, t: float) -> None:
        ee_pos = np.array(data.site_xpos[self.ee_site_id])
        ee_mat = np.array(data.site_xmat[self.ee_site_id]).reshape(3, 3)
        f_world = ee_mat @ data.sensordata[self.ee_force_adr : self.ee_force_adr + 3]

        for i, done in enumerate(self.completed):
            if done:
                self._press_t[i] = None
                continue
            pos, _, normal = self.spot_pose(data, i)
            if np.linalg.norm(ee_pos - pos) > SPOT_DIST_TOL:
                self._press_t[i] = None
                continue
            press_mag = abs(float(f_world @ normal))
            if abs(press_mag - SPOT_FORCE_TARGET) <= SPOT_FORCE_TOL:
                if self._press_t[i] is None:
                    self._press_t[i] = t
                elif t - self._press_t[i] >= SPOT_HOLD_TIME:
                    self.mark_completed(i)
                    self._press_t[i] = None
            else:
                self._press_t[i] = None


# ==============================
# App wiring
# ==============================
class FinalProjectApp(HWAppBase):
    model_path = PROJECT_ROOT / "project" / "models" / "panda" / "project3.xml"
    title = "Final Project Task 3 — Spot Welding"
    site_name = "ee_site"
    camera = {
        "lookat": [0.55, 0.0, 0.25],
        "distance": 1.5,
        "azimuth": -150.0,
        "elevation": -20.0,
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
        self.finger_open_position = 0.0
        self.duration = duration
        self.plot_enabled = plot_enabled
        self.plot_api = None
        self._log_button: ttk.Button | None = None

        self.spots = SpotTracker(self.model)
        self.controller = Controller(self.robot, self.model, self.spots)

        self.ee_force_adr = self._sensor_adr("ee_force")
        self.ee_torque_adr = self._sensor_adr("ee_torque")

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
        self.sig_ee_force = Signal(
            "ee_force",
            lambda d: d.sensordata[self.ee_force_adr : self.ee_force_adr + 3].copy(),
        )
        self.sig_ee_torque = Signal(
            "ee_torque",
            lambda d: d.sensordata[self.ee_torque_adr : self.ee_torque_adr + 3].copy(),
        )
        self._all_signals = [
            self.sig_q,
            self.sig_dq,
            self.sig_ddq,
            self.sig_tau,
            self.sig_power,
            self.sig_ee_pos,
            self.sig_ee_rpy,
            self.sig_ee_force,
            self.sig_ee_torque,
        ]

        self.data_logger = DataLogger(
            signals=self._all_signals,
            output_dir=self.data_dir,
            filename_prefix="p3_log",
            enabled=start_recording,
        )

    def _sensor_adr(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"sensor '{name}' not found")
        return int(self.model.sensor_adr[sid])

    def sim_step(self) -> bool:
        if self.duration is not None and self.sim_time > self.duration:
            return False
        mujoco.mj_forward(self.model, self.data)
        self.spots.update(self.data, self.sim_time)
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
        self.spots.reset()
        self.controller.reset()
        if self.plot_api is not None:
            self.plot_api.clear()

    def build_gui_content(self, parent: tk.Widget, api) -> None:
        if self.plot_enabled:
            plot_frame, self.plot_api = api.add_plot_area(
                parent,
                signals=[self.sig_tau, self.sig_dq, self.sig_ee_force],
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
