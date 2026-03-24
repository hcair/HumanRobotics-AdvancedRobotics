"""HW1: Open Manipulator X — Jacobian-based gravity compensation."""

import logging
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer

_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "core").is_dir():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
PROJECT_ROOT = _p

from core import load_model, KeyboardHandler, robotics, UnifiedGUI, make_qpos_parameter_from_model
from core.realtime import RealtimeSync, RateLimiter, RealtimeConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Constants
# ==============================
DEG2RAD = np.pi / 180.0
INITIAL_QPOS_DEG = np.array([-45.0, -40.0, 20.0, 20.0, 0.0, 0.0])  # degrees


# ==============================
# Controller
# ==============================
def compute_gravity_vector(model: mujoco.MjModel, data: mujoco.MjData, robot: robotics.RobotWrapper) -> np.ndarray:
    nv = model.nv
    gravity = np.array(model.opt.gravity, dtype=np.float64)
    tau = np.zeros(nv)
    for body_id in range(1, model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if body_name is None:
            continue
        _, Jp, _ = robot.get_body_jacobian(
            data, body_name, translational=True, rotational=False
        )
        m = model.body_mass[body_id]
        # --- BEGIN: HW1 Problem 2(e) — modify the block below (gravity compensation) ---
        # Replace the placeholder with the formula from part (d).
        tau += 0  # placeholder
        # --- END: HW1 Problem 2(e) ---
    return tau


# ==============================
# Application
# ==============================
class OpenManipulatorApp:
    def __init__(self) -> None:
        model_path = PROJECT_ROOT / "models" / "robotis_open_manipulator_x" / "open_manipulator_x_simple.xml"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model, self.data = load_model(model_path)
        self.robot = robotics.RobotWrapper(self.model, site_name="ee_site")
        nq = self.model.nq
        nv = self.model.nv

        self.initial_qpos = (INITIAL_QPOS_DEG[:nq] * DEG2RAD).astype(float)
        self.initial_qvel = np.zeros(nv)

        self.render_dt = RealtimeConfig.render_dt()
        self.sync = RealtimeSync()
        self.render_tick = RateLimiter(0.0, self.render_dt)

        self.gravity_compensation_enabled = True
        self.keyboard = KeyboardHandler()

    # ------------------------------------------------------------------
    def _print_dynamics(self, which: str) -> None:
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        mujoco.mj_forward(self.model, self.data)
        if which == "ee_pose":
            pos, R = self.robot.get_ee_pose(self.data)
            logger.info("\n=== End-effector pose (FK) ===\nPosition p (3,): %s\nRotation matrix R (3x3):\n%s\n\n", pos, R)
        elif which == "jac":
            jac_bodies = [
                ("link2", "J1"),
                ("link3", "J2"),
                ("link4", "J3"),
                ("link5", "J4"),
                ("end_effector", "Jee"),
            ]
            logger.info("\n=== Jacobian per body ===\n")
            for body_name, label in jac_bodies:
                J_full, jacp, jacr = self.robot.get_body_jacobian(
                    self.data, body_name, translational=True, rotational=True
                )
                logger.info("--- %s (%s) ---\nJ (6 x nv):\n%s\n", label, body_name, J_full)
            logger.info("")
        elif which == "M":
            M = self.robot.get_inertia_matrix(self.data)
            M_rigid = M - np.diag(np.array(self.model.dof_armature, dtype=np.float64))
            logger.info("\n=== Inertia matrix M(q) (nv x nv) ===\n%s\n\n", M_rigid)
        elif which == "C":
            C = self.robot.get_coriolis_centrifugal(self.data)
            logger.info("\n=== Coriolis/centrifugal C(q,qd) (nv,) ===\n%s\n\n", C)
        elif which == "g":
            g = self.robot.get_gravity_vector(self.data)
            logger.info("\n=== Gravity vector g(q) (nv,) ===\n%s\n\n", g)

    def _reset_simulation(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.model, self.data)
        self.sync.reset(self.data.time)
        self.render_tick.next_time = self.data.time
        logger.info("Reset: back to initial pose.")

    def _toggle_gravity_compensation(self) -> None:
        self.gravity_compensation_enabled = not self.gravity_compensation_enabled
        logger.info("Gravity compensation: %s", "ON" if self.gravity_compensation_enabled else "OFF")

    def _build_content(self, parent, api) -> None:
        status_param = make_qpos_parameter_from_model(
            self.model, default=list(self.initial_qpos), name="Current q (Status)"
        )
        status_param.read_only = True
        status_param.setter = None
        status_param.getter = lambda m, d: list(d.qpos)
        status_panel = api.add_status_panel(parent, [status_param], self.model, self.data)
        status_panel.pack(side=tk.TOP, fill=tk.X)

        cmd_params = [
            make_qpos_parameter_from_model(
                self.model, default=list(self.initial_qpos), name="Target q (Command)"
            )
        ]
        cmd_panel = api.add_parameter_panel(parent, cmd_params, self.model, self.data)
        cmd_panel.pack(side=tk.TOP, fill=tk.X)

        btns = api.add_button_row(parent, [
            ("EE pose", lambda: self._print_dynamics("ee_pose")),
            ("Jacobian", lambda: self._print_dynamics("jac")),
            ("Inertia matrix", lambda: self._print_dynamics("M")),
            ("Coriolis/centrifugal", lambda: self._print_dynamics("C")),
            ("Gravity vector", lambda: self._print_dynamics("g")),
            ("Gravity comp. ON/OFF", self._toggle_gravity_compensation),
            ("Exit", lambda: self.keyboard.request_exit()),
        ])
        btns.pack(side=tk.TOP, fill=tk.X)

    # ------------------------------------------------------------------
    def run(self) -> None:
        gui = UnifiedGUI(
            self.model, self.data,
            title="Open Manipulator X",
            build_content=self._build_content,
            auto_start=False,
        )
        gui.set_reset_callback(self._reset_simulation)
        gui.start()

        self._reset_simulation()

        self.keyboard.set_reset_callback(self._reset_simulation)
        logger.info("Keyboard: %s", self.keyboard.get_keyboard_help())

        try:
            self.keyboard.set_exit_callback(lambda: logger.info("Simulation exited."))

            with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.keyboard.create_key_callback()
            ) as viewer:
                self.keyboard.set_viewer(viewer)
                viewer.cam.lookat[:] = [0.14, -0.04, 0.09]
                viewer.cam.distance = 1.25
                viewer.cam.azimuth = 83.0
                viewer.cam.elevation = -35.0
                try:
                    if hasattr(viewer, "opt") and hasattr(viewer.opt, "background_rgb"):
                        viewer.opt.background_rgb[:] = [0.45, 0.45, 0.5]
                    if hasattr(viewer, "opt") and hasattr(viewer.opt, "ambient"):
                        viewer.opt.ambient = 0.4
                except Exception:
                    pass

                while viewer.is_running() and not self.keyboard.should_exit:
                    gui.update()
                    gui.check_and_apply_pending_update()
                    gui.check_and_apply_pending_reset()

                    if self.keyboard.paused:
                        viewer.sync()
                        time.sleep(0.01)
                        continue

                    self.sync.set_speed_factor(self.keyboard.speed_factor)
                    target_t = self.sync.target_sim_time()

                    while self.data.time < target_t:
                        mujoco.mj_forward(self.model, self.data)
                        if self.gravity_compensation_enabled:
                            tau = np.array(
                                compute_gravity_vector(self.model, self.data, self.robot)[: self.model.nu],
                                dtype=float,
                            )
                        else:
                            tau = np.zeros(self.model.nu)

                        self.robot.set_torque(self.data, tau)
                        mujoco.mj_step(self.model, self.data)

                    if self.render_tick.ready(self.data.time):
                        viewer.sync()

                    time.sleep(0.0005)

        finally:
            try:
                gui.stop()
            except Exception:
                pass


# ==============================
# main
# ==============================
def main() -> None:
    app = OpenManipulatorApp()
    app.run()


if __name__ == "__main__":
    main()
