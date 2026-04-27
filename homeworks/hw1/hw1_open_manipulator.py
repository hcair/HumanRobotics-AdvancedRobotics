"""HW1: Open Manipulator X — Jacobian-based gravity compensation."""

import logging
import sys
import tkinter as tk
from pathlib import Path

import mujoco
import numpy as np

_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "core").is_dir():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
PROJECT_ROOT = _p

from core import robotics, make_qpos_parameter_from_model
from homeworks.helpers import HWAppBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Problem constants
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
        _, Jv, _ = robot.get_body_jacobian(
            data, body_name, translational=True, rotational=False
        )
        m = model.body_mass[body_id]
        # --- BEGIN: HW1 Problem 2(e) — modify the block below (gravity compensation) ---
        # Replace the placeholder with the formula from part (d).
        tau += 0  # placeholder
        # --- END: HW1 Problem 2(e) ---
    return tau


# ==============================
# Demo wiring
# ==============================
class OpenManipulatorApp(HWAppBase):
    model_path = PROJECT_ROOT / "models" / "robotis_open_manipulator_x" / "open_manipulator_x_simple.xml"
    title = "Open Manipulator X"
    camera = {"lookat": [0.14, -0.04, 0.09], "distance": 1.25, "azimuth": 83.0, "elevation": -35.0}

    def __init__(self) -> None:
        super().__init__()
        nq = self.model.nq
        self.q_home = (INITIAL_QPOS_DEG[:nq] * DEG2RAD).astype(float)
        self.gravity_compensation_enabled = True

    def sim_step(self) -> bool:
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
        return True

    # ---- GUI button callbacks ----
    def _toggle_gravity_compensation(self) -> None:
        self.gravity_compensation_enabled = not self.gravity_compensation_enabled
        logger.info("Gravity compensation: %s", "ON" if self.gravity_compensation_enabled else "OFF")

    def _print_dynamics(self, which: str) -> None:
        np.set_printoptions(precision=4, suppress=True, linewidth=120)
        mujoco.mj_forward(self.model, self.data)
        if which == "ee_pose":
            pos, R = self.robot.get_ee_pose(self.data)
            logger.info("\n=== End-effector pose (FK) ===\nPosition p (3,): %s\nRotation matrix R (3x3):\n%s\n\n", pos, R)
        elif which == "jac":
            jac_bodies = [
                ("link2", "J1"), ("link3", "J2"), ("link4", "J3"),
                ("link5", "J4"), ("end_effector", "Jee"),
            ]
            logger.info("\n=== Jacobian per body ===\n")
            for body_name, label in jac_bodies:
                J_full, _, _ = self.robot.get_body_jacobian(
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

    def build_gui_content(self, parent: tk.Widget, api) -> None:
        status_param = make_qpos_parameter_from_model(
            self.model, default=list(self.q_home), name="Current q (Status)",
        )
        status_param.read_only = True
        status_param.setter = None
        status_param.getter = lambda m, d: list(d.qpos)
        status_panel = api.add_status_panel(parent, [status_param], self.model, self.data)
        status_panel.pack(side=tk.TOP, fill=tk.X)

        cmd_params = [
            make_qpos_parameter_from_model(
                self.model, default=list(self.q_home), name="Target q (Command)",
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


# ==============================
# main
# ==============================
def main() -> None:
    OpenManipulatorApp().run()


if __name__ == "__main__":
    main()
