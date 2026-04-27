"""HW2-3: Franka Panda collision avoidance with null-space task priority."""

import logging
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_p = Path(__file__).resolve().parent
while _p != _p.parent and not (_p / "core").is_dir():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
PROJECT_ROOT = _p

from core import robotics
from core.math_utils import orientation_error
from core.viewer_utils import draw_frame_at
from homeworks.helpers import AvoidanceProbe, HWAppBase, MovableObstacle, ObstacleAutoMover

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Problem constants
# ==============================
DEG2RAD = np.pi / 180.0
Q_HOME_DEG = np.array([0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0])

ARM_DOFS = 7
OBSTACLE_BODY_NAME = "ball"
OBSTACLE_NUDGE_STEP = 0.02


@dataclass(frozen=True)
class TaskGains:
    kp_pos: float = 50.0
    kd_pos: float = 20.0
    kp_ori: float = 50.0
    kd_ori: float = 20.0
    kp_posture: float = 20.0
    kd_posture: float = 10.0


@dataclass(frozen=True)
class AvoidanceConfig:
    obs_radius: float = 0.05
    soft_margin: float = 0.22  # soft avoidance when penetration measure <= this (larger = earlier)
    avoid_margin: float = 0.2
    kp_hard: float = 300.0
    kd_hard: float = 40.0
    kp_soft: float = 80.0
    kd_soft: float = 20.0


TASK_GAINS = TaskGains()
AVOID_CFG = AvoidanceConfig()


# ==============================
# Controller
# ==============================
class CAController:
    def __init__(
        self,
        robot: robotics.RobotWrapper,
        model: mujoco.MjModel,
        q_home: np.ndarray,
        ball_body_id: int,
    ) -> None:
        self.robot = robot
        self.model = model
        self.q_home = np.array(q_home, dtype=float)
        self.ball_body_id = ball_body_id
        self.avoidance = AvoidanceProbe(
            robot, model,
            arm_dofs=ARM_DOFS,
            safety_margin=AVOID_CFG.avoid_margin + AVOID_CFG.obs_radius,
        )
        self.last_avoid_info: tuple[int, bool] | None = None
        self._last_avoid_time: float = -1.0

    def compute_torque(
        self,
        data: mujoco.MjData,
        xp_des: np.ndarray,
        dxp_des: np.ndarray,
        R_des: np.ndarray,
        dxr_des: np.ndarray,
    ) -> np.ndarray:
        q = data.qpos[:ARM_DOFS].copy()
        dq = data.qvel[:ARM_DOFS].copy()
        M = self.robot.get_inertia_matrix(data)[:ARM_DOFS, :ARM_DOFS]
        g = self.robot.get_gravity_vector(data)[:ARM_DOFS]
        c = self.robot.get_coriolis_centrifugal(data)[:ARM_DOFS]

        # ---- Avoidance setup ----
        # Pick the body closest to the obstacle and build the 1D constraint quantities.
        #   delta = ||p_body - p_obs|| - safety_margin   (signed clearance, < 0 = penetrating)
        #   n     = (p_body - p_obs) / ||...||           (unit vector, obstacle -> body)
        #   Jv    = body's 3D translational Jacobian     (3, ARM_DOFS)
        obs_pos = data.xpos[self.ball_body_id].copy() if self.ball_body_id >= 0 else None
        nearest_body, delta, n, Jv = self.avoidance.query(data, obs_pos)
        if nearest_body is not None and delta <= AVOID_CFG.soft_margin:
            # 3D -> 1D projection. Only motion along n changes the clearance phi(q),
            # so project the 3D body Jacobian onto n as a row vector:
            #     J_avoid = n^T @ Jv      (1, ARM_DOFS)   — 1D constraint Jacobian
            #     dphi/dt = J_avoid @ dq  (scalar)        — clearance change rate
            J_avoid = (n @ Jv).reshape(1, ARM_DOFS)

            # Approach speed = max(-dphi/dt, 0) — only damp when getting closer.
            phi_dot = float(J_avoid @ dq)
            approach_speed = max(-phi_dot, 0.0)

            # Scalar repulsive force along n, packed as a (1,) vector parallel to F_cmd in hw2-2.
            #   Hard branch (delta <= 0, penetrating): position push + damping.
            #     f_avoid = kp_hard * (-delta) + kd_hard * approach_speed
            #   Soft branch (delta > 0, inside soft margin): damping only.
            if delta <= 0.0:
                f_avoid = AVOID_CFG.kp_hard * (-delta) + AVOID_CFG.kd_hard * approach_speed
            else:
                f_avoid = AVOID_CFG.kd_soft * approach_speed

            self.last_avoid_info = (self.avoidance.body_ids[nearest_body], delta <= 0.0)
            self._last_avoid_time = data.time
        else:
            J_avoid = np.zeros((1, ARM_DOFS), dtype=float)
            f_avoid = 0.0
        F_avoid = np.array([f_avoid])  # (1,) force vector along n

        if self._last_avoid_time >= 0 and (data.time - self._last_avoid_time) > 0.15:
            self.last_avoid_info = None
            self._last_avoid_time = -1.0

        # --- BEGIN: HW2-3 Problem (a) ---
        # Compute tau_avoid from the avoidance Jacobian J_avoid (1, ARM_DOFS) and the
        # repulsive force vector F_avoid (1,).

        tau_avoid = np.zeros(ARM_DOFS, dtype=float)
        # --- END: HW2-3 Problem (a) ---


        # --- BEGIN: HW2-3 Problem (b) ---
        # Build the avoidance null-space projector N1 (dynamically consistent).

        N1 = np.zeros((ARM_DOFS, ARM_DOFS), dtype=float)
        # --- END: HW2-3 Problem (b) ---


        # EE task setup: pose and the full 6D Jacobian (translational + rotational stacked).
        xp_ee, R_ee = self.robot.get_ee_pose(data)
        _, Jv_ee, Jw_ee = self.robot.get_ee_jacobian(
            data, translational=True, rotational=True
        )
        Jv_ee = Jv_ee[:, :ARM_DOFS]
        Jw_ee = Jw_ee[:, :ARM_DOFS]
        J_ee = np.vstack([Jv_ee, Jw_ee])


        # --- BEGIN: HW2-3 Problem (c) ---
        # Compute the desired tracking task acceleration acc_track (6,) via PD on
        # translational and rotational pose errors (use orientation_error for the
        # rotational term).

        acc_track = np.zeros(6, dtype=float)
        # --- END: HW2-3 Problem (c) ---


        # --- BEGIN: HW2-3 Problem (d) ---
        # Project the EE Jacobian into the null-space of avoidance to obtain the
        # tracking task Jacobian J_track. Then compute its operational-space inertia
        # Lambda_track and the tracking task torque tau_track from acc_track.

        tau_track = np.zeros(ARM_DOFS, dtype=float)
        # --- END: HW2-3 Problem (d) ---


        # --- BEGIN: HW2-3 Problem (e) ---
        # Build the null-space projector N2 of the projected tracking task.

        N2 = np.zeros((ARM_DOFS, ARM_DOFS), dtype=float)
        # --- END: HW2-3 Problem (e) ---


        # tau_posture: joint-space PD posture torque toward self.q_home.
        tau_posture = TASK_GAINS.kp_posture * (self.q_home - q) - TASK_GAINS.kd_posture * dq


        # --- BEGIN: HW2-3 Problem (f) ---
        # Synthesize the total torque tau with hierarchical priority
        # (avoidance > tracking > posture) using tau_avoid, tau_track, tau_posture,
        # the null-space projectors N1, N2, and gravity/Coriolis compensation.

        tau = np.zeros(ARM_DOFS, dtype=float)
        # --- END: HW2-3 Problem (f) ---

        return tau


# ==============================
# Demo wiring
# ==============================
class CollisionAvoidanceApp(HWAppBase):
    model_path = PROJECT_ROOT / "models" / "franka_emika_panda" / "hw2_3.xml"
    title = "HW2-3 Collision Avoidance"
    camera = {"lookat": [0.2, 0.0, 0.3], "distance": 2.0, "azimuth": 140.0, "elevation": -40.0}

    def __init__(self) -> None:
        super().__init__()
        self.q_home = Q_HOME_DEG * DEG2RAD
        self.arm_dofs = ARM_DOFS
        self.finger_open_position = 0.04

        self.obstacle = MovableObstacle(
            self.model, body_name=OBSTACLE_BODY_NAME, radius=AVOID_CFG.obs_radius,
        )
        self.auto_mover = ObstacleAutoMover()
        self.auto_mode = False
        self.controller = CAController(self.robot, self.model, self.q_home, self.obstacle.body_id)

        self.xp_des = np.zeros(3, dtype=float)
        self.R_des = np.eye(3, dtype=float)

    def sim_step(self) -> bool:
        if self.auto_mode:
            # Drive obstacle by the auto sinusoidal pattern; apply() updates the model
            # and runs forward kinematics so the controller sees the new position.
            self.obstacle.pos = self.auto_mover.position_at(self.data.time)
            self.obstacle.apply(self.data)
        else:
            mujoco.mj_forward(self.model, self.data)
        tau = self.controller.compute_torque(
            self.data,
            self.xp_des,
            np.zeros(3, dtype=float),
            self.R_des,
            np.zeros(3, dtype=float),
        )
        self.robot.set_torque(self.data, tau)
        mujoco.mj_step(self.model, self.data)
        return True

    def on_reset(self) -> None:
        self.obstacle.reset(self.data)
        p_ee, R_ee = self.robot.get_ee_pose(self.data)
        self.xp_des[:] = p_ee
        self.R_des[:, :] = R_ee
        self.auto_mode = False

    def _toggle_auto_mode(self) -> None:
        if not self.auto_mode:
            # Reset robot, then center the auto-motion on the EE position at the home
            # pose (so the obstacle oscillates around where the robot actually is).
            # Start the sin pattern 1.0 s into its phase so the obstacle is already
            # offset from the EE at the moment auto mode kicks in (otherwise base+sin(0)
            # would put the obstacle exactly on top of the EE).
            self.base_reset()
            p_ee, _ = self.robot.get_ee_pose(self.data)
            self.auto_mover.start(p_ee.copy(), t0=self.data.time - 1.0)
            self.obstacle.pos = self.auto_mover.position_at(self.data.time)
            self.obstacle.apply(self.data)
            self.auto_mode = True
            logger.info("Auto mode: ON (obstacle moves automatically around EE home)")
        else:
            self.auto_mode = False
            logger.info("Auto mode: OFF (manual nudge)")

    def draw_overlay(self, viewer) -> None:
        info = self.controller.last_avoid_info
        self.obstacle.draw(
            viewer, self.data,
            highlight_body_id=info[0] if info is not None else None,
            highlight_is_hard=info[1] if info is not None else False,
        )
        # EE target pose: shows where the operational-space EE task is trying to hold
        # (RGB axes = X/Y/Z of R_des at xp_des).
        draw_frame_at(viewer, self.xp_des, self.R_des)

    def build_gui_content(self, parent: tk.Widget, api) -> None:
        step = OBSTACLE_NUDGE_STEP
        nudge_btns = api.add_button_row(
            parent,
            [
                ("+X", lambda: self.obstacle.nudge(np.array([step, 0.0, 0.0]), self.data)),
                ("-X", lambda: self.obstacle.nudge(np.array([-step, 0.0, 0.0]), self.data)),
                ("+Y", lambda: self.obstacle.nudge(np.array([0.0, step, 0.0]), self.data)),
                ("-Y", lambda: self.obstacle.nudge(np.array([0.0, -step, 0.0]), self.data)),
                ("+Z", lambda: self.obstacle.nudge(np.array([0.0, 0.0, step]), self.data)),
                ("-Z", lambda: self.obstacle.nudge(np.array([0.0, 0.0, -step]), self.data)),
            ],
        )
        nudge_btns.pack(side=tk.TOP, fill=tk.X)

        ctrl_btns = api.add_button_row(
            parent,
            [
                ("Auto / Manual", self._toggle_auto_mode),
                ("Reset simulation", self.base_reset),
                ("Exit", lambda: self.keyboard.request_exit()),
            ],
        )
        ctrl_btns.pack(side=tk.TOP, fill=tk.X)


# ==============================
# main
# ==============================
def main() -> None:
    CollisionAvoidanceApp().run()


if __name__ == "__main__":
    main()
