"""HW2-2: Franka Panda unified motion/force control via selection matrix."""

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
from core.signal import Signal
from core.trajectory import MultiSegmentTrajectory
from core.viewer_utils import (
    StrokeRecorder,
    draw_status_markers,
    draw_trajectory_preview,
)
from homeworks.helpers import (
    CHECKPOINT_PALETTE,
    HW22AutoForceToggle,
    HWAppBase,
    make_hw2_2_force_grading,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Problem constants
# ==============================
DEG2RAD = np.pi / 180.0
Q_HOME_DEG = np.array([0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0])

Z_HEIGHT = 0.3  # Writing height (world z) above table top

# --- BEGIN: HW2-2 Problem (a) ---
# Task-space trajectory: end-effector path spelling "ROBOT" at height Z_HEIGHT.
# Each waypoint is Pi = np.array([x, y, z]). Build WAYPOINTS_OSC starting from home P0,
# tracing R-O-B-O-T, then closing by returning to P0.
#
# SEGMENT_TIMES_OSC: numpy array of length (len(WAYPOINTS_OSC) - 1), one duration
# per segment. Pass both into MultiSegmentTrajectory.

P0 = np.array([0.30689059, 0.0, 0.5318822])   # home pose (EE at Q_HOME via FK)
P1 = np.array([0.3, -0.35, Z_HEIGHT])         # R start (first checkpoint, auto force ON)
P2 = np.array([0.45, -0.2, Z_HEIGHT])         # arbitrary intermediate sample
P3 = np.array([0.5, 0.3, Z_HEIGHT])           # T end (last checkpoint, auto force OFF)

WAYPOINTS_OSC = [
    P0, 
    P1, P2, P3, 
    P0
]

SEGMENT_TIMES_OSC = np.array([
    2.5,
    4.0, 4.0,
    2.5
])

TRAJ_OSC = MultiSegmentTrajectory(WAYPOINTS_OSC, SEGMENT_TIMES_OSC)
# --- END: HW2-2 Problem (a) ---

ARM_DOFS = 7

R_DES = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)  # Pencil tip pointing down toward table


@dataclass(frozen=True)
class TaskGains:
    kp_pos: float = 100.0
    kd_pos: float = 40.0
    kp_ori: float = 50.0
    kd_ori: float = 20.0
    kp_posture: float = 20.0
    kd_posture: float = 10.0


@dataclass(frozen=True)
class ForceControlConfig:
    kp: float = 0.5
    ki: float = 1.5
    fz_desired: float = -5.0
    sensor_filter_alpha: float = 0.98
    err_integral_max: float = 30.0


TASK_GAINS = TaskGains()
FORCE_CFG = ForceControlConfig()

# ==============================
# Controller
# ==============================
class UMFCController:
    def __init__(
        self,
        robot: robotics.RobotWrapper,
        model: mujoco.MjModel,
        q_home: np.ndarray,
        ee_force_sensor_adr: int | None = None,
    ) -> None:
        self.robot = robot
        self.model = model
        self.q_home = np.array(q_home, dtype=float)
        self.ee_force_sensor_adr = ee_force_sensor_adr

        self.force_control_mode = False
        self.Fz_filtered: float | None = None
        self.Fz_err_integral = 0.0

    def reset_force_estimator(self) -> None:
        self.Fz_filtered = None
        self.Fz_err_integral = 0.0

    def get_Fz_filtered(self) -> float | None:
        return self.Fz_filtered

    def toggle_force_control_mode(self) -> None:
        self.force_control_mode = not self.force_control_mode
        logger.info(
            "Force control mode: %s",
            "ON" if self.force_control_mode else "OFF",
        )

    def get_force_measurements(self, data: mujoco.MjData) -> np.ndarray:
        F_sensor_world = np.zeros(6)
        if self.ee_force_sensor_adr is None:
            return F_sensor_world

        # Local 3D contact force at EE; rotate to world. F_sensor_world has shape (6,) [Fx, Fy, Fz, Mx, My, Mz]; only Fz is filled.
        force_local = np.array(
            [
                data.sensordata[self.ee_force_sensor_adr + 0],
                data.sensordata[self.ee_force_sensor_adr + 1],
                data.sensordata[self.ee_force_sensor_adr + 2],
            ],
            dtype=float,
        )

        R_ee = self.robot.get_ee_pose(data)[1]
        force_world = R_ee @ force_local

        Fz_world_raw = force_world[2]

        if self.Fz_filtered is None:
            self.Fz_filtered = float(Fz_world_raw)
        else:
            self.Fz_filtered = float(
                FORCE_CFG.sensor_filter_alpha * self.Fz_filtered
                + (1.0 - FORCE_CFG.sensor_filter_alpha) * Fz_world_raw
            )

        F_sensor_world[2] = self.Fz_filtered
        return F_sensor_world

    def compute_torque(
        self,
        data: mujoco.MjData,
        xp_des: np.ndarray,
        dxp_des: np.ndarray,
        ddxp_des: np.ndarray,
        R_des: np.ndarray,
        dxr_des: np.ndarray,
    ) -> np.ndarray:
        q = data.qpos[:ARM_DOFS].copy()
        dq = data.qvel[:ARM_DOFS].copy()
        xp_ee, R_ee = self.robot.get_ee_pose(data)
        _, Jv_ee, Jw_ee = self.robot.get_ee_jacobian(
            data, translational=True, rotational=True
        )
        Jv_ee = Jv_ee[:, :ARM_DOFS]
        Jw_ee = Jw_ee[:, :ARM_DOFS]
        J_ee = np.vstack([Jv_ee, Jw_ee])
        M = self.robot.get_inertia_matrix(data)[:ARM_DOFS, :ARM_DOFS]
        g = self.robot.get_gravity_vector(data)[:ARM_DOFS]
        c = self.robot.get_coriolis_centrifugal(data)[:ARM_DOFS]

        dxp_ee = Jv_ee @ dq
        dxr_ee = Jw_ee @ dq
        e_xp = xp_des - xp_ee
        e_dxp = dxp_des - dxp_ee
        e_phi = orientation_error(R_des, R_ee)
        e_dxr = dxr_des - dxr_ee

        # --- BEGIN: HW2-2 Problem (b) ---
        # Compute:
        #   1) the desired translational acceleration ddxp_star using PD tracking,
        #   2) the desired rotational acceleration ddxr_star using PD tracking,
        #   3) the task-space motion command F_motion_star (6,) combining ddxp_star
        #      and ddxr_star into a single 6D vector.
        # Use module gains TASK_GAINS.kp_pos, TASK_GAINS.kd_pos, TASK_GAINS.kp_ori, TASK_GAINS.kd_ori.
        
        F_motion_star = np.zeros(6, dtype=float)
        # --- END: HW2-2 Problem (b) ---


        dt = float(self.model.opt.timestep)
        F_sensor = self.get_force_measurements(data)
        Fz_err = FORCE_CFG.fz_desired - F_sensor[2]
        if self.force_control_mode:
            self.Fz_err_integral += Fz_err * dt
            self.Fz_err_integral = np.clip(
                self.Fz_err_integral,
                -FORCE_CFG.err_integral_max,
                FORCE_CFG.err_integral_max,
            )


        # --- BEGIN: HW2-2 Problem (c) ---
        # F_sensor has shape (6,) [Fx, Fy, Fz, Mx, My, Mz]; only Fz is populated.
        # Compute:
        #   1) selection matrices SIGMA (6x6) and SIGMA_BAR = I - SIGMA,
        #      with z-axis released (SIGMA[2,2]=0) when self.force_control_mode is True, else SIGMA = I.
        #   2) the task-space force F_force_star (6,): only the z component is non-zero.
        #      Combine the feedforward FORCE_CFG.fz_desired with PI feedback on the force error
        #      (FORCE_CFG.kp on Fz_err, FORCE_CFG.ki on self.Fz_err_integral).
        #   3) the combined task-space command F_cmd (6,): superpose the motion command
        #      projected through SIGMA with the force command projected through SIGMA_BAR.
        
        F_cmd = np.zeros(6, dtype=float)
        # --- END: HW2-2 Problem (c) ---


        # --- BEGIN: HW2-2 Problem (d) ---
        # Compute:
        #   1) operational-space inertia matrix Lambda (6x6) for the 6D task.
        #      Add a small damping term (epsilon * I) when inverting J M^-1 J^T
        #      to stay well-conditioned near singular configurations.
        #   2) task-space torque tau_task from Lambda and F_cmd.
        #
        # Hint: prefer np.linalg.solve(M, X) over np.linalg.inv(M) @ X for numerical stability.
        
        tau_task = np.zeros(ARM_DOFS, dtype=float)
        # --- END: HW2-2 Problem (d) ---


        # tau_posture: joint-space PD posture torque toward self.q_home.
        tau_posture = TASK_GAINS.kp_posture * (self.q_home - q) - TASK_GAINS.kd_posture * dq


        # --- BEGIN: HW2-2 Problem (e) ---
        # Compute:
        #   1) dynamically consistent null-space projector N_task,
        #   2) total tau: superpose the task-space torque with the posture torque
        #      projected into the null-space of the task, and add gravity/Coriolis
        #      compensation.
        
        tau = np.zeros(ARM_DOFS, dtype=float)
        # --- END: HW2-2 Problem (e) ---

        return tau


# ==============================
# Demo wiring
# ==============================
class UMFCApp(HWAppBase):
    model_path = PROJECT_ROOT / "models" / "franka_emika_panda" / "hw2_2.xml"
    title = "HW2-2 Unified Motion/Force Control"
    camera = {"lookat": [0.0, 0.1, 0.1], "distance": 2.0, "azimuth": 155.0, "elevation": -24.0}

    def __init__(self) -> None:
        super().__init__()
        self.q_home = Q_HOME_DEG * DEG2RAD
        self.arm_dofs = ARM_DOFS
        self.finger_open_position = 0.04

        ee_force_sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_force"
        )
        self.ee_force_sensor_adr = (
            int(self.model.sensor_adr[ee_force_sensor_id])
            if ee_force_sensor_id >= 0
            else None
        )
        self.controller = UMFCController(
            self.robot, self.model, self.q_home, ee_force_sensor_adr=self.ee_force_sensor_adr,
        )

        self.fz_signal = Signal(
            name="Fz [N]",
            getter=lambda _d: np.array([0.0 if self.controller.get_Fz_filtered() is None
                                        else self.controller.get_Fz_filtered()]),
            ylim_min=-7.0,
            ylim_max=-3.0,
            reference=FORCE_CFG.fz_desired,
            reference_label=f"Fz_desired = {FORCE_CFG.fz_desired}",
        )
        self.plot_api = None

        self.trace = StrokeRecorder()
        self._desired_geom_count = 0

        self.checkpoints, self._cp_markers = make_hw2_2_force_grading(Z_HEIGHT)
        self.auto_force = HW22AutoForceToggle()

    def sim_step(self) -> bool:
        if self.sim_time > TRAJ_OSC.total_time:
            if not self.checkpoints.finalized:
                self.checkpoints.report(logger)
            return False
        mujoco.mj_forward(self.model, self.data)
        xp_des, dxp_des, ddxp_des = TRAJ_OSC.evaluate(self.sim_time)
        tau = self.controller.compute_torque(
            self.data, xp_des, dxp_des, ddxp_des, R_DES, np.zeros(3, dtype=float),
        )
        self.robot.set_torque(self.data, tau)
        if self.plot_api is not None:
            self.plot_api.push(self.data.time, self.data)
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt
        ee_pos = self.robot.get_ee_pose(self.data)[0]
        self.auto_force.update(
            ee_pos,
            self.controller.force_control_mode,
            self.controller.toggle_force_control_mode,
        )
        self.checkpoints.update(ee_pos, self.controller.get_Fz_filtered())
        return True

    def on_reset(self) -> None:
        self.controller.reset_force_estimator()
        self.controller.force_control_mode = False
        self.trace.clear()
        self.checkpoints.clear()
        self.auto_force.clear()
        if self.plot_api is not None:
            self.plot_api.clear()

    def visualize_trajectory(self, viewer) -> None:
        draw_trajectory_preview(viewer, TRAJ_OSC)
        self._desired_geom_count = viewer.user_scn.ngeom

    def pre_render(self) -> None:
        ee_pos = self.robot.get_ee_pose(self.data)[0]
        self.trace.sample(self.controller.force_control_mode, ee_pos)
        if self.plot_api is not None:
            self.plot_api.update()

    def draw_overlay(self, viewer) -> None:
        viewer.user_scn.ngeom = self._desired_geom_count
        draw_status_markers(
            viewer, self._cp_markers, self.checkpoints.states(), CHECKPOINT_PALETTE, size=0.015,
        )
        self.trace.draw(viewer)

    def register_keys(self) -> None:
        self.keyboard.register_callback("f", self.controller.toggle_force_control_mode)

    def build_gui_content(self, parent: tk.Widget, api) -> None:
        if self.ee_force_sensor_adr is not None:
            plot_frame, self.plot_api = api.add_plot_area(
                parent,
                signals=[self.fz_signal],
                dt=self.dt,
                window_sec=10.0,
                figsize=(6, 3),
            )
            plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        btns = api.add_button_row(parent, [
            ("Toggle Force Mode (F)", self.controller.toggle_force_control_mode),
            ("Reset", self.base_reset),
            ("Exit", lambda: self.keyboard.request_exit()),
        ])
        btns.pack(side=tk.TOP, fill=tk.X)


# ==============================
# main
# ==============================
def main() -> None:
    UMFCApp().run()


if __name__ == "__main__":
    main()
