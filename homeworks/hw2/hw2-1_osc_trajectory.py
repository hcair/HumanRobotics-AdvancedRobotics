"""HW2-1 OSC: Franka Panda operational-space control with task-space trajectory."""

import logging
import sys
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
from core.trajectory import MultiSegmentTrajectory
from core.viewer_utils import draw_trajectory_preview
from homeworks.helpers import HWAppBase, make_hw2_1_scene_monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================
# Problem constants
# ==============================
DEG2RAD = np.pi / 180.0
Q_HOME_DEG = np.array([0.0, -45.0, 0.0, -135.0, 0.0, 90.0, 45.0])

# --- BEGIN: HW2-1 Problem (d) ---
# Add the remaining task-space waypoints.
# Each waypoint should be a 3D position [x, y, z] in meters (world frame).
# The values below are arbitrary examples — replace each number with your own
# end-effector positions to design the trajectory you want.

X0 = np.array([0.30689059, 0.0, 0.5318822])  # EE position at Q_HOME (FK, do not change)
X1 = np.array([ 0.4,  0.1, 0.6])
X2 = np.array([ 0.5,  0.0, 0.5])
X3 = np.array([ 0.4, -0.2, 0.4])
# X4 = ...

WAYPOINTS_OSC = [
    X0,
    X1,
    X2,
    X3,
    # ...
    X0,
]
# --- END: HW2-1 Problem (d) ---

TRAJ_OSC = MultiSegmentTrajectory(WAYPOINTS_OSC, segment_time=3.0)

ARM_DOFS = 7
KP_OSC = np.diag([50.0, 50.0, 50.0])
KD_OSC = np.diag([10.0, 10.0, 10.0])


# ==============================
# Controller
# ==============================
class OSCController:
    def __init__(self, robot: robotics.RobotWrapper, model: mujoco.MjModel) -> None:
        self.robot = robot
        self.model = model

    def compute_torque(
        self, data: mujoco.MjData, xp_des: np.ndarray, dxp_des: np.ndarray, ddxp_des: np.ndarray
    ) -> np.ndarray:
        xp_ee = self.robot.get_ee_pose(data)[0].copy()
        _, Jv_ee, _ = self.robot.get_ee_jacobian(data, translational=True, rotational=False)
        Jv_ee = Jv_ee[:, :ARM_DOFS]
        dq = data.qvel[:ARM_DOFS].copy()

        M = self.robot.get_inertia_matrix(data)[:ARM_DOFS, :ARM_DOFS]
        c = self.robot.get_coriolis_centrifugal(data)[:ARM_DOFS]
        g = self.robot.get_gravity_vector(data)[:ARM_DOFS]

        # Operational-space inertia matrix Lambda for the 3D translational task.
        # The (epsilon * I) damping keeps the inverse well-conditioned near singularities.
        Jv_M_inv_JvT = Jv_ee @ np.linalg.solve(M, Jv_ee.T)        # = Jv M^-1 Jv^T   (3x3)
        Lambda = np.linalg.inv(Jv_M_inv_JvT + 1e-2 * np.eye(3))   # = (Jv M^-1 Jv^T + epsilon * I)^-1

        # --- BEGIN: HW2-1 Problem (e) ---
        # Compute:
        #   1) the desired task-space acceleration ddx_star using PD tracking
        #      on (xp_des, dxp_des, ddxp_des) vs. the current (xp_ee, Jv_ee @ dq),
        #      with the position gain KP_OSC and the velocity gain KD_OSC,
        #   2) the control torque tau using operational-space formulation
        #      (combine Jv_ee, Lambda, ddx_star, and add c, g for gravity/Coriolis compensation).
        # Shapes: ddx_star is (3,), tau is (ARM_DOFS,).
        
        ddx_star = np.zeros(3, dtype=float)
        tau = np.zeros(ARM_DOFS, dtype=float)
        # --- END: HW2-1 Problem (e) ---

        return tau


# ==============================
# Demo wiring
# ==============================
class OSCApp(HWAppBase):
    model_path = PROJECT_ROOT / "models" / "franka_emika_panda" / "hw2_1.xml"
    title = "HW2-1 OSC Trajectory"
    camera = {"lookat": [0.0, 0.0, 0.0], "distance": 3.0, "azimuth": -180.0, "elevation": -25.0}

    def __init__(self) -> None:
        super().__init__()
        self.q_home = Q_HOME_DEG * DEG2RAD
        self.arm_dofs = ARM_DOFS
        self.finger_open_position = 0.04
        self.controller = OSCController(self.robot, self.model)
        self.scene = make_hw2_1_scene_monitor(self.model)

    def sim_step(self) -> bool:
        if self.sim_time > TRAJ_OSC.total_time:
            return False
        mujoco.mj_forward(self.model, self.data)
        xp_des, dxp_des, ddxp_des = TRAJ_OSC.evaluate(self.sim_time)
        tau = self.controller.compute_torque(self.data, xp_des, dxp_des, ddxp_des)
        self.robot.set_torque(self.data, tau)
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt
        self.scene.update(self.data, self.robot.get_ee_pose(self.data)[0])
        return True

    def on_reset(self) -> None:
        self.scene.reset()

    def visualize_trajectory(self, viewer) -> None:
        draw_trajectory_preview(viewer, TRAJ_OSC)


# ==============================
# main
# ==============================
def main() -> None:
    OSCApp().run()


if __name__ == "__main__":
    main()
