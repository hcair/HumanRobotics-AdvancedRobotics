"""HW2-1 JSC: Franka Panda joint-space control with joint-space trajectory."""

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

# --- BEGIN: HW2-1 Problem (b) ---
# Add the remaining joint-space waypoints in degrees.
# Each waypoint should be a 7-dimensional joint vector [q1, q2, q3, q4, q5, q6, q7].
# The values below are arbitrary examples — replace each number with your own
# joint angles (degrees) to design the trajectory you want.

Q1 = np.array([10.0, -30.0,  0.0,  -45.0,  0.0, 60.0, 0.0])
Q2 = np.array([25.0, -20.0,  5.0,  -85.0,  0.0, 70.0, 0.0])
Q3 = np.array([-15.0,-50.0, -5.0, -100.0,  5.0, 55.0, 0.0])
# Q4 = ...

WAYPOINTS_JSC = [
    Q_HOME_DEG * DEG2RAD,
    Q1 * DEG2RAD,
    Q2 * DEG2RAD,
    Q3 * DEG2RAD,
    # ...
    Q_HOME_DEG * DEG2RAD,
]
# --- END: HW2-1 Problem (b) ---

TRAJ_JSC = MultiSegmentTrajectory(WAYPOINTS_JSC, segment_time=3.0)

ARM_DOFS = 7
KP_JSC = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
KD_JSC = np.diag([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])


# ==============================
# Controller
# ==============================
class JSCController:
    def __init__(self, robot: robotics.RobotWrapper, model: mujoco.MjModel) -> None:
        self.robot = robot
        self.model = model

    def compute_torque(self, data: mujoco.MjData, q_des: np.ndarray, dq_des: np.ndarray, ddq_des: np.ndarray) -> np.ndarray:
        q = data.qpos[:ARM_DOFS].copy()
        dq = data.qvel[:ARM_DOFS].copy()

        M = self.robot.get_inertia_matrix(data)[:ARM_DOFS, :ARM_DOFS]
        c = self.robot.get_coriolis_centrifugal(data)[:ARM_DOFS]
        g = self.robot.get_gravity_vector(data)[:ARM_DOFS]

        # --- BEGIN: HW2-1 Problem (c) ---
        # Compute:
        #   1) the desired joint acceleration ddq_star using PD tracking
        #      on (q_des, dq_des, ddq_des) vs. the current (q, dq),
        #      with the position gain KP_JSC and the velocity gain KD_JSC,
        #   2) the control torque tau using joint-space computed torque control
        #      (combine M, ddq_star, and add c, g for gravity/Coriolis compensation).
        # Shapes: ddq_star and tau are (ARM_DOFS,) vectors.
        
        ddq_star = np.zeros(ARM_DOFS, dtype=float)
        tau = np.zeros(ARM_DOFS, dtype=float)
        # --- END: HW2-1 Problem (c) ---

        return tau


# ==============================
# Demo wiring
# ==============================
class JSCApp(HWAppBase):
    model_path = PROJECT_ROOT / "models" / "franka_emika_panda" / "hw2_1.xml"
    title = "HW2-1 JSC Trajectory"
    camera = {"lookat": [0.0, 0.0, 0.0], "distance": 3.0, "azimuth": -180.0, "elevation": -25.0}

    def __init__(self) -> None:
        super().__init__()
        self.q_home = Q_HOME_DEG * DEG2RAD
        self.arm_dofs = ARM_DOFS
        self.finger_open_position = 0.04
        self.controller = JSCController(self.robot, self.model)
        self.scene = make_hw2_1_scene_monitor(self.model)

    def sim_step(self) -> bool:
        if self.sim_time > TRAJ_JSC.total_time:
            return False
        mujoco.mj_forward(self.model, self.data)
        q_des, dq_des, ddq_des = TRAJ_JSC.evaluate(self.sim_time)
        tau = self.controller.compute_torque(self.data, q_des, dq_des, ddq_des)
        self.robot.set_torque(self.data, tau)
        mujoco.mj_step(self.model, self.data)
        self.sim_time += self.dt
        self.scene.update(self.data, self.robot.get_ee_pose(self.data)[0])
        return True

    def on_reset(self) -> None:
        self.scene.reset()

    def visualize_trajectory(self, viewer) -> None:
        def fk(q):
            self.data.qpos[:ARM_DOFS] = q
            mujoco.mj_forward(self.model, self.data)
            return self.robot.get_ee_pose(self.data)[0]
        draw_trajectory_preview(viewer, TRAJ_JSC, point_fn=fk)


# ==============================
# main
# ==============================
def main() -> None:
    JSCApp().run()


if __name__ == "__main__":
    main()
