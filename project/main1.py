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

N_BOXES = 10
SPAWN_INTERVAL = 1.5
BELT_SPEED = 0.10
BELT_TOP_Z = 0.205
BELT_END_Y = -0.30
STASH_POS = np.array([0.50, 0.25, 0.40])

BASKET_XY_CENTER = np.array([0.0, 0.45])
BASKET_XY_HALF = 0.15
DETACH_EE_SPEED_MAX = 0.5


# ==============================
# Controller
# ==============================
# Conveyor / suction (use inside compute_torque):
#   self.conveyor.on_belt[i]          — True while box i is still on the belt
#   self.conveyor.box_qpos_adr[i]     — qpos index of box i; data.qpos[adr:adr+3] is its world pos
#   self.suction.attached_idx         — index of the currently held box, or None
#   self.suction.detach(data)         — release the held box
class Controller:
    def __init__(
        self,
        robot: robotics.RobotWrapper,
        model: mujoco.MjModel,
        suction: SuctionGripper,
        conveyor: ConveyorSystem,
    ) -> None:
        self.robot = robot
        self.model = model
        self.suction = suction
        self.conveyor = conveyor

    def reset(self) -> None:
        pass

    def compute_torque(self, data: mujoco.MjData, t: float) -> np.ndarray:
        g = self.robot.get_gravity_vector(data)[:ARM_DOFS]

        tau = g
        return tau


# ==============================
# Conveyor system
# ==============================
class ConveyorSystem:
    def __init__(self, model: mujoco.MjModel, n_boxes: int = N_BOXES) -> None:
        self.model = model
        self.n_boxes = n_boxes
        self.box_geom_ids: list[int] = []
        self.box_qpos_adr: list[int] = []
        self.box_qvel_adr: list[int] = []
        self.box_half_min: list[float] = []
        for i in range(n_boxes):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"box{i+1}")
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"box{i+1}_joint")
            if bid < 0 or jid < 0:
                raise ValueError(f"box{i+1} or its joint not found")
            self.box_qpos_adr.append(int(model.jnt_qposadr[jid]))
            self.box_qvel_adr.append(int(model.jnt_dofadr[jid]))
            gadr = int(model.body_geomadr[bid])
            self.box_geom_ids.append(gadr)
            self.box_half_min.append(float(np.min(model.geom_size[gadr, :3])))
        self._orig_contype = {
            gid: int(model.geom_contype[gid]) for gid in self.box_geom_ids
        }
        self._orig_conaffinity = {
            gid: int(model.geom_conaffinity[gid]) for gid in self.box_geom_ids
        }
        self.on_belt: list[bool] = [False] * n_boxes
        self._spawned: list[bool] = [False] * n_boxes
        self._next_spawn_t: float = 0.0
        self._next_box_idx: int = 0
        self._suction: SuctionGripper | None = None

    def bind_suction(self, suction: SuctionGripper) -> None:
        self._suction = suction

    def reset(self, data: mujoco.MjData) -> None:
        for i in range(self.n_boxes):
            adr = self.box_qpos_adr[i]
            data.qpos[adr : adr + 3] = STASH_POS
            data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
            v = self.box_qvel_adr[i]
            data.qvel[v : v + 6] = 0.0
            gid = self.box_geom_ids[i]
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0
        self.on_belt = [False] * self.n_boxes
        self._spawned = [False] * self.n_boxes
        self._next_spawn_t = 0.0
        self._next_box_idx = 0

    def update(self, t: float, data: mujoco.MjData) -> None:
        if self._suction is not None and self._suction.attached_idx is not None:
            self.on_belt[self._suction.attached_idx] = False

        for i in range(self.n_boxes):
            if self._spawned[i]:
                continue
            adr = self.box_qpos_adr[i]
            data.qpos[adr : adr + 3] = STASH_POS
            data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
            v = self.box_qvel_adr[i]
            data.qvel[v : v + 6] = 0.0

        if t >= self._next_spawn_t and self._next_box_idx < self.n_boxes:
            i = self._next_box_idx
            gid = self.box_geom_ids[i]
            self.model.geom_contype[gid] = self._orig_contype[gid]
            self.model.geom_conaffinity[gid] = self._orig_conaffinity[gid]
            v = self.box_qvel_adr[i]
            data.qvel[v : v + 6] = 0.0
            self._spawned[i] = True
            self.on_belt[i] = True
            self._next_box_idx += 1
            self._next_spawn_t = t + SPAWN_INTERVAL

        for i, on_belt in enumerate(self.on_belt):
            if not on_belt:
                continue
            adr = self.box_qpos_adr[i]
            if data.qpos[adr + 2] < BELT_TOP_Z + 0.10:
                v = self.box_qvel_adr[i]
                data.qvel[v + 1] = -BELT_SPEED
            if data.qpos[adr + 1] < BELT_END_Y:
                self.on_belt[i] = False


# ==============================
# Suction gripper
# ==============================
class SuctionGripper:
    def __init__(
        self,
        model: mujoco.MjModel,
        n_boxes: int = N_BOXES,
    ) -> None:
        self.model = model
        self.box_body_ids: list[int] = []
        self.box_geom_ids: list[int] = []
        self.box_qpos_adr: list[int] = []
        self.box_qvel_adr: list[int] = []
        self.weld_eq_ids: list[int] = []
        for i in range(n_boxes):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"box{i+1}")
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"box{i+1}_joint"
            )
            wid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, f"suction_box{i+1}"
            )
            if bid < 0 or jid < 0 or wid < 0:
                raise ValueError(f"box{i+1} body / joint / weld not found")
            self.box_body_ids.append(bid)
            self.box_geom_ids.append(int(model.body_geomadr[bid]))
            self.box_qpos_adr.append(int(model.jnt_qposadr[jid]))
            self.box_qvel_adr.append(int(model.jnt_dofadr[jid]))
            self.weld_eq_ids.append(wid)
        self.pad_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "suction_pad")
        if self.pad_geom_id < 0:
            raise ValueError("suction_pad geom not found in model")
        self.hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.hand_body_id < 0:
            raise ValueError("hand body not found in model")
        self._geom_to_box: dict[int, int] = {
            g: i for i, g in enumerate(self.box_geom_ids)
        }
        self._orig_contype: dict[int, int] = {
            gid: int(model.geom_contype[gid]) for gid in self.box_geom_ids
        }
        self._orig_conaffinity: dict[int, int] = {
            gid: int(model.geom_conaffinity[gid]) for gid in self.box_geom_ids
        }
        self.attached_idx: int | None = None

    def try_attach(self, data: mujoco.MjData) -> bool:
        if self.attached_idx is not None:
            return False
        for k in range(data.ncon):
            g1 = data.contact[k].geom1
            g2 = data.contact[k].geom2
            if g1 == self.pad_geom_id and g2 in self._geom_to_box:
                box_idx = self._geom_to_box[g2]
            elif g2 == self.pad_geom_id and g1 in self._geom_to_box:
                box_idx = self._geom_to_box[g1]
            else:
                continue
            if data.contact[k].dist > -0.001:
                continue

            gid = self.box_geom_ids[box_idx]
            v = self.box_qvel_adr[box_idx]
            wid = self.weld_eq_ids[box_idx]

            data.qvel[v : v + 6] = 0.0

            hand_pos = np.array(data.xpos[self.hand_body_id])
            hand_quat = np.array(data.xquat[self.hand_body_id])
            box_pos = np.array(data.xpos[self.box_body_ids[box_idx]])
            box_quat = np.array(data.xquat[self.box_body_ids[box_idx]])

            hand_mat = np.zeros(9)
            mujoco.mju_quat2Mat(hand_mat, hand_quat)
            rel_pos = hand_mat.reshape(3, 3).T @ (box_pos - hand_pos)

            hand_quat_conj = np.array(
                [hand_quat[0], -hand_quat[1], -hand_quat[2], -hand_quat[3]]
            )
            rel_quat = np.zeros(4)
            mujoco.mju_mulQuat(rel_quat, hand_quat_conj, box_quat)

            self.model.eq_data[wid, 3:6] = rel_pos
            self.model.eq_data[wid, 6:10] = rel_quat

            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0

            data.eq_active[wid] = 1
            self.attached_idx = box_idx
            return True
        return False

    def detach(self, data: mujoco.MjData) -> None:
        if self.attached_idx is None:
            return
        gid = self.box_geom_ids[self.attached_idx]
        self.model.geom_contype[gid] = self._orig_contype[gid]
        self.model.geom_conaffinity[gid] = self._orig_conaffinity[gid]
        data.eq_active[self.weld_eq_ids[self.attached_idx]] = 0
        self.attached_idx = None

    def reset(self, data: mujoco.MjData) -> None:
        for gid in self.box_geom_ids:
            self.model.geom_contype[gid] = self._orig_contype[gid]
            self.model.geom_conaffinity[gid] = self._orig_conaffinity[gid]
        for wid in self.weld_eq_ids:
            data.eq_active[wid] = 0
        self.attached_idx = None

    def update(self, data: mujoco.MjData) -> None:
        if self.attached_idx is None:
            self.try_attach(data)
            return
        adr = self.box_qpos_adr[self.attached_idx]
        bx = float(data.qpos[adr])
        by = float(data.qpos[adr + 1])
        if (
            abs(bx - BASKET_XY_CENTER[0]) < BASKET_XY_HALF
            and abs(by - BASKET_XY_CENTER[1]) < BASKET_XY_HALF
        ):
            ee_speed = float(np.linalg.norm(data.qvel[:ARM_DOFS]))
            if ee_speed < DETACH_EE_SPEED_MAX:
                self.detach(data)


# ==============================
# App wiring
# ==============================
class FinalProjectApp(HWAppBase):
    model_path = PROJECT_ROOT / "project" / "models" / "panda" / "project1.xml"
    title = "Final Project Task 1 — Pick & Place"
    site_name = "ee_site"
    camera = {
        "lookat": [0.4, -0.1, 0.3],
        "distance": 2.2,
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

        self.conveyor = ConveyorSystem(self.model)
        self.suction = SuctionGripper(self.model)
        self.conveyor.bind_suction(self.suction)
        self.controller = Controller(self.robot, self.model, self.suction, self.conveyor)

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
            filename_prefix="p1_log",
            enabled=start_recording,
        )

    def sim_step(self) -> bool:
        if self.duration is not None and self.sim_time > self.duration:
            return False
        mujoco.mj_forward(self.model, self.data)
        self.conveyor.update(self.sim_time, self.data)
        self.suction.update(self.data)
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
        self.suction.reset(self.data)
        self.conveyor.reset(self.data)
        self.controller.reset()
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
