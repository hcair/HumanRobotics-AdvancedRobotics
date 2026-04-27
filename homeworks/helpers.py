"""Shared utilities for the homework demo apps: app base class, scene checks,
trackers, visualization helpers, and pre-configured tasks."""

from __future__ import annotations

import logging
import time
import tkinter as tk
from pathlib import Path
from typing import Any, Callable, Sequence

import mujoco
import mujoco.viewer
import numpy as np

from core import load_model, KeyboardHandler, robotics, UnifiedGUI
from core.realtime import RateLimiter, RealtimeConfig, RealtimeSync
from core.viewer_utils import draw_arrow, draw_wireframe_sphere

logger = logging.getLogger(__name__)

# Standard Franka Panda link names; override in SimMonitor if using a different robot.
PANDA_ROBOT_BODIES: tuple[str, ...] = (
    "link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7",
    "hand", "left_finger", "right_finger",
)

# Subset used as movable obstacle-avoidance bodies (excludes the fixed base link0
# and the finger sub-bodies which already share the hand's location).
PANDA_AVOIDANCE_BODIES: tuple[str, ...] = (
    "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand",
)

# Palette matching CheckpointTracker.states() output (0=pending, 1=pass, 2=fail).
CHECKPOINT_PALETTE: list[np.ndarray] = [
    np.array([0.3, 0.5, 1.0, 1.0], dtype=np.float32),  # 0: pending (blue)
    np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),  # 1: pass (green)
    np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),  # 2: fail (red)
]


class CheckpointTracker:
    """Tracks whether the end-effector passes through each of a list of target points.

    Two modes, selected by whether ``is_pass`` is provided:

    - **Proximity-only** (``is_pass=None``): a checkpoint passes the first time
      EE enters ``radius``. Use ``on_first_pass`` for immediate per-checkpoint
      logging; ``report`` is optional.

    - **Conditional** (``is_pass`` given): per-step ``update(ee_pos, value)``
      marks a checkpoint as passed only if ``is_pass(value)`` is True at some
      moment while EE is within ``radius``. ``score(value)`` (lower is better)
      is used to record the "best" observed value for the end-of-run report.

    ``points`` and ``ee_pos`` dimensions must match (either both 2D or both 3D
    — EE pos is truncated to points' dimensionality).
    """

    def __init__(
        self,
        points: np.ndarray | list,
        radius: float,
        *,
        is_pass: Callable[[float], bool] | None = None,
        score: Callable[[float], float] | None = None,
        name: str = "CP",
        value_fmt: str = "{:+.2f}",
        on_first_pass: Callable[[int], None] | None = None,
    ) -> None:
        self.points = np.asarray(points, dtype=float).reshape(len(points), -1)
        self.radius = float(radius)
        self.is_pass = is_pass
        self.score = score
        self.name = name
        self.value_fmt = value_fmt
        self.on_first_pass = on_first_pass
        n = len(self.points)
        self.passed = np.zeros(n, dtype=bool)
        self.best_value = np.full(n, np.nan, dtype=float)
        self.best_score = np.full(n, np.inf, dtype=float)
        self.finalized = False

    def update(self, ee_pos: np.ndarray, value: float | None = None) -> None:
        """Per-sim-step update. Call every step during the trajectory."""
        k = self.points.shape[1]
        ee = np.asarray(ee_pos, dtype=float).ravel()[:k]
        dists = np.linalg.norm(self.points - ee, axis=1)
        near_idx = np.where(dists <= self.radius)[0]
        if near_idx.size == 0:
            return

        if self.is_pass is None:
            for i in near_idx:
                if not self.passed[i]:
                    self.passed[i] = True
                    if self.on_first_pass is not None:
                        self.on_first_pass(int(i))
            return

        if value is None:
            return
        v = float(value)
        s = float(self.score(v)) if self.score is not None else 0.0
        accepted = bool(self.is_pass(v))
        for i in near_idx:
            if self.score is not None:
                if s < self.best_score[i]:
                    self.best_score[i] = s
                    self.best_value[i] = v
            elif np.isnan(self.best_value[i]):
                self.best_value[i] = v
            if accepted and not self.passed[i]:
                self.passed[i] = True
                if self.on_first_pass is not None:
                    self.on_first_pass(int(i))

    def clear(self) -> None:
        self.passed[:] = False
        self.best_value[:] = np.nan
        self.best_score[:] = np.inf
        self.finalized = False

    def states(self) -> np.ndarray:
        """Integer states suitable for ``core.viewer_utils.draw_status_markers``.

        Before ``report()``: ``0`` (pending) or ``1`` (passed).
        After ``report()``: ``1`` (passed) or ``2`` (failed).
        """
        if self.finalized:
            return np.where(self.passed, 1, 2)
        return self.passed.astype(int)

    def report(self, logger: logging.Logger) -> None:
        """Log end-of-run summary and flip ``states()`` into finalized (pass/fail) mode."""
        n_total = len(self.points)
        n_pass = int(self.passed.sum())
        if n_pass == n_total:
            logger.info("All %d %ss passed.", n_total, self.name)
        else:
            logger.info("%d/%d %ss passed. Missed:", n_pass, n_total, self.name)
            for i, passed in enumerate(self.passed):
                if passed:
                    continue
                coord = ", ".join(f"{c:.2f}" for c in self.points[i])
                if self.is_pass is not None and not np.isnan(self.best_value[i]):
                    info = f"best value while near = {self.value_fmt.format(self.best_value[i])}"
                else:
                    info = "never entered radius"
                logger.info("  %s%02d at (%s): %s  [FAIL]", self.name, i + 1, coord, info)
        self.finalized = True


class CollisionWarner:
    """Detects contacts between robot bodies and obstacle bodies, logs with cooldown.

    Resolves body names to IDs at construction; ignores missing bodies. Call
    ``check(data, logger)`` after each ``mj_step`` to emit a warning for each
    new collision event (rate-limited by ``cooldown`` seconds).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        robot_body_names: Sequence[str],
        obstacle_body_names: Sequence[str],
        cooldown: float = 0.5,
    ) -> None:
        self.model = model
        self.cooldown = float(cooldown)
        self.robot_ids = self._resolve(model, robot_body_names)
        self.obstacle_ids = self._resolve(model, obstacle_body_names)
        self._last_warn_time: float = -1.0

    @staticmethod
    def _resolve(model: mujoco.MjModel, names: Sequence[str]) -> set[int]:
        ids: set[int] = set()
        for n in names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid >= 0:
                ids.add(bid)
        return ids

    def reset(self) -> None:
        self._last_warn_time = -1.0

    def check(self, data: mujoco.MjData, logger: logging.Logger) -> None:
        for i in range(data.ncon):
            g1, g2 = data.contact[i].geom1, data.contact[i].geom2
            b1, b2 = self.model.geom_bodyid[g1], self.model.geom_bodyid[g2]
            if b1 in self.robot_ids and b2 in self.obstacle_ids:
                robot_bid, obstacle_bid = b1, b2
            elif b2 in self.robot_ids and b1 in self.obstacle_ids:
                robot_bid, obstacle_bid = b2, b1
            else:
                continue
            t = float(data.time)
            if self._last_warn_time < 0 or (t - self._last_warn_time) >= self.cooldown:
                rname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, robot_bid) or "?"
                oname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, obstacle_bid) or "?"
                logger.warning("Collision: %s vs %s", rname, oname)
                self._last_warn_time = t
            return


class MovableObstacle:
    """Single-obstacle position management + wireframe-sphere visualization.

    Stores the obstacle's initial ``body_pos`` at construction so ``reset()``
    can restore it after nudging. ``draw()`` resets ``user_scn`` and renders
    the obstacle sphere; optionally adds an arrow to a "highlighted" robot
    body (e.g. the nearest link during active avoidance).
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        body_name: str,
        radius: float,
        sphere_rgba: tuple[float, float, float, float] = (1.0, 0.3, 0.3, 0.5),
    ) -> None:
        self.model = model
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.radius = float(radius)
        self.sphere_rgba = np.array(sphere_rgba, dtype=np.float32)
        self.pos: np.ndarray | None = None
        self._home_pos: np.ndarray | None = None
        if self.body_id >= 0:
            self._home_pos = np.array(model.body_pos[self.body_id], dtype=float).copy()
            self.pos = self._home_pos.copy()
        else:
            logger.warning("MovableObstacle: body '%s' not found.", body_name)

    @property
    def active(self) -> bool:
        return self.body_id >= 0 and self.pos is not None

    def apply(self, data: mujoco.MjData) -> None:
        """Write ``self.pos`` into ``model.body_pos`` and re-run forward kinematics."""
        if not self.active:
            return
        self.model.body_pos[self.body_id][:] = self.pos  # type: ignore[index]
        mujoco.mj_forward(self.model, data)

    def nudge(self, delta_xyz: np.ndarray, data: mujoco.MjData) -> None:
        if not self.active:
            return
        self.pos = self.pos + np.asarray(delta_xyz, dtype=float)  # type: ignore[operator]
        self.apply(data)

    def reset(self, data: mujoco.MjData) -> None:
        """Restore obstacle to its initial position (captured at construction)."""
        if self._home_pos is None or self.body_id < 0:
            return
        self.pos = self._home_pos.copy()
        self.apply(data)

    def draw(
        self,
        viewer: Any,
        data: mujoco.MjData,
        *,
        highlight_body_id: int | None = None,
        highlight_is_hard: bool = False,
    ) -> None:
        """Reset ``user_scn`` and render obstacle wireframe + optional highlight arrow."""
        viewer.user_scn.ngeom = 0
        if not self.active:
            return
        obs_center = data.xpos[self.body_id]
        n = draw_wireframe_sphere(
            viewer, obs_center, self.radius, 0,
            n_segments=16, line_radius=0.002, rgba=self.sphere_rgba,
        )
        viewer.user_scn.ngeom = n
        if highlight_body_id is not None and highlight_body_id >= 0:
            rgba = (
                np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
                if highlight_is_hard
                else np.array([1.0, 0.6, 0.0, 1.0], dtype=np.float32)
            )
            draw_arrow(viewer, obs_center, data.xipos[highlight_body_id], rgba, width=0.006)


class ObstacleAutoMover:
    """Drives an obstacle position along a sum of three independent sinusoids (one per axis).

    Periods are mutually incommensurate so the trajectory does not repeat in short windows,
    producing a non-trivial test pattern for collision avoidance demos.
    """

    def __init__(
        self,
        amp_xyz: tuple[float, float, float] = (0.15, 0.15, 0.10),
        period_xyz: tuple[float, float, float] = (5.3, 3.7, 7.1),
    ) -> None:
        self.amp_xyz = np.array(amp_xyz, dtype=float)
        self.period_xyz = np.array(period_xyz, dtype=float)
        self._base: np.ndarray | None = None
        self._t0: float = 0.0

    def start(self, base_pos: np.ndarray, t0: float = 0.0) -> None:
        """Set the base (offset = 0) position and the time origin."""
        self._base = np.asarray(base_pos, dtype=float).copy()
        self._t0 = float(t0)

    def position_at(self, t: float) -> np.ndarray:
        """Return base + sinusoidal offset at sim time t."""
        if self._base is None:
            return np.zeros(3, dtype=float)
        dt = float(t) - self._t0
        two_pi = 2.0 * np.pi
        offsets = self.amp_xyz * np.sin(two_pi * dt / self.period_xyz)
        return self._base + offsets


class SimMonitor:
    """Bundles collision warning + goal checkpoint tracking.

    Both pieces are optional: pass empty obstacle list to skip collision warnings,
    pass ``goal_positions=None`` to skip goal tracking.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        goal_positions: np.ndarray | list | None = None,
        goal_radius: float = 0.06,
        robot_body_names: Sequence[str] = PANDA_ROBOT_BODIES,
        obstacle_body_names: Sequence[str] = (),
        collision_cooldown: float = 0.5,
        goal_name: str = "Goal",
    ) -> None:
        self.collision = CollisionWarner(
            model, robot_body_names, obstacle_body_names, cooldown=collision_cooldown,
        )
        if goal_positions is not None and len(goal_positions) > 0:
            self.goals: CheckpointTracker | None = CheckpointTracker(
                goal_positions,
                radius=goal_radius,
                name=goal_name,
                on_first_pass=lambda i: logger.info("Reached %s %d.", goal_name, i + 1),
            )
        else:
            self.goals = None

    def update(self, data: mujoco.MjData, ee_pos: np.ndarray) -> None:
        self.collision.check(data, logger)
        if self.goals is not None:
            self.goals.update(ee_pos)

    def reset(self) -> None:
        self.collision.reset()
        if self.goals is not None:
            self.goals.clear()


def make_hw2_1_scene_monitor(model: mujoco.MjModel) -> "SimMonitor":
    """Build the HW2-1 SimMonitor: 3 goal points + collision warning vs plates.

    Goal positions are read from the model bodies named ``goal1``, ``goal2``, ``goal3``.
    Obstacle body names are bound to ``("plate", "plate2")``.
    """
    goal_positions: list[np.ndarray] = []
    for name in ("goal1", "goal2", "goal3"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            goal_positions.append(np.array(model.body_pos[bid], dtype=float))
    return SimMonitor(
        model,
        goal_positions=goal_positions,
        obstacle_body_names=("plate", "plate2"),
    )


class AvoidanceProbe:
    """Finds the body closest to an obstacle and returns its translational Jacobian.

    ``query(data, obs_pos)`` returns ``(name, delta, n, Jv)`` for the body with
    the smallest signed clearance ``delta = ‖p_body - p_obs‖ - safety_margin``
    (most penetrating). ``n`` is the unit vector from obstacle to body, ``Jv``
    is the 3D translational Jacobian of that body (3, arm_dofs). ``name`` is
    None if no body is found or ``obs_pos`` is None — in that case ``n`` and
    ``Jv`` are zeros.
    """

    def __init__(
        self,
        robot: robotics.RobotWrapper,
        model: mujoco.MjModel,
        arm_dofs: int,
        safety_margin: float,
        body_names: Sequence[str] = PANDA_AVOIDANCE_BODIES,
    ) -> None:
        self.robot = robot
        self.arm_dofs = int(arm_dofs)
        self.safety_margin = float(safety_margin)
        self.body_ids: dict[str, int] = {}
        for name in body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.body_ids[name] = bid

    def query(
        self,
        data: mujoco.MjData,
        obs_pos: np.ndarray | None,
    ) -> tuple[str | None, float, np.ndarray, np.ndarray]:
        if obs_pos is None:
            return None, float("inf"), np.zeros(3, dtype=float), np.zeros((3, self.arm_dofs), dtype=float)
        best_name: str | None = None
        best_delta = float("inf")
        best_n = np.zeros(3, dtype=float)
        best_Jv = np.zeros((3, self.arm_dofs), dtype=float)
        for body_name, body_id in self.body_ids.items():
            p = data.xipos[body_id].copy()
            r = p - obs_pos
            d = float(np.linalg.norm(r))
            if d < 1e-6:
                continue
            delta = d - self.safety_margin
            if delta < best_delta:
                _, Jv, _ = self.robot.get_body_jacobian(
                    data, body_name, translational=True, rotational=False,
                )
                best_delta = delta
                best_name = body_name
                best_n = r / d
                best_Jv = Jv[:, : self.arm_dofs]
        return best_name, best_delta, best_n, best_Jv


class HW22AutoForceToggle:
    """Per-step auto-toggle of force-control mode based on EE XY proximity.

    - When EE first enters the proximity of ``(0.3, -0.35)`` with force mode OFF,
      calls ``toggle()`` to flip it ON.
    - When EE first enters the proximity of ``(0.5, 0.3)`` with force mode ON,
      calls ``toggle()`` to flip it OFF.

    Each direction fires at most once between ``clear()`` calls.
    """

    def __init__(self) -> None:
        self._on_xy = np.array([0.3, -0.35])
        self._off_xy = np.array([0.5, 0.3])
        self._radius = 0.025
        self._on_fired = False
        self._off_fired = False

    def clear(self) -> None:
        self._on_fired = False
        self._off_fired = False

    def update(
        self,
        ee_pos: np.ndarray,
        force_mode_on: bool,
        toggle: Callable[[], None],
    ) -> None:
        """Per-sim-step update. ``toggle()`` is called when a trigger fires."""
        ee_xy = np.asarray(ee_pos, dtype=float).ravel()[:2]
        if not self._on_fired and not force_mode_on:
            if np.linalg.norm(ee_xy - self._on_xy) <= self._radius:
                toggle()
                self._on_fired = True
                return
        if not self._off_fired and force_mode_on:
            if np.linalg.norm(ee_xy - self._off_xy) <= self._radius:
                toggle()
                self._off_fired = True


def make_hw2_2_force_grading(z_height: float) -> tuple["CheckpointTracker", np.ndarray]:
    """Build the HW2-2 Fz checkpoint task (Fz convergence at fixed XY positions).

    Returns ``(tracker, markers_3d)`` — the tracker for per-step ``update(ee_pos, fz)``
    + ``report(logger)`` calls, and ``markers_3d`` (N, 3) world-frame points for
    ``draw_status_markers`` to render pending/pass/fail spheres.
    """
    target_fz = -5.0
    tolerance = 0.2
    radius = 0.025
    xy = np.array(
        [
            [0.3, -0.35], [0.4, -0.35], [0.5, -0.35],
            [0.35, -0.25], [0.5, -0.25],
            [0.4, -0.2],
            [0.3, -0.15], [0.5, -0.15],
            [0.4, -0.1],
            [0.3, -0.05], [0.4, -0.05], [0.5, -0.05],
            [0.35, 0.05], [0.45, 0.05],
            [0.4, 0.1],
            [0.3, 0.15], [0.5, 0.15],
            [0.4, 0.2],
            [0.3, 0.25], [0.3, 0.3], [0.5, 0.3],
            [0.3, 0.35],
        ],
        dtype=float,
    )
    tracker = CheckpointTracker(
        xy,
        radius=radius,
        is_pass=lambda v: abs(v - target_fz) <= tolerance,
        score=lambda v: abs(v - target_fz),
        name="CP",
        value_fmt="{:+.2f}N",
    )
    markers_3d = np.hstack([xy, np.full((len(xy), 1), z_height)])
    return tracker, markers_3d


class HWAppBase:
    """Base class for demo apps. Subclasses implement task-specific logic via hooks.

    Class attributes to override:
        model_path (Path):         MuJoCo XML path (required).
        title (str):               Window title.
        site_name (str):           EE site name in the MJCF.
        camera (dict | None):      Optional {"lookat", "distance", "azimuth", "elevation"}.
        bg_rgb (tuple):            Viewer background color.
        ambient (float):           Viewer ambient light.

    Required override:
        sim_step() -> bool         One sim step. Return False to pause stepping (e.g. trajectory ended).

    Optional hooks:
        on_reset()                 Extra reset logic (scene, tracker, buffers).
        visualize_trajectory(viewer)   Drawn once before the main loop.
        pre_render()               Called each render tick BEFORE draw_overlay (sampling, etc.).
        draw_overlay(viewer)       Called each render tick to (re)build non-persistent scene geoms.
        build_gui_content(parent, api)   Called by UnifiedGUI to populate the panel.
        register_keys()            Called once after keyboard handler is constructed.
    """

    model_path: Path | None = None
    title: str = "HW Demo"
    site_name: str = "ee_site"
    camera: dict[str, Any] | None = None
    bg_rgb: tuple[float, float, float] = (0.45, 0.45, 0.5)
    ambient: float = 0.4

    def __init__(self) -> None:
        if self.model_path is None:
            raise NotImplementedError("Subclass must set class attribute `model_path`.")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model, self.data = load_model(self.model_path)
        self.robot = robotics.RobotWrapper(self.model, site_name=self.site_name)
        self.dt = self.model.opt.timestep
        self.render_dt = RealtimeConfig.render_dt()
        self.sync = RealtimeSync()
        self.render_tick = RateLimiter(0.0, self.render_dt)
        self.keyboard = KeyboardHandler()
        self.sim_time = 0.0

    # ---- Required override ----
    def sim_step(self) -> bool:
        """Execute one sim step. Return False to pause inner stepping (e.g. trajectory finished)."""
        raise NotImplementedError

    # ---- Optional hooks ----
    def on_reset(self) -> None:
        """Called at the end of ``base_reset``. Override to clear trackers/buffers."""
        pass

    def visualize_trajectory(self, viewer: Any) -> None:
        """Called once after viewer launches. Override to draw trajectory preview."""
        pass

    def pre_render(self) -> None:
        """Called each render tick, before ``draw_overlay``. Override for sampling (e.g. EE trace)."""
        pass

    def draw_overlay(self, viewer: Any) -> None:
        """Called each render tick. Override to rebuild dynamic scene geoms."""
        pass

    def build_gui_content(self, parent: tk.Widget, api: Any) -> None:
        """Populate the UnifiedGUI panel (plots, parameters, buttons)."""
        pass

    def register_keys(self) -> None:
        """Register custom keyboard callbacks on ``self.keyboard``."""
        pass

    # ---- Base-provided ----
    def base_reset(self) -> None:
        """Standard reset: ``mj_resetData`` → write ``q_home`` (if set) → zero qvel → resync."""
        mujoco.mj_resetData(self.model, self.data)
        q_home = getattr(self, "q_home", None)
        if q_home is not None:
            q = np.asarray(q_home, dtype=float)
            n = min(len(q), self.model.nq)
            self.data.qpos[:n] = q[:n]
        finger_open = getattr(self, "finger_open_position", None)
        arm_dofs = getattr(self, "arm_dofs", None)
        if finger_open is not None and arm_dofs is not None and self.model.nq > arm_dofs:
            self.data.qpos[arm_dofs : self.model.nq] = finger_open
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.sync.reset(self.data.time)
        self.render_tick.next_time = self.data.time
        self.sim_time = 0.0
        self.on_reset()
        logger.info("Reset: back to initial pose.")

    def _apply_camera(self, viewer: Any) -> None:
        if self.camera is not None:
            viewer.cam.lookat[:] = self.camera.get("lookat", [0.0, 0.0, 0.0])
            viewer.cam.distance = float(self.camera.get("distance", 2.0))
            viewer.cam.azimuth = float(self.camera.get("azimuth", 0.0))
            viewer.cam.elevation = float(self.camera.get("elevation", -30.0))
        try:
            if hasattr(viewer, "opt") and hasattr(viewer.opt, "background_rgb"):
                viewer.opt.background_rgb[:] = self.bg_rgb
            if hasattr(viewer, "opt") and hasattr(viewer.opt, "ambient"):
                viewer.opt.ambient = self.ambient
        except Exception:
            pass

    def run(self) -> None:
        # Only spawn the UnifiedGUI window if the subclass actually populates it.
        has_gui_content = type(self).build_gui_content is not HWAppBase.build_gui_content
        gui = None
        if has_gui_content:
            gui = UnifiedGUI(
                self.model, self.data,
                title=self.title,
                build_content=self.build_gui_content,
                auto_start=False,
            )
            gui.set_reset_callback(self.base_reset)
            gui.start()

        self.base_reset()
        self.keyboard.set_reset_callback(self.base_reset)
        self.register_keys()
        logger.info("Keyboard: %s", self.keyboard.get_keyboard_help())
        self.keyboard.set_exit_callback(lambda: logger.info("Simulation exited."))

        try:
            with mujoco.viewer.launch_passive(
                self.model, self.data, key_callback=self.keyboard.create_key_callback(),
            ) as viewer:
                self.keyboard.set_viewer(viewer)
                self._apply_camera(viewer)
                self.visualize_trajectory(viewer)
                # visualize_trajectory may dirty qpos (e.g. FK sampling); restore clean state.
                self.base_reset()

                while viewer.is_running() and not self.keyboard.should_exit:
                    if gui is not None:
                        gui.update()
                        if hasattr(gui, "check_and_apply_pending_update"):
                            gui.check_and_apply_pending_update()
                        if hasattr(gui, "check_and_apply_pending_reset"):
                            gui.check_and_apply_pending_reset()

                    if self.keyboard.paused:
                        viewer.sync()
                        time.sleep(0.01)
                        continue

                    self.sync.set_speed_factor(self.keyboard.speed_factor)
                    target_t = self.sync.target_sim_time()

                    while self.data.time < target_t:
                        if not self.sim_step():
                            break

                    if self.render_tick.ready(self.data.time):
                        self.pre_render()
                        self.draw_overlay(viewer)
                        viewer.sync()

                    time.sleep(0.0005)
        finally:
            if gui is not None:
                try:
                    gui.stop()
                except Exception:
                    pass
