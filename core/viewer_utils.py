"""MuJoCo passive viewer drawing helpers: coordinate frames, circle segments, wireframe spheres, trajectory points."""

from typing import Any

import numpy as np
import mujoco


def draw_circle_segments(
    scn: Any,
    n: int,
    pts: list[np.ndarray],
    line_radius: float,
    rgba: np.ndarray,
    maxgeom: int,
    closed: bool = True,
) -> int:
    """Draw line segments between consecutive points as thin cylinders in the viewer scene.

    Args:
        scn: MuJoCo viewer user scene (viewer.user_scn).
        n: Starting geom index.
        pts: List of 3D points (world frame).
        line_radius: Radius of each cylinder segment.
        rgba: Color (4,) RGBA float32.
        maxgeom: Maximum number of geoms (stop if n >= maxgeom).
        closed: If True, connect last point to first; otherwise open polyline.

    Returns:
        Next geom index after the drawn segments.
    """
    n_pts = len(pts)
    n_iter = n_pts if closed else n_pts - 1
    for i in range(n_iter):
        if n >= maxgeom:
            return n
        p0 = pts[i]
        p1 = pts[(i + 1) % n_pts]
        geom = scn.geoms[n]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.zeros(3, dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).reshape(9),
            rgba=rgba,
        )
        from_col = np.asarray(p0, dtype=np.float64).reshape(3, 1)
        to_col = np.asarray(p1, dtype=np.float64).reshape(3, 1)
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            line_radius,
            from_col,
            to_col,
        )
        geom.rgba[:] = rgba
        n += 1
    return n


def draw_wireframe_sphere(
    viewer: Any,
    center: np.ndarray,
    radius: float,
    start_n: int,
    n_segments: int = 24,
    line_radius: float = 0.002,
    rgba: np.ndarray | None = None,
) -> int:
    """Draw a wireframe sphere (latitude and longitude circles) in the viewer scene.

    Args:
        viewer: MuJoCo passive viewer (e.g. from mujoco.viewer.launch_passive).
        center: Center of the sphere (3,) in world frame.
        radius: Sphere radius.
        start_n: First geom index to use.
        n_segments: Number of points per circle.
        line_radius: Radius of each cylinder segment.
        rgba: Color (4,) RGBA float32; default light blue semi-transparent.

    Returns:
        Next geom index after the sphere.
    """
    scn = viewer.user_scn
    n = start_n
    center = np.asarray(center, dtype=np.float64).reshape(3)
    if rgba is None:
        rgba = np.array([0.5, 0.6, 0.9, 0.5], dtype=np.float32)
    n_lat = 8
    theta_list = np.linspace(0.15 * np.pi, 0.85 * np.pi, n_lat)
    phi = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    for theta in theta_list:
        pts = [
            center
            + radius
            * np.array(
                [np.sin(theta) * np.cos(p), np.sin(theta) * np.sin(p), np.cos(theta)],
                dtype=np.float64,
            )
            for p in phi
        ]
        n = draw_circle_segments(scn, n, pts, line_radius, rgba, scn.maxgeom)
    n_meridian = 12
    phi_list = np.linspace(0, 2 * np.pi, n_meridian, endpoint=False)
    theta = np.linspace(0, np.pi, n_segments)
    for phi0 in phi_list:
        pts = [
            center
            + radius
            * np.array(
                [np.sin(t) * np.cos(phi0), np.sin(t) * np.sin(phi0), np.cos(t)],
                dtype=np.float64,
            )
            for t in theta
        ]
        n = draw_circle_segments(scn, n, pts, line_radius, rgba, scn.maxgeom, closed=False)
    return n


def draw_trajectory_points(
    viewer: Any,
    points: list[np.ndarray | list],
    size: float = 0.01,
    rgba: np.ndarray | None = None,
) -> int:
    """Draw trajectory as a sequence of small spheres in the viewer scene.

    Args:
        viewer: MuJoCo passive viewer (e.g. from mujoco.viewer.launch_passive).
        points: List of 3D points (world frame), each (3,) or length-3 list.
        size: Radius of each sphere.
        rgba: Color (4,) RGBA float32; default green opaque.

    Returns:
        Number of geoms added (may be less than len(points) if maxgeom is reached).
    """
    scn = viewer.user_scn
    if rgba is None:
        rgba = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    n_start = scn.ngeom
    n = n_start
    for p in points:
        if n >= scn.maxgeom:
            break
        pos = np.asarray(p, dtype=np.float64).reshape(3)
        geom = scn.geoms[n]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0.0, 0.0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        n += 1
    scn.ngeom = n
    return n - n_start


def draw_trajectory_preview(
    viewer: Any,
    traj: Any,
    *,
    point_fn: Any = None,
    dt: float = 0.05,
    size: float = 0.01,
    rgba: np.ndarray | None = None,
) -> int:
    """Sample a trajectory at uniform ``dt`` and draw each resulting world point as a sphere.

    Args:
        viewer: MuJoCo passive viewer.
        traj: Object with ``total_time`` and ``evaluate(t) -> (sample, ...)``.
        point_fn: Optional callable converting a sample to a 3D world point. If ``None``,
            the trajectory's first output is taken to be the 3D point itself (task-space
            trajectory). For joint-space trajectories, pass a forward-kinematics callable.
        dt: Sampling interval (seconds).
        size: Sphere radius for each sample.
        rgba: Sphere color (4,) RGBA float32; default green opaque.

    Returns:
        Number of geoms added.
    """
    points: list[np.ndarray] = []
    t = 0.0
    while t < traj.total_time:
        sample, _, _ = traj.evaluate(t)
        p = sample if point_fn is None else point_fn(sample)
        points.append(np.asarray(p, dtype=float).copy())
        t += dt
    return draw_trajectory_points(viewer, points, size=size, rgba=rgba)


def draw_polyline(
    viewer: Any,
    points: list[np.ndarray],
    line_radius: float = 0.003,
    rgba: np.ndarray | None = None,
) -> int:
    """Draw an open polyline as cylinder segments between consecutive points in the viewer scene.

    Args:
        viewer: MuJoCo passive viewer (e.g. from mujoco.viewer.launch_passive).
        points: List of 3D points (world frame). Fewer than 2 points draws nothing.
        line_radius: Radius of each cylinder segment.
        rgba: Color (4,) RGBA float32; default orange-red opaque.

    Returns:
        Number of geoms added (may be less than len(points)-1 if maxgeom is reached).
    """
    scn = viewer.user_scn
    if rgba is None:
        rgba = np.array([1.0, 0.3, 0.1, 1.0], dtype=np.float32)
    if len(points) < 2:
        return 0
    n_start = scn.ngeom
    n = draw_circle_segments(scn, n_start, points, line_radius, rgba, scn.maxgeom, closed=False)
    scn.ngeom = n
    return n - n_start


def draw_status_markers(
    viewer: Any,
    points: np.ndarray,
    states: Any,
    palette: list[np.ndarray],
    size: float = 0.015,
) -> int:
    """Draw spheres at each point colored by palette[states[i]] in the viewer scene.

    Args:
        viewer: MuJoCo passive viewer.
        points: (N, 3) array (or list) of world-frame positions.
        states: (N,) integer-like indices into ``palette``.
        palette: List of RGBA (4,) float32 colors.
        size: Sphere radius.

    Returns:
        Number of geoms added (may be less than N if maxgeom is reached).
    """
    scn = viewer.user_scn
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    sts = np.asarray(states, dtype=int).ravel()
    n_start = scn.ngeom
    n = n_start
    for pos, state in zip(pts, sts):
        if n >= scn.maxgeom:
            break
        rgba = palette[int(state)]
        geom = scn.geoms[n]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0.0, 0.0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        n += 1
    scn.ngeom = n
    return n - n_start


class StrokeRecorder:
    """Records 3D points into separate strokes and draws them as polylines.

    Each call to :meth:`sample` with ``condition=True`` appends to the current
    stroke; a ``False`` → ``True`` transition starts a new stroke. ``False``
    samples are ignored. Useful for conditional EE-path tracing (e.g. only
    while a pen is pressed).
    """

    def __init__(
        self,
        line_radius: float = 0.003,
        rgba: np.ndarray | None = None,
    ) -> None:
        self.line_radius = line_radius
        self.rgba = (
            rgba
            if rgba is not None
            else np.array([1.0, 0.3, 0.1, 1.0], dtype=np.float32)
        )
        self._strokes: list[list[np.ndarray]] = []
        self._was_on = False

    def sample(self, condition: bool, pos: np.ndarray) -> None:
        """Record ``pos`` into the current stroke if ``condition`` is True."""
        if condition:
            if not self._was_on:
                self._strokes.append([])
            self._strokes[-1].append(np.asarray(pos, dtype=float).copy())
        self._was_on = bool(condition)

    def draw(self, viewer: Any) -> None:
        """Draw every stroke as an open polyline in the viewer scene."""
        for stroke in self._strokes:
            draw_polyline(viewer, stroke, line_radius=self.line_radius, rgba=self.rgba)

    def clear(self) -> None:
        """Discard all recorded strokes and reset the transition state."""
        self._strokes.clear()
        self._was_on = False


def draw_frame_at(
    viewer: Any,
    pos: np.ndarray,
    R: np.ndarray,
    axis_length: float = 0.08,
    axis_radius: float = 0.004,
    alpha: float = 1.0,
    in_place: bool = False,
) -> None:
    """Draw or update a coordinate frame (three cylinders along X, Y, Z) in the viewer scene.

    Args:
        viewer: MuJoCo passive viewer.
        pos: Origin of the frame (3,) in world frame.
        R: Rotation matrix (3x3) columns are X, Y, Z axis directions.
        axis_length: Length of each axis cylinder.
        axis_radius: Radius of each cylinder.
        alpha: Opacity (0–1).
        in_place: If True, overwrite the first three geoms; otherwise append and set scn.ngeom.
    """
    scn = viewer.user_scn
    start = 0 if in_place else scn.ngeom
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    colors = [
        np.array([1.0, 0.0, 0.0, alpha], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, alpha], dtype=np.float32),
        np.array([0.0, 0.0, 1.0, alpha], dtype=np.float32),
    ]
    drawn = 0
    for i, rgba in enumerate(colors):
        if not in_place and start + i >= scn.maxgeom:
            break
        geom = scn.geoms[start + i]
        p0 = pos
        p1 = pos + axis_length * R[:, i]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=np.zeros(3, dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).reshape(9),
            rgba=rgba,
        )
        from_col = np.asarray(p0, dtype=np.float64).reshape(3, 1)
        to_col = np.asarray(p1, dtype=np.float64).reshape(3, 1)
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            axis_radius,
            from_col,
            to_col,
        )
        geom.rgba[:] = rgba
        drawn += 1
    if not in_place:
        scn.ngeom = start + drawn


def draw_arrow(
    viewer: Any,
    start: np.ndarray,
    end: np.ndarray,
    rgba: np.ndarray,
    width: float = 0.012,
) -> int:
    """Draw an arrow (cylinder shaft + sphere tip) from start to end in the viewer scene.

    Args:
        viewer: MuJoCo passive viewer.
        start: Arrow tail (3,) in world frame.
        end: Arrow head (3,) in world frame.
        rgba: Color (4,) RGBA float32.
        width: Shaft radius (sphere tip radius is 2x).

    Returns:
        Updated scn.ngeom after the arrow is added.
    """
    scn = viewer.user_scn
    if scn.ngeom + 2 > scn.maxgeom:
        return scn.ngeom

    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    if float(np.linalg.norm(end - start)) < 1e-9:
        return scn.ngeom

    n = scn.ngeom
    geom = scn.geoms[n]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=np.zeros(3, dtype=np.float64),
        pos=np.zeros(3, dtype=np.float64),
        mat=np.eye(3).reshape(9),
        rgba=rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        width,
        start.reshape(3, 1),
        end.reshape(3, 1),
    )
    geom.rgba[:] = rgba
    n += 1

    geom = scn.geoms[n]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([width * 2.0, 0.0, 0.0], dtype=np.float64),
        pos=end,
        mat=np.eye(3).reshape(9),
        rgba=rgba,
    )
    n += 1

    scn.ngeom = n
    return n
