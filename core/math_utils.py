"""General-purpose math helpers: SO(3) rotation matrices, RPY, axis-angle, orientation error."""

import numpy as np


def rot_x(angle: float) -> np.ndarray:
    """Rotation matrix about world x-axis by angle (rad).

    Returns:
        (3, 3) rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(angle: float) -> np.ndarray:
    """Rotation matrix about world y-axis by angle (rad).

    Returns:
        (3, 3) rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    """Rotation matrix about world z-axis by angle (rad).

    Returns:
        (3, 3) rotation matrix.
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Build rotation matrix from roll, pitch, yaw (fixed-frame ZYX convention).

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Args:
        roll: Rotation about x (rad).
        pitch: Rotation about y (rad).
        yaw: Rotation about z (rad).

    Returns:
        (3, 3) rotation matrix.
    """
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Compute axis-angle representation from a rotation matrix.

    Args:
        R: (3, 3) rotation matrix.

    Returns:
        (3,) axis-angle; norm is the angle in rad, direction is the rotation axis.
    """
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-10:
        return np.zeros(3)
    return axis * (angle / n)


def orientation_error(R_des: np.ndarray, R_cur: np.ndarray) -> np.ndarray:
    """Orientation error as axis-angle of R_err = R_des @ R_cur.T.

    Magnitude is clipped to pi for stability.

    Args:
        R_des: Desired (3, 3) rotation matrix.
        R_cur: Current (3, 3) rotation matrix.

    Returns:
        (3,) axis-angle error; norm at most pi.
    """
    R_err = R_des @ R_cur.T
    e = rotation_matrix_to_axis_angle(R_err)
    mag = np.linalg.norm(e)
    if mag > np.pi:
        e = e * (np.pi / mag)
    return e
