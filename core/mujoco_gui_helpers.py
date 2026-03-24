"""Helpers to build GUIParameter lists for MuJoCo qpos (from model or fixed size)."""

import math
from typing import Any

import mujoco

from core.gui import GUIParameter

_JNT_FREE, _JNT_BALL, _JNT_SLIDE, _JNT_HINGE = 0, 1, 2, 3
_JNT_NQPOS = {_JNT_FREE: 7, _JNT_BALL: 4, _JNT_SLIDE: 1, _JNT_HINGE: 1}


def _set_qpos(model: Any, data: Any, value: Any) -> None:
    """Write value into data.qpos (first nq elements), zero qvel, run mj_forward.

    Args:
        model: MuJoCo model.
        data: MuJoCo data to modify.
        value: Array-like of length at least model.nq.
    """
    data.qpos[:] = value[: model.nq]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _get_qpos(model: Any, data: Any) -> list[float]:
    """Return current generalized position as a list.

    Args:
        model: MuJoCo model (unused; for signature compatibility with setter).
        data: MuJoCo data.

    Returns:
        list(data.qpos).
    """
    return list(data.qpos)


def make_qpos_parameter(
    default: list[float],
    name: str = "q",
    labels: list[str] | None = None,
    min_val: float = -math.pi,
    max_val: float = math.pi,
) -> GUIParameter:
    """Build a GUIParameter for a generic joint position vector with uniform limits.

    Args:
        default: Default qpos values; length defines dimension.
        name: Parameter name for the GUI.
        labels: Optional list of labels per dimension; default "q0", "q1", ...
        min_val: Minimum for all dimensions.
        max_val: Maximum for all dimensions.

    Returns:
        GUIParameter with setter/getter bound to _set_qpos/_get_qpos.
    """
    n = len(default)
    if labels is None:
        labels = [f"q{i}" for i in range(n)]
    return GUIParameter(
        name=name,
        param_type="vector",
        default=default,
        min_val=min_val,
        max_val=max_val,
        labels=labels,
        setter=_set_qpos,
        getter=_get_qpos,
    )


def make_qpos_parameter_from_model(
    model: Any,
    default: list[float] | None = None,
    name: str = "q",
    labels: list[str] | None = None,
    use_degrees: bool = True,
) -> GUIParameter:
    """Build a GUIParameter for model.qpos using joint types and ranges from the model.

    Hinge joints get min/max from jnt_range (or ±π if unlimited); slide and others
    get per-DOF limits. Optionally displays hinge DOFs in degrees (angle_indices).

    Args:
        model: MuJoCo model (must have nq, njnt, jnt_qposadr, jnt_type, jnt_range, qpos0).
        default: Initial qpos; if None, uses model.qpos0.
        name: Parameter name for the GUI.
        labels: Optional labels; default "q0", "q1", ...
        use_degrees: If True, hinge DOFs use angle_indices so GUI shows degrees.

    Returns:
        GUIParameter with per-DOF min_vals/max_vals and optional angle_indices.
    """
    nq = model.nq
    default = list(default) if default is not None else list(model.qpos0)
    if labels is None:
        labels = [f"q{i}" for i in range(nq)]
    min_vals = [-math.pi] * nq
    max_vals = [math.pi] * nq
    angle_indices: list[int] = []
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        jtype = int(model.jnt_type[j])
        nqpos = _JNT_NQPOS.get(jtype, 1)
        lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
        unlimited = abs(lo) > 1e6 or abs(hi) > 1e6
        if jtype == _JNT_HINGE:
            if unlimited:
                lo, hi = -math.pi, math.pi
            for k in range(nqpos):
                min_vals[adr + k] = lo
                max_vals[adr + k] = hi
            angle_indices.extend(range(adr, adr + nqpos))
        elif jtype == _JNT_SLIDE:
            if unlimited:
                lo, hi = -0.1, 0.1
            for k in range(nqpos):
                min_vals[adr + k] = lo
                max_vals[adr + k] = hi
        else:
            for k in range(nqpos):
                min_vals[adr + k] = lo if not unlimited else -math.pi
                max_vals[adr + k] = hi if not unlimited else math.pi
    return GUIParameter(
        name=name,
        param_type="vector",
        default=default,
        min_val=None,
        max_val=None,
        min_vals=min_vals,
        max_vals=max_vals,
        labels=labels,
        setter=_set_qpos,
        getter=_get_qpos,
        angle_indices=angle_indices if use_degrees else None,
    )
