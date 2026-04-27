"""Trajectory generators: quintic polynomial segments and multi-segment trajectories."""

import logging

import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class Trajectory(ABC):
    """Abstract base for trajectory generators returning position, velocity, acceleration."""

    @abstractmethod
    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return position, velocity, acceleration at time t.

        Args:
            t: Time (global or segment-relative depending on implementation).

        Returns:
            Tuple of (position, velocity, acceleration) arrays; shape matches trajectory dimension.
        """
        pass


class FifthOrderTrajectory(Trajectory):
    """Quintic polynomial segment: position, velocity, acceleration at ti and tf. x(tau) = sum c_k*tau^k; six BCs give 6x6 system A*c = b."""

    def __init__(
        self,
        ti: float,
        tf: float,
        xi: float | np.ndarray,
        xf: float | np.ndarray,
        vi: float | np.ndarray = 0.0,
        vf: float | np.ndarray = 0.0,
        ai: float | np.ndarray = 0.0,
        af: float | np.ndarray = 0.0,
    ) -> None:
        self.ti, self.tf = float(ti), float(tf)
        self.T = self.tf - self.ti
        self.xi = np.atleast_1d(np.asarray(xi, dtype=float))
        self.xf = np.atleast_1d(np.asarray(xf, dtype=float))
        d = len(self.xi)
        T = self.T
        xi = self.xi
        xf = self.xf
        vi = np.broadcast_to(np.asarray(vi, dtype=float), d)
        vf = np.broadcast_to(np.asarray(vf, dtype=float), d)
        ai = np.broadcast_to(np.asarray(ai, dtype=float), d)
        af = np.broadcast_to(np.asarray(af, dtype=float), d)
        T2, T3, T4, T5 = T**2, T**3, T**4, T**5

        # --- BEGIN: HW2-1 Problem (a) ---
        # Construct the 6x6 matrix A and the boundary-condition matrix b
        # for the quintic polynomial trajectory.
        # Boundary conditions:
        #   x(ti)=xi, xdot(ti)=vi, xddot(ti)=ai
        #   x(tf)=xf, xdot(tf)=vf, xddot(tf)=af

        A = np.zeros((6, 6), dtype=float)
        A[0, :] = [1, 0, 0, 0, 0, 0]
        A[1, :] = [0, 1, 0, 0, 0, 0]
        A[2, :] = [0, 0, 2, 0, 0, 0]
        # A[3, :] = ...
        # A[4, :] = ...
        # A[5, :] = ...
        b = np.zeros((6, d), dtype=float)
        b[0, :] = xi
        b[1, :] = vi
        b[2, :] = ai
        # b[3, :] = ...
        # b[4, :] = ...
        # b[5, :] = ...

        # Compute coefficients for the quintic polynomial trajectory here.
        # You can use np.linalg.solve(A, b) to compute coefficients.
        
        self.coeffs = np.zeros((6, d), dtype=float)
        # --- END: HW2-1 Problem (a) ---

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_rel = np.clip(t - self.ti, 0.0, self.T)
        t2, t3, t4, t5 = t_rel**2, t_rel**3, t_rel**4, t_rel**5
        T_vec = np.array([1, t_rel, t2, t3, t4, t5])
        V_vec = np.array([0, 1, 2 * t_rel, 3 * t2, 4 * t3, 5 * t4])
        A_vec = np.array([0, 0, 2, 6 * t_rel, 12 * t2, 20 * t3])
        return T_vec @ self.coeffs, V_vec @ self.coeffs, A_vec @ self.coeffs


class MultiSegmentTrajectory(Trajectory):
    """Piecewise quintic path. Waypoint: ``(d,)``, ``(2,d)`` (position, velocity), or ``(3,d)`` (+ acceleration).

    If every waypoint is ``(d,)``, ``vel_start`` / ``vel_end`` and ``acc_start`` / ``acc_end`` apply at the
    ends only; interior velocity and acceleration are zero. Otherwise missing rows are zero.
    """

    @staticmethod
    def _unpack_waypoints(waypoints_raw: list) -> tuple[list, list, list, int]:
        positions: list = []
        vel_embed: list = []
        acc_embed: list = []
        d: int | None = None
        for w in waypoints_raw:
            a = np.asarray(w, dtype=float)
            if a.ndim == 1:
                if d is None:
                    d = int(a.size)
                elif int(a.size) != d:
                    raise ValueError(f"All waypoints must have dimension {d}, got {a.size}")
                positions.append(np.copy(a))
                vel_embed.append(None)
                acc_embed.append(None)
            elif a.ndim == 2:
                r, c = a.shape
                if d is None:
                    d = int(c)
                elif int(c) != d:
                    raise ValueError(f"All waypoints must have dimension {d}, got {c}")
                if r == 2:
                    positions.append(np.copy(a[0]))
                    vel_embed.append(np.copy(a[1]))
                    acc_embed.append(None)
                elif r == 3:
                    positions.append(np.copy(a[0]))
                    vel_embed.append(np.copy(a[1]))
                    acc_embed.append(np.copy(a[2]))
                else:
                    raise ValueError(
                        "Stacked waypoint must have shape (2, d) or (3, d); got "
                        f"{a.shape}"
                    )
            else:
                raise ValueError(
                    "Each waypoint must be a 1d position (d,) or stacked (2, d) / (3, d) array"
                )
        if d is None:
            raise ValueError("waypoints list is empty")
        return positions, vel_embed, acc_embed, d

    def __init__(
        self,
        waypoints: list,
        segment_time: float | np.ndarray,
        vel_start: float | np.ndarray | None = None,
        vel_end: float | np.ndarray | None = None,
        acc_start: float | np.ndarray | None = None,
        acc_end: float | np.ndarray | None = None,
    ) -> None:
        if len(waypoints) == 0:
            self.segments = []
            self.t_starts = np.array([])
            self.total_time = 0.0
            self.num_segments = 0
            return
        positions, vel_embed, acc_embed, d = self._unpack_waypoints(waypoints)
        n_wp = len(positions)
        n = n_wp - 1
        if n <= 0:
            self.segments = []
            self.t_starts = np.array([])
            self.total_time = 0.0
            self.num_segments = 0
            return

        segment_time = np.atleast_1d(np.asarray(segment_time, dtype=float))
        if segment_time.size == 1:
            durations = np.full(n, float(segment_time.flat[0]))
        else:
            if segment_time.size != n:
                logger.warning(
                    "segment_time length must be 1 or %d (len(waypoints)-1), got %d",
                    n, segment_time.size,
                )
                raise ValueError(f"segment_time length must be 1 or {n} (len(waypoints)-1)")
            durations = segment_time

        has_emb_v = any(v is not None for v in vel_embed)
        has_emb_a = any(a is not None for a in acc_embed)

        if has_emb_v:
            vels = [
                np.zeros(d) if vel_embed[i] is None else np.asarray(vel_embed[i], dtype=float).copy()
                for i in range(n_wp)
            ]
        else:
            if vel_start is None:
                vel_start = np.zeros(d)
            else:
                vel_start = np.broadcast_to(np.atleast_1d(np.asarray(vel_start, dtype=float)), d)
            if vel_end is None:
                vel_end = np.zeros(d)
            else:
                vel_end = np.broadcast_to(np.atleast_1d(np.asarray(vel_end, dtype=float)), d)
            vels = [np.zeros(d) for _ in range(n_wp)]
            vels[0] = np.asarray(vel_start, dtype=float).copy()
            vels[-1] = np.asarray(vel_end, dtype=float).copy()

        if has_emb_a:
            accs = [
                (
                    np.zeros(d)
                    if acc_embed[i] is None
                    else np.asarray(acc_embed[i], dtype=float).copy()
                )
                for i in range(n_wp)
            ]
        elif has_emb_v:
            accs = [np.zeros(d) for _ in range(n_wp)]
        else:
            if acc_start is None:
                acc_start = np.zeros(d)
            else:
                acc_start = np.broadcast_to(np.atleast_1d(np.asarray(acc_start, dtype=float)), d)
            if acc_end is None:
                acc_end = np.zeros(d)
            else:
                acc_end = np.broadcast_to(np.atleast_1d(np.asarray(acc_end, dtype=float)), d)
            accs = [np.zeros(d) for _ in range(n_wp)]
            accs[0] = np.asarray(acc_start, dtype=float).copy()
            accs[-1] = np.asarray(acc_end, dtype=float).copy()

        t_ends = np.cumsum(durations)
        t_starts = np.concatenate([[0.0], t_ends[:-1]])
        self.segments = []
        for i in range(n):
            t_start, t_end = float(t_starts[i]), float(t_ends[i])
            vi = vels[i]
            vf = vels[i + 1]
            ai = accs[i]
            af = accs[i + 1]
            self.segments.append(
                FifthOrderTrajectory(
                    t_start,
                    t_end,
                    positions[i],
                    positions[i + 1],
                    vi=vi,
                    vf=vf,
                    ai=ai,
                    af=af,
                )
            )
        self.t_starts = t_starts
        self.total_time = float(t_ends[-1])
        self.num_segments = len(self.segments)

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.segments:
            return np.array([]), np.array([]), np.array([])
        if t >= self.total_time:
            last = self.segments[-1]
            return np.copy(last.xf), np.zeros_like(last.xf), np.zeros_like(last.xf)
        if t <= self.segments[0].ti:
            return self.segments[0].evaluate(self.segments[0].ti)
        idx = np.searchsorted(self.t_starts, t, side="right") - 1
        idx = int(np.clip(idx, 0, self.num_segments - 1))
        return self.segments[idx].evaluate(t)
