"""Real-time sync and rate limiting: map wall time to simulation time, throttle render/plot."""

import threading
import time


class RealtimeSync:
    """Maps wall-clock time to simulation time with a speed factor for real-time or faster/slower playback."""

    def __init__(self, speed_factor: float = 1.0) -> None:
        """Create sync; call reset(sim_time) to anchor wall time to simulation time.

        Args:
            speed_factor: Simulation time advance per second of wall time (1.0 = real time).
        """
        self._lock = threading.Lock()
        self.wall_start: float | None = None
        self.sim_start: float | None = None
        self._speed_factor = speed_factor

    def reset(self, sim_time: float) -> None:
        """Anchor current wall time to given simulation time.

        Args:
            sim_time: Simulation time that corresponds to "now".
        """
        with self._lock:
            self.wall_start = time.perf_counter()
            self.sim_start = sim_time

    @property
    def speed_factor(self) -> float:
        """Current multiplier: sim time per second of wall time."""
        with self._lock:
            return self._speed_factor

    def set_speed_factor(self, factor: float) -> None:
        """Set speed factor; re-anchors wall/sim so target_sim_time() stays continuous.

        Args:
            factor: New multiplier (clamped to at least 0.01).
        """
        with self._lock:
            if self.wall_start is not None:
                current_sim_time = self._target_sim_time_internal()
                self.wall_start = time.perf_counter()
                self.sim_start = current_sim_time
            self._speed_factor = max(0.01, factor)

    def target_sim_time(self) -> float:
        """Simulation time that should be "current" given elapsed wall time and speed factor.

        Returns:
            sim_start + (elapsed_wall * speed_factor), or 0.0 if reset() never called.
        """
        with self._lock:
            return self._target_sim_time_internal()

    def _target_sim_time_internal(self) -> float:
        if self.wall_start is None:
            return 0.0
        wall_now = time.perf_counter()
        elapsed_wall_time = wall_now - self.wall_start
        return self.sim_start + (elapsed_wall_time * self._speed_factor)


class RateLimiter:
    """Emits ready=True at most once per period in simulation time."""

    def __init__(self, start_time: float, period: float) -> None:
        """Initialize next trigger at start_time; then every period.

        Args:
            start_time: First time at which ready() will return True.
            period: Minimum sim-time interval between True returns.
        """
        self.next_time = start_time
        self.period = period

    def ready(self, t: float) -> bool:
        """Return True if t has reached or passed next_time, then advance next_time by period.

        Args:
            t: Current simulation time.

        Returns:
            True if this call is the one that crosses the threshold (and advances next_time).
        """
        if t >= self.next_time:
            self.next_time += self.period
            return True
        return False

    def reset(self, t: float) -> None:
        """Set next trigger time to t.

        Args:
            t: New next_time value.
        """
        self.next_time = t


class RealtimeConfig:
    """Default rates and helpers for render/plot timesteps."""

    DEFAULT_RENDER_HZ = 60.0
    DEFAULT_PLOT_HZ = 60.0

    @staticmethod
    def hz_to_dt(hz: float | None, default: float) -> float:
        """Convert frequency (Hz) to period (seconds). Invalid or None uses default.

        Args:
            hz: Desired frequency, or None / non-positive to use default.
            default: Fallback period in seconds.

        Returns:
            Period in seconds (1/hz or default).
        """
        actual_hz = hz if (hz is not None and hz > 0) else default
        return 1.0 / actual_hz

    @classmethod
    def render_dt(cls, render_hz: float | None = None) -> float:
        """Period for render updates; uses DEFAULT_RENDER_HZ if render_hz not given.

        Args:
            render_hz: Optional override in Hz.

        Returns:
            Period in seconds.
        """
        return cls.hz_to_dt(render_hz, cls.DEFAULT_RENDER_HZ)

    @classmethod
    def plot_dt(cls, plot_hz: float | None = None) -> float:
        """Period for plot updates; uses DEFAULT_PLOT_HZ if plot_hz not given.

        Args:
            plot_hz: Optional override in Hz.

        Returns:
            Period in seconds.
        """
        return cls.hz_to_dt(plot_hz, cls.DEFAULT_PLOT_HZ)
