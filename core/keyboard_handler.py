"""Keyboard input handler for MuJoCo viewer: pause, exit, reset, speed, and custom key callbacks."""

import threading
from collections.abc import Callable


class KeyboardHandler:
    """Central handler for keyboard input: pause, exit, speed, and custom key callbacks.

    System keys (space, q/esc, r, +/-, etc.) are mapped to pause, exit, reset, and speed;
    additional keys can be registered via register_callback. Use create_key_callback()
    as the MuJoCo viewer key callback.
    """

    def __init__(self) -> None:
        """Initialize with default system handlers (pause, exit, reset, speed, camera)."""
        self._paused = False
        self._should_exit = False
        self._lock = threading.Lock()
        self._callbacks: dict[str, Callable] = {}
        self._reset_callback: Callable | None = None
        self._exit_callback: Callable | None = None
        self._viewer: object | None = None
        self._speed_factors = [0.25, 0.5, 0.75, 1.0, 2.0, 4.0, 8.0]
        self._speed_index = 3
        self._system_handlers: dict[str, Callable[[], None]] = {}
        self._setup_default_handlers()

    @property
    def paused(self) -> bool:
        """True if simulation is paused."""
        with self._lock:
            return self._paused

    def pause(self) -> None:
        """Set paused to True."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """Set paused to False."""
        with self._lock:
            self._paused = False

    def toggle_pause(self) -> None:
        """Flip paused state."""
        with self._lock:
            self._paused = not self._paused

    @property
    def should_exit(self) -> bool:
        """True if exit was requested (e.g. Q or ESC)."""
        with self._lock:
            return self._should_exit

    def request_exit(self) -> None:
        """Set should_exit to True and invoke exit callback if set."""
        with self._lock:
            self._should_exit = True

    @property
    def speed_factor(self) -> float:
        """Current playback speed multiplier (e.g. 1.0 = real time)."""
        with self._lock:
            return self._speed_factors[self._speed_index]

    def increase_speed(self) -> float | None:
        """Move to next higher speed. Returns new factor or None if already max.

        Returns:
            New speed factor, or None if at maximum.
        """
        with self._lock:
            if self._speed_index < len(self._speed_factors) - 1:
                self._speed_index += 1
                return self._speed_factors[self._speed_index]
        return None

    def decrease_speed(self) -> float | None:
        """Move to next lower speed. Returns new factor or None if already min.

        Returns:
            New speed factor, or None if at minimum.
        """
        with self._lock:
            if self._speed_index > 0:
                self._speed_index -= 1
                return self._speed_factors[self._speed_index]
        return None

    def set_speed_index(self, index: int) -> None:
        """Set speed by index into the speed factors list.

        Args:
            index: Valid index in [0, len(speed_factors)); ignored if out of range.
        """
        with self._lock:
            if 0 <= index < len(self._speed_factors):
                self._speed_index = index

    def register_callback(self, key: str, callback: Callable) -> None:
        """Register a callback for a key (lowercased). Overwrites any existing callback for that key.

        Args:
            key: Key character (e.g. "a"); stored lowercased.
            callback: Callable with no arguments, invoked when key is pressed.
        """
        self._callbacks[key.lower()] = callback

    def set_reset_callback(self, callback: Callable) -> None:
        """Set callback invoked on reset (e.g. R key).

        Args:
            callback: Callable with no arguments.
        """
        self._reset_callback = callback

    def set_exit_callback(self, callback: Callable) -> None:
        """Set callback invoked when exit is requested (e.g. Q/ESC).

        Args:
            callback: Callable with no arguments.
        """
        self._exit_callback = callback

    def set_viewer(self, viewer: object) -> None:
        """Set viewer reference used by the C key to print camera info.

        Args:
            viewer: MuJoCo passive viewer. Call inside the launch_passive context after the viewer is created.
        """
        self._viewer = viewer

    def _normalize_key(self, keycode: int) -> str | None:
        if keycode == 27:
            return "esc"
        if not (0 <= keycode <= 0x10FFFF):
            return None
        try:
            c = chr(keycode)
        except (ValueError, OverflowError):
            return None
        return c.lower() if len(c) == 1 else c

    def _trigger_reset(self) -> None:
        if self._reset_callback is not None:
            self._reset_callback()

    def _trigger_exit(self) -> None:
        self.request_exit()
        if self._exit_callback is not None:
            self._exit_callback()

    def _print_camera_info(self) -> None:
        """Print current viewer.cam (lookat, distance, azimuth, elevation) to terminal for copying into code."""
        v = self._viewer
        if v is None or not hasattr(v, "cam"):
            print("[Camera] Viewer not set. Call keyboard.set_viewer(viewer) after launch_passive.")
            return
        cam = v.cam
        lookat = getattr(cam, "lookat", None)
        distance = getattr(cam, "distance", None)
        azimuth = getattr(cam, "azimuth", None)
        elevation = getattr(cam, "elevation", None)
        print("=== Camera (viewer.cam) ===")
        if lookat is not None:
            try:
                la = list(lookat)
                print("  lookat:  [{:.6g}, {:.6g}, {:.6g}]".format(la[0], la[1], la[2]))
            except (TypeError, IndexError):
                print("  lookat:", lookat)
        if distance is not None:
            print("  distance: {:.6g}".format(float(distance)))
        if azimuth is not None:
            print("  azimuth: {:.6g}".format(float(azimuth)))
        if elevation is not None:
            print("  elevation: {:.6g}".format(float(elevation)))
        print("----------------------------")

    def _setup_default_handlers(self) -> None:
        self._system_handlers = {
            " ": self.toggle_pause,
            "q": self._trigger_exit,
            "esc": self._trigger_exit,
            "r": self._trigger_reset,
            "c": self._print_camera_info,
            "+": self.increase_speed,
            "=": self.increase_speed,
            "-": self.decrease_speed,
            "_": self.decrease_speed,
        }

    def _handle_key(self, keycode: int) -> bool:
        key_char = self._normalize_key(keycode)
        if key_char is None:
            return False
        if key_char in self._system_handlers:
            self._system_handlers[key_char]()
            return True
        if key_char in self._callbacks:
            self._callbacks[key_char]()
            return True
        return False

    def create_key_callback(self, viewer: object = None) -> Callable[[int], bool]:
        """Return a callback suitable for MuJoCo viewer: (keycode) -> bool.

        Args:
            viewer: Unused; for signature compatibility with viewer API.

        Returns:
            self._handle_key.
        """
        return self._handle_key

    def get_keyboard_help(self) -> str:
        """Return a multi-line string describing system and custom keys and current speed.

        Returns:
            Human-readable help text.
        """
        _key_descriptions: dict[str, str] = {
            " ": "Space: Pause/Resume",
            "r": "R: Reset",
            "q": "Q / ESC: Exit",
            "esc": "Q / ESC: Exit",
            "c": "C: Print camera info",
            "+": "+/=: Speed up",
            "=": "+/=: Speed up",
            "-": "-/_: Speed down",
            "_": "-/_: Speed down",
        }
        seen_descriptions: set[str] = set()
        lines = ["=== Keyboard (system) ==="]
        for k in self._system_handlers:
            desc = _key_descriptions.get(k, "")
            if desc and desc not in seen_descriptions:
                seen_descriptions.add(desc)
                lines.append(desc)
        if self._callbacks:
            lines.append("Custom: " + ", ".join(sorted(self._callbacks.keys())).upper())
        lines.append("----------------------")
        lines.append("Current speed: {:.2f}x".format(self.speed_factor))
        return "\n".join(lines)
