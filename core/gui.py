"""Unified Tk GUI: parameter panels, plots, buttons; sync with model/data and apply/reset."""

import logging
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from core.signal import Signal

logger = logging.getLogger(__name__)


_GUI_FONT_SIZE = 12
_GUI_PAD = 12
_GUI_ENTRY_WIDTH = 14
_GUI_BTN_WIDTH = 20


@dataclass
class GUIParameter:
    """Single parameter for the unified GUI: scalar or vector, with optional setter/getter and angle display.

    Attributes:
        name: Label for the parameter panel.
        param_type: "float", "int", "bool", or "vector".
        default: Default value (number, bool, or list of numbers for vector).
        min_val: Optional scalar min (float/int/bool).
        max_val: Optional scalar max.
        min_vals: Per-element min for vector (overrides min_val).
        max_vals: Per-element max for vector (overrides max_val).
        step: Optional step (unused by current widgets).
        labels: Optional list of labels per vector element.
        setter: Optional (model, data, value) callable to apply value.
        getter: Optional (model, data) -> value callable to read current value.
        angle_indices: Indices of vector elements to show in degrees (radians internally).
        read_only: If True, widget is display-only (Label/read-only Entry); getter is used to sync each frame.
    """

    name: str
    param_type: Literal["float", "int", "bool", "vector"]
    default: Any
    min_val: float | None = None
    max_val: float | None = None
    min_vals: list[float] | None = None
    max_vals: list[float] | None = None
    step: float | None = None
    labels: list[str] | None = None
    setter: Callable | None = None
    getter: Callable | None = None
    angle_indices: list[int] | None = None
    read_only: bool = False

    _value: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self._value is None:
            self._value = self.default

    def value_to_display(self, value: Any) -> Any:
        if self.param_type != "vector" or not self.angle_indices:
            return value
        value = list(value) if isinstance(value, (list, tuple)) else [value]
        angle_set = set(self.angle_indices)
        return [float(x) * (180.0 / math.pi) if i in angle_set else float(x) for i, x in enumerate(value)]

    def display_to_value(self, display: Any) -> Any:
        if self.param_type != "vector" or not self.angle_indices:
            return display
        display = list(display) if isinstance(display, (list, tuple)) else [display]
        angle_set = set(self.angle_indices)
        return [float(x) * (math.pi / 180.0) if i in angle_set else float(x) for i, x in enumerate(display)]

    def value_to_display_element(self, value: float, index: int) -> float:
        if self.angle_indices and index in set(self.angle_indices):
            return float(value) * (180.0 / math.pi)
        return float(value)

    def display_to_value_element(self, display: float, index: int) -> float:
        if self.angle_indices and index in set(self.angle_indices):
            return float(display) * (math.pi / 180.0)
        return float(display)


class _ParameterWidgetManager:
    """Builds and manages Tk widgets for a list of GUIParameters and syncs with model/data."""

    def __init__(
        self,
        parameters: list[GUIParameter],
        model: Any,
        data: Any,
        *,
        status_only: bool = False,
    ) -> None:
        self._parameters = parameters
        self._model = model
        self._data = data
        self._param_widgets: list[tuple[list[Any], list[Any]]] = []
        self._status_only = status_only
        self._read_only_widgets: list[tuple[list[Any], list[Any]]] = []

    def _get_float_from_widget(self, widget_or_var: Any, default: float) -> float:
        try:
            return float(widget_or_var.get())
        except (ValueError, TypeError, AttributeError):
            return default

    def create_panel(self, parent: tk.Widget, ctrl: "UnifiedGUI") -> ttk.Frame:
        """Build parameter widgets and Apply/Reset buttons; sync initial values from getters."""
        for p in self._parameters:
            p._value = p.default
            if p.getter is not None:
                try:
                    p._value = p.getter(self._model, self._data)
                except Exception as e:
                    logger.warning("Getter failed for parameter '%s': %s", p.name, e)
            if isinstance(p._value, (list, tuple)):
                p._value = list(p._value)

        frame = ttk.Frame(parent)
        self._param_widgets.clear()
        for param in self._parameters:
            vars_list, entry_list = self._create_param_widget(frame, param)
            self._param_widgets.append((vars_list, entry_list))
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=(_GUI_PAD, 0))
        ttk.Button(btn_frame, text="Apply Changes", command=ctrl.request_apply, width=_GUI_BTN_WIDTH).pack(side=tk.LEFT, padx=_GUI_PAD)
        ttk.Button(btn_frame, text="Reset to Default", command=ctrl.request_reset, width=_GUI_BTN_WIDTH).pack(side=tk.LEFT, padx=_GUI_PAD)
        return frame

    def create_status_panel(self, parent: tk.Widget) -> ttk.Frame:
        """Build read-only parameter widgets (Label/read-only Entry) for monitoring; no Apply/Reset.
        All parameters in this manager are expected to have read_only=True and a getter.
        """
        for p in self._parameters:
            p._value = p.default
            if p.getter is not None:
                try:
                    p._value = p.getter(self._model, self._data)
                except Exception as e:
                    logger.warning("Getter failed for parameter '%s': %s", p.name, e)
            if isinstance(p._value, (list, tuple)):
                p._value = list(p._value)

        frame = ttk.Frame(parent)
        self._param_widgets.clear()
        self._read_only_widgets.clear()
        for param in self._parameters:
            vars_list, widget_list = self._create_read_only_param_widget(frame, param)
            self._param_widgets.append((vars_list, widget_list))
            self._read_only_widgets.append((vars_list, widget_list))
        return frame

    def _create_read_only_param_widget(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        """Create display-only widget (ttk.Label with StringVar or DoubleVar) for one parameter."""
        frame = ttk.LabelFrame(parent, text=param.name, padding=_GUI_PAD)
        frame.pack(fill=tk.X, pady=_GUI_PAD)
        if param.param_type == "float":
            return self._create_read_only_float(frame, param)
        if param.param_type == "int":
            return self._create_read_only_int(frame, param)
        if param.param_type == "bool":
            return self._create_read_only_bool(frame, param)
        if param.param_type == "vector":
            return self._create_read_only_vector(frame, param)
        return ([], [])

    def _create_read_only_float(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        var = tk.DoubleVar(value=float(param._value))
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=_GUI_PAD // 2)
        ttk.Label(row, text="Value:").pack(side=tk.LEFT)
        ttk.Label(row, textvariable=var, width=_GUI_ENTRY_WIDTH).pack(side=tk.LEFT, padx=_GUI_PAD)
        return ([var], [])

    def _create_read_only_int(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        var = tk.IntVar(value=int(param._value))
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=_GUI_PAD // 2)
        ttk.Label(row, text="Value:").pack(side=tk.LEFT)
        ttk.Label(row, textvariable=var, width=_GUI_ENTRY_WIDTH).pack(side=tk.LEFT, padx=_GUI_PAD)
        return ([var], [])

    def _create_read_only_bool(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        var = tk.BooleanVar(value=bool(param._value))
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=_GUI_PAD // 2)
        ttk.Label(row, text="Value:").pack(side=tk.LEFT)
        ttk.Label(row, textvariable=var, width=_GUI_ENTRY_WIDTH).pack(side=tk.LEFT, padx=_GUI_PAD)
        return ([var], [])

    def _create_read_only_vector(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        if not isinstance(param._value, (list, tuple)):
            param._value = list(param.default)
        display_values = param.value_to_display(param._value)
        value_vars: list[Any] = []
        angle_set = set(param.angle_indices or [])
        labels = param.labels or [f"[{i}]" for i in range(len(param._value))]
        for i, (label, display_val) in enumerate(zip(labels, display_values)):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=_GUI_PAD // 2)
            var = tk.DoubleVar(value=float(display_val))
            value_vars.append(var)
            unit = " (°)" if i in angle_set else ""
            ttk.Label(row, text=f"{label}{unit}:").pack(side=tk.LEFT, padx=_GUI_PAD)
            ttk.Label(row, textvariable=var, width=_GUI_ENTRY_WIDTH).pack(side=tk.LEFT)
        return (value_vars, [])

    def read_from_widgets(self) -> list[Any]:
        result: list[Any] = []
        for i, param in enumerate(self._parameters):
            vars_list, entry_list = self._param_widgets[i]
            if param.param_type == "vector":
                raw_list = []
                default_vals = list(param.default) if isinstance(param.default, (list, tuple)) else []
                for j in range(len(vars_list)):
                    default = default_vals[j] if j < len(default_vals) else 0.0
                    w = entry_list[j] if (entry_list and j < len(entry_list)) else vars_list[j]
                    raw_list.append(self._get_float_from_widget(w, default))
                result.append(param.display_to_value(raw_list))
            else:
                result.append(vars_list[0].get())
        return result

    def write_to_widgets(self, values: list[Any]) -> None:
        for i, param in enumerate(self._parameters):
            vars_list, entry_list = self._param_widgets[i]
            val = values[i] if i < len(values) else param.default
            if param.param_type == "vector":
                val = list(val) if isinstance(val, (list, tuple)) else [val]
                display_list = param.value_to_display(val)
                for j, var in enumerate(vars_list):
                    if j < len(display_list):
                        var.set(display_list[j])
            else:
                vars_list[0].set(val)

    def _create_param_widget(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        frame = ttk.LabelFrame(parent, text=param.name, padding=_GUI_PAD)
        frame.pack(fill=tk.X, pady=_GUI_PAD)
        if param.param_type == "float":
            return self._create_float_widget(frame, param)
        if param.param_type == "int":
            return self._create_int_widget(frame, param)
        if param.param_type == "bool":
            return self._create_bool_widget(frame, param)
        if param.param_type == "vector":
            return self._create_vector_widget(frame, param)
        return ([], [])

    def _create_float_widget(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        value_var = tk.DoubleVar(value=param._value)
        if param.min_val is not None and param.max_val is not None:
            ttk.Scale(parent, from_=param.min_val, to=param.max_val, variable=value_var, orient=tk.HORIZONTAL, length=200).pack(fill=tk.X, pady=_GUI_PAD // 2)
        entry_frame = ttk.Frame(parent)
        entry_frame.pack(fill=tk.X)
        ttk.Label(entry_frame, text="Value:").pack(side=tk.LEFT)
        ttk.Entry(entry_frame, textvariable=value_var, width=_GUI_ENTRY_WIDTH).pack(side=tk.LEFT, padx=_GUI_PAD)
        return ([value_var], [])

    def _create_int_widget(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        value_var = tk.IntVar(value=int(param._value))
        ttk.Entry(parent, textvariable=value_var, width=_GUI_ENTRY_WIDTH).pack(fill=tk.X, pady=_GUI_PAD // 2)
        return ([value_var], [])

    def _create_bool_widget(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        value_var = tk.BooleanVar(value=bool(param._value))
        ttk.Checkbutton(parent, text="Enabled", variable=value_var).pack(pady=_GUI_PAD // 2)
        return ([value_var], [])

    def _create_vector_widget(self, parent: ttk.Frame, param: GUIParameter) -> tuple[list[Any], list[Any]]:
        if not isinstance(param._value, (list, tuple)):
            param._value = list(param.default)
        display_values = param.value_to_display(param._value)
        value_vars: list[Any] = []
        entry_widgets: list[Any] = []
        angle_set = set(param.angle_indices or [])
        labels = param.labels or [f"[{i}]" for i in range(len(param._value))]
        for i, (label, display_val) in enumerate(zip(labels, display_values)):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=_GUI_PAD // 2)
            var = tk.DoubleVar(value=float(display_val))
            value_vars.append(var)
            min_v = param.min_vals[i] if param.min_vals and i < len(param.min_vals) else param.min_val
            max_v = param.max_vals[i] if param.max_vals and i < len(param.max_vals) else param.max_val
            unit = " (°)" if i in angle_set else ""
            ttk.Label(row, text=f"{label}{unit}:").pack(side=tk.LEFT, padx=_GUI_PAD)
            if min_v is not None and max_v is not None:
                sl_min = param.value_to_display_element(min_v, i)
                sl_max = param.value_to_display_element(max_v, i)
                ttk.Scale(row, from_=sl_min, to=sl_max, variable=var, orient=tk.HORIZONTAL, length=180).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=_GUI_PAD)
            entry = ttk.Entry(row, textvariable=var, width=_GUI_ENTRY_WIDTH)
            entry.pack(side=tk.LEFT)
            entry_widgets.append(entry)
        return (value_vars, entry_widgets)


class _EmbeddedPlot:
    """Matplotlib figure embedded in Tk: rolling time-series of signals with fixed time window."""

    def __init__(
        self,
        parent: tk.Widget,
        signals: list[Signal],
        dt: float,
        window_sec: float = 5.0,
        figsize: tuple[float, float] = (6, 4),
    ) -> None:
        self.signals = signals
        self.dt = dt
        self.window_sec = window_sec
        self.maxlen = max(10, int(window_sec / dt))
        self.t_buf: deque = deque(maxlen=self.maxlen)
        self.y_buf: list[deque] = [deque(maxlen=self.maxlen) for _ in signals]
        self.lines: list[list] = []

        self.fig = Figure(figsize=figsize)
        n = len(signals)
        if n > 1:
            axes_arr = self.fig.subplots(n, 1, sharex=True)
            self.axes = list(axes_arr.flat)
        else:
            self.axes = [self.fig.subplots(1, 1)]
        for ax, sig in zip(self.axes, signals):
            ax.set_ylabel(sig.name)
            ax.grid(True)
            if sig.reference is not None:
                label = sig.reference_label or f"ref = {sig.reference}"
                ax.axhline(y=sig.reference, color="gray", linestyle="--", label=label)
            self.lines.append([])
        self.axes[-1].set_xlabel("Time (s)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def push(self, t: float, data: Any) -> None:
        self.t_buf.append(float(t))
        for buf, sig in zip(self.y_buf, self.signals):
            buf.append(sig.extract(data))

    def update(self) -> None:
        if not self.t_buf:
            return
        t = list(self.t_buf)
        for i, (ax, sig, buf) in enumerate(zip(self.axes, self.signals, self.y_buf)):
            y = np.asarray(buf)
            if not self.lines[i]:
                for k in range(y.shape[1]):
                    label = sig.name if y.shape[1] == 1 else f"{sig.name}[{k}]"
                    (line,) = ax.plot([], [], label=label)
                    self.lines[i].append(line)
                ax.legend(loc="upper left")
            for k, line in enumerate(self.lines[i]):
                line.set_data(t, y[:, k])
            ax.set_xlim(t[-1] - self.window_sec, t[-1])
            if sig.ylim_min is not None or sig.ylim_max is not None:
                y_min = sig.ylim_min if sig.ylim_min is not None else ax.get_ylim()[0]
                y_max = sig.ylim_max if sig.ylim_max is not None else ax.get_ylim()[1]
                ax.set_ylim(y_min, y_max)
            else:
                ax.relim()
                ax.autoscale_view(scaley=True)
                if sig.reference is not None:
                    ymin, ymax = ax.get_ylim()
                    ax.set_ylim(min(ymin, sig.reference - 0.5), max(ymax, sig.reference + 0.5))
        try:
            self.canvas.draw_idle()
            self.canvas.flush_events()
        except tk.TclError as e:
            logger.warning("Plot canvas update failed: %s", e)

    def clear(self) -> None:
        self.t_buf.clear()
        for buf in self.y_buf:
            buf.clear()


class _BuildAPI:
    """Thin API passed to build_content: forwards to UnifiedGUI for panels, buttons, plots."""

    def __init__(self, ctrl: "UnifiedGUI") -> None:
        self._ctrl = ctrl

    def add_parameter_panel(
        self,
        parent: tk.Widget,
        parameters: list[GUIParameter],
        model: Any,
        data: Any,
    ) -> ttk.Frame:
        return self._ctrl.add_parameter_panel(parent, parameters, model, data)

    def add_status_panel(
        self,
        parent: tk.Widget,
        parameters: list[GUIParameter],
        model: Any,
        data: Any,
    ) -> ttk.Frame:
        return self._ctrl.add_status_panel(parent, parameters, model, data)

    def add_button_row(
        self,
        parent: tk.Widget,
        buttons: list[tuple[str, Callable[[], None]]],
    ) -> ttk.Frame:
        return self._ctrl.add_button_row(parent, buttons)

    def add_plot_area(
        self,
        parent: tk.Widget,
        signals: list[Signal],
        dt: float,
        window_sec: float = 5.0,
        figsize: tuple[float, float] = (6, 4),
    ) -> tuple[ttk.Frame, Any]:
        return self._ctrl.add_plot_area(parent, signals, dt, window_sec, figsize)


class UnifiedGUI:
    """Tk + Matplotlib GUI: parameter panels, buttons, optional plots; syncs with model/data and reset/apply."""

    def __init__(
        self,
        model: Any,
        data: Any,
        title: str = "Simulation",
        build_content: Callable[[tk.Widget, _BuildAPI], None] | None = None,
        auto_start: bool = True,
    ) -> None:
        """Create GUI; if auto_start and build_content given, start() is called immediately."""
        self.model = model
        self.data = data
        self.title = title
        self.build_content = build_content or (lambda p, api: None)
        self.parameters: list[GUIParameter] = []
        self._reset_callback: Callable[[], None] | None = None
        self.root: tk.Tk | None = None
        self._lock = threading.Lock()
        self._running = False
        self._pending_update = False
        self._pending_reset = False
        self._panel_managers: list[_ParameterWidgetManager] = []
        self._status_managers: list[_ParameterWidgetManager] = []

        if auto_start and build_content:
            self.start()

    def start(self) -> None:
        """Create Tk window, run build_content, and show GUI. Idempotent if already running."""
        if self._running:
            return
        self._running = True
        try:
            self.root = tk.Tk()
            self.root.title(self.title)
            self.root.protocol("WM_DELETE_WINDOW", self.stop)
            font = ("Segoe UI", _GUI_FONT_SIZE)
            self.root.option_add("*Font", font)
            style = ttk.Style()
            style.configure("TButton", padding=(_GUI_PAD, _GUI_PAD // 2))
            style.configure("TLabelframe.Label", font=font)
            style.configure("TLabel", font=font)
            main_frame = ttk.Frame(self.root, padding=_GUI_PAD * 2)
            main_frame.pack(fill=tk.BOTH, expand=True)
            api = _BuildAPI(self)
            self.build_content(main_frame, api)
            self.root.minsize(400, 300)
            self.root.update_idletasks()
            self.root.update()
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.after(100, self._clear_topmost)
        except Exception as e:
            logger.exception("GUI start failed: %s", e)
            self._running = False

    def _clear_topmost(self) -> None:
        """Clear topmost so window can go behind others; safe if root already destroyed."""
        if self.root is not None:
            try:
                self.root.attributes("-topmost", False)
            except tk.TclError:
                pass

    def stop(self) -> None:
        """Quit and destroy Tk root; set _running False."""
        self._running = False
        if self.root is not None:
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass
        self.root = None

    def update(self) -> None:
        """Process Tk events (update_idletasks + update). Call from simulation loop."""
        if self.root is not None and self._running:
            try:
                self.sync_status_widgets()
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                self._running = False
            except Exception:
                pass

    def sync_status_widgets(self) -> None:
        """Read current values from all read-only (status) parameters via getters and update their widgets."""
        if not self._status_managers:
            return
        with self._lock:
            for manager in self._status_managers:
                values: list[Any] = []
                for param in manager._parameters:
                    if param.getter is not None:
                        try:
                            val = param.getter(self.model, self.data)
                            values.append(list(val) if isinstance(val, (list, tuple)) else val)
                        except Exception as e:
                            logger.warning("Getter failed for parameter '%s': %s", param.name, e)
                            values.append(
                                list(param.default) if isinstance(param.default, (list, tuple)) else param.default
                            )
                    else:
                        values.append(
                            list(param.default) if isinstance(param.default, (list, tuple)) else param.default
                        )
                manager.write_to_widgets(values)

    def set_reset_callback(self, callback: Callable[[], None] | None) -> None:
        """Set callback invoked when user presses Reset (and on next check_and_apply_pending_reset)."""
        self._reset_callback = callback

    def add_parameter_panel(
        self,
        parent: tk.Widget,
        parameters: list[GUIParameter],
        model: Any,
        data: Any,
    ) -> ttk.Frame:
        """Add a panel of parameter widgets and Apply/Reset; returns the frame. Updates self.model/data."""
        self.model = model
        self.data = data
        self.parameters.extend(parameters)
        manager = _ParameterWidgetManager(parameters, model, data, status_only=False)
        self._panel_managers.append(manager)
        return manager.create_panel(parent, self)

    def add_status_panel(
        self,
        parent: tk.Widget,
        parameters: list[GUIParameter],
        model: Any,
        data: Any,
    ) -> ttk.Frame:
        """Add a read-only panel that displays current values via getters; no Apply/Reset. Synced each frame."""
        self.model = model
        self.data = data
        self.parameters.extend(parameters)
        manager = _ParameterWidgetManager(parameters, model, data, status_only=True)
        self._status_managers.append(manager)
        return manager.create_status_panel(parent)

    def add_button_row(
        self,
        parent: tk.Widget,
        buttons: list[tuple[str, Callable[[], None]]],
    ) -> ttk.Frame:
        """Add a row of (label, command) buttons; returns the frame."""
        frame = ttk.Frame(parent)
        for label, cmd in buttons:
            ttk.Button(frame, text=label, command=cmd, width=_GUI_BTN_WIDTH).pack(side=tk.LEFT, padx=_GUI_PAD)
        return frame

    def add_plot_area(
        self,
        parent: tk.Widget,
        signals: list[Signal],
        dt: float,
        window_sec: float = 5.0,
        figsize: tuple[float, float] = (6, 4),
    ) -> tuple[ttk.Frame, Any]:
        """Add rolling time-series plot; returns (frame, plot_api). Call plot_api.push(t, data) and plot_api.update()."""
        if not signals:
            return (ttk.Frame(parent), None)
        frame = ttk.Frame(parent)
        plot_api = _EmbeddedPlot(frame, signals, dt, window_sec, figsize)
        return (frame, plot_api)

    def request_apply(self) -> None:
        """Copy widget values into parameters and set _pending_update so next check_and_apply_pending_update runs setters."""
        with self._lock:
            for manager in self._panel_managers:
                values = manager.read_from_widgets()
                for i, param in enumerate(manager._parameters):
                    if i < len(values):
                        param._value = values[i]
            if self._panel_managers:
                self._pending_update = True

    def request_reset(self) -> None:
        """Reset parameter values to defaults, write to widgets, set _pending_update and _pending_reset."""
        with self._lock:
            for manager in self._panel_managers:
                for param in manager._parameters:
                    param._value = list(param.default) if isinstance(param.default, (list, tuple)) else param.default
                manager.write_to_widgets([p._value for p in manager._parameters])
            if self._panel_managers:
                self._pending_update = True
            if self._reset_callback is not None:
                self._pending_reset = True

    def get_value(self, param_name: str) -> Any:
        """Return current value of the first parameter with the given name, or None."""
        with self._lock:
            for param in self.parameters:
                if param.name == param_name:
                    if isinstance(param._value, (list, tuple)):
                        return list(param._value)
                    return param._value
        return None

    def check_and_apply_pending_update(self) -> bool:
        """If Apply was requested, run all parameter setters with current _value and clear pending. Returns True if applied."""
        if not self.parameters:
            return False
        with self._lock:
            if not self._pending_update:
                return False
            self._pending_update = False
            for param in self.parameters:
                try:
                    if param.setter is not None:
                        param.setter(self.model, self.data, param._value)
                except Exception as e:
                    logger.warning("Setter failed for parameter '%s': %s", param.name, e)
        return True

    def check_and_apply_pending_reset(self) -> bool:
        """If Reset was requested, call reset callback and clear pending. Returns True if reset was pending."""
        with self._lock:
            if not self._pending_reset:
                return False
            self._pending_reset = False
        if self._reset_callback is not None:
            try:
                self._reset_callback()
            except Exception:
                pass
        return True