"""Time-series logging from named signals; save to CSV or XLSX."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.signal import Signal

logger = logging.getLogger(__name__)


class DataLogger:
    """Buffers time-series data from named signals and writes to CSV or XLSX on save."""

    def __init__(
        self,
        signals: list[Signal],
        output_dir: Path,
        filename_prefix: str = "",
        enabled: bool = False,
    ) -> None:
        """Initialize logger with signal definitions and output directory.

        Args:
            signals: List of Signal instances defining what to extract from each log() call.
            output_dir: Directory for saved files; created if missing.
            filename_prefix: Prepended to saved filenames (e.g. "run1" -> "run1_20250101_120000.csv").
            enabled: If False, log() is a no-op until set_enabled(True).
        """
        self.signals = signals
        self.output_dir = Path(output_dir)
        self.filename_prefix = filename_prefix
        self.enabled = enabled

        self.times: list[float] = []
        self.data_buffers: list[list[np.ndarray]] = [[] for _ in signals]
        self._last_saved_count = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable logging. If disabling and buffer has data, saves and clears first.

        Args:
            enabled: True to record on log(), False to stop and optionally flush.
        """
        was_enabled = self.enabled
        self.enabled = enabled

        if not enabled and was_enabled and self.times:
            self.save_and_clear()

    def log(self, t: float, data: Any) -> None:
        """Append one time point and extracted signal values to the buffer.

        Args:
            t: Timestamp (e.g. simulation time).
            data: Arbitrary object passed to each Signal.extract(data).
        """
        if not self.enabled:
            return
        self.times.append(float(t))
        for i, sig in enumerate(self.signals):
            self.data_buffers[i].append(sig.extract(data).copy())

    def save(self, clear_after_save: bool = False, format: str = "csv") -> Path | None:
        """Write buffered data to a file. Skips if buffer empty or unchanged since last save.

        Args:
            clear_after_save: If True, clear buffers after writing.
            format: "csv" or "xlsx". Default "csv".

        Returns:
            Path to the saved file, or None if nothing was written.
        """
        if not self.times:
            return None
        if len(self.times) == self._last_saved_count:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format == "csv":
            ext = ".csv"
        else:
            ext = ".xlsx"
        name = f"{self.filename_prefix}_{timestamp}{ext}" if self.filename_prefix else f"data_{timestamp}{ext}"
        filepath = self.output_dir / name
        times_array = np.array(self.times)

        data_dict: dict = {"time": times_array}
        for i, sig in enumerate(self.signals):
            arr = np.array(self.data_buffers[i])
            if arr.ndim == 1:
                data_dict[sig.name] = arr
            else:
                for j in range(arr.shape[1]):
                    data_dict[f"{sig.name}[{j}]"] = arr[:, j]
        df = pd.DataFrame(data_dict)
        if format == "csv":
            df.to_csv(filepath, index=False)
        else:
            df.to_excel(filepath, index=False, engine="openpyxl")

        self._last_saved_count = len(self.times)
        if clear_after_save:
            self.clear()
            self._last_saved_count = 0

        logger.info("Saved: %s", filepath)
        return filepath

    def save_and_clear(self) -> Path | None:
        """Save buffer to file and clear buffers. Equivalent to save(clear_after_save=True).

        Returns:
            Path to saved file or None.
        """
        return self.save(clear_after_save=True)

    def clear(self) -> None:
        """Clear time and signal buffers without saving."""
        self.times.clear()
        for buf in self.data_buffers:
            buf.clear()
        self._last_saved_count = 0
