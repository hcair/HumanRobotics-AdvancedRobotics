"""Named extractors of numeric vectors from data (e.g. for plotting or logging)."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class Signal:
    """Named extractor of a numeric vector from arbitrary data (e.g. simulation state).

    Attributes:
        name: Display name and column prefix in logged/saved data.
        getter: Callable that takes one argument (e.g. data) and returns an array-like.
        indices: Optional indices to slice the getter output (e.g. first 3 DOF).
        ylim_min: Optional fixed y-axis minimum for plotting.
        ylim_max: Optional fixed y-axis maximum for plotting.
    """

    name: str
    getter: Callable
    indices: Iterable[int] | None = None
    ylim_min: float | None = None
    ylim_max: float | None = None

    def extract(self, data: object) -> np.ndarray:
        """Compute signal value from data via getter, optionally subselect indices.

        Args:
            data: Passed to self.getter(data).

        Returns:
            1D float array (possibly subset of getter output if indices is set).
        """
        values = np.asarray(self.getter(data), dtype=float)
        if self.indices is not None:
            values = values[list(self.indices)]
        return values
