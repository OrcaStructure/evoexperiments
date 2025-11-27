from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional
import json

import numpy as np

ArrayLike = np.ndarray


@dataclass
class GridSimulation:
    """
    Wraps a 2D grid with an update rule and records its history.

    The update rule should accept the current grid (numpy array) and return the
    next grid of the same shape.
    """

    initial_state: ArrayLike
    update_rule: Callable[[ArrayLike], ArrayLike]
    dtype: Optional[np.dtype] = None
    history: list[ArrayLike] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        state = np.asarray(self.initial_state)
        self.dtype = self.dtype or state.dtype
        self.initial_state = state.astype(self.dtype, copy=False)
        # Store a copy so external references can't mutate our history.
        self.history.append(self.initial_state.copy())

    @property
    def state(self) -> ArrayLike:
        return self.history[-1]

    def step(self) -> ArrayLike:
        """Advance the simulation by one time step and record the state."""
        next_state = np.asarray(self.update_rule(self.state)).astype(self.dtype, copy=False)
        if next_state.shape != self.state.shape:
            raise ValueError("Update rule must preserve grid shape")
        self.history.append(next_state.copy())
        return next_state

    def run(self, steps: int) -> list[ArrayLike]:
        """Run the simulation for a number of steps, recording each state."""
        for _ in range(steps):
            self.step()
        return self.history

    def save_history(self, path: str | Path, metadata: Optional[dict] = None) -> Path:
        """
        Save the full history to disk as a compressed NumPy archive.

        The archive stores an array shaped (frames, rows, cols) and an optional
        JSON metadata string for small descriptors (rule name, seed, etc.).
        """
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        history_array = np.stack(self.history)
        metadata_json = json.dumps(metadata or {})
        np.savez_compressed(dest, history=history_array, metadata=metadata_json)
        return dest

    @staticmethod
    def load_history(path: str | Path) -> tuple[np.ndarray, dict]:
        """Load a saved history file produced by ``save_history``."""
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(str(archive.get("metadata", "{}")))
        return archive["history"], metadata

    @classmethod
    def from_history(
        cls, history: Iterable[ArrayLike], update_rule: Callable[[ArrayLike], ArrayLike], dtype: Optional[np.dtype] = None
    ) -> "GridSimulation":
        """
        Create a simulation seeded with an existing history.

        The returned simulation contains the provided frames; subsequent steps
        continue from the last one.
        """
        history_list = [np.asarray(frame).astype(dtype or np.asarray(frame).dtype, copy=True) for frame in history]
        if not history_list:
            raise ValueError("History must contain at least one frame")
        sim = cls(initial_state=history_list[0], update_rule=update_rule, dtype=dtype)
        # Replace the automatically added first frame with the full provided history.
        sim.history = history_list
        return sim
