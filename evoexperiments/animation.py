from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from .grid import GridSimulation


def _prepare_history(history: Iterable[np.ndarray]) -> np.ndarray:
    arr = np.asarray(list(history))
    if arr.ndim != 3:
        raise ValueError("History must be an array shaped (frames, rows, cols)")
    return arr


def animate_history(
    history: Iterable[np.ndarray],
    interval: int = 200,
    cmap: str = "magma",
    title: Optional[str] = None,
    repeat: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Render a saved history as an animation.

    ``show=False`` is useful for headless runs when only saving output.
    """
    frames = _prepare_history(history)
    fig, ax = plt.subplots()
    im = ax.imshow(frames[0], cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame_idx: int):
        im.set_data(frames[frame_idx])
        if title:
            ax.set_title(f"{title} (t={frame_idx})")
        return [im]

    animation = FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=repeat, blit=False)

    if save_path:
        dest = Path(save_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        animation.save(dest)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return animation


def animate_simulation(
    simulation: GridSimulation,
    steps: int,
    interval: int = 200,
    cmap: str = "magma",
    title: Optional[str] = None,
    repeat: bool = True,
    show: bool = True,
    save_path: str | Path | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """
    Run a simulation and animate the recorded frames.

    The simulation keeps its expanded history after this call so you can save
    and reload it later.
    """
    simulation.run(steps)
    return animate_history(
        simulation.history,
        interval=interval,
        cmap=cmap,
        title=title,
        repeat=repeat,
        show=show,
        save_path=save_path,
        vmin=vmin,
        vmax=vmax,
    )
