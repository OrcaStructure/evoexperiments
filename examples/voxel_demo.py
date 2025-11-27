from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  needed for 3D proj

# Allow running directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evoexperiments import VoxelWorld  # noqa: E402
from evoexperiments.voxel_world import (
    BLOCK_OBSERVER,
    BLOCK_PISTON,
    BLOCK_REDSTONE,
    BLOCK_SLIME,
    BLOCK_STICKY_PISTON,
)


PALETTE = [
    "#8d6e63",  # wood
    "#607d8b",  # stone
    "#4caf50",  # grass
    "#03a9f4",  # water
    "#ff9800",  # sand
    "#9c27b0",  # purple block
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple voxel (Minecraft-like) scene viewer")
    p.add_argument("--example", choices=["piston", "sticky_line", "observer", "random"], default="piston")
    p.add_argument("--blocks", type=int, default=30, help="Number of random blocks to place (example=random)")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--extent", type=int, default=8, help="Half-span for random positions (centered at origin)")
    p.add_argument("--no-show", action="store_true", help="Skip interactive window")
    p.add_argument("--video", type=Path, help="Optional gif/mp4 output path for rotation sweep")
    p.add_argument("--frames", type=int, default=90, help="Frames for rotation sweep when saving video")
    return p.parse_args()


def populate_random(world: VoxelWorld, n: int, extent: int, rng: np.random.Generator) -> None:
    xs = rng.integers(-extent, extent + 1, size=n)
    ys = rng.integers(-extent, extent + 1, size=n)
    zs = rng.integers(0, extent + 1, size=n)  # stack upward from ground-ish
    tab20 = colormaps.get_cmap("tab20")
    for i in range(n):
        color = to_rgba(PALETTE[i % len(PALETTE)]) if i < len(PALETTE) else tab20(i % 20)
        world.add_block(int(xs[i]), int(ys[i]), int(zs[i]), color=color)


def populate_example(world: VoxelWorld, name: str, rng: np.random.Generator, extent: int, n: int) -> None:
    if name == "random":
        populate_random(world, n, extent, rng)
        return

    if name == "piston":
        # Redstone -> piston pushing a stone block
        world.add_block(0, 0, 0, color=PALETTE[1], block_type=BLOCK_REDSTONE)
        world.add_block(1, 0, 0, color=PALETTE[2], block_type=BLOCK_PISTON, facing=(1, 0, 0))
        world.add_block(2, 0, 0, color=PALETTE[0], block_type="stone")
    elif name == "sticky_line":
        # Sticky piston + slime chain of three blocks
        world.add_block(0, 0, 0, color=PALETTE[1], block_type=BLOCK_REDSTONE)
        world.add_block(1, 0, 0, color=PALETTE[3], block_type=BLOCK_STICKY_PISTON, facing=(1, 0, 0))
        for i in range(3):
            world.add_block(2 + i, 0, 0, color=PALETTE[(i + 2) % len(PALETTE)], block_type=BLOCK_SLIME)
    elif name == "observer":
        # Observer watching a slime block; pulse will power piston upward.
        world.add_block(0, 0, 0, color=PALETTE[0], block_type="stone")
        world.add_block(0, 1, 0, color=PALETTE[3], block_type=BLOCK_SLIME)
        world.add_block(0, 2, 0, color=PALETTE[4], block_type=BLOCK_OBSERVER, facing=(0, -1, 0))
        world.add_block(0, 1, 1, color=PALETTE[2], block_type=BLOCK_PISTON, facing=(0, 0, 1))
        world.add_block(0, 1, 2, color=PALETTE[5], block_type="stone")
    else:
        populate_random(world, n, extent, rng)


def render(world: VoxelWorld, save_path: Path | None, frames: int, show: bool, interactive: bool = True):
    occupied, colors = world.to_voxel_arrays()
    if occupied.size == 0:
        print("No blocks to render.")
        return None

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    def draw_scene():
        occ, col = world.to_voxel_arrays()
        ax.clear()
        ax.voxels(occ, facecolors=col, edgecolor="k", linewidth=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect([max(1, occ.shape[0]), max(1, occ.shape[1]), max(1, occ.shape[2])])
        ax.set_title("Voxel world (drag to orbit)")
        ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
        fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))
        fig.canvas.draw_idle()

    draw_scene()

    # Add an update button to step logic ticks.
    anim = None
    if interactive and show:
        ax_button = plt.axes([0.8, 0.02, 0.15, 0.05])
        btn = Button(ax_button, "Tick / Update")

        def on_click(event):
            world.tick()
            draw_scene()

        btn.on_clicked(on_click)

    if save_path:
        from matplotlib.animation import FuncAnimation

        def update(frame_idx: int):
            ax.view_init(elev=30, azim=frame_idx * (360 / frames))
            return ax,

        anim = FuncAnimation(fig, update, frames=frames, interval=60, repeat=True, blit=False)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    world = VoxelWorld()
    populate_example(world, args.example, rng, args.extent, args.blocks)

    render(world, save_path=args.video, frames=args.frames, show=not args.no_show, interactive=True)


if __name__ == "__main__":
    main()
