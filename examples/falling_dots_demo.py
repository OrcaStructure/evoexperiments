from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.animation import FuncAnimation


# Allow running directly via ``python examples/falling_dots_demo.py``
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evoexperiments import FallingDotsWorld, animate_history
from evoexperiments.falling_dots import CREATURE_BASE, CREATURE_COLORS, COLOR_BINS, COLOR_DIMS


def build_cmap() -> tuple[ListedColormap, BoundaryNorm]:
    # 0 empty, 1 pellet, 2 ground, 3+ creature hues
    base = [
        "#f5f5f5",  # empty
        "#ffb74d",  # pellet
        "#8d6e63",  # ground
    ]
    # Creature palette follows the same bin packing used in the simulation.
    creature_colors = []
    cmap_source = colormaps.get_cmap("magma")
    for r_bin in range(COLOR_BINS):
        for g_bin in range(COLOR_BINS):
            for b_bin in range(COLOR_BINS):
                r = r_bin / max(1, COLOR_BINS - 1)
                g = g_bin / max(1, COLOR_BINS - 1)
                b = b_bin / max(1, COLOR_BINS - 1)
                # Mix with magma to keep some visual cohesion.
                blend = cmap_source((r + g + b) / (3 * max(1, COLOR_BINS - 1)))
                creature_colors.append((r * 0.6 + blend[0] * 0.4, g * 0.6 + blend[1] * 0.4, b * 0.6 + blend[2] * 0.4, 1.0))
    colors = base + creature_colors
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1.0), cmap.N)
    return cmap, norm


def animate_with_leaderboard(
    frames,
    stats,
    cmap,
    interval: int,
    show: bool,
    save_path: Path | None,
):
    fig, (ax_grid, ax_table) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2.5, 1.1]})
    im = ax_grid.imshow(frames[0], cmap=cmap, interpolation="nearest", vmin=0, vmax=len(cmap.colors) - 1)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_table.axis("off")
    table = [None]

    def swatch_color(val: int):
        if 0 <= val < len(cmap.colors):
            col = cmap.colors[val]
            if isinstance(col, str):
                return col
            return tuple(col)
        return (1, 1, 1, 1)

    def update(frame_idx: int):
        im.set_data(frames[frame_idx])
        ax_grid.set_title(f"Frame {frame_idx}")
        rows = stats[frame_idx] if stats is not None and frame_idx < len(stats) else []
        top = sorted(rows, key=lambda r: r.get("lifetime_energy", 0.0), reverse=True)[:5]
        if not top:
            cell_text = [["", "", ""]]
            cell_colors = [["#ffffff", "#ffffff", "#ffffff"]]
        else:
            cell_text = [["", str(r["id"]), f"{r.get('lifetime_energy', 0.0):.1f}"] for r in top]
            cell_colors = [
                [swatch_color(CREATURE_BASE + int(r.get("color_idx", 0))), "#ffffff", "#ffffff"] for r in top
            ]
        if table[0] is not None:
            table[0].remove()
        table[0] = ax_table.table(
            cellText=cell_text,
            cellColours=cell_colors,
            colLabels=["color", "id", "lifetime_energy"],
            loc="center",
        )
        table[0].scale(1.2, 1.4)
        table[0].auto_set_font_size(False)
        table[0].set_fontsize(9)
        ax_table.set_title("Top 5 energy (lifetime)")
        return [im, table[0]]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, repeat=True, blit=False)
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return anim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Falling dots with gravity, pellets, and neural control")
    parser.add_argument("--width", type=int, default=60)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2, help="Random seed")
    parser.add_argument("--pellet-chance", type=float, default=0.05, help="Spawn chance per column each step")
    parser.add_argument("--pellet-fade", type=float, default=0.04, help="Per-step fade chance for pellets")
    parser.add_argument("--pellet-energy", type=float, default=50.0)
    parser.add_argument("--initial-energy", type=float, default=30.0, help="Starting energy for seeded creatures")
    parser.add_argument("--spawn-count", type=int, default=8, help="How many creatures to seed at start")
    parser.add_argument("--maintenance", type=float, default=0.05)
    parser.add_argument("--move-cost", type=float, default=0.35)
    parser.add_argument("--up-extra", type=float, default=0.3, help="Additional cost for upward moves")
    parser.add_argument("--color-mutation", type=float, default=0.01, help="Stddev for per-channel color drift")
    parser.add_argument("--interval", type=int, default=60, help="Milliseconds between frames")
    parser.add_argument("--video", type=Path, help="Optional gif/mp4 output path")
    parser.add_argument("--save", type=Path, help="Where to save run history (.npz)")
    parser.add_argument("--load", type=Path, help="Load a saved history instead of simulating")
    parser.add_argument("--no-show", action="store_true", help="Skip opening a window")
    return parser.parse_args()


def init_world(args: argparse.Namespace) -> FallingDotsWorld:
    world = FallingDotsWorld(
        width=args.width,
        height=args.height,
        rng_seed=args.seed,
        pellet_spawn_chance=args.pellet_chance,
        pellet_energy=args.pellet_energy,
        pellet_fade_chance=args.pellet_fade,
        maintenance_cost=args.maintenance,
        move_cost=args.move_cost,
        upward_extra_cost=args.up_extra,
        color_mutation=args.color_mutation,
    )

    # Seed creatures near the ground so they have to climb a little for pellets.
    count = max(1, args.spawn_count)
    cols = np.linspace(1, args.width - 2, num=count, dtype=int)
    row = max(1, args.height - 3)
    for col in np.unique(cols):
        world.add_creature(row=row, col=int(col), energy=args.initial_energy)
    return world


def run_sim(args: argparse.Namespace):
    world = init_world(args)
    world.run(args.steps)
    if args.save:
        world.save_history(args.save, metadata={"type": "falling_dots", "steps": args.steps})
        print(f"Saved {len(world.history)} frames to {args.save}")
    return world.history, world.stats_history


def main():
    args = parse_args()
    cmap, norm = build_cmap()

    if args.load:
        history, metadata = FallingDotsWorld.load_history(args.load)
        title = metadata.get("type", "falling_dots")
        frames, stats = history, None
    else:
        frames, stats = run_sim(args)
        title = "Falling dots"

    # Skip creating a Matplotlib animation when nothing will render; avoids noisy warnings.
    if args.no_show and args.video is None:
        return None

    if stats is None:
        anim = animate_history(
            frames,
            interval=args.interval,
            cmap=cmap,
            vmin=0,
            vmax=len(cmap.colors) - 1,
            title=title.replace("_", " ").title(),
            show=not args.no_show,
            save_path=args.video,
        )
    else:
        anim = animate_with_leaderboard(
            frames,
            stats=stats,
            cmap=cmap,
            interval=args.interval,
            show=not args.no_show,
            save_path=args.video,
        )
    return anim


if __name__ == "__main__":
    main()
