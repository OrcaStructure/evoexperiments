from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from evoexperiments import PlantWorld, animate_history
from evoexperiments.plant_world import PLANT_BASE, PALETTE_SPAN


def build_cmap():
    # 0 sky, 1 ground, 2 sun, 3+ plants (cycled palette)
    colors = [
        "#6ec6ff",  # sky
        "#8d6e63",  # ground
        "#ffd54f",  # sun
        "#66bb6a",
        "#43a047",
        "#81c784",
        "#2e7d32",
        "#9ccc65",
        "#7cb342",
        "#558b2f",
        "#8bc34a",
        "#c5e1a5",
        "#aed581",
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1.0), cmap.N)
    return cmap, norm


def color_for_pid(pid: int, cmap: ListedColormap) -> str:
    palette_offset = PLANT_BASE
    span = min(PALETTE_SPAN, len(cmap.colors) - palette_offset)
    if span <= 0:
        return "n/a"
    idx = palette_offset + (pid % span)
    color = cmap.colors[idx]
    # Convert rgba floats to hex.
    if isinstance(color, str):
        return color
    r, g, b = (int(255 * c) for c in color[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def print_plant_table(world: PlantWorld, cmap: ListedColormap) -> None:
    if not world.plants:
        print("(no plants)")
        return
    rows = []
    for pid in sorted(world.plants):
        plant = world.plants[pid]
        rows.append(
            (
                pid,
                len(plant.cells),
                f"{plant.energy_after_sun:.2f}",
                f"{plant.energy_after_maintenance:.2f}",
                color_for_pid(pid, cmap),
            )
        )
    header = ("id", "cells", "energy_sun", "energy_post_maint", "color")
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    def fmt_row(row):
        return " | ".join(str(val).ljust(w) for val, w in zip(row, widths))
    print(fmt_row(header))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def animate_with_table(
    frames,
    stats,
    cmap,
    interval: int,
    show: bool,
    save_path: Path | None,
):
    if stats is None or len(stats) != len(frames):
        # Fallback to plain animation if stats are missing.
        return animate_history(
            frames,
            interval=interval,
            cmap=cmap,
            vmin=0,
            vmax=len(cmap.colors) - 1,
            show=show,
            save_path=save_path,
        )

    fig, (ax_grid, ax_table) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"width_ratios": [2.5, 1.5]})
    im = ax_grid.imshow(frames[0], cmap=cmap, interpolation="nearest", vmin=0, vmax=len(cmap.colors) - 1)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    ax_table.axis("off")
    table = [None]  # mutable holder

    def update(frame_idx: int):
        im.set_data(frames[frame_idx])
        ax_grid.set_title(f"Day frame {frame_idx}")
        # Build table rows.
        rows = stats[frame_idx]
        if not rows:
            cell_text = [["(no plants)", "", "", "", ""]]
        else:
            cell_text = [
                [
                    row["id"],
                    row["cells"],
                    f"{row.get('energy_sun', row.get('energy', 0.0)):.2f}",
                    f"{row.get('energy_post_maint', row.get('energy', 0.0)):.2f}",
                    color_for_pid(row["id"], cmap),
                ]
                for row in sorted(rows, key=lambda r: r["id"])
            ]
        col_labels = ["id", "cells", "energy_sun", "energy_post_maint", "color"]
        # Remove prior table.
        if table[0] is not None:
            table[0].remove()
        table[0] = ax_table.table(cellText=cell_text, colLabels=col_labels, loc="center")
        table[0].scale(1.2, 1.4)
        table[0].auto_set_font_size(False)
        table[0].set_fontsize(9)
        ax_table.set_title("Plant stats")
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
    parser = argparse.ArgumentParser(description="Plant world demo with sunlight and neural genomes")
    parser.add_argument("--width", type=int, default=50)
    parser.add_argument("--height", type=int, default=50)
    parser.add_argument("--days", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for world initialization (omit for random)")
    parser.add_argument("--sun-energy", type=float, default=3.0)
    parser.add_argument("--maintenance", type=float, default=0.2)
    parser.add_argument("--growth-cost", type=float, default=1.0)
    parser.add_argument("--seed-cost", type=float, default=2.0)
    parser.add_argument("--save", type=Path, help="Save run history to .npz")
    parser.add_argument("--load", type=Path, help="Load a saved .npz history and replay it")
    parser.add_argument("--video", type=Path, help="Optional gif/mp4 output for the animation")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive window")
    parser.add_argument("--interval", type=int, default=5, help="Milliseconds between frames in animation")
    parser.add_argument("--log-table", action="store_true", help="Print plant stats at start/end of each day")
    parser.add_argument("--table-in-fig", action="store_true", help="Render per-plant table next to the grid animation")
    return parser.parse_args()


def init_world(args: argparse.Namespace) -> PlantWorld:
    world = PlantWorld(
        width=args.width,
        height=args.height,
        rng_seed=args.seed,
        sunlight_energy=args.sun_energy,
        maintenance_cost=args.maintenance,
        growth_cost=args.growth_cost,
        seed_cost=args.seed_cost,
    )

    # Start a few plants along the ground.
    for col in (args.width // 3, args.width // 2, (2 * args.width) // 3):
        world.add_plant(column=col, energy=5.0)
    return world


def run_new_world(args: argparse.Namespace):
    world = init_world(args)

    cmap, norm = build_cmap()

    for day in range(args.days):
        if args.log_table:
            print(f"\nDay {day+1} start")
            print_plant_table(world, cmap)
        world.advance_day()
        if args.log_table:
            print(f"Day {day+1} end")
            print_plant_table(world, cmap)

    if args.save:
        world.save_history(args.save, metadata={"type": "plant_world", "days": args.days})
        print(f"Saved {len(world.history)} frames to {args.save}")

    if args.table_in_fig:
        anim = animate_with_table(
            world.history,
            world.stats_history,
            cmap=cmap,
            interval=args.interval,
            show=not args.no_show,
            save_path=args.video,
        )
    else:
        anim = animate_history(
            world.history,
            interval=args.interval,
            cmap=cmap,
            vmin=0,
            vmax=len(cmap.colors) - 1,
            show=not args.no_show,
            save_path=args.video,
        )
    # Keep reference to avoid matplotlib GC warnings when show=False and not saving.
    return anim


def replay_saved(args: argparse.Namespace):
    history, metadata = PlantWorld.load_history(args.load)
    cmap, norm = build_cmap()
    anim = animate_history(
        history,
        interval=args.interval,
        cmap=cmap,
        vmin=0,
        vmax=len(cmap.colors) - 1,
        show=not args.no_show,
        save_path=args.video,
        title=str(metadata.get("type", "Saved run")).replace("_", " ").title(),
    )
    return anim


def main():
    args = parse_args()
    if args.load:
        replay_saved(args)
    else:
        run_new_world(args)


if __name__ == "__main__":
    main()
