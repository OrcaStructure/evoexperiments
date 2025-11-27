from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt

# Allow running directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evoexperiments.voxel_evolution import (  # noqa: E402
    blocks_to_world,
    load_generation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a saved voxel machine by id.")
    p.add_argument("--run-dir", type=Path, required=True, help="Run directory produced by voxel_evolution.")
    p.add_argument("--machine-id", required=True, help="Machine id to render.")
    p.add_argument("--generation", type=int, help="Optional generation number to search within.")
    p.add_argument("--ticks", type=int, default=0, help="Ticks to step the machine before rendering.")
    p.add_argument("--save", type=Path, help="Optional path to save a PNG of the render.")
    p.add_argument("--no-show", action="store_true", help="Do not open an interactive window.")
    return p.parse_args()


def render_world(world, save_path: Optional[Path], show: bool) -> None:
    occupied, colors = world.to_voxel_arrays()
    if occupied.size == 0:
        print("No blocks to render.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(occupied, facecolors=colors, edgecolor="k", linewidth=0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([max(1, occupied.shape[0]), max(1, occupied.shape[1]), max(1, occupied.shape[2])])
    ax.set_title("Voxel machine")
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    fig.patch.set_facecolor((1.0, 1.0, 1.0, 0.0))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    if show and not plt.isinteractive():
        plt.show()
    elif show:
        plt.draw()
    else:
        plt.close(fig)


def find_machine(run_dir: Path, machine_id: str, generation: Optional[int]):
    if generation is not None:
        gen_path = run_dir / f"generation_{generation:03d}.json"
        if not gen_path.exists():
            raise FileNotFoundError(f"No generation file at {gen_path}")
        gens = [gen_path]
    else:
        gens = sorted(run_dir.glob("generation_*.json"))
        if not gens:
            raise FileNotFoundError(f"No generation files found in {run_dir}")

    for path in gens:
        for record in load_generation(path):
            if record.machine_id == machine_id:
                return record
    raise ValueError(f"Machine id {machine_id} not found in {run_dir}")


def main() -> None:
    args = parse_args()
    record = find_machine(args.run_dir, args.machine_id, args.generation)
    world = blocks_to_world(record.blocks)
    for _ in range(max(0, args.ticks)):
        world.tick()
    render_world(world, save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
