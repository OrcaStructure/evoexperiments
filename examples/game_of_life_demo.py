from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evoexperiments import GridSimulation, animate_history, animate_simulation


def game_of_life_step(grid: np.ndarray) -> np.ndarray:
    """Conway's Game of Life update rule using numpy rolls for neighbors."""
    neighbors = sum(np.roll(np.roll(grid, dx, axis=0), dy, axis=1) for dx, dy in _NEIGHBOR_OFFSETS)
    return ((neighbors == 3) | ((grid == 1) & (neighbors == 2))).astype(grid.dtype, copy=False)


_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def random_seed(shape: tuple[int, int], fill_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < fill_ratio).astype(np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Game of Life demo with save/replay support")
    parser.add_argument("--size", type=int, default=40, help="Grid side length (square grid)")
    parser.add_argument("--steps", type=int, default=100, help="Steps to simulate when generating a run")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for initial grid")
    parser.add_argument("--density", type=float, default=0.25, help="Probability a cell starts alive")
    parser.add_argument("--interval", type=int, default=120, help="Milliseconds between frames in the animation")
    parser.add_argument("--cmap", type=str, default="Greys", help="Matplotlib colormap")
    parser.add_argument("--save", type=Path, help="Where to save the generated history (.npz)")
    parser.add_argument("--load", type=Path, help="Load a saved history instead of generating a new run")
    parser.add_argument("--continue-steps", type=int, default=0, help="After loading, extend the run by this many steps")
    parser.add_argument("--video", type=Path, help="Optional path to save an animation (gif/mp4) if writer is available")
    parser.add_argument("--no-show", action="store_true", help="Skip opening a window; useful on headless machines")
    return parser.parse_args()


def generate_run(args: argparse.Namespace):
    grid = random_seed((args.size, args.size), fill_ratio=args.density, seed=args.seed)
    metadata = {"rule": "game_of_life", "seed": args.seed, "density": args.density}

    sim = GridSimulation(initial_state=grid, update_rule=game_of_life_step, dtype=np.uint8)
    animate_simulation(
        sim,
        steps=args.steps,
        interval=args.interval,
        cmap=args.cmap,
        title="Game of Life",
        show=not args.no_show,
        save_path=args.video,
    )
    if args.save:
        sim.save_history(args.save, metadata=metadata)
        print(f"Saved {len(sim.history)} frames to {args.save}")


def replay_run(args: argparse.Namespace):
    history, metadata = GridSimulation.load_history(args.load)
    title = metadata.get("rule", "Loaded run").replace("_", " ").title()

    if args.continue_steps > 0:
        sim = GridSimulation.from_history(history, game_of_life_step, dtype=np.uint8)
        animate_simulation(
            sim,
            steps=args.continue_steps,
            interval=args.interval,
            cmap=args.cmap,
            title=title,
            show=not args.no_show,
            save_path=args.video,
        )
        if args.save:
            sim.save_history(args.save, metadata=metadata)
            print(f"Extended and saved {len(sim.history)} frames to {args.save}")
    else:
        animate_history(
            history,
            interval=args.interval,
            cmap=args.cmap,
            title=title,
            show=not args.no_show,
            save_path=args.video,
        )


def main():
    args = parse_args()
    if args.load:
        replay_run(args)
    else:
        generate_run(args)


if __name__ == "__main__":
    main()
