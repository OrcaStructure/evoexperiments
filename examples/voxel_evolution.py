from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import List

import numpy as np

# Allow running directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evoexperiments.voxel_evolution import (  # noqa: E402
    MachineRecord,
    blocks_to_world,
    build_random_world,
    measure_average_movement,
    mutate_world,
    save_generation,
    world_to_blocks,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless evolution of random voxel machines.")
    p.add_argument("--population", type=int, default=20, help="Population size.")
    p.add_argument("--survivors", type=int, default=5, help="Top performers kept each generation.")
    p.add_argument("--generations", type=int, default=10, help="Number of generations to run.")
    p.add_argument("--initial-steps", type=int, default=12, help="Random build steps for initial machines.")
    p.add_argument("--ticks", type=int, default=12, help="Ticks to simulate for fitness.")
    p.add_argument("--seed", type=int, default=0, help="Random seed.")
    p.add_argument("--run-dir", type=Path, default=Path("runs"), help="Directory to save run results.")
    p.add_argument("--run-id", type=str, help="Optional run id (defaults to timestamp).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "population": args.population,
        "survivors": args.survivors,
        "generations": args.generations,
        "initial_steps": args.initial_steps,
        "ticks": args.ticks,
        "seed": args.seed,
        "run_id": run_id,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    population: List = [build_random_world(rng, steps=args.initial_steps) for _ in range(args.population)]

    for gen in range(args.generations):
        records: List[MachineRecord] = []
        fitnesses: List[float] = []
        for idx, world in enumerate(population):
            fitness = measure_average_movement(world, args.ticks)
            fitnesses.append(fitness)
            mid = f"{run_id}_g{gen:03d}_i{idx:04d}"
            records.append(MachineRecord(machine_id=mid, fitness=fitness, blocks=world_to_blocks(world)))

        # Save current generation.
        gen_path = run_dir / f"generation_{gen:03d}.json"
        save_generation(gen_path, records)

        # Select survivors.
        records_sorted = sorted(records, key=lambda r: r.fitness, reverse=True)
        survivors = records_sorted[: max(1, args.survivors)]
        best = survivors[0]
        print(f"Generation {gen}: best={best.fitness:.3f} avg={sum(fitnesses)/len(fitnesses):.3f} id={best.machine_id}")

        # Prepare next generation.
        new_population: List = []
        # Keep best unmutated to preserve elite.
        new_population.append(blocks_to_world(best.blocks))
        # Remaining population are mutations of survivors (cycled).
        for i in range(1, args.population):
            parent = survivors[i % len(survivors)]
            world = blocks_to_world(parent.blocks)
            new_population.append(mutate_world(world, rng))
        population = new_population

    print(f"Run complete. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
