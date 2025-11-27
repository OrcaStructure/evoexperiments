from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .voxel_world import (
    VoxelWorld,
    BLOCK_PISTON,
    BLOCK_REDSTONE,
    BLOCK_SLIME,
    BLOCK_STICKY_PISTON,
    BLOCK_STONE,
    FACES,
)

BlockSpec = Tuple[int, int, int, str, Tuple[int, int, int]]

# Order matters for random selection.
DEFAULT_BLOCK_TYPES: Tuple[str, ...] = (
    BLOCK_STONE,
    BLOCK_REDSTONE,
    BLOCK_PISTON,
    BLOCK_STICKY_PISTON,
    BLOCK_SLIME,
)

_FACES_LIST: Tuple[Tuple[int, int, int], ...] = tuple(FACES.values())
_DIRECTIONAL: Tuple[str, ...] = (BLOCK_PISTON, BLOCK_STICKY_PISTON)


def clone_world(world: VoxelWorld) -> VoxelWorld:
    """Deep copy a world to simulate without mutating the source."""
    return copy.deepcopy(world)


def world_to_blocks(world: VoxelWorld) -> List[BlockSpec]:
    """Serialize a world to a simple list of block specs."""
    blocks: List[BlockSpec] = []
    for (x, y, z), b in world.blocks.items():
        blocks.append((int(x), int(y), int(z), b.type, tuple(int(v) for v in b.facing)))
    return blocks


def blocks_to_world(blocks: Iterable[BlockSpec]) -> VoxelWorld:
    """Create a world from serialized blocks."""
    world = VoxelWorld()
    for x, y, z, block_type, facing in blocks:
        world.add_block(x, y, z, block_type=block_type, facing=facing)
    return world


def _random_facing(rng: np.random.Generator) -> Tuple[int, int, int]:
    return _FACES_LIST[int(rng.integers(0, len(_FACES_LIST)))]


def add_random_block(
    world: VoxelWorld,
    rng: np.random.Generator,
    block_types: Tuple[str, ...] = DEFAULT_BLOCK_TYPES,
    max_attempts: int = 16,
) -> bool:
    """
    Add a random block to a random face of an existing block (or origin if empty).

    Returns True if a block was placed, False if no free spot was found.
    """
    if not world.blocks:
        x = y = z = 0
        target_pos = (x, y, z)
    else:
        keys = list(world.blocks.keys())
        target_pos = None
        for _ in range(max_attempts):
            base = keys[int(rng.integers(0, len(keys)))]
            face = _random_facing(rng)
            candidate = tuple(int(a + b) for a, b in zip(base, face))
            if candidate not in world.blocks:
                target_pos = candidate
                break
        if target_pos is None:
            return False
        x, y, z = target_pos

    block_type = block_types[int(rng.integers(0, len(block_types)))]
    facing = _random_facing(rng) if block_type in _DIRECTIONAL else (0, 0, 1)
    world.add_block(x, y, z, block_type=block_type, facing=facing)
    return True


def build_random_world(
    rng: np.random.Generator,
    steps: int,
    block_types: Tuple[str, ...] = DEFAULT_BLOCK_TYPES,
) -> VoxelWorld:
    world = VoxelWorld()
    for _ in range(steps):
        add_random_block(world, rng, block_types=block_types)
    return world


def mutate_world(
    base_world: VoxelWorld,
    rng: np.random.Generator,
    add_range: Tuple[int, int] = (1, 3),
    remove_range: Tuple[int, int] = (0, 2),
) -> VoxelWorld:
    """Return a mutated copy by adding and removing a few blocks."""
    world = clone_world(base_world)
    # Random removals
    n_remove = int(rng.integers(remove_range[0], remove_range[1] + 1))
    for _ in range(n_remove):
        if not world.blocks:
            break
        pos = list(world.blocks.keys())[int(rng.integers(0, len(world.blocks)))]
        world.remove_block(*pos)
    # Random additions
    n_add = int(rng.integers(add_range[0], add_range[1] + 1))
    for _ in range(n_add):
        add_random_block(world, rng)
    return world


def measure_average_movement(world: VoxelWorld, ticks: int) -> float:
    """
    Simulate for `ticks` and return average number of blocks that changed position per tick.

    Movement is approximated as half the symmetric difference in occupied cells between
    consecutive ticks (piston pushes shift by one cell).
    """
    sim = clone_world(world)
    prev_positions = set(sim.blocks.keys())
    if ticks <= 0:
        return 0.0
    moved_total = 0.0
    for _ in range(ticks):
        sim.tick()
        curr_positions = set(sim.blocks.keys())
        moved_this_tick = len(prev_positions.symmetric_difference(curr_positions)) / 2.0
        moved_total += moved_this_tick
        prev_positions = curr_positions
    return moved_total / float(ticks)


@dataclass
class MachineRecord:
    machine_id: str
    fitness: float
    blocks: List[BlockSpec]

    def to_dict(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "fitness": self.fitness,
            "blocks": [list(b) for b in self.blocks],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MachineRecord":
        blocks = [tuple(item) for item in data["blocks"]]  # type: ignore[list-item]
        return cls(machine_id=data["machine_id"], fitness=float(data["fitness"]), blocks=blocks)  # type: ignore[arg-type]


def save_generation(path: Path, records: Iterable[MachineRecord]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2)


def load_generation(path: Path) -> List[MachineRecord]:
    import json

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [MachineRecord.from_dict(item) for item in data]
