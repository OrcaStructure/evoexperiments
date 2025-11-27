from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple, Optional, Set, List

import numpy as np


Color = Tuple[float, float, float, float]

# Default colors by block type for convenience (used when no color is provided).
BLOCK_DEFAULT_COLORS: Dict[str, Color] = {
    "stone": (0.55, 0.45, 0.4, 1.0),
    "redstone_block": (0.6, 0.1, 0.1, 1.0),
    "piston": (0.35, 0.6, 0.35, 1.0),
    "sticky_piston": (0.15, 0.55, 0.7, 1.0),
    "slime": (0.8, 0.6, 0.2, 1.0),
    "observer": (0.45, 0.2, 0.6, 1.0),
}


def _resolve_color(color: Optional[Iterable[float]] | None, block_type: str = BLOCK_STONE) -> Color:
    if color is None:
        return BLOCK_DEFAULT_COLORS.get(block_type, (0.6, 0.6, 0.6, 1.0))
    if isinstance(color, str):
        hex_str = color.lstrip("#")
        if len(hex_str) in (6, 8):
            r = int(hex_str[0:2], 16) / 255.0
            g = int(hex_str[2:4], 16) / 255.0
            b = int(hex_str[4:6], 16) / 255.0
            a = 1.0 if len(hex_str) == 6 else int(hex_str[6:8], 16) / 255.0
            return (r, g, b, a)
        raise ValueError("String colors must be hex like #rrggbb or #rrggbbaa")
    arr = np.asarray(list(color), dtype=float)
    if arr.shape[0] == 3:
        arr = np.append(arr, 1.0)
    if arr.shape[0] != 4:
        raise ValueError("Color must have 3 or 4 components")
    return tuple(np.clip(arr, 0.0, 1.0))  # type: ignore[return-value]


BLOCK_STONE = "stone"
BLOCK_REDSTONE = "redstone_block"
BLOCK_PISTON = "piston"
BLOCK_STICKY_PISTON = "sticky_piston"
BLOCK_SLIME = "slime"
BLOCK_OBSERVER = "observer"

ALL_BLOCKS = {
    BLOCK_STONE,
    BLOCK_REDSTONE,
    BLOCK_PISTON,
    BLOCK_STICKY_PISTON,
    BLOCK_SLIME,
    BLOCK_OBSERVER,
}


FACES = {
    "up": (0, 0, 1),
    "down": (0, 0, -1),
    "north": (0, -1, 0),
    "south": (0, 1, 0),
    "west": (-1, 0, 0),
    "east": (1, 0, 0),
}


@dataclass(frozen=True)
class Block:
    pos: Tuple[int, int, int]
    color: Color
    type: str = BLOCK_STONE
    facing: Tuple[int, int, int] = (0, 0, 1)  # default up
    powered: bool = False
    extended: bool = False  # for pistons


class VoxelWorld:
    """Minimal voxel world: a set of colored blocks in 3D space."""

    def __init__(self) -> None:
        self.blocks: Dict[Tuple[int, int, int], Block] = {}
        self._prev_state: Dict[Tuple[int, int, int], Block] = {}

    def add_block(
        self,
        x: int,
        y: int,
        z: int,
        *,
        color: Optional[Iterable[float]] = None,
        block_type: str = BLOCK_STONE,
        facing: Tuple[int, int, int] | str = (0, 0, 1),
    ) -> None:
        c = _resolve_color(color, block_type)
        f = FACES.get(facing, facing) if isinstance(facing, str) else facing
        if block_type not in ALL_BLOCKS:
            raise ValueError(f"Unknown block type: {block_type}")
        pos = (x, y, z)
        self.blocks[pos] = Block(pos=pos, color=c, type=block_type, facing=tuple(f))

    def remove_block(self, x: int, y: int, z: int) -> None:
        self.blocks.pop((x, y, z), None)

    def clear(self) -> None:
        self.blocks.clear()

    # --- Simulation -----------------------------------------------------
    def tick(self) -> None:
        """Advance one logic tick (power, observers, pistons, slime pushes)."""
        prev_state = self._prev_state
        self._prev_state = self.blocks.copy()

        power_sources: Set[Tuple[int, int, int]] = set()
        # Redstone blocks always powered.
        for pos, b in self.blocks.items():
            if b.type == BLOCK_REDSTONE:
                power_sources.add(pos)

        # Observers detect changes.
        observer_pulses: Set[Tuple[int, int, int]] = set()
        for pos, b in self.blocks.items():
            if b.type != BLOCK_OBSERVER:
                continue
            target_pos = tuple(np.add(pos, b.facing))
            prev = prev_state.get(target_pos)
            curr = self.blocks.get(target_pos)
            if prev != curr:
                observer_pulses.add(pos)
                power_sources.add(pos)

        # Determine which blocks are powered (adjacent to a power source).
        powered_blocks: Set[Tuple[int, int, int]] = set()
        for pos in power_sources:
            for npos in self._adjacent(pos):
                if npos in self.blocks:
                    powered_blocks.add(npos)

        # Update pistons.
        moves: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
        new_blocks = self.blocks.copy()
        for pos, b in list(self.blocks.items()):
            is_piston = b.type in (BLOCK_PISTON, BLOCK_STICKY_PISTON)
            if not is_piston:
                # Update powered flag for non-pistons as well.
                if b.powered != (pos in powered_blocks or pos in power_sources):
                    new_blocks[pos] = replace(b, powered=(pos in powered_blocks or pos in power_sources))
                continue
            currently_powered = pos in powered_blocks or pos in power_sources
            if currently_powered and not b.extended:
                # Try to extend and push.
                facing = b.facing
                head_pos = tuple(np.add(pos, facing))
                cluster = self._gather_push_cluster(head_pos, facing)
                if self._can_move(cluster, facing):
                    moves.extend(self._plan_moves(cluster, facing))
                    new_blocks[pos] = replace(b, powered=True, extended=True)
            elif not currently_powered and b.extended:
                # Retract
                head_pos = tuple(np.add(pos, b.facing))
                if b.type == BLOCK_STICKY_PISTON and head_pos in new_blocks:
                    pulled_target = head_pos  # where the head was
                    pulled_from = tuple(np.add(head_pos, b.facing))
                    if pulled_target not in new_blocks:
                        block_to_pull = new_blocks.pop(pulled_from, None)
                        if block_to_pull:
                            new_blocks[pulled_target] = replace(block_to_pull, pos=pulled_target)
                new_blocks[pos] = replace(b, powered=False, extended=False)
            else:
                # Just update powered flag
                if b.powered != currently_powered:
                    new_blocks[pos] = replace(b, powered=currently_powered)

        # Apply movement (sorted farthest first).
        for src, dst in sorted(moves, key=lambda p: self._move_sort_key(p[0], p[1]), reverse=True):
            block = new_blocks.pop(src, None)
            if block:
                new_blocks[dst] = replace(block, pos=dst)

        self.blocks = new_blocks

    def _solid_positions(self) -> Set[Tuple[int, int, int]]:
        return set(self.blocks.keys())

    def _adjacent(self, pos: Tuple[int, int, int]) -> Iterable[Tuple[int, int, int]]:
        x, y, z = pos
        for dx, dy, dz in FACES.values():
            yield (x + dx, y + dy, z + dz)

    def _gather_push_cluster(self, start: Tuple[int, int, int], direction: Tuple[int, int, int]) -> Set[Tuple[int, int, int]]:
        """Collect blocks to move; slime chains pull adjacent blocks (including other slime blocks)."""
        if start not in self.blocks:
            return set()
        cluster: Set[Tuple[int, int, int]] = set()
        queue = [start]
        while queue:
            pos = queue.pop()
            if pos in cluster or pos not in self.blocks:
                continue
            cluster.add(pos)
            block = self.blocks[pos]
            if block.type == BLOCK_SLIME:
                for npos in self._adjacent(pos):
                    if npos in self.blocks:
                        queue.append(npos)
        # Limit push length to avoid infinite; mimic MC 12 blocks loosely.
        if len(cluster) > 12:
            cluster = set(list(cluster)[:12])
        return cluster

    def _can_move(self, cluster: Set[Tuple[int, int, int]], direction: Tuple[int, int, int]) -> bool:
        for pos in cluster:
            target = tuple(np.add(pos, direction))
            if target in self.blocks and target not in cluster:
                return False
        return True

    def _plan_moves(self, cluster: Set[Tuple[int, int, int]], direction: Tuple[int, int, int]) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        moves = []
        for pos in cluster:
            target = tuple(np.add(pos, direction))
            moves.append((pos, target))
        return moves

    def _move_sort_key(self, src: Tuple[int, int, int], dst: Tuple[int, int, int]) -> int:
        # Move farthest blocks first along the move direction to avoid overwrites.
        move_vec = (dst[0] - src[0], dst[1] - src[1], dst[2] - src[2])
        return src[0] * move_vec[0] + src[1] * move_vec[1] + src[2] * move_vec[2]

    # --- Rendering helpers ------------------------------------------------
    def to_voxel_arrays(self, margin: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Build boolean occupancy and RGBA color arrays suitable for matplotlib's ax.voxels().

        Includes piston heads when pistons are extended (rendered only).
        The arrays are tightly bounded around existing blocks plus an optional margin.
        Empty worlds return two empty arrays.
        """
        if not self.blocks:
            return np.zeros((0, 0, 0), dtype=bool), np.zeros((0, 0, 0, 4), dtype=float)

        # Gather positions including piston heads for bounding box.
        head_positions: list[Tuple[Tuple[int, int, int], Color]] = []
        for pos, b in self.blocks.items():
            if b.type in (BLOCK_PISTON, BLOCK_STICKY_PISTON) and b.extended:
                head_pos = tuple(np.add(pos, b.facing))
                head_color = self._piston_head_color(b.color)
                head_positions.append((head_pos, head_color))

        all_positions = list(self.blocks.keys()) + [hp[0] for hp in head_positions]
        coords = np.array(all_positions, dtype=int)
        mins = coords.min(axis=0) - margin
        maxs = coords.max(axis=0) + margin
        shape = (maxs - mins + 1).astype(int)
        occupied = np.zeros(shape, dtype=bool)
        colors = np.zeros(shape.tolist() + [4], dtype=float)

        for pos, block in self.blocks.items():
            idx = tuple((np.array(pos) - mins).astype(int).tolist())
            occupied[idx] = True
            colors[idx] = block.color

        # Render piston heads.
        for head_pos, head_color in head_positions:
            idx = tuple((np.array(head_pos) - mins).astype(int).tolist())
            occupied[idx] = True
            colors[idx] = head_color

        return occupied, colors

    def _piston_head_color(self, base: Color) -> Color:
        # Lighten the piston color to differentiate the head.
        r, g, b, a = base
        blend = 0.7
        return (min(1.0, r * blend + 0.3), min(1.0, g * blend + 0.3), min(1.0, b * blend + 0.3), a)

    def bounding_box(self) -> tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Return inclusive min/max coordinates covering all blocks; defaults to zeros if empty."""
        if not self.blocks:
            return (0, 0, 0), (0, 0, 0)
        coords = np.array(list(self.blocks.keys()), dtype=int)
        mins = tuple(coords.min(axis=0).tolist())
        maxs = tuple(coords.max(axis=0).tolist())
        return mins, maxs
