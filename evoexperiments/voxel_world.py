from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple, Optional, Set, List

import numpy as np


Color = Tuple[float, float, float, float]

# Default colors by block type for convenience (used when no color is provided).
BLOCK_DEFAULT_COLORS: Dict[str, Color] = {
    # Closer-to-material hues.
    "stone": (0.75, 0.77, 0.78, 1.0),  # light grey
    "redstone_block": (0.76, 0.09, 0.09, 1.0),  # redstone block
    "piston": (0.42, 0.44, 0.47, 1.0),  # dark grey piston body
    "sticky_piston": (0.32, 0.34, 0.37, 1.0),  # darker sticky piston body
    "slime": (0.62, 0.86, 0.60, 1.0),  # slime green
}


def _resolve_color(color: Optional[Iterable[float]] | None, block_type: str = "stone") -> Color:
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
BLOCK_PISTON_HEAD = "piston_head"

ALL_BLOCKS = {
    BLOCK_STONE,
    BLOCK_REDSTONE,
    BLOCK_PISTON,
    BLOCK_STICKY_PISTON,
    BLOCK_SLIME,
    BLOCK_PISTON_HEAD,
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
        """Advance one logic tick (power, pistons, slime pushes)."""
        power_sources: Set[Tuple[int, int, int]] = {pos for pos, b in self.blocks.items() if b.type == BLOCK_REDSTONE}

        powered_blocks: Set[Tuple[int, int, int]] = set()
        for pos in power_sources:
            for npos in self._adjacent(pos):
                if npos in self.blocks:
                    powered_blocks.add(npos)

        moves: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
        head_additions: list[Tuple[Tuple[int, int, int], Block]] = []
        new_blocks = self.blocks.copy()

        for pos, b in list(self.blocks.items()):
            is_piston = b.type in (BLOCK_PISTON, BLOCK_STICKY_PISTON)
            if not is_piston:
                if b.powered != (pos in powered_blocks or pos in power_sources):
                    new_blocks[pos] = replace(b, powered=(pos in powered_blocks or pos in power_sources))
                continue
            currently_powered = pos in powered_blocks or pos in power_sources
            if currently_powered and not b.extended:
                facing = b.facing
                head_pos = tuple(np.add(pos, facing))
                cluster = self._gather_push_cluster(head_pos, facing, forbidden={pos})
                if self._can_move(cluster, facing, occupied=new_blocks):
                    moves.extend(self._plan_moves(cluster, facing))
                    new_blocks[pos] = replace(b, powered=True, extended=True)
                    head_additions.append((head_pos, b))
                else:
                    if not b.powered:
                        new_blocks[pos] = replace(b, powered=True)
            elif not currently_powered and b.extended:
                head_pos = tuple(np.add(pos, b.facing))
                new_blocks.pop(head_pos, None)
                if b.type == BLOCK_STICKY_PISTON:
                    pulled_from = tuple(np.add(head_pos, b.facing))
                    cluster = self._gather_push_cluster(pulled_from, b.facing, forbidden={pos, head_pos})
                    back_dir = tuple(-x for x in b.facing)
                    if cluster and self._can_move(cluster, back_dir, occupied=new_blocks, allow_override={head_pos}):
                        moves.extend(self._plan_moves(cluster, back_dir))
                new_blocks[pos] = replace(b, powered=False, extended=False)
            else:
                if b.powered != currently_powered:
                    new_blocks[pos] = replace(b, powered=currently_powered)

        for src, dst in sorted(moves, key=lambda p: self._move_sort_key(p[0], p[1]), reverse=True):
            block = new_blocks.pop(src, None)
            if block:
                new_blocks[dst] = replace(block, pos=dst)

        for head_pos, piston_block in head_additions:
            new_blocks[head_pos] = Block(
                pos=head_pos,
                color=piston_block.color,
                type=BLOCK_PISTON_HEAD,
                facing=piston_block.facing,
                powered=True,
                extended=True,
            )

        self.blocks = new_blocks

    def _solid_positions(self) -> Set[Tuple[int, int, int]]:
        return set(self.blocks.keys())

    def _adjacent(self, pos: Tuple[int, int, int]) -> Iterable[Tuple[int, int, int]]:
        x, y, z = pos
        for dx, dy, dz in FACES.values():
            yield (x + dx, y + dy, z + dz)

    def _gather_push_cluster(
        self,
        start: Tuple[int, int, int],
        direction: Tuple[int, int, int],
        forbidden: Optional[Set[Tuple[int, int, int]]] = None,
    ) -> Set[Tuple[int, int, int]]:
        """
        Collect the full set of blocks that will be pushed by a piston.

        We always include the line of blocks in front of the piston (so pushes
        propagate even without slime), and slime blocks also grab any adjacent
        blocks to move with them.
        """
        if start not in self.blocks:
            return set()
        cluster: Set[Tuple[int, int, int]] = set()
        queue = [start]
        while queue:
            pos = queue.pop()
            if pos in cluster or pos not in self.blocks or (forbidden and pos in forbidden):
                continue
            cluster.add(pos)
            block = self.blocks[pos]
            # Always follow the chain directly in front of the piston so pushes propagate.
            forward = tuple(np.add(pos, direction))
            if forward in self.blocks:
                queue.append(forward)
            if block.type == BLOCK_SLIME:
                for npos in self._adjacent(pos):
                    if npos in self.blocks:
                        queue.append(npos)
        # Limit push length to avoid infinite; mimic MC 12 blocks loosely.
        if len(cluster) > 12:
            cluster = set(list(cluster)[:12])
        return cluster

    def _can_move(
        self,
        cluster: Set[Tuple[int, int, int]],
        direction: Tuple[int, int, int],
        *,
        occupied: Optional[Dict[Tuple[int, int, int], Block]] = None,
        allow_override: Optional[Set[Tuple[int, int, int]]] = None,
    ) -> bool:
        occupied = occupied if occupied is not None else self.blocks
        allow_override = allow_override or set()
        for pos in cluster:
            target = tuple(np.add(pos, direction))
            if target in occupied and target not in cluster and target not in allow_override:
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

        all_positions = list(self.blocks.keys())
        coords = np.array(all_positions, dtype=int)
        mins = coords.min(axis=0) - margin
        maxs = coords.max(axis=0) + margin
        shape = (maxs - mins + 1).astype(int)
        occupied = np.zeros(shape, dtype=bool)
        colors = np.zeros(shape.tolist() + [4], dtype=float)

        for pos, block in self.blocks.items():
            idx = tuple((np.array(pos) - mins).astype(int).tolist())
            occupied[idx] = True
            colors[idx] = self._piston_head_color(block.color) if block.type == BLOCK_PISTON_HEAD else block.color

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
