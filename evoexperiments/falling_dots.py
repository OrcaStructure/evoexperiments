from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json

import numpy as np


# Rendering categories
EMPTY = 0
PELLET = 1
GROUND = 2
CREATURE_BASE = 3  # actual creature value is CREATURE_BASE + color_idx
COLOR_DIMS = 3
COLOR_BINS = 5  # per-dimension quantization
CREATURE_COLORS = COLOR_BINS ** COLOR_DIMS  # total bins available for per-creature color

# Action indices
LEFT, RIGHT, UP_LEFT, UP, UP_RIGHT = range(5)
ACTION_OFFSETS = {
    LEFT: (0, -1),
    RIGHT: (0, 1),
    UP_LEFT: (-1, -1),
    UP: (-1, 0),
    UP_RIGHT: (-1, 1),
}


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass
class Genome:
    """Two-layer perceptron controlling movement."""

    w1: np.ndarray  # (hidden, input)
    b1: np.ndarray  # (hidden,)
    w2: np.ndarray  # (5, hidden)
    b2: np.ndarray  # (5,)

    @classmethod
    def random(cls, rng: np.random.Generator, input_dim: int, hidden: int = 24, scale: float = 0.4) -> "Genome":
        w1 = rng.normal(scale=scale, size=(hidden, input_dim))
        b1 = rng.normal(scale=scale, size=(hidden,))
        w2 = rng.normal(scale=scale, size=(5, hidden))
        b2 = rng.normal(scale=scale, size=(5,))
        return cls(w1, b1, w2, b2)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(self.w1 @ x + self.b1)
        return self.w2 @ h + self.b2

    def mutate(self, rng: np.random.Generator, sigma: float = 0.05) -> "Genome":
        return Genome(
            w1=self.w1 + rng.normal(scale=sigma, size=self.w1.shape),
            b1=self.b1 + rng.normal(scale=sigma, size=self.b1.shape),
            w2=self.w2 + rng.normal(scale=sigma, size=self.w2.shape),
            b2=self.b2 + rng.normal(scale=sigma, size=self.b2.shape),
        )


@dataclass
class Creature:
    creature_id: int
    position: Tuple[int, int]
    energy: float
    genome: Genome
    reproduction_split: float  # fraction of pellet energy given to offspring
    color_gene: np.ndarray  # values in [0,1] per channel
    lifetime_energy: float  # total energy acquired over life


class FallingDotsWorld:
    """
    Creatures are black dots that move via a neural net and are pulled down by gravity.

    Each step:
      * Pellets spawn at random heights.
      * Creatures pay maintenance, take one movement action if they can afford it, then gravity pulls them down.
      * Eating a pellet grants energy and can spawn a child directly above using a mutated genome.
    """

    def __init__(
        self,
        width: int,
        height: int,
        rng_seed: Optional[int] = None,
        pellet_spawn_chance: float = 0.05,
        pellet_energy: float = 6.0,
        pellet_fade_chance: float = 0.02,
        maintenance_cost: float = 0.2,
        move_cost: float = 0.3,
        upward_extra_cost: float = 0.4,
        reproduction_mutation: float = 0.05,
        color_mutation: float = 0.01,
    ) -> None:
        if width < 4 or height < 6:
            raise ValueError("World too small; need room for gravity and sensing")
        self.width = width
        self.height = height
        self.ground_y = height - 1
        self.rng = np.random.default_rng(rng_seed)

        self.pellet_spawn_chance = pellet_spawn_chance
        self.pellet_energy = pellet_energy
        self.pellet_fade_chance = pellet_fade_chance
        self.maintenance_cost = maintenance_cost
        self.move_cost = move_cost
        self.upward_extra_cost = upward_extra_cost
        self.reproduction_mutation = reproduction_mutation
        self.color_mutation = color_mutation

        # Occupancy grid only tracks creatures; pellets are separate.
        self.occupancy = np.zeros((height, width), dtype=np.int32)
        self.pellets: Dict[Tuple[int, int], int] = {}
        self.creatures: Dict[int, Creature] = {}
        self.next_creature_id = 1

        self.history: List[np.ndarray] = []
        self.stats_history: List[list] = []
        self.record_frame()

    # --- Setup helpers -----------------------------------------------------
    def add_creature(
        self,
        row: int,
        col: int,
        energy: float,
        genome: Optional[Genome] = None,
        split: Optional[float] = None,
        color_gene: Optional[Iterable[float]] = None,
    ) -> Optional[int]:
        if not (0 <= row < self.ground_y and 0 <= col < self.width):
            return None
        if self.occupancy[row, col] != 0:
            return None
        creature_id = self.next_creature_id
        self.next_creature_id += 1
        genome = genome or Genome.random(self.rng, input_dim=self._feature_dim())
        reproduction_split = float(np.clip(split if split is not None else self.rng.uniform(0.2, 0.8), 0.0, 1.0))
        if color_gene is None:
            color_gene_arr = self.rng.random(COLOR_DIMS)
        else:
            color_gene_arr = np.asarray(color_gene, dtype=float)
        if color_gene_arr.shape != (COLOR_DIMS,):
            color_gene_arr = np.resize(color_gene_arr, (COLOR_DIMS,))
        color_gene_arr = np.clip(color_gene_arr, 0.0, 1.0)
        creature = Creature(
            creature_id=creature_id,
            position=(row, col),
            energy=energy,
            genome=genome,
            reproduction_split=reproduction_split,
            color_gene=color_gene_arr,
            lifetime_energy=energy,
        )
        self.creatures[creature_id] = creature
        self.occupancy[row, col] = creature_id
        return creature_id

    # --- Simulation --------------------------------------------------------
    def step(self) -> None:
        self._age_pellets()
        self._spawn_pellets()
        creature_ids = list(self.creatures.keys())
        self.rng.shuffle(creature_ids)

        for cid in creature_ids:
            creature = self.creatures.get(cid)
            if creature is None:
                continue

            creature.energy -= self.maintenance_cost
            if creature.energy <= 0.0:
                self._remove_creature(cid)
                continue

            self._take_action(creature)

        self._apply_gravity()
        self.record_frame()

    def run(self, steps: int) -> List[np.ndarray]:
        for _ in range(steps):
            self.step()
        return self.history

    # --- Actions -----------------------------------------------------------
    def _take_action(self, creature: Creature) -> None:
        if creature.energy < self.move_cost:
            return

        features = self._sense(creature.position)
        logits = creature.genome.forward(features)
        probs = softmax(logits)
        action_idx = int(self.rng.choice(len(ACTION_OFFSETS), p=probs))

        dr, dc = ACTION_OFFSETS[action_idx]
        target = (creature.position[0] + dr, creature.position[1] + dc)

        # Determine cost; upward moves cost more.
        cost = self.move_cost + (self.upward_extra_cost if dr < 0 else 0.0)
        if creature.energy < cost:
            return

        # Require support directly beneath the creature for any upward move.
        if dr < 0 and not self._has_support(creature.position):
            return

        if not self._is_walkable(target):
            return

        creature.energy -= cost
        self._move_creature(creature, target)

    def _move_creature(self, creature: Creature, target: Tuple[int, int]) -> None:
        old_pos = creature.position
        occupant = self.occupancy[target] if self._in_bounds(target) else -1
        if occupant != 0:
            return  # cannot move into another creature or ground sentinel

        self.occupancy[old_pos] = 0
        creature.position = target
        self.occupancy[target] = creature.creature_id

        self._consume_if_on_pellet(creature)

    def _eat_pellet(self, creature: Creature, pos: Tuple[int, int]) -> None:
        self.pellets.pop(pos, None)
        gain = self.pellet_energy
        split = float(np.clip(creature.reproduction_split, 0.0, 1.0))
        child_energy = gain * split
        creature.energy += gain - child_energy
        creature.lifetime_energy += gain

        above = (pos[0] - 1, pos[1])
        if child_energy <= 0.0 or not self._is_walkable(above):
            return

        mutated = creature.genome.mutate(self.rng)
        new_split = float(np.clip(creature.reproduction_split + self.rng.normal(scale=self.reproduction_mutation), 0.0, 1.0))
        new_color = np.clip(
            creature.color_gene + self.rng.normal(scale=self.color_mutation, size=COLOR_DIMS),
            0.0,
            1.0,
        )
        self.add_creature(
            row=above[0],
            col=above[1],
            energy=child_energy,
            genome=mutated,
            split=new_split,
            color_gene=new_color,
        )

    def _consume_if_on_pellet(self, creature: Creature) -> None:
        pos = creature.position
        if pos in self.pellets:
            self._eat_pellet(creature, pos)

    # --- Gravity and pellets ----------------------------------------------
    def _apply_gravity(self) -> None:
        # Move each creature down by one if the cell below is empty.
        for cid in list(self.creatures.keys()):
            creature = self.creatures.get(cid)
            if creature is None:
                continue
            below = (creature.position[0] + 1, creature.position[1])
            if self._is_walkable(below):
                self.occupancy[creature.position] = 0
                creature.position = below
                self.occupancy[below] = cid
                self._consume_if_on_pellet(creature)

    def _spawn_pellets(self) -> None:
        for col in range(self.width):
            if self.rng.random() < self.pellet_spawn_chance:
                row = int(self.rng.integers(0, self.ground_y))
                pos = (row, col)
                if self.occupancy[pos] == 0 and pos not in self.pellets:
                    self.pellets[pos] = self._sample_pellet_ttl()

    def _age_pellets(self) -> None:
        to_delete: List[Tuple[int, int]] = []
        for pos, ttl in list(self.pellets.items()):
            ttl -= 1
            if ttl <= 0:
                to_delete.append(pos)
            else:
                self.pellets[pos] = ttl
        for pos in to_delete:
            self.pellets.pop(pos, None)

    def _sample_pellet_ttl(self) -> int:
        # Geometric lifetime keeps expected pellet count roughly stable.
        ttl = int(self.rng.geometric(self.pellet_fade_chance))
        return max(1, ttl)

    # --- Helpers -----------------------------------------------------------
    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def _is_walkable(self, pos: Tuple[int, int]) -> bool:
        if not self._in_bounds(pos):
            return False
        r, c = pos
        if r >= self.ground_y:
            return False
        return self.occupancy[r, c] == 0

    def _has_support(self, pos: Tuple[int, int]) -> bool:
        below = (pos[0] + 1, pos[1])
        if below[0] == self.ground_y:
            return True
        return self._in_bounds(below) and self.occupancy[below] != 0

    def _sense(self, pos: Tuple[int, int]) -> np.ndarray:
        values: List[float] = []
        for dr in (-2, -1, 0, 1):
            for dc in (-2, -1, 0, 1):
                r, c = pos[0] + dr, pos[1] + dc
                values.append(self._cell_value((r, c)))
        return np.asarray(values, dtype=float)

    def _cell_value(self, pos: Tuple[int, int]) -> float:
        if not self._in_bounds(pos):
            return 2.0  # treat outside as ground
        r, c = pos
        if r >= self.ground_y:
            return 2.0
        if self.occupancy[r, c] != 0:
            return 1.0
        if (r, c) in self.pellets:
            return -1.0
        return 0.0

    def _feature_dim(self) -> int:
        return 16

    def _remove_creature(self, cid: int) -> None:
        creature = self.creatures.pop(cid, None)
        if creature:
            self.occupancy[creature.position] = 0

    # --- Recording ---------------------------------------------------------
    def record_frame(self) -> None:
        frame = np.full_like(self.occupancy, fill_value=EMPTY)
        frame[self.ground_y, :] = GROUND
        for r, c in self.pellets:
            if 0 <= r < self.height and 0 <= c < self.width:
                frame[r, c] = PELLET
        for cid, creature in self.creatures.items():
            r, c = creature.position
            color_idx = self._color_index(creature.color_gene)
            frame[r, c] = CREATURE_BASE + color_idx

        self.history.append(frame.copy())
        stats = [
            {
                "id": cid,
                "row": creature.position[0],
                "col": creature.position[1],
                "energy": creature.energy,
                "split": creature.reproduction_split,
                "color": creature.color_gene.tolist(),
                "color_idx": self._color_index(creature.color_gene),
                "lifetime_energy": creature.lifetime_energy,
            }
            for cid, creature in self.creatures.items()
        ]
        self.stats_history.append(stats)

    def _color_index(self, gene: np.ndarray) -> int:
        bins = np.clip(np.round(gene * (COLOR_BINS - 1)), 0, COLOR_BINS - 1).astype(int)
        # Pack into a single index so palette order is deterministic.
        idx = 0
        for b in bins:
            idx = idx * COLOR_BINS + int(b)
        return int(idx)

    # --- Persistence -------------------------------------------------------
    def save_history(self, path: str | Path, metadata: Optional[dict] = None) -> Path:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        history_array = np.stack(self.history)
        metadata_json = json.dumps(metadata or {})
        np.savez_compressed(dest, history=history_array, metadata=metadata_json)
        return dest

    @staticmethod
    def load_history(path: str | Path) -> tuple[np.ndarray, dict]:
        archive = np.load(path, allow_pickle=False)
        metadata = json.loads(str(archive.get("metadata", "{}")))
        return archive["history"], metadata
