from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import json

import numpy as np

# Visual categories for rendering.
EMPTY = 0
GROUND = 1
SUN = 2
PLANT_BASE = 3  # actual value is PLANT_BASE + (plant_id % palette_span)
PALETTE_SPAN = 10

# Action indices
UP, DOWN, LEFT, RIGHT, SEED = range(5)
DIRECTION_OFFSETS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass
class Genome:
    """Two-layer perceptron genome controlling growth actions."""

    w1: np.ndarray  # (hidden, input)
    b1: np.ndarray  # (hidden,)
    w2: np.ndarray  # (5, hidden)
    b2: np.ndarray  # (5,)

    @classmethod
    def random(cls, rng: np.random.Generator, input_dim: int, hidden: int = 12, scale: float = 0.3) -> "Genome":
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
class Plant:
    plant_id: int
    cells: set[Tuple[int, int]]
    energy: float
    genome: Genome
    used_for_growth: set[Tuple[int, int]] = field(default_factory=set)
    last_direction: Optional[int] = None
    units_since_dir_change: int = 0
    units_since_branch: int = 0
    energy_after_sun: float = 0.0
    energy_after_maintenance: float = 0.0


@dataclass
class Seed:
    column: int
    genome: Genome


class PlantWorld:
    """
    Simulates plants on a grid with descending sunlight and neural-network-driven growth.

    Each day:
      1. Four sunlight passes descend from the sky (two vertical, one 45° left, one 45° right).
      2. Plants are processed in random order: pay upkeep, decide growth/seed actions until out of energy.
      3. Seeds become new plants on the ground with perturbed genomes.
    """

    def __init__(
        self,
        width: int,
        height: int,
        ground_y: Optional[int] = None,
        rng_seed: Optional[int] = None,
        sunlight_energy: float = 3.0,
        maintenance_cost: float = 0.2,
        growth_cost: float = 1.0,
        seed_cost: float = 2.0,
        seed_energy: float = 2.0,
        max_actions_per_plant: int = 6,
    ):
        if width < 4 or height < 4:
            raise ValueError("World must be at least 4x4")
        self.width = width
        self.height = height
        self.ground_y = ground_y if ground_y is not None else height - 1
        if not (0 <= self.ground_y < height):
            raise ValueError("ground_y must lie inside the grid")

        self.rng = np.random.default_rng(rng_seed)
        self.sunlight_energy = sunlight_energy
        self.maintenance_cost = maintenance_cost
        self.growth_cost = growth_cost
        self.seed_cost = seed_cost
        self.seed_energy = seed_energy
        self.max_actions_per_plant = max_actions_per_plant

        # Occupancy grid: 0 empty, >0 plant id.
        self.occupancy = np.zeros((height, width), dtype=np.int32)
        self.plants: Dict[int, Plant] = {}
        self.next_plant_id = 1

        self.pending_seeds: List[Seed] = []
        self.history: List[np.ndarray] = []
        self.stats_history: List[list] = []
        self.record_frame()

        # Precompute sunlight attenuation by row (strongly dim near ground).
        self._sun_row_factors = np.ones(self.height, dtype=float)
        decay_start = max(0, self.ground_y - 9)  # last 10 rows include ground
        for r in range(decay_start, self.height):
            ratio = (self.ground_y - r) / max(1, self.ground_y - decay_start)  # 1 at decay_start, 0 at ground
            factor = 0.05 + 0.95 * max(0.0, ratio) ** 2  # quadratic fade; floor at 5%
            self._sun_row_factors[r] = factor

    # --- Initialization helpers -------------------------------------------------
    def add_plant(self, column: int, energy: float, genome: Optional[Genome] = None) -> Optional[int]:
        """Place a plant as a single cell on the ground in the given column."""
        if not (0 <= column < self.width):
            return None
        pos = (self.ground_y, column)
        if self.occupancy[pos] != 0:
            return None
        plant_id = self.next_plant_id
        self.next_plant_id += 1
        genome = genome or Genome.random(self.rng, input_dim=self._feature_dim())
        plant = Plant(plant_id=plant_id, cells={pos}, energy=energy, genome=genome)
        self.plants[plant_id] = plant
        self.occupancy[pos] = plant_id
        return plant_id

    # --- Simulation steps ------------------------------------------------------
    def advance_day(self) -> None:
        """Run sunlight passes, plant upkeep/growth, seed resolution, and record frames."""
        self._reset_daily_energy_tracking()
        # Sunlight passes: one vertical per column, plus two diagonal sweeps that each hit every ground cell once.
        self._sun_sweep_vertical()
        self._sun_sweep_diag_left()   # down-right
        self._sun_sweep_diag_right()  # down-left
        self._capture_energy_after_sun()

        self._process_plants()
        self._resolve_seeds()
        self.record_frame()
        self._wipe_energy_end_of_day()

    def run(self, days: int) -> List[np.ndarray]:
        for _ in range(days):
            self.advance_day()
        return self.history

    # --- Sunlight --------------------------------------------------------------
    def _sun_sweep_vertical(self) -> None:
        active = set(range(self.width))
        for row in range(self.ground_y + 1):
            sun_positions: List[Tuple[int, int]] = []
            next_active = set()
            for col in active:
                pos = (row, col)
                sun_positions.append(pos)
                occupant = self.occupancy[pos]
                if occupant > 0:
                    self._apply_sun_to_plant(occupant, row)
                    # beam stops on hit
                    continue
                next_active.add(col)
            self.record_frame(sun_path=sun_positions, capture_stats=False)
            active = next_active

    def _sun_sweep_diag_left(self) -> None:
        """Sun from top-left moving down-right; front is a line sloping down-left to up-right (perpendicular to motion)."""
        active: set[int] = {0}  # index i corresponds to position (t - i, i)
        max_steps = self.ground_y + self.width
        for t in range(max_steps):
            sun_positions: List[Tuple[int, int]] = []
            next_active: set[int] = set()
            for i in active:
                r = t - i
                c = i
                if not (0 <= r < self.height and 0 <= c < self.width):
                    continue
                sun_positions.append((r, c))
                occupant = self.occupancy[r, c]
                if occupant > 0:
                    self._apply_sun_to_plant(occupant, r)
                    # beam stops
                    continue
                # continue this beam
                next_active.add(i)
            # spawn a new beam at the top edge each step (widens the front)
            if t + 1 < self.width:
                next_active.add(t + 1)
            self.record_frame(sun_path=sun_positions, capture_stats=False)
            # stop once beams are below ground and no new beams will appear
            active = {i for i in next_active if (t + 1 - i) <= self.ground_y}
            if not active and t + 1 >= self.width:
                break

    def _sun_sweep_diag_right(self) -> None:
        """Sun from top-right moving down-left; front is a line sloping up-left to down-right (perpendicular to motion)."""
        active: set[int] = {0}  # index i corresponds to position (t - i, self.width - 1 - i)
        max_steps = self.ground_y + self.width
        for t in range(max_steps):
            sun_positions: List[Tuple[int, int]] = []
            next_active: set[int] = set()
            for i in active:
                r = t - i
                c = self.width - 1 - i
                if not (0 <= r < self.height and 0 <= c < self.width):
                    continue
                sun_positions.append((r, c))
                occupant = self.occupancy[r, c]
                if occupant > 0:
                    self._apply_sun_to_plant(occupant, r)
                    continue
                next_active.add(i)
            if t + 1 < self.width:
                next_active.add(t + 1)
            self.record_frame(sun_path=sun_positions, capture_stats=False)
            active = {i for i in next_active if (t + 1 - i) <= self.ground_y}
            if not active and t + 1 >= self.width:
                break

    # --- Plant updates ---------------------------------------------------------
    def _process_plants(self) -> None:
        plant_ids = list(self.plants.keys())
        self.rng.shuffle(plant_ids)

        for pid in plant_ids:
            plant = self.plants.get(pid)
            if plant is None:
                continue

            upkeep = self.maintenance_cost * len(plant.cells)
            plant.energy -= upkeep
            plant.energy_after_maintenance = max(0.0, plant.energy)
            if plant.energy < 0:
                self._remove_plant(pid)
                continue

            actions = 0
            while plant.energy >= min(self.growth_cost, self.seed_cost) and actions < self.max_actions_per_plant:
                actions += 1
                if not self._attempt_action(plant):
                    break

    def _attempt_action(self, plant: Plant) -> bool:
        features = self._features_for_plant(plant)
        logits = plant.genome.forward(features)
        probs = softmax(logits)
        order = list(np.argsort(probs))[::-1]  # descending

        max_prob = probs[order[0]]
        primary_candidates = [idx for idx in order if max_prob - probs[idx] <= 0.1]

        # Execute tied primaries (could be multiple actions).
        tried = set()
        successes = 0
        growth_successes = 0
        for action_idx in primary_candidates:
            tried.add(action_idx)
            result = self._execute_action(plant, action_idx, order, probs, branch_group=len(primary_candidates) > 1)
            if result.success:
                successes += 1
                if result.grew:
                    growth_successes += 1
        if successes > 0:
            # Branch counter reset happens inside _execute_action when branch_group True.
            return True

        # If primaries failed, try other actions in order.
        for action_idx in order:
            if action_idx in tried:
                continue
            result = self._execute_action(plant, action_idx, order, probs, branch_group=False)
            if result.success:
                return True
        return False

    def _execute_action(
        self, plant: Plant, action_idx: int, order: List[int], probs: np.ndarray, branch_group: bool
    ):
        Result = lambda success, grew=False: type("Result", (), {"success": success, "grew": grew})
        if action_idx == SEED:
            if plant.energy < self.seed_cost:
                return Result(False)
            # Choose a direction by the highest remaining outputs.
            for dir_idx in order:
                if dir_idx == SEED:
                    continue
                target = self._pick_growth_target(plant, dir_idx)
                if target is None:
                    continue
                plant.energy -= self.seed_cost
                target_pos, _parent = target
                self.pending_seeds.append(Seed(column=target_pos[1], genome=plant.genome.mutate(self.rng)))
                return Result(True, grew=False)
            return Result(False)

        # Growth actions
        if plant.energy < self.growth_cost:
            return Result(False)
        target = self._pick_growth_target(plant, action_idx)
        if target is None:
            return Result(False)

        plant.energy -= self.growth_cost
        target_pos, parent_cell = target
        plant.cells.add(target_pos)
        plant.used_for_growth.add(parent_cell)
        self.occupancy[target_pos] = plant.plant_id
        self._register_growth(plant, action_idx, branch_group)
        return Result(True, grew=True)

    def _pick_growth_target(self, plant: Plant, direction_idx: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        dr, dc = DIRECTION_OFFSETS.get(direction_idx, (0, 0))
        candidates: list[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for r, c in plant.cells:
            if (r, c) in plant.used_for_growth:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and self.occupancy[nr, nc] == 0:
                candidates.append(((nr, nc), (r, c)))
        if not candidates:
            return None
        return candidates[self.rng.integers(len(candidates))]

    def _remove_plant(self, plant_id: int) -> None:
        plant = self.plants.pop(plant_id, None)
        if not plant:
            return
        for cell in plant.cells:
            if self.occupancy[cell] == plant_id:
                self.occupancy[cell] = 0

    # --- Seeding ----------------------------------------------------------------
    def _resolve_seeds(self) -> None:
        seeds = self.pending_seeds
        self.pending_seeds = []
        for seed in seeds:
            pos = (self.ground_y, seed.column)
            if 0 <= seed.column < self.width and self.occupancy[pos] == 0:
                self.add_plant(column=seed.column, energy=self.seed_energy, genome=seed.genome)

    # --- Features and rendering -------------------------------------------------
    def _feature_dim(self) -> int:
        # total cells, energy, since direction change, since branch
        return 4

    def _register_growth(self, plant: Plant, direction_idx: int, branch_group: bool) -> None:
        # Track direction change distance.
        if plant.last_direction is None or plant.last_direction != direction_idx:
            plant.units_since_dir_change = 0
            plant.last_direction = direction_idx
        else:
            plant.units_since_dir_change += 1

        # Track branching distance (reset when a tie led to a successful growth).
        if branch_group:
            plant.units_since_branch = 0
        else:
            plant.units_since_branch += 1

    def _features_for_plant(self, plant: Plant) -> np.ndarray:
        return np.asarray(
            [
                float(len(plant.cells)),
                float(plant.energy),
                float(plant.units_since_dir_change),
                float(plant.units_since_branch),
            ],
            dtype=float,
        )

    def record_frame(self, sun_path: Optional[Iterable[Tuple[int, int]]] = None, capture_stats: bool = True) -> None:
        frame = np.full_like(self.occupancy, fill_value=EMPTY)
        frame[self.ground_y, :] = GROUND

        # Plants
        for pid, plant in self.plants.items():
            val = PLANT_BASE + (pid % PALETTE_SPAN)
            for r, c in plant.cells:
                frame[r, c] = val

        if sun_path:
            for r, c in sun_path:
                if 0 <= r < self.height and 0 <= c < self.width:
                    frame[r, c] = SUN

        self.history.append(frame.copy())
        if capture_stats:
            stats = []
            for pid, plant in self.plants.items():
                stats.append(
                    {
                        "id": pid,
                        "cells": len(plant.cells),
                        "energy": plant.energy,
                        "energy_sun": plant.energy_after_sun,
                        "energy_post_maint": plant.energy_after_maintenance,
                    }
                )
            self.stats_history.append(stats)
        else:
            # Reuse last known stats to keep table values stable across intra-day frames.
            if self.stats_history:
                self.stats_history.append(self.stats_history[-1])
            else:
                self.stats_history.append([])

    # --- Energy tracking helpers ---------------------------------------------
    def _reset_daily_energy_tracking(self) -> None:
        for plant in self.plants.values():
            plant.energy_after_sun = 0.0
            plant.energy_after_maintenance = 0.0
            # ensure new day doesn't reset growth usage; a cell can only be used once ever.

    def _capture_energy_after_sun(self) -> None:
        for plant in self.plants.values():
            plant.energy_after_sun = plant.energy

    def _wipe_energy_end_of_day(self) -> None:
        for plant in self.plants.values():
            plant.energy = 0.0

    def _apply_sun_to_plant(self, plant_id: int, row: int) -> None:
        plant = self.plants.get(plant_id)
        if not plant:
            return
        factor = self._sun_row_factors[row] if 0 <= row < len(self._sun_row_factors) else 1.0
        plant.energy += self.sunlight_energy * factor

    # --- Persistence -----------------------------------------------------------
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
