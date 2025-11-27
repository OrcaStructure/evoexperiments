# evoexperiments

Python utilities for experimenting with grid-based simulations (cellular automata,
lattice games, etc.), including animation and save/replay of full histories.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Run the included Game of Life demo:

```bash
python examples/game_of_life_demo.py --size 60 --steps 200 --save runs/life_run.npz
```

- A matplotlib window shows the animation (use `--no-show` for headless runs).
- Histories are stored as compressed `.npz` archives you can reload later.
- Add `--video runs/life.gif` to save an animation (requires a writer such as ffmpeg or ImageMagick).

Replay or extend a previous run:

```bash
python examples/game_of_life_demo.py --load runs/life_run.npz --continue-steps 50 --save runs/extended.npz
```

### Plant world (sunlight + genomes)

```bash
python examples/plant_world_demo.py --days 80 --save runs/plants.npz
```

- Blue sky/ground background with a yellow sun row descending four times per day (two straight down, one 45° left, one 45° right).
- Plants own connected tiles and hold energy. Sunlight collisions add energy; upkeep drains it; low energy kills the plant.
- Each plant has a tiny neural net genome that chooses actions (grow up/down/left/right or seed). Ties within 0.1 do multiple actions; impossible choices fall back to the next best. Seeds spawn new plants on the ground with mutated genomes.
- View or replay runs: `python examples/plant_world_demo.py --load runs/plants.npz --no-show --video runs/plants.gif`

## Library usage

```python
import numpy as np
from evoexperiments import GridSimulation, animate_history

def rule(grid: np.ndarray) -> np.ndarray:
    # Toy rule: toggle cells if the sum of their neighbors is odd
    neighbors = (
        np.roll(grid, 1, 0)
        + np.roll(grid, -1, 0)
        + np.roll(grid, 1, 1)
        + np.roll(grid, -1, 1)
    )
    return (neighbors % 2).astype(np.uint8)

sim = GridSimulation(initial_state=np.zeros((8, 8)), update_rule=rule, dtype=np.uint8)
sim.run(20)
sim.save_history("runs/toy.npz", metadata={"rule": "odd-sum"})
animate_history(sim.history, interval=150, title="Odd rule")
```

Key pieces:

- `GridSimulation` manages state, applies an update rule, and records every frame.
- `GridSimulation.save_history` / `load_history` persist complete runs (with optional metadata).
- `animate_history` and `animate_simulation` render histories or live runs. Use `show=False` if you only want to save output.
