"""Generate shared two-moons datasets for both benchmark implementations.

Run from the repository root:
    cd examples/karpathy && uv run python ../data/datasets.py
"""

import json
from pathlib import Path

import numpy as np

DATASET_SIZES = [100, 200, 500, 1000, 5000, 10000]
SEED = 17
NOISE = 0.2


def make_moons(n: int, noise: float = NOISE, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_top = n // 2
    n_bot = n - n_top

    theta_top = np.linspace(0, np.pi, n_top)
    theta_bot = np.linspace(0, np.pi, n_bot)

    X = np.vstack(
        [
            np.column_stack([np.cos(theta_top), np.sin(theta_top)]),
            np.column_stack([1 - np.cos(theta_bot), 0.5 - np.sin(theta_bot)]),
        ]
    )
    X += rng.standard_normal((n, 2)) * noise
    y = np.array([1] * n_top + [-1] * n_bot)
    return X, y


def dataset(n: int) -> dict:
    X, y = make_moons(n)
    return {
        "n": n,
        "noise": NOISE,
        "seed": SEED,
        "X": X.tolist(),
        "y": y.tolist(),
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    for n in DATASET_SIZES:
        path = out_dir / f"moons_{n}.json"
        path.write_text(json.dumps(dataset(n), indent=2))
        print(f"{n}: {path}")
