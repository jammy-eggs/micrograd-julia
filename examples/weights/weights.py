"""Generate deterministic initial weights for all benchmark architectures.

Run from the repository root:
    cd examples/karpathy && uv run python ../weights/weights.py
Outputs: weights/ directory with one JSON per architecture.
"""

import json
import random
from itertools import pairwise
from pathlib import Path


ARCHITECTURES = [
    [2, 8, 1],
    [2, 16, 16, 1],
    [2, 32, 32, 1],
    [2, 16, 32, 16, 1],
]
SEED = 17


def generate_weights(arch: list[int], seed: int) -> dict:
    random.seed(seed)
    return {
        "architecture": arch,
        "layers": [
            {
                "weights": [
                    [random.uniform(-1, 1) for _ in range(nin)] for _ in range(nout)
                ],
                "biases": [0.0] * nout,
            }
            for nin, nout in pairwise(arch)
        ],
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(exist_ok=True)
    for arch in ARCHITECTURES:
        data = generate_weights(arch, SEED)
        name = "_".join(str(s) for s in arch)
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2))
        n_params = sum(
            len(layer["weights"]) * len(layer["weights"][0]) + len(layer["biases"])
            for layer in data["layers"]
        )
        print(f"{name}: {n_params} params → {path}")
