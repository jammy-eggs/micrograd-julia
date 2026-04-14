"""Train Karpathy's micrograd on two-moons classification."""

import time
from collections.abc import Callable

from nn import MLP
from utils import cross_entropy_loss, hinge_loss, load_dataset, load_weights, update_params, weights_path


def train(
    n_samples: int = 200,
    arch: list[int] = [16, 16, 1],  # noqa: B006
    n_steps: int = 100,
    lr: float = 1.0,
    activation: str = "relu",
    loss_fn: Callable = hinge_loss,
) -> None:
    X, y = load_dataset(n_samples)

    model = MLP(2, arch, activation=activation)

    load_weights(model, weights_path(arch))

    t_start = time.perf_counter()
    for k in range(n_steps):
        total_loss, acc = loss_fn(model, X, y)

        model.zero_grad()
        total_loss.backward()
        update_params(model, k, n_steps, lr)

        if k % 10 == 0 or k == n_steps - 1:
            print(f"step {k:3d}  loss {total_loss.data:.4f}  acc {acc * 100:.1f}%")

    elapsed = time.perf_counter() - t_start
    print(f"\n{n_steps} steps in {elapsed:.3f}s ({elapsed / n_steps * 1000:.1f}ms/step)")


if __name__ == "__main__":
    train(n_samples=200, arch=[16, 16, 1])
