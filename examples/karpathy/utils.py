"""Shared helpers for the Python micrograd experiments."""

import json
from pathlib import Path

from engine import Value
from nn import MLP

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

__all__ = [
    "accuracy",
    "cross_entropy_loss",
    "data_path",
    "full_arch",
    "hinge_loss",
    "load_dataset",
    "load_weights",
    "scores_and_reg",
    "update_params",
    "weights_path",
]


def full_arch(arch: list[int]) -> list[int]:
    return [2, *arch]


def weights_path(arch: list[int]) -> Path:
    name = "_".join(str(dim) for dim in full_arch(arch)) + ".json"
    return WEIGHTS_DIR / name


def data_path(n_samples: int) -> Path:
    return DATA_DIR / f"moons_{n_samples}.json"


def load_dataset(n_samples: int) -> tuple[list[list[float]], list[int]]:
    with open(data_path(n_samples)) as f:
        data = json.load(f)
    return data["X"], data["y"]


def load_weights(model: MLP, path: str | Path) -> None:
    with open(path) as f:
        data = json.load(f)

    for layer, layer_data in zip(model.layers, data["layers"], strict=True):
        for neuron, weights, bias in zip(
            layer.neurons,
            layer_data["weights"],
            layer_data["biases"],
            strict=True,
        ):
            for w, val in zip(neuron.w, weights, strict=True):
                w.data = val
            neuron.b.data = bias


def scores_and_reg(model: MLP, X: list, alpha: float = 1e-4) -> tuple[list[Value], Value]:
    inputs = [list(map(Value, xrow)) for xrow in X]
    scores = list(map(model, inputs))
    reg_loss = alpha * sum(p * p for p in model.parameters())
    return scores, reg_loss


def accuracy(y: list, scores: list[Value]) -> float:
    correct = sum((yi > 0) == (score.data > 0) for yi, score in zip(y, scores, strict=True))
    return correct / len(y)


def hinge_loss(model: MLP, X: list, y: list) -> tuple[Value, float]:
    scores, reg_loss = scores_and_reg(model, X)
    losses = [(1 + -yi * scorei).relu() for yi, scorei in zip(y, scores, strict=True)]
    data_loss = sum(losses) * (1.0 / len(losses))
    total_loss = data_loss + reg_loss
    return total_loss, accuracy(y, scores)


def sigmoid(score: Value) -> Value:
    return Value(1.0) / (Value(1.0) + (-score).exp())


def binary_cross_entropy(score: Value, label: int) -> Value:
    target = (1 + label) / 2
    prob = sigmoid(score)
    return -(target * prob.log() + (1 - target) * (Value(1.0) - prob).log())


def cross_entropy_loss(model: MLP, X: list, y: list) -> tuple[Value, float]:
    scores, reg_loss = scores_and_reg(model, X)
    losses = [
        binary_cross_entropy(score, label) for score, label in zip(scores, y, strict=True)
    ]
    data_loss = sum(losses) * (1.0 / len(losses))
    total_loss = data_loss + reg_loss
    return total_loss, accuracy(y, scores)


def update_params(model: MLP, step: int, n_steps: int, lr: float) -> None:
    step_lr = lr - 0.9 * step / n_steps
    for p in model.parameters():
        p.data -= step_lr * p.grad
