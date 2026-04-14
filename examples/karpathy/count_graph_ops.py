import json
from collections import Counter
from pathlib import Path
from statistics import median

from nn import MLP
from utils import full_arch, hinge_loss, load_dataset, load_weights, update_params, weights_path

DEFAULT_N_SAMPLES = 200
DEFAULT_ARCH = [16, 16, 1]
DEFAULT_N_STEPS = 100
DEFAULT_ACTIVATION = "relu"
DEFAULT_LOSS = "hinge"
DEFAULT_LR = 1.0
BENCHMARK_JSON = Path(__file__).resolve().parent.parent / "results" / "bench_python.json"


def op_label(node):
    if not node._op:
        return "leaf"
    if node._op.startswith("**"):
        return "pow"
    return node._op


def topo(root):
    order = []
    visited = set()
    stack = [(root, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if node in visited:
            continue
        visited.add(node)
        stack.append((node, True))
        for child in node._prev:
            if child not in visited:
                stack.append((child, False))
    return order


def count_graph(root):
    counts = Counter()
    for node in topo(root):
        counts[op_label(node)] += 1
    return counts


def benchmark_median_seconds(path: Path) -> float:
    with path.open() as f:
        data = json.load(f)
    target_arch = full_arch(DEFAULT_ARCH)
    for result in data["results"]:
        if (
            result["n_samples"] == DEFAULT_N_SAMPLES
            and result["n_steps"] == DEFAULT_N_STEPS
            and result["arch"] == target_arch
            and result.get("activation", DEFAULT_ACTIVATION) == DEFAULT_ACTIVATION
            and result.get("loss", DEFAULT_LOSS) == DEFAULT_LOSS
        ):
            return median(trial["elapsed_seconds"] for trial in result["trials"])
    raise RuntimeError(f"Could not find matching benchmark result in {path}")


def main() -> None:
    X, y = load_dataset(DEFAULT_N_SAMPLES)
    model = MLP(2, DEFAULT_ARCH, activation=DEFAULT_ACTIVATION)
    load_weights(model, weights_path(DEFAULT_ARCH))

    total_counts = Counter()
    for step in range(DEFAULT_N_STEPS):
        total_loss, _ = hinge_loss(model, X, y)
        total_counts.update(count_graph(total_loss))
        model.zero_grad()
        total_loss.backward()
        update_params(model, step, DEFAULT_N_STEPS, DEFAULT_LR)

    leaf_nodes = total_counts.get("leaf", 0)
    total_nodes = sum(total_counts.values())
    op_nodes = total_nodes - leaf_nodes
    median_seconds = benchmark_median_seconds(BENCHMARK_JSON)

    summary = {
        "language": "python",
        "config": {
            "n_samples": DEFAULT_N_SAMPLES,
            "arch": full_arch(DEFAULT_ARCH),
            "n_steps": DEFAULT_N_STEPS,
            "activation": DEFAULT_ACTIVATION,
            "loss": DEFAULT_LOSS,
        },
        "graph_node_counts": dict(sorted(total_counts.items())),
        "total_graph_nodes": total_nodes,
        "leaf_nodes": leaf_nodes,
        "operation_nodes": op_nodes,
        "benchmark_median_elapsed_seconds": median_seconds,
        "effective_total_nodes_per_second": total_nodes / median_seconds,
        "effective_operation_nodes_per_second": op_nodes / median_seconds,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
