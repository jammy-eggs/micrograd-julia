"""Benchmark Karpathy's micrograd across dataset sizes and architectures."""

import argparse
import gc
import json
import platform
import time
import tracemalloc
from pathlib import Path

from nn import MLP
from utils import (
    cross_entropy_loss,
    full_arch,
    hinge_loss,
    load_dataset,
    load_weights,
    update_params,
    weights_path,
)

DATASET_SIZES = [100, 200, 500, 1000, 5000, 10000]
ARCHITECTURES = [[8, 1], [16, 16, 1], [32, 32, 1], [16, 32, 16, 1]]
N_TRIALS = 5
N_STEPS = 100
STEP_COUNTS = [1, 10, 100, 500, 1000]
STEP_SWEEP_N_SAMPLES = 200
STEP_SWEEP_ARCH = [16, 16, 1]
LR = 1.0
PHASES = ("forward", "backward", "update")
ACTIVATIONS = {"relu", "tanh"}
LOSSES = {"hinge", "cross_entropy"}


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_activation(value: str) -> str:
    activation = value.strip().lower()
    if activation not in ACTIVATIONS:
        raise argparse.ArgumentTypeError(f"unsupported activation: {value}")
    return activation


def parse_loss(value: str) -> str:
    loss = value.strip().lower().replace("-", "_")
    if loss in {"crossentropy", "bce"}:
        loss = "cross_entropy"
    if loss not in LOSSES:
        raise argparse.ArgumentTypeError(f"unsupported loss: {value}")
    return loss


def resolve_loss_fn(loss: str):
    return hinge_loss if loss == "hinge" else cross_entropy_loss


def write_output(path: Path, output: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2)
    tmp_path.replace(path)


def run_trial(
    n_samples: int,
    arch: list[int],
    n_steps: int = N_STEPS,
    lr: float = LR,
    activation: str = "relu",
    loss: str = "hinge",
) -> dict:
    X, y = load_dataset(n_samples)
    model = MLP(2, arch, activation=activation)
    load_weights(model, weights_path(arch))
    loss_fn = resolve_loss_fn(loss)

    gc.collect()

    phase_seconds = dict.fromkeys(PHASES, 0.0)
    tracemalloc.start()
    t_start = time.perf_counter()
    for k in range(n_steps):
        phase_start = time.perf_counter()
        total_loss, acc = loss_fn(model, X, y)
        phase_seconds["forward"] += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        model.zero_grad()
        total_loss.backward()
        phase_seconds["backward"] += time.perf_counter() - phase_start

        phase_start = time.perf_counter()
        update_params(model, k, n_steps, lr)
        phase_seconds["update"] += time.perf_counter() - phase_start

    elapsed = time.perf_counter() - t_start
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / n_steps,
        "phase_seconds": phase_seconds,
        "phase_seconds_per_step": {
            phase: seconds / n_steps for phase, seconds in phase_seconds.items()
        },
        "memory": {
            "tracking": "tracemalloc",
            "current_bytes": current_bytes,
            "peak_bytes": peak_bytes,
            "total_bytes": peak_bytes,
            "bytes_per_step": peak_bytes / n_steps,
            "phase_bytes": None,
            "phase_bytes_per_step": None,
        },
        "final_loss": total_loss.data,
        "final_acc": acc,
    }


def run_config(
    n_samples: int,
    arch: list[int],
    n_steps: int = N_STEPS,
    n_trials: int = N_TRIALS,
    lr: float = LR,
    activation: str = "relu",
    loss: str = "hinge",
    section: str = "config",
    config_index: int = 1,
    total_configs: int = 1,
    quiet: bool = False,
) -> dict:
    label = (
        f"n={n_samples} arch={full_arch(arch)} steps={n_steps} "
        f"activation={activation} loss={loss}"
    )
    if not quiet:
        print("", flush=True)
        print(f"[{section} {config_index:02d}/{total_configs:02d}] {label}", flush=True)

    trials = []
    for t in range(n_trials):
        trial = run_trial(
            n_samples,
            arch,
            n_steps=n_steps,
            lr=lr,
            activation=activation,
            loss=loss,
        )
        trials.append(trial)
        if not quiet:
            print(
                f"    trial {t + 1}/{n_trials}  {trial['elapsed_seconds']:.2f}s",
                flush=True,
            )

    return {
        "n_samples": n_samples,
        "arch": full_arch(arch),
        "n_steps": n_steps,
        "lr": lr,
        "activation": activation,
        "loss": loss,
        "trials": trials,
    }


def throughput_sweep(
    results: list[dict],
    n_steps: int = N_STEPS,
    n_trials: int = N_TRIALS,
    lr: float = LR,
    activation: str = "relu",
    loss: str = "hinge",
    checkpoint=None,
    quiet: bool = False,
) -> None:
    total_configs = len(ARCHITECTURES) * len(DATASET_SIZES)
    config_index = 0

    for arch in ARCHITECTURES:
        for n_samples in DATASET_SIZES:
            config_index += 1
            results.append(run_config(
                n_samples,
                arch,
                n_steps=n_steps,
                n_trials=n_trials,
                lr=lr,
                activation=activation,
                loss=loss,
                section="throughput",
                config_index=config_index,
                total_configs=total_configs,
                quiet=quiet,
            ))
            if checkpoint is not None:
                checkpoint()


def step_sweep(
    results: list[dict],
    n_samples: int = STEP_SWEEP_N_SAMPLES,
    arch: list[int] | None = None,
    step_counts: list[int] | None = None,
    n_trials: int = N_TRIALS,
    lr: float = LR,
    activation: str = "relu",
    loss: str = "hinge",
    checkpoint=None,
    quiet: bool = False,
) -> None:
    arch = STEP_SWEEP_ARCH if arch is None else arch
    step_counts = STEP_COUNTS if step_counts is None else step_counts
    total_configs = len(step_counts)
    for config_index, n_steps in enumerate(step_counts, start=1):
        results.append(run_config(
            n_samples,
            arch,
            n_steps=n_steps,
            n_trials=n_trials,
            lr=lr,
            activation=activation,
            loss=loss,
            section="step",
            config_index=config_index,
            total_configs=total_configs,
            quiet=quiet,
        ))
        if checkpoint is not None:
            checkpoint()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["all", "throughput", "step-sweep", "single"],
        default="all",
        help="which benchmark section to run",
    )
    parser.add_argument("--n-samples", type=int, default=STEP_SWEEP_N_SAMPLES)
    parser.add_argument("--arch", type=parse_int_list, default=STEP_SWEEP_ARCH)
    parser.add_argument("--n-steps", type=int, default=N_STEPS)
    parser.add_argument("--step-counts", type=parse_int_list, default=STEP_COUNTS)
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--activation", type=parse_activation, default="relu")
    parser.add_argument("--loss", type=parse_loss, default="hinge")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "bench_python.json",
    )
    return parser


def bench(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    results = []
    step_sweep_results = []
    output = {
        "language": "python",
        "version": platform.python_version(),
        "platform": platform.machine(),
        "mode": args.mode,
        "n_trials": args.trials,
        "results": results,
        "step_sweep_results": step_sweep_results,
        "status": "running",
    }

    def checkpoint() -> None:
        write_output(args.out, output)

    checkpoint()

    if args.mode == "all":
        throughput_sweep(
            results,
            n_trials=args.trials,
            lr=args.lr,
            activation=args.activation,
            loss=args.loss,
            checkpoint=checkpoint,
            quiet=args.quiet,
        )
        step_sweep(
            step_sweep_results,
            n_trials=args.trials,
            lr=args.lr,
            activation=args.activation,
            loss=args.loss,
            checkpoint=checkpoint,
            quiet=args.quiet,
        )
    elif args.mode == "throughput":
        throughput_sweep(
            results,
            n_steps=args.n_steps,
            n_trials=args.trials,
            lr=args.lr,
            activation=args.activation,
            loss=args.loss,
            checkpoint=checkpoint,
            quiet=args.quiet,
        )
    elif args.mode == "step-sweep":
        step_sweep(
            step_sweep_results,
            n_samples=args.n_samples,
            arch=args.arch,
            step_counts=args.step_counts,
            n_trials=args.trials,
            lr=args.lr,
            activation=args.activation,
            loss=args.loss,
            checkpoint=checkpoint,
            quiet=args.quiet,
        )
    elif args.mode == "single":
        results.append(
            run_config(
                args.n_samples,
                args.arch,
                n_steps=args.n_steps,
                n_trials=args.trials,
                lr=args.lr,
                activation=args.activation,
                loss=args.loss,
                section="single",
                config_index=1,
                total_configs=1,
                quiet=args.quiet,
            )
        )
        checkpoint()

    output["status"] = "complete"
    checkpoint()
    if not args.quiet:
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    bench()
