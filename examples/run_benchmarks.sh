#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES_DIR="$ROOT_DIR/examples"
PYTHON_DIR="$EXAMPLES_DIR/karpathy"
RESULTS_DIR="$EXAMPLES_DIR/results"
FIGURES_DIR="$EXAMPLES_DIR/figures"
COLD_RESULTS_DIR="$RESULTS_DIR/cold_process"
LOG_PATH="$RESULTS_DIR/run.log"

cd "$ROOT_DIR"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log_line() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

section() {
  printf '\n[%s] ==== %s ====\n' "$(timestamp)" "$*"
}

mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR/bench_julia.json"
rm -f "$RESULTS_DIR/bench_python.json"
rm -f "$RESULTS_DIR"/bench_python_smoke.json
rm -f "$LOG_PATH"
rm -rf "$COLD_RESULTS_DIR"
rm -rf "$FIGURES_DIR"

exec > >(tee -a "$LOG_PATH") 2>&1
log_line "Logging benchmark output to $LOG_PATH"

section "Instantiating Julia environments"
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=examples -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

section "Syncing Python benchmark environment"
(
  cd "$PYTHON_DIR"
  uv sync
)

section "Running Julia in-process benchmark sweep"
julia --project=examples examples/jeggs/bench.jl

section "Running Python in-process benchmark sweep"
(
  cd "$PYTHON_DIR"
  uv run python bench.py
)

section "Running cold-process step sweep"
julia --project=examples examples/cold_process_sweep.jl --trials 5

section "Rendering figures"
julia --project=examples examples/plot_results.jl
julia --project=examples examples/plot_results.jl examples/results/cold_process

section "Done"
log_line "Results: $RESULTS_DIR"
log_line "Figures: $FIGURES_DIR"
