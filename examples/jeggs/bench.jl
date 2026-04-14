using Printf
using Micrograd

include(joinpath(@__DIR__, "..", "script_utils.jl"))
include("utils.jl")
using .MicrogradExampleScriptUtils
using .MicrogradExampleUtils

const DATASET_SIZES = [100, 200, 500, 1000, 5000, 10000]
const ARCHITECTURES = [[8, 1], [16, 16, 1], [32, 32, 1], [16, 32, 16, 1]]
const N_TRIALS = 5
const N_STEPS = 100
const STEP_COUNTS = [1, 10, 100, 500, 1000]
const STEP_SWEEP_N_SAMPLES = 200
const STEP_SWEEP_ARCH = [16, 16, 1]
const LR = 1.0
const PHASES = ("forward", "backward", "update")

option_activation(options, key, default) = haskey(options, key) ? activation_symbol(options[key]) : activation_symbol(default)
option_loss(options, key, default) = haskey(options, key) ? loss_symbol(options[key]) : loss_symbol(default)

function phase_dict(value)
    return Dict(phase => value for phase in PHASES)
end

# ─── Single trial ────────────────────────────────────────────────────

function run_trial(
    n_samples::Int,
    arch::Vector{Int};
    n_steps::Int=N_STEPS,
    lr::Real=LR,
    activation::Symbol=:relu,
    loss::Symbol=:hinge,
)
    X, y = load_dataset(n_samples)
    model = MLP(2, arch; activation=activation)
    load_weights!(model, weights_path(arch))
    loss_fn = loss_function(loss)

    GC.gc()

    local total_loss, acc
    phase_seconds = phase_dict(0.0)
    phase_bytes = phase_dict(0)

    start_ns = time_ns()
    for step in 0:(n_steps - 1)
        forward = @timed loss_fn(model, X, y)
        total_loss, acc = forward.value
        phase_seconds["forward"] += forward.time
        phase_bytes["forward"] += forward.bytes

        backward = @timed begin
            zero_grad!(model)
            backward!(total_loss)
        end
        phase_seconds["backward"] += backward.time
        phase_bytes["backward"] += backward.bytes

        update = @timed begin
            update_params!(model, step, n_steps, lr)
        end
        phase_seconds["update"] += update.time
        phase_bytes["update"] += update.bytes
    end
    elapsed = (time_ns() - start_ns) / 1e9

    total_bytes = sum(values(phase_bytes))

    return Dict(
        "elapsed_seconds" => elapsed,
        "seconds_per_step" => elapsed / n_steps,
        "phase_seconds" => phase_seconds,
        "phase_seconds_per_step" => Dict(
            phase => phase_seconds[phase] / n_steps for phase in PHASES
        ),
        "memory" => Dict(
            "tracking" => "timed_bytes",
            "current_bytes" => nothing,
            "peak_bytes" => nothing,
            "total_bytes" => total_bytes,
            "bytes_per_step" => total_bytes / n_steps,
            "phase_bytes" => phase_bytes,
            "phase_bytes_per_step" => Dict(
                phase => phase_bytes[phase] / n_steps for phase in PHASES
            ),
        ),
        "final_loss" => total_loss.data,
        "final_acc" => acc,
    )
end

function run_config(
    n_samples::Int,
    arch::Vector{Int};
    n_steps::Int=N_STEPS,
    n_trials::Int=N_TRIALS,
    lr::Real=LR,
    activation::Symbol=:relu,
    loss::Symbol=:hinge,
    section::AbstractString="config",
    config_index::Int=1,
    total_configs::Int=1,
    quiet::Bool=false,
)
    arch_label = full_arch(arch)
    label = "n=$n_samples arch=$arch_label steps=$n_steps activation=$activation loss=$loss"
    if !quiet
        println()
        @printf("[%s %02d/%02d] %s\n", section, config_index, total_configs, label)
    end

    trials = []
    for t in 1:n_trials
        trial = run_trial(n_samples, arch; n_steps=n_steps, lr=lr, activation=activation, loss=loss)
        quiet || @printf("    trial %d/%d  %.2fs\n", t, n_trials, trial["elapsed_seconds"])
        push!(trials, trial)
    end

    return Dict(
        "n_samples" => n_samples,
        "arch" => arch_label,
        "n_steps" => n_steps,
        "lr" => lr,
        "activation" => String(activation),
        "loss" => String(loss),
        "trials" => trials,
    )
end

# ─── Benchmark sweep ─────────────────────────────────────────────────

function throughput_sweep!(
    results::Vector{Any};
    n_steps::Int=N_STEPS,
    n_trials::Int=N_TRIALS,
    lr::Real=LR,
    activation::Symbol=:relu,
    loss::Symbol=:hinge,
    checkpoint=nothing,
    quiet::Bool=false,
)
    total_configs = length(ARCHITECTURES) * length(DATASET_SIZES)
    config_index = 0
    for arch in ARCHITECTURES
        for n_samples in DATASET_SIZES
            config_index += 1
            push!(results, run_config(
                n_samples,
                arch;
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
            checkpoint === nothing || checkpoint()
        end
    end
end

function step_sweep!(
    results::Vector{Any};
    n_samples::Int=STEP_SWEEP_N_SAMPLES,
    arch::Vector{Int}=STEP_SWEEP_ARCH,
    step_counts::Vector{Int}=STEP_COUNTS,
    n_trials::Int=N_TRIALS,
    lr::Real=LR,
    activation::Symbol=:relu,
    loss::Symbol=:hinge,
    checkpoint=nothing,
    quiet::Bool=false,
)
    total_configs = length(step_counts)
    for (config_index, n_steps) in enumerate(step_counts)
        push!(results, run_config(
            n_samples,
            arch;
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
        checkpoint === nothing || checkpoint()
    end
end

function bench(args=ARGS)
    options = parse_options(args)
    mode = get(options, "mode", "all")
    n_trials = option_int(options, "trials", N_TRIALS)
    n_steps = option_int(options, "n-steps", N_STEPS)
    n_samples = option_int(options, "n-samples", STEP_SWEEP_N_SAMPLES)
    arch = option_int_list(options, "arch", STEP_SWEEP_ARCH)
    lr = option_float(options, "lr", LR)
    activation = option_activation(options, "activation", :relu)
    loss = option_loss(options, "loss", :hinge)
    step_counts = option_int_list(options, "step-counts", STEP_COUNTS)
    quiet = option_bool(options, "quiet", false)
    out_path = get(options, "out", joinpath(@__DIR__, "..", "results", "bench_julia.json"))

    results = Any[]
    step_sweep_results = Any[]
    output = Dict{String,Any}(
        "language" => "julia",
        "version" => string(VERSION),
        "platform" => Sys.MACHINE,
        "mode" => mode,
        "n_trials" => n_trials,
        "results" => results,
        "step_sweep_results" => step_sweep_results,
        "status" => "running",
    )
    checkpoint() = write_json_output(output, out_path)
    checkpoint()

    if mode == "all"
        throughput_sweep!(results; n_trials=n_trials, lr=lr, activation=activation, loss=loss, checkpoint=checkpoint, quiet=quiet)
        step_sweep!(step_sweep_results; n_trials=n_trials, lr=lr, activation=activation, loss=loss, checkpoint=checkpoint, quiet=quiet)
    elseif mode == "throughput"
        throughput_sweep!(results; n_steps=n_steps, n_trials=n_trials, lr=lr, activation=activation, loss=loss, checkpoint=checkpoint, quiet=quiet)
    elseif mode == "step-sweep"
        step_sweep!(step_sweep_results;
            n_samples=n_samples,
            arch=arch,
            step_counts=step_counts,
            n_trials=n_trials,
            lr=lr,
            activation=activation,
            loss=loss,
            checkpoint=checkpoint,
            quiet=quiet,
        )
    elseif mode == "single"
        push!(results, run_config(
            n_samples,
            arch;
            n_steps=n_steps,
            n_trials=n_trials,
            lr=lr,
            activation=activation,
            loss=loss,
            section="single",
            config_index=1,
            total_configs=1,
            quiet=quiet,
        ))
        checkpoint()
    else
        error("unknown mode: $mode")
    end

    output["status"] = "complete"
    checkpoint()
    quiet || println("\nSaved $out_path")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    bench()
end
