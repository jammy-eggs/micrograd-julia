using JSON3
using Printf

include("script_utils.jl")
using .MicrogradExampleScriptUtils

const EXAMPLES_DIR = @__DIR__
const DEFAULT_RESULTS_DIR = joinpath(EXAMPLES_DIR, "results", "cold_process")
const DEFAULT_PYTHON = joinpath(EXAMPLES_DIR, "karpathy", ".venv", "bin", "python")
const STEP_COUNTS = [1, 10, 100, 500, 1000]
const STEP_SWEEP_N_SAMPLES = 200
const STEP_SWEEP_ARCH = [16, 16, 1]
const DEFAULT_COLD_CONFIGS = [
    (n_samples=200, arch=[16, 16, 1]),
    (n_samples=1000, arch=[16, 16, 1]),
]
const N_TRIALS = 5
const LR = 1.0
const LANGUAGES = ("julia", "python")

function benchmark_command(
    language::String,
    tmp_path::AbstractString;
    n_samples::Int,
    arch::Vector{Int},
    n_steps::Int,
    lr::Real,
    activation::String,
    loss::String,
    python::AbstractString,
)
    arch_arg = join(arch, ",")
    if language == "julia"
        return `julia --project=$EXAMPLES_DIR $(joinpath(EXAMPLES_DIR, "jeggs", "bench.jl")) --mode single --n-samples $n_samples --arch $arch_arg --n-steps $n_steps --trials 1 --lr $lr --activation $activation --loss $loss --quiet --out $tmp_path`
    elseif language == "python"
        return `$python $(joinpath(EXAMPLES_DIR, "karpathy", "bench.py")) --mode single --n-samples $n_samples --arch $arch_arg --n-steps $n_steps --trials 1 --lr $lr --activation $activation --loss $loss --quiet --out $tmp_path`
    end
    error("unsupported language: $language")
end

function merge_trial!(trial::Dict{String,Any}, external_elapsed::Float64, n_steps::Int)
    benchmark_elapsed = Float64(trial["elapsed_seconds"])
    benchmark_per_step = Float64(trial["seconds_per_step"])
    trial["benchmark_elapsed_seconds"] = benchmark_elapsed
    trial["benchmark_seconds_per_step"] = benchmark_per_step
    trial["process_elapsed_seconds"] = external_elapsed
    trial["process_seconds_per_step"] = external_elapsed / n_steps
    trial["elapsed_seconds"] = external_elapsed
    trial["seconds_per_step"] = external_elapsed / n_steps
    trial["measurement"] = "fresh_process"
    return trial
end

function run_single_trial(
    language::String;
    n_samples::Int,
    arch::Vector{Int},
    n_steps::Int,
    lr::Real,
    activation::String,
    loss::String,
    python::AbstractString,
)
    tmp_path, io = mktemp()
    close(io)
    json_path = tmp_path * ".json"
    mv(tmp_path, json_path; force=true)
    cmd = benchmark_command(
        language,
        json_path;
        n_samples=n_samples,
        arch=arch,
        n_steps=n_steps,
        lr=lr,
        activation=activation,
        loss=loss,
        python=python,
    )

    elapsed = @elapsed run(cmd)
    data = try
        JSON3.read(read(json_path, String), Dict{String,Any})
    finally
        rm(json_path; force=true)
    end

    result = only(data["results"])
    trial = only(result["trials"])
    merge_trial!(trial, elapsed, n_steps)
    return (
        language=String(data["language"]),
        version=String(data["version"]),
        platform=String(data["platform"]),
        result=result,
        trial=trial,
    )
end

function run_config(
    language::String;
    n_samples::Int,
    arch::Vector{Int},
    n_steps::Int,
    n_trials::Int,
    lr::Real,
    activation::String,
    loss::String,
    python::AbstractString,
    config_index::Int,
    total_configs::Int,
)
    arch_label = join([2; arch], ",")
    println()
    @printf("[cold:%s %02d/%02d] n=%d arch=[%s] steps=%d activation=%s loss=%s\n", language, config_index, total_configs, n_samples, arch_label, n_steps, activation, loss)

    trials = Dict{String,Any}[]
    metadata = nothing
    language_version = ""
    language_platform = ""
    for trial_index in 1:n_trials
        run_data = run_single_trial(
            language;
            n_samples=n_samples,
            arch=arch,
            n_steps=n_steps,
            lr=lr,
            activation=activation,
            loss=loss,
            python=python,
        )
        metadata === nothing && (metadata = run_data.result)
        language_version = run_data.version
        language_platform = run_data.platform
        push!(trials, run_data.trial)
        @printf("    trial %d/%d  %.2fs process  %.2fs loop\n", trial_index, n_trials, run_data.trial["process_elapsed_seconds"], run_data.trial["benchmark_elapsed_seconds"])
    end

    config = Dict{String,Any}()
    for key in ("n_samples", "arch", "n_steps", "lr", "activation", "loss")
        config[key] = metadata[key]
    end
    config["trials"] = trials
    return (
        language=language,
        version=language_version,
        platform=language_platform,
        config=config,
    )
end

function output_template(
    language::AbstractString,
    version::String,
    platform::String,
    configs::Vector{Dict{String,Any}},
    n_trials::Int,
)
    return Dict(
        "language" => language,
        "version" => version,
        "platform" => platform,
        "mode" => "cold-process-step-sweep",
        "measurement" => "fresh_process",
        "n_trials" => n_trials,
        "results" => Any[],
        "step_sweep_results" => configs,
        "status" => "running",
    )
end

function cold_process_sweep(args=ARGS)
    options = parse_options(args)
    step_counts = option_int_list(options, "step-counts", STEP_COUNTS)
    n_trials = option_int(options, "trials", N_TRIALS)
    lr = option_float(options, "lr", LR)
    activation = option_string(options, "activation", "relu")
    loss = option_string(options, "loss", "hinge")
    out_dir = abspath(option_string(options, "results-dir", DEFAULT_RESULTS_DIR))
    python = abspath(option_string(options, "python", DEFAULT_PYTHON))
    languages = option_string_list(options, "languages", collect(LANGUAGES))

    isfile(python) || error("Python interpreter not found: $python")

    configs_to_run =
        if haskey(options, "n-samples") || haskey(options, "arch")
            [(;
                n_samples=option_int(options, "n-samples", STEP_SWEEP_N_SAMPLES),
                arch=option_int_list(options, "arch", STEP_SWEEP_ARCH),
            )]
        else
            DEFAULT_COLD_CONFIGS
        end

    for language in languages
        language_str = String(language)
        language_str ∈ LANGUAGES || error("unsupported language: $language_str")
        configs = Dict{String,Any}[]
        version = ""
        platform = ""
        out_path = joinpath(out_dir, "bench_$(language_str)_cold.json")
        output = output_template(language_str, version, platform, configs, n_trials)
        write_json_output(output, out_path)
        total_configs = length(configs_to_run) * length(step_counts)
        config_index = 0
        for config in configs_to_run
            for n_steps in step_counts
                config_index += 1
                result = run_config(
                    language_str;
                    n_samples=config.n_samples,
                    arch=config.arch,
                    n_steps=n_steps,
                    n_trials=n_trials,
                    lr=lr,
                    activation=activation,
                    loss=loss,
                    python=python,
                    config_index=config_index,
                    total_configs=total_configs,
                )
                version = result.version
                platform = result.platform
                push!(configs, result.config)
                output["version"] = version
                output["platform"] = platform
                write_json_output(output, out_path)
            end
        end
        output["status"] = "complete"
        write_json_output(output, out_path)
        println("\nSaved $out_path")
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    cold_process_sweep()
end
