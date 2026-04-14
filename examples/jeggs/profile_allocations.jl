using JSON3
using Printf
using Profile
using Statistics
using Micrograd

include(joinpath(@__DIR__, "..", "script_utils.jl"))
include("utils.jl")
using .MicrogradExampleScriptUtils
using .MicrogradExampleUtils

const EXAMPLES_DIR = normpath(joinpath(@__DIR__, ".."))
const REPO_ROOT = normpath(joinpath(EXAMPLES_DIR, ".."))
const DEFAULT_RESULTS = joinpath(EXAMPLES_DIR, "results", "bench_julia.json")
const DEFAULT_OUT_DIR = joinpath(EXAMPLES_DIR, "results", "allocation_profiles")
const PHASES = (:forward, :backward, :update)
const DEFAULT_SAMPLE_RATE = 0.01
const DEFAULT_TOP = 20
const DEFAULT_N_STEPS = 100
const DEFAULT_N_SAMPLES = 200
const DEFAULT_ARCH = [16, 16, 1]
const DEFAULT_LR = 1.0
const DEFAULT_REPEATS = 1

function phase_symbol(value)
    phase = Symbol(lowercase(replace(String(value), '-' => '_')))
    phase ∈ (:auto, PHASES...) || throw(ArgumentError("unsupported phase: $value"))
    return phase
end

option_phase(options, key, default) = haskey(options, key) ? phase_symbol(options[key]) : phase_symbol(default)

function load_results(path::AbstractString)
    isfile(path) || return nothing
    return JSON3.read(read(path, String), Dict{String,Any})
end

function benchmark_status(results_data)
    results_data === nothing && return nothing
    return haskey(results_data, "status") ? String(results_data["status"]) : nothing
end

function median_total_bytes(result)
    values = Float64[
        Float64(trial["memory"]["bytes_per_step"])
        for trial in result["trials"]
    ]
    return median(values)
end

function median_phase_bytes(result, phase::Symbol)
    phase_key = String(phase)
    values = Float64[
        Float64(trial["memory"]["phase_bytes_per_step"][phase_key])
        for trial in result["trials"]
    ]
    return median(values)
end

function config_matches(result, n_samples, arch, n_steps, activation, loss)
    Int(result["n_samples"]) == n_samples || return false
    Int.(result["arch"]) == [2; arch] || return false
    Int(result["n_steps"]) == n_steps || return false
    String(get(result, "activation", "relu")) == String(activation) || return false
    String(get(result, "loss", "hinge")) == String(loss) || return false
    return true
end

function select_result(results_data, n_samples, arch, n_steps, activation, loss)
    results_data === nothing && return nothing
    candidates = get(results_data, "results", Any[])
    isempty(candidates) && return nothing

    for result in candidates
        config_matches(result, n_samples, arch, n_steps, activation, loss) && return result
    end

    return nothing
end

function heaviest_result(results_data)
    results_data === nothing && return nothing
    candidates = get(results_data, "results", Any[])
    isempty(candidates) && return nothing
    return argmax(median_total_bytes, candidates)
end

argmax(f, items) = begin
    best_item = first(items)
    best_value = f(best_item)
    for item in Iterators.drop(items, 1)
        value = f(item)
        if value > best_value
            best_value = value
            best_item = item
        end
    end
    best_item
end

function auto_phase(result)
    return argmax(phase -> median_phase_bytes(result, phase), collect(PHASES))
end

function result_config(result)
    return (
        n_samples=Int(result["n_samples"]),
        arch=Int.(result["arch"])[2:end],
        n_steps=Int(result["n_steps"]),
        activation=activation_symbol(get(result, "activation", "relu")),
        loss=loss_symbol(get(result, "loss", "hinge")),
    )
end

function build_context(n_samples, arch, n_steps, activation, loss, lr)
    X, y = load_dataset(n_samples)
    model = MLP(2, arch; activation=activation)
    load_weights!(model, weights_path(arch))
    return (
        X=X,
        y=y,
        model=model,
        n_steps=n_steps,
        lr=lr,
        loss_fn=loss_function(loss),
    )
end

function warmup!(ctx)
    total_loss, _ = ctx.loss_fn(ctx.model, ctx.X, ctx.y)
    zero_grad!(ctx.model)
    backward!(total_loss)
    update_params!(ctx.model, 0, max(ctx.n_steps, 1), ctx.lr)
    return nothing
end

function profile_phase!(ctx, phase::Symbol; sample_rate::Float64, repeats::Int=DEFAULT_REPEATS)
    Profile.Allocs.clear()
    GC.gc()
    repeats >= 1 || throw(ArgumentError("repeats must be at least 1"))

    if phase == :forward
        Profile.Allocs.@profile sample_rate=sample_rate begin
            for _ in 1:repeats
                ctx.loss_fn(ctx.model, ctx.X, ctx.y)
            end
        end
    elseif phase == :backward
        Profile.Allocs.@profile sample_rate=sample_rate begin
            for _ in 1:repeats
                total_loss, _ = ctx.loss_fn(ctx.model, ctx.X, ctx.y)
                zero_grad!(ctx.model)
                backward!(total_loss)
            end
        end
    else
        Profile.Allocs.@profile sample_rate=sample_rate begin
            for _ in 1:repeats
                total_loss, _ = ctx.loss_fn(ctx.model, ctx.X, ctx.y)
                zero_grad!(ctx.model)
                backward!(total_loss)
                update_params!(ctx.model, 0, max(ctx.n_steps, 1), ctx.lr)
            end
        end
    end

    return Profile.Allocs.fetch()
end

function local_frame(stacktrace)
    for frame in stacktrace
        file = String(frame.file)
        if startswith(file, REPO_ROOT) && !endswith(file, "profile_allocations.jl")
            return frame
        end
    end
    return nothing
end

function site_key(frame)
    if frame === nothing
        return ("<julia internals>", "<julia>", 0)
    end
    file = relpath(String(frame.file), REPO_ROOT)
    return (String(frame.func), file, Int(frame.line))
end

function top_types(type_counts::Dict{String,Int}; top_n::Int=3)
    ordered = sort(collect(type_counts); by=last, rev=true)
    return [
        Dict("type" => type_name, "samples" => count)
        for (type_name, count) in Iterators.take(ordered, top_n)
    ]
end

function summarize_allocations(allocs; top_n::Int=DEFAULT_TOP)
    total_sampled_bytes = sum(alloc.size for alloc in allocs)
    sites = Dict{Tuple{String,String,Int},Dict{String,Any}}()

    for alloc in allocs
        frame = local_frame(alloc.stacktrace)
        func, file, line = site_key(frame)
        key = (func, file, line)
        site = get!(sites, key) do
            Dict(
                "function" => func,
                "file" => file,
                "line" => line,
                "sampled_bytes" => 0,
                "samples" => 0,
                "types" => Dict{String,Int}(),
            )
        end
        site["sampled_bytes"] += alloc.size
        site["samples"] += 1
        type_name = string(alloc.type)
        types = site["types"]
        types[type_name] = get(types, type_name, 0) + 1
    end

    ordered = sort(collect(values(sites)); by=site -> site["sampled_bytes"], rev=true)
    top_sites = Dict{String,Any}[]
    for site in Iterators.take(ordered, top_n)
        push!(top_sites, Dict(
            "function" => site["function"],
            "file" => site["file"],
            "line" => site["line"],
            "sampled_bytes" => site["sampled_bytes"],
            "sampled_fraction" => total_sampled_bytes == 0 ? 0.0 : site["sampled_bytes"] / total_sampled_bytes,
            "samples" => site["samples"],
            "top_types" => top_types(site["types"]),
        ))
    end

    return Dict(
        "sampled_allocations" => length(allocs),
        "sampled_bytes" => total_sampled_bytes,
        "top_sites" => top_sites,
    )
end

function default_out_path(out_dir, phase, n_samples, arch, activation, loss)
    arch_label = join([2; arch], "-")
    filename = "julia_allocs_$(phase)_n$(n_samples)_arch$(arch_label)_$(activation)_$(loss).json"
    return joinpath(out_dir, filename)
end

function print_summary(report)
    println("Julia allocation profile")
    config = report["config"]
    selection = report["selection"]
    @printf(
        "selection: config=%s phase=%s source=%s\n",
        selection["config_selection"],
        selection["phase_selection"],
        something(selection["results_source"], "<none>"),
    )
    @printf(
        "config: n=%d arch=%s steps=%d activation=%s loss=%s phase=%s\n",
        config["n_samples"],
        join(config["arch"], ","),
        config["n_steps"],
        config["activation"],
        config["loss"],
        report["phase"],
    )
    @printf("profiling: sample_rate=%.4f repeats=%d\n", report["sample_rate"], report["repeats"])
    if haskey(report, "benchmark_bytes_per_step")
        @printf("benchmark total bytes/step: %.0f\n", report["benchmark_bytes_per_step"])
    end
    if haskey(report, "benchmark_phase_bytes_per_step")
        phase_bytes = report["benchmark_phase_bytes_per_step"]
        @printf(
            "benchmark bytes/step: forward=%.0f backward=%.0f update=%.0f\n",
            phase_bytes["forward"],
            phase_bytes["backward"],
            phase_bytes["update"],
        )
    end
    summary = report["summary"]
    @printf("sampled allocations: %d, sampled bytes: %d\n", summary["sampled_allocations"], summary["sampled_bytes"])
    for (index, site) in enumerate(summary["top_sites"])
        @printf(
            "%2d. %6.2f%% %10dB %6d samples  %s (%s:%d)\n",
            index,
            100 * site["sampled_fraction"],
            site["sampled_bytes"],
            site["samples"],
            site["function"],
            site["file"],
            site["line"],
        )
    end
end

function main(args=ARGS)
    options = parse_options(args)
    results_path = abspath(option_string(options, "results", DEFAULT_RESULTS))
    results_data = load_results(results_path)
    status = benchmark_status(results_data)

    explicit_config = any(haskey(options, key) for key in ("n-samples", "arch", "n-steps", "activation", "loss"))
    if !explicit_config && status == "running"
        error("benchmark results at $results_path are still marked as running; wait for completion or pass an explicit config")
    end
    selected_result = explicit_config ? nothing : heaviest_result(results_data)
    selected_config = selected_result === nothing ? nothing : result_config(selected_result)

    n_samples = selected_config === nothing ? option_int(options, "n-samples", DEFAULT_N_SAMPLES) : selected_config.n_samples
    arch = selected_config === nothing ? option_int_list(options, "arch", DEFAULT_ARCH) : selected_config.arch
    n_steps = selected_config === nothing ? option_int(options, "n-steps", DEFAULT_N_STEPS) : selected_config.n_steps
    activation = selected_config === nothing ? activation_symbol(option_string(options, "activation", "relu")) : selected_config.activation
    loss = selected_config === nothing ? loss_symbol(option_string(options, "loss", "hinge")) : selected_config.loss

    if explicit_config
        selected_result = select_result(results_data, n_samples, arch, n_steps, activation, loss)
    end

    requested_phase = option_phase(options, "phase", :auto)
    phase = requested_phase == :auto ? (selected_result === nothing ? :forward : auto_phase(selected_result)) : requested_phase
    sample_rate = option_float(options, "sample-rate", DEFAULT_SAMPLE_RATE)
    repeats = option_int(options, "repeats", DEFAULT_REPEATS)
    top_n = option_int(options, "top", DEFAULT_TOP)
    out_dir = abspath(option_string(options, "out-dir", DEFAULT_OUT_DIR))
    out_path = abspath(option_string(options, "out", default_out_path(out_dir, phase, n_samples, arch, activation, loss)))
    lr = option_float(options, "lr", DEFAULT_LR)

    warmup!(build_context(n_samples, arch, n_steps, activation, loss, lr))
    profile_ctx = build_context(n_samples, arch, n_steps, activation, loss, lr)
    alloc_results = profile_phase!(profile_ctx, phase; sample_rate=sample_rate, repeats=repeats)
    summary = summarize_allocations(alloc_results.allocs; top_n=top_n)

    report = Dict{String,Any}(
        "phase" => String(phase),
        "sample_rate" => sample_rate,
        "repeats" => repeats,
        "config" => Dict(
            "n_samples" => n_samples,
            "arch" => [2; arch],
            "n_steps" => n_steps,
            "activation" => String(activation),
            "loss" => String(loss),
            "lr" => lr,
        ),
        "selection" => Dict(
            "results_source" => isfile(results_path) ? results_path : nothing,
            "results_status" => status,
            "config_selection" => explicit_config ? "explicit" : (selected_result === nothing ? "default" : "heaviest_total_bytes_per_step"),
            "phase_selection" => requested_phase == :auto ? (selected_result === nothing ? "default_forward" : "auto_from_benchmark") : "explicit",
        ),
        "summary" => summary,
    )

    if selected_result !== nothing
        report["benchmark_bytes_per_step"] = median_total_bytes(selected_result)
        report["benchmark_phase_bytes_per_step"] = Dict(
            String(phase_name) => median_phase_bytes(selected_result, phase_name) for phase_name in PHASES
        )
    end

    write_json_output(report, out_path)

    print_summary(report)
    println("\nSaved $out_path")
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
