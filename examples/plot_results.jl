using CairoMakie
using JSON3
using Statistics

const EXAMPLES_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(EXAMPLES_DIR, ".."))
const RESULTS_DIR = joinpath(EXAMPLES_DIR, "results")
const ALLOCATION_PROFILE = joinpath(RESULTS_DIR, "allocation_profiles", "julia_allocs_forward_n10000_arch2-32-32-1_relu_hinge.json")
const DOCS_FIGURES_DIR = joinpath(REPO_ROOT, "docs", "figures")
const DEFAULT_STEPS = 100
const DEFAULT_ACTIVATION = "relu"
const DEFAULT_LOSS = "hinge"
const LANGUAGE_COLORS = Dict(
    "julia" => colorant"#2B6CB0",
    "python" => colorant"#C05621",
)
const METRIC_COLORS = Dict(
    :overall => colorant"#355C7D",
    :forward => colorant"#C06C84",
    :backward => colorant"#6C9A8B",
)

function benchmark_files(results_dir)
    isdir(results_dir) || error("No benchmark results directory found: $results_dir")
    return [
        joinpath(results_dir, name)
        for name in sort(readdir(results_dir))
        if startswith(name, "bench_") && endswith(name, ".json")
    ]
end

function load_outputs(results_dir=RESULTS_DIR)
    outputs = Dict{String,Any}()
    for path in benchmark_files(results_dir)
        data = JSON3.read(read(path, String), Dict{String,Any})
        outputs[String(data["language"])] = data
    end
    isempty(outputs) && error("No benchmark JSON files found in $results_dir")
    return outputs
end

function matches_story_surface(result)
    return Int(result["n_steps"]) == DEFAULT_STEPS &&
        String(get(result, "activation", DEFAULT_ACTIVATION)) == DEFAULT_ACTIVATION &&
        String(get(result, "loss", DEFAULT_LOSS)) == DEFAULT_LOSS
end

function throughput_results(output; preserve_order=false)
    results = [result for result in get(output, "results", Any[]) if matches_story_surface(result)]
    if preserve_order
        return results
    end
    return sort(results; by=result -> (length(result["arch"]), Tuple(Int.(result["arch"])), Int(result["n_samples"])))
end

function config_key(result)
    return (
        Int(result["n_samples"]),
        Tuple(Int.(result["arch"])),
        Int(result["n_steps"]),
        String(get(result, "activation", DEFAULT_ACTIVATION)),
        String(get(result, "loss", DEFAULT_LOSS)),
    )
end

function result_index(output)
    indexed = Dict{Any,Any}()
    for result in throughput_results(output)
        indexed[config_key(result)] = result
    end
    return indexed
end

median_elapsed(result) = median(Float64[trial["elapsed_seconds"] for trial in result["trials"]])
median_phase(result, phase) = median(Float64[trial["phase_seconds"][phase] for trial in result["trials"]])

function cumulative_trial_hours(results)
    hours = Float64[]
    running = 0.0
    for result in results
        running += sum(Float64[trial["elapsed_seconds"] for trial in result["trials"]]) / 3600
        push!(hours, running)
    end
    return hours
end

function save_png(fig, basename)
    mkpath(DOCS_FIGURES_DIR)
    out = joinpath(DOCS_FIGURES_DIR, "$basename.png")
    save(out, fig, px_per_unit=2)
    println("saved $out")
end

function shared_story_tables(outputs)
    jl_results = throughput_results(outputs["julia"]; preserve_order=true)
    py_results = throughput_results(outputs["python"]; preserve_order=true)
    jlidx = result_index(outputs["julia"])
    pyidx = result_index(outputs["python"])
    shared = sort(collect(intersect(Set(keys(jlidx)), Set(keys(pyidx)))); by=key -> (length(key[2]), key[2], key[1]))
    return jl_results, py_results, jlidx, pyidx, shared
end

function shared_architectures(shared)
    archs = unique([key[2] for key in shared])
    return sort(archs; by=arch -> (length(arch), arch))
end

function base_figure(; size=(1200, 760), fontsize=22)
    return Figure(size=size, fontsize=fontsize, backgroundcolor=:white)
end

function styled_axis(figpos; title, xlabel, ylabel, xgridvisible=false, ygridvisible=true, kwargs...)
    Axis(
        figpos;
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        titlealign=:left,
        titlesize=30,
        xlabelsize=20,
        ylabelsize=20,
        xticklabelsize=16,
        yticklabelsize=16,
        xgridvisible=xgridvisible,
        ygridvisible=ygridvisible,
        xminorgridvisible=false,
        yminorgridvisible=false,
        xgridcolor=:gray92,
        ygridcolor=:gray90,
        leftspinecolor=:gray35,
        bottomspinecolor=:gray35,
        topspinevisible=false,
        rightspinevisible=false,
        kwargs...,
    )
end

function plot_sweep_completion(outputs)
    jl_results, py_results, _, _, _ = shared_story_tables(outputs)
    jl_hours = cumulative_trial_hours(jl_results)
    py_hours = cumulative_trial_hours(py_results)

    next_trial_hours = haskey(outputs["python"], "truncation") ?
        Float64(outputs["python"]["truncation"]["live_log_only_trial"]["elapsed_seconds"]) / 3600 : 0.0
    xmax = max(jl_hours[end], py_hours[end] + 5.5)

    fig = base_figure(size=(1100, 900))
    ax = styled_axis(
        fig[1, 1];
        title="Julia finished the sweep; Python did not",
        xlabel="Cumulative measured trial time (hours)",
        ylabel="Completed throughput configs (out of 24)",
        xgridvisible=false,
        ygridvisible=true,
        xticks=0:5:35,
        yticks=0:4:24,
    )

    stairs!(ax, [0.0; jl_hours], [0; collect(1:length(jl_results))]; step=:post, color=LANGUAGE_COLORS["julia"], linewidth=4)
    stairs!(ax, [0.0; py_hours], [0; collect(1:length(py_results))]; step=:post, color=LANGUAGE_COLORS["python"], linewidth=4)

    scatter!(ax, [jl_hours[end]], [length(jl_results)]; color=LANGUAGE_COLORS["julia"], markersize=18)
    scatter!(ax, [py_hours[end]], [length(py_results)]; color=LANGUAGE_COLORS["python"], markersize=18)

    text!(
        ax,
        jl_hours[end] + 0.35,
        length(jl_results) - 0.15;
        text="Julia\n24/24 · 3.59h",
        align=(:left, :top),
        color=LANGUAGE_COLORS["julia"],
        fontsize=18,
    )
    text!(
        ax,
        py_hours[end] + 0.35,
        length(py_results) - 0.1;
        text="Python\n16/24 · 27.52h\nnext trial alone: 5.08h",
        align=(:left, :top),
        color=LANGUAGE_COLORS["python"],
        fontsize=18,
    )

    xlims!(ax, 0, xmax)
    ylims!(ax, 0, 24.8)
    save_png(fig, "sweep_completion")
end

function architecture_summary(outputs)
    _, _, jlidx, pyidx, shared = shared_story_tables(outputs)
    archs = shared_architectures(shared)
    labels = String[]
    totals = Float64[]
    forwards = Float64[]
    backwards = Float64[]

    for arch in archs
        keys = [key for key in shared if key[2] == arch]
        overall = Float64[]
        forward = Float64[]
        backward = Float64[]
        for key in keys
            jr = jlidx[key]
            pr = pyidx[key]
            push!(overall, median_elapsed(pr) / median_elapsed(jr))
            push!(forward, median_phase(pr, "forward") / median_phase(jr, "forward"))
            push!(backward, median_phase(pr, "backward") / median_phase(jr, "backward"))
        end
        push!(labels, join(arch, '-'))
        push!(totals, median(overall))
        push!(forwards, median(forward))
        push!(backwards, median(backward))
    end
    return labels, totals, forwards, backwards
end

function plot_slowdown_by_architecture(outputs)
    labels, totals, forwards, backwards = architecture_summary(outputs)
    centers = collect(1:length(labels))
    offsets = [-0.18, 0.18]
    width = 0.32

    fig = base_figure(size=(1360, 760))
    Label(
        fig[0, 1:2],
        "Python's penalty grows with graph size",
        tellwidth=false,
        halign=:left,
        fontsize=32,
        font=:bold,
        padding=(0, 0, 8, 0),
    )
    Label(
        fig[1, 1:2],
        "Bars show how many times longer Python takes than Julia on matched configs",
        tellwidth=false,
        halign=:left,
        fontsize=17,
        color=:gray35,
        padding=(0, 0, 10, 0),
    )

    ax_total = styled_axis(
        fig[2, 1];
        title="Whole step median",
        xlabel="Network architecture",
        ylabel="Times longer than Julia (log scale)",
        xgridvisible=false,
        ygridvisible=true,
        yscale=log10,
        xticks=(centers, labels),
        yticks=([1, 3, 10, 30, 100, 300], ["1x", "3x", "10x", "30x", "100x", "300x"]),
    )

    ax_phase = styled_axis(
        fig[2, 2];
        title="Phase medians",
        xlabel="Network architecture",
        ylabel="",
        xgridvisible=false,
        ygridvisible=true,
        yscale=log10,
        xticks=(centers, labels),
        yticks=([1, 3, 10, 30, 100, 300], ["1x", "3x", "10x", "30x", "100x", "300x"]),
    )

    barplot!(ax_total, centers, totals; width=0.5, fillto=1.0, color=METRIC_COLORS[:overall], strokecolor=:white, strokewidth=1.5)
    for (x, value) in zip(centers, totals)
        text!(ax_total, x, value * 1.08; text="$(Int(round(value)))x", align=(:center, :bottom), color=:gray20, fontsize=15)
    end

    phase_series = [
        ("Forward pass", forwards, METRIC_COLORS[:forward]),
        ("Backward pass", backwards, METRIC_COLORS[:backward]),
    ]
    for (offset, (label, values, color)) in zip(offsets, phase_series)
        xs = centers .+ offset
        barplot!(ax_phase, xs, values; width=width, fillto=1.0, color=color, strokecolor=:white, strokewidth=1.5, label=label)
        for (x, value) in zip(xs, values)
            text!(ax_phase, x, value * 1.08; text="$(Int(round(value)))x", align=(:center, :bottom), color=:gray20, fontsize=15)
        end
    end

    Legend(fig[3, 2], ax_phase; orientation=:horizontal, framevisible=false, tellwidth=false, halign=:left, labelsize=15)
    ylims!(ax_total, 1, 500)
    ylims!(ax_phase, 1, 500)
    colgap!(fig.layout, 28)
    rowgap!(fig.layout, 10)
    save_png(fig, "slowdown_by_architecture")
end

function pretty_function_name(name::AbstractString)
    if name == "Value"
        return "Value constructor"
    elseif name == "+"
        return "+ primitive"
    elseif name == "*"
        return "* primitive"
    elseif name == "relu"
        return "relu"
    elseif name == "-"
        return "neg"
    elseif name == "MLP"
        return "MLP internals"
    elseif name == "#scores_and_reg#3"
        return "scores_and_reg"
    else
        return name
    end
end

function plot_julia_allocation_hotspots()
    isfile(ALLOCATION_PROFILE) || return @warn "Skipping allocation hotspots: profile not found at $ALLOCATION_PROFILE"
    data = JSON3.read(read(ALLOCATION_PROFILE, String), Dict{String,Any})
    totals = Dict{String,Float64}()
    for site in data["summary"]["top_sites"]
        fn = String(site["function"])
        totals[fn] = get(totals, fn, 0.0) + Float64(site["sampled_fraction"])
    end
    ranked = sort(collect(totals); by=last, rev=true)
    keep = ranked[1:min(5, length(ranked))]
    labels = [pretty_function_name(name) for (name, _) in keep]
    values = [100 * frac for (_, frac) in keep]

    fig = base_figure(size=(980, 620), fontsize=20)
    ax = styled_axis(
        fig[1, 1];
        title="Julia allocation hotspots on the heaviest completed config",
        xlabel="Function / allocation site",
        ylabel="Share of sampled allocation bytes (%)",
        xgridvisible=false,
        ygridvisible=true,
        xticks=(1:length(labels), labels),
    )
    barplot!(ax, 1:length(labels), values; color=LANGUAGE_COLORS["julia"])
    for (i, value) in enumerate(values)
        text!(ax, i, value + 1.0; text="$(round(value; digits=1))%", align=(:center, :bottom), color=:black, fontsize=15)
    end
    ylims!(ax, 0, maximum(values) + 8)
    save_png(fig, "julia_allocation_hotspots")
end

function main(args=ARGS)
    set_theme!(theme_minimal())
    outputs = load_outputs(isempty(args) ? RESULTS_DIR : abspath(args[1]))
    plot_sweep_completion(outputs)
    plot_slowdown_by_architecture(outputs)
    plot_julia_allocation_hotspots()
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
