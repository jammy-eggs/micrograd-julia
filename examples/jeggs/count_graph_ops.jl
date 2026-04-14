using JSON3
using Micrograd
using Statistics

include("utils.jl")
using .MicrogradExampleUtils

const DEFAULT_N_SAMPLES = 200
const DEFAULT_ARCH = [16, 16, 1]
const DEFAULT_N_STEPS = 100
const DEFAULT_ACTIVATION = :relu
const DEFAULT_LOSS = :hinge
const DEFAULT_LR = 1.0
const BENCHMARK_JSON = normpath(joinpath(@__DIR__, "..", "results", "bench_julia.json"))

function op_label(node::Value)
    name = String(nameof(typeof(node._backward)))
    if occursin("add_backward", name)
        return "+"
    elseif occursin("mul_backward", name)
        return "*"
    elseif occursin("pow_backward", name)
        return "pow"
    elseif occursin("inv_backward", name)
        return "inv"
    elseif occursin("neg_backward", name)
        return "neg"
    elseif occursin("exp_backward", name)
        return "exp"
    elseif occursin("log_backward", name)
        return "log"
    elseif occursin("tanh_backward", name)
        return "tanh"
    elseif occursin("relu_backward", name)
        return "ReLU"
    else
        return "leaf"
    end
end

function count_graph(root::Value)
    counts = Dict{String,Int}()
    for node in topo(root)
        label = op_label(node)
        counts[label] = get(counts, label, 0) + 1
    end
    return counts
end

function merge_counts!(dest::Dict{String,Int}, src::Dict{String,Int})
    for (key, value) in src
        dest[key] = get(dest, key, 0) + value
    end
    return dest
end

function benchmark_median_seconds(path::AbstractString)
    data = JSON3.read(read(path, String), Dict{String,Any})
    target_arch = full_arch(DEFAULT_ARCH)
    for result in data["results"]
        if Int(result["n_samples"]) == DEFAULT_N_SAMPLES &&
           Int(result["n_steps"]) == DEFAULT_N_STEPS &&
           Int.(result["arch"]) == target_arch &&
           String(get(result, "activation", String(DEFAULT_ACTIVATION))) == String(DEFAULT_ACTIVATION) &&
           String(get(result, "loss", String(DEFAULT_LOSS))) == String(DEFAULT_LOSS)
            return median(Float64[trial["elapsed_seconds"] for trial in result["trials"]])
        end
    end
    error("Could not find matching benchmark result in $path")
end

function main()
    X, y = load_dataset(DEFAULT_N_SAMPLES)
    model = MLP(2, DEFAULT_ARCH; activation=DEFAULT_ACTIVATION)
    load_weights!(model, weights_path(DEFAULT_ARCH))
    loss_fn = loss_function(DEFAULT_LOSS)

    total_counts = Dict{String,Int}()
    for step in 0:(DEFAULT_N_STEPS - 1)
        total_loss, _ = loss_fn(model, X, y)
        merge_counts!(total_counts, count_graph(total_loss))
        zero_grad!(model)
        backward!(total_loss)
        update_params!(model, step, DEFAULT_N_STEPS, DEFAULT_LR)
    end

    leaf_nodes = get(total_counts, "leaf", 0)
    total_nodes = sum(values(total_counts))
    op_nodes = total_nodes - leaf_nodes
    median_seconds = benchmark_median_seconds(BENCHMARK_JSON)

    summary = Dict(
        "language" => "julia",
        "config" => Dict(
            "n_samples" => DEFAULT_N_SAMPLES,
            "arch" => full_arch(DEFAULT_ARCH),
            "n_steps" => DEFAULT_N_STEPS,
            "activation" => String(DEFAULT_ACTIVATION),
            "loss" => String(DEFAULT_LOSS),
        ),
        "graph_node_counts" => Dict(total_counts),
        "total_graph_nodes" => total_nodes,
        "leaf_nodes" => leaf_nodes,
        "operation_nodes" => op_nodes,
        "benchmark_median_elapsed_seconds" => median_seconds,
        "effective_total_nodes_per_second" => total_nodes / median_seconds,
        "effective_operation_nodes_per_second" => op_nodes / median_seconds,
    )

    JSON3.pretty(stdout, summary)
    println()
end

main()
