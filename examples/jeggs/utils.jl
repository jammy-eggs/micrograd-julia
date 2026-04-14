module MicrogradExampleUtils

using JSON3
using Micrograd

export DATA_DIR,
    WEIGHTS_DIR,
    accuracy,
    activation_symbol,
    cross_entropy_loss,
    data_path,
    full_arch,
    hinge_loss,
    load_dataset,
    load_weights!,
    loss_function,
    loss_symbol,
    scores_and_reg,
    update_params!,
    weights_path

const EXAMPLES_DIR = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(EXAMPLES_DIR, "data")
const WEIGHTS_DIR = joinpath(EXAMPLES_DIR, "weights")
const ACTIVATIONS = (:relu, :tanh)
const LOSSES = (:hinge, :cross_entropy)

function activation_symbol(value)
    activation = value isa Symbol ? value : Symbol(String(value))
    activation ∈ ACTIVATIONS || throw(ArgumentError("unsupported activation: $value"))
    return activation
end

function loss_symbol(value)
    raw = lowercase(replace(String(value), '-' => '_'))
    loss = raw in ("crossentropy", "cross_entropy", "bce") ? :cross_entropy : Symbol(raw)
    loss ∈ LOSSES || throw(ArgumentError("unsupported loss: $value"))
    return loss
end

loss_function(loss::Symbol) = loss == :hinge ? hinge_loss : cross_entropy_loss

full_arch(arch::AbstractVector{<:Integer}) = [2, arch...]

function weights_path(arch::AbstractVector{<:Integer})
    name = join(string.(full_arch(arch)), "_") * ".json"
    return joinpath(WEIGHTS_DIR, name)
end

data_path(n_samples::Integer) = joinpath(DATA_DIR, "moons_$n_samples.json")

function load_dataset(n_samples::Integer)
    data = JSON3.read(read(data_path(n_samples), String))
    X = [Float64(data.X[i][j]) for i in eachindex(data.X), j in eachindex(data.X[1])]
    y = [Int(label) for label in data.y]
    return X, y
end

function load_weights!(model::MLP, path::AbstractString)
    data = JSON3.read(read(path, String))

    for (layer, layer_data) in zip(model.layers, data.layers)
        for (neuron, weights, bias) in zip(layer.neurons, layer_data.weights, layer_data.biases)
            for (w, val) in zip(neuron.weights, weights)
                w.data = val
            end
            neuron.bias.data = bias
        end
    end

    return model
end

function scores_and_reg(model::MLP, X::AbstractMatrix{<:Real}; alpha::Real=1e-4)
    inputs = [Value.(row) for row in eachrow(X)]
    scores = model.(inputs)
    reg_loss = alpha * sum(p * p for p in params(model))
    return scores, reg_loss
end

function accuracy(y, scores)
    correct = count((label > 0) == (score.data > 0) for (label, score) in zip(y, scores))
    return correct / length(y)
end

function hinge_loss(model::MLP, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer})
    scores, reg_loss = scores_and_reg(model, X)
    losses = [relu(1 - label * score) for (label, score) in zip(y, scores)]
    data_loss = sum(losses) / length(losses)
    return data_loss + reg_loss, accuracy(y, scores)
end

sigmoid(x::Value) = 1 / (1 + exp(-x))

function binary_cross_entropy(score::Value, label::Integer)
    target = (1 + label) / 2
    prob = sigmoid(score)
    return -(target * log(prob) + (1 - target) * log(1 - prob))
end

function cross_entropy_loss(model::MLP, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer})
    scores, reg_loss = scores_and_reg(model, X)
    losses = [binary_cross_entropy(score, label) for (score, label) in zip(scores, y)]
    data_loss = sum(losses) / length(losses)
    return data_loss + reg_loss, accuracy(y, scores)
end

function update_params!(model::MLP, step::Integer, n_steps::Integer, lr::Real)
    step_lr = lr - 0.9 * step / n_steps
    for p in params(model)
        p.data -= step_lr * p.grad
    end
end

end
