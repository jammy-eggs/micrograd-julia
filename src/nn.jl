using LinearAlgebra
LinearAlgebra.dot(a::Value, b::Value) = a * b

# ─── Neuron ──────────────────────────────────────────────────────────

struct Neuron
    weights::Vector{Value}
    bias::Value
    nonlin::Bool
    activation::Symbol

    function Neuron(nin; nonlin=true, activation::Symbol=:relu)
        if activation ∉ (:relu, :tanh)
            throw(ArgumentError("unsupported activation: $activation"))
        end
        return new(
        [Value(2 * rand() - 1) for _ in 1:nin],
        Value(0.0),
        nonlin,
        activation,
    )
    end
end

function (n::Neuron)(x)
    act = n.weights ⋅ x + n.bias
    if !n.nonlin
        return act
    elseif n.activation == :relu
        return relu(act)
    else
        return tanh(act)
    end
end

params(n::Neuron) = [n.weights..., n.bias]

# ─── Layer ───────────────────────────────────────────────────────────

struct Layer
    neurons::Vector{Neuron}

    Layer(nin, nout; kwargs...) = new([Neuron(nin; kwargs...) for _ in 1:nout])
end

function (l::Layer)(x)
    outs = [n(x) for n in l.neurons]
    return length(outs) == 1 ? outs[1] : outs
end

params(l::Layer) = mapreduce(params, vcat, l.neurons; init=Value[])

# ─── MLP ─────────────────────────────────────────────────────────────

struct MLP
    layers::Vector{Layer}

    function MLP(nin, nouts; activation::Symbol=:relu)
        sz = [nin; nouts]
        n = length(nouts)
        new([Layer(sz[i], sz[i+1]; nonlin=(i != n), activation=activation) for i in 1:n])
    end
end

function (m::MLP)(x)
    for layer in m.layers
        x = layer(x)
    end
    return x
end

params(m::MLP) = mapreduce(params, vcat, m.layers; init=Value[])

function zero_grad!(model)
    for param in params(model)
        param.grad = 0.0
    end
end
