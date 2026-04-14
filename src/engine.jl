# ─── Value ────────────────────────────────────────────────────────────

mutable struct Value
    data::Float64
    grad::Float64
    _backward::Function
    _prev::Tuple{Vararg{Value}}

    Value(data::Float64) = new(data, 0.0, () -> nothing, ())
end

Value(x::Number) = Value(Float64(x))

# ─── Utils ──────────────────────────────────────────────────────────

Base.show(io::IO, x::Value) = print(io, "Value(data=", x.data, ",grad=", x.grad, ")")
Base.:(==)(a::Value, b::Number) = a.data == b
Base.zero(::Type{Value}) = Value(0.0)
Base.zero(::Value) = Value(0.0)
Base.one(::Type{Value}) = Value(1.0)
Base.one(::Value) = Value(1.0)
Base.isless(a::Value, b::Value) = isless(a.data, b.data)
Base.float(v::Value) = v.data

# ─── Promotion & conversion ──────────────────────────────────────────

Base.convert(::Type{Value}, x::Number) = Value(x)
Base.promote_rule(::Type{Value}, ::Type{<:Number}) = Value

Base.:(+)(a::Value, b::Number) = +(promote(a, b)...)
Base.:(+)(a::Number, b::Value) = +(promote(a, b)...)
Base.:(*)(a::Value, b::Number) = *(promote(a, b)...)
Base.:(*)(a::Number, b::Value) = *(promote(a, b)...)
Base.:(-)(a::Value, b::Number) = -(promote(a, b)...)
Base.:(-)(a::Number, b::Value) = -(promote(a, b)...)
Base.:(/)(a::Value, b::Number) = /(promote(a, b)...)
Base.:(/)(a::Number, b::Value) = /(promote(a, b)...)
function Base.:(^)(a::Value, b::Number)
    out = Value(a.data^b)
    function pow_backward()
        a.grad += (b * a.data^(b - 1)) * out.grad
    end
    out._backward = pow_backward
    out._prev = (a,)
    return out
end

# ─── Gradient primitives ─────────────────────────────────────────────

function Base.:(+)(a::Value, b::Value)
    out = Value(a.data + b.data)
    function add_backward()
        a.grad += out.grad
        b.grad += out.grad
    end
    out._backward = add_backward
    out._prev = (a, b)
    return out
end

function Base.:(*)(a::Value, b::Value)
    out = Value(a.data * b.data)
    function mul_backward()
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad
    end
    out._backward = mul_backward
    out._prev = (a, b)
    return out
end

function Base.inv(a::Value)
    out = Value(1.0 / a.data)
    function inv_backward()
        a.grad += (-1.0 * a.data^(-2)) * out.grad
    end
    out._backward = inv_backward
    out._prev = (a,)
    return out
end

function Base.:(-)(a::Value)
    out = Value(-a.data)
    function neg_backward()
        a.grad += -out.grad
    end
    out._backward = neg_backward
    out._prev = (a,)
    return out
end

function Base.exp(a::Value)
    out = Value(exp(a.data))
    function exp_backward()
        a.grad += out.data * out.grad
    end
    out._backward = exp_backward
    out._prev = (a,)
    return out
end

function Base.log(a::Value)
    out = Value(log(a.data))
    function log_backward()
        a.grad += out.grad / a.data
    end
    out._backward = log_backward
    out._prev = (a,)
    return out
end

# ─── Delegated ops ───────────────────────────────────────────────────

Base.:(-)(a::Value, b::Value) = a + (-b)

Base.:(/)(a::Value, b::Value) = a * (b^-1)

# ─── Activations ─────────────────────────────────────────────────────

function Base.:(tanh)(a::Value)
    out = Value(tanh(a.data))
    function tanh_backward()
        a.grad += (1.0 - out.data^2) * out.grad
    end
    out._backward = tanh_backward
    out._prev = (a,)
    return out
end

function relu(x::Value)
    out = Value(max(0.0, x.data))
    function relu_backward()
        x.grad += (out.data > 0 ? 1.0 : 0.0) * out.grad
    end
    out._backward = relu_backward
    out._prev = (x,)
    return out
end

# ─── Backpropagation ─────────────────────────────────────────────────

function topo(root::Value)
    visited = Set{Value}()
    order = Value[]
    stack = Tuple{Value,Bool}[(root, false)]

    while !isempty(stack)
        node, expanded = pop!(stack)
        if expanded
            push!(order, node)
            continue
        end
        if node ∈ visited
            continue
        end
        push!(visited, node)
        push!(stack, (node, true))
        for child in node._prev
            if child ∉ visited
                push!(stack, (child, false))
            end
        end
    end

    return order
end

function backward!(v::Value)
    v.grad = 1.0
    for node in reverse!(topo(v))
        node._backward()
    end
end
