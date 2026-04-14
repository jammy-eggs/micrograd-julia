@testset "Neural Network" begin

    # ─── Neuron ──────────────────────────────────────────────────────

    @testset "Neuron constructor" begin
        n = Neuron(3)
        @test length(n.weights) == 3
        @test n.bias.data == 0.0
        @test n.nonlin
        @test n.activation == :relu
        @test all(w -> -1 ≤ w.data ≤ 1, n.weights)
    end

    @testset "Neuron nonlin=false" begin
        n = Neuron(2; nonlin=false)
        @test !n.nonlin
    end

    @testset "Neuron tanh activation" begin
        n = Neuron(2; activation=:tanh)
        @test n.activation == :tanh
        out = n([Value(1.0), Value(1.0)])
        @test out isa Value
        @test -1.0 <= out.data <= 1.0
    end

    @testset "Neuron params" begin
        n = Neuron(4)
        p = params(n)
        @test length(p) == 5  # 4 weights + 1 bias
        @test p[end] === n.bias
    end

    @testset "Neuron forward" begin
        n = Neuron(3)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        out = n(x)
        @test out isa Value
        @test out.data >= 0.0
    end

    @testset "Neuron forward linear" begin
        n = Neuron(2; nonlin=false)
        x = [Value(1.0), Value(1.0)]
        out = n(x)
        @test out isa Value
        expected = sum(w.data for w in n.weights) + n.bias.data
        @test isapprox(out.data, expected; atol=1e-10)
    end

    # ─── Layer ───────────────────────────────────────────────────────

    @testset "Layer constructor" begin
        l = Layer(3, 4)
        @test length(l.neurons) == 4
        @test all(n -> length(n.weights) == 3, l.neurons)
    end

    @testset "Layer params" begin
        l = Layer(3, 4)
        p = params(l)
        @test length(p) == 4 * (3 + 1)  # 4 neurons × (3 weights + 1 bias)
    end

    @testset "Layer forward" begin
        l = Layer(3, 4)
        x = [Value(1.0), Value(2.0), Value(3.0)]
        out = l(x)
        @test length(out) == 4
        @test all(o -> o isa Value, out)
    end

    @testset "Layer single output" begin
        l = Layer(2, 1)
        x = [Value(1.0), Value(2.0)]
        out = l(x)
        @test out isa Value  # unwrapped, not a 1-element vector
    end

    # ─── MLP ─────────────────────────────────────────────────────────

    @testset "MLP constructor" begin
        m = MLP(3, [4, 4, 1])
        @test length(m.layers) == 3
        @test length(m.layers[1].neurons) == 4
        @test length(m.layers[2].neurons) == 4
        @test length(m.layers[3].neurons) == 1
    end

    @testset "MLP configurable activation" begin
        m = MLP(3, [4, 1]; activation=:tanh)
        @test m.layers[1].neurons[1].activation == :tanh
        @test m.layers[end].neurons[1].nonlin == false
    end

    @testset "MLP last layer linear" begin
        m = MLP(3, [4, 1])
        @test m.layers[end].neurons[1].nonlin == false
        @test m.layers[1].neurons[1].nonlin == true
    end

    @testset "MLP params" begin
        m = MLP(3, [4, 4, 1])
        p = params(m)
        expected = 4 * (3 + 1) + 4 * (4 + 1) + 1 * (4 + 1)  # 41
        @test length(p) == expected
    end

    @testset "MLP forward" begin
        m = MLP(3, [4, 4, 1])
        x = [Value(1.0), Value(2.0), Value(3.0)]
        out = m(x)
        @test out isa Value
    end

    @testset "MLP backward" begin
        m = MLP(2, [1])
        x = [Value(1.0), Value(2.0)]
        out = m(x)
        backward!(out)
        p = params(m)
        @test [v.grad for v in p] == [1.0, 2.0, 1.0]
    end

    # ─── zero_grad! ──────────────────────────────────────────────────

    @testset "zero_grad! clears grads" begin
        m = MLP(3, [4, 4, 1])
        x = [Value(1.0), Value(2.0), Value(3.0)]
        out = m(x)
        backward!(out)
        zero_grad!(m)
        @test all(v -> v.grad == 0.0, params(m))
    end

    @testset "zero_grad! on Neuron" begin
        n = Neuron(3)
        out = n([Value(1.0), Value(2.0), Value(3.0)])
        backward!(out)
        zero_grad!(n)
        @test all(v -> v.grad == 0.0, params(n))
    end

end
