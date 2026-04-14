@testset "Engine" begin

    # === Karpathy's tests (from karpathy/micrograd test_engine.py) ===

    @testset "sanity check" begin
        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = relu(z) + z * x
        h = relu(z * z)
        y = h + q + q * x
        backward!(y)

        @test y.data == -20.0
        @test x.grad == 46.0
    end

    @testset "more ops" begin
        a = Value(-4.0)
        b = Value(2.0)
        c = a + b
        d = a * b + b^3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + relu(b + a)
        d += 3 * d + relu(b - a)
        e = c - d
        f = e^2
        g = f / 2.0
        g += 10.0 / f
        backward!(g)

        tol = 1e-6
        @test abs(g.data - 24.70408163265306) < tol
        @test abs(a.grad - 138.83381924198252) < tol
        @test abs(b.grad - 645.5772594752186) < tol
    end

    # === Forward pass ===

    @testset "value creation" begin
        v = Value(3.0)
        @test v.data == 3.0
        @test v.grad == 0.0
        @test v._prev == ()
    end

    @testset "addition forward" begin
        @test Value(5.0) + Value(2.0) == 7
        @test Value(5.0) + 2.0 == 7
        @test 5 + Value(-12.0) == -7
    end

    @testset "negation forward" begin
        @test -Value(12.0) == -12.0
    end

    @testset "subtraction forward" begin
        @test Value(33.0) - Value(-10.0) == 43
        @test Value(33.0) - Value(10.0) == 23
        @test 33.0 - Value(-10.0) == 43
        @test Value(10.0) - 2 == 8
    end

    @testset "multiplication forward" begin
        @test Value(-10.0) * Value(10.0) == -100
        @test 33 * Value(10.0) == 330
        @test Value(-10.0) * 2 == -20
    end

    @testset "power forward" begin
        @test Value(2.0)^2 == 4.0
        @test Value(9.0)^2.0 == 81
    end

    @testset "inverse forward" begin
        @test inv(Value(2.0)) == 1 / 2
        @test inv(Value(5.0)) == 1.0 / 5.0
        @test inv(Value(4.0)) == 0.25
    end

    @testset "division forward" begin
        @test Value(33.0) / Value(-3.0) == -11.0
        @test Value(-33.0) / Value(-3.0) == 11.0
        @test Value(10.0) / Value(2.0) == 5
        @test Value(33.0) / -3 == -11.0
        @test -33 / Value(-3.0) == 11.0
        @test Value(10.0) / 2.0 == 5
    end

    @testset "tanh forward" begin
        @test tanh(Value(0.0)).data == 0.0
        @test abs(tanh(Value(1.0)).data - tanh(1.0)) < 1e-10
        @test abs(tanh(Value(-2.0)).data - tanh(-2.0)) < 1e-10
    end

    @testset "relu forward" begin
        @test relu(Value(5.0)).data == 5.0
        @test relu(Value(-3.0)).data == 0.0
        @test relu(Value(0.0)).data == 0.0
    end

    @testset "exp forward" begin
        @test exp(Value(0.0)).data == 1.0
        @test isapprox(exp(Value(2.0)).data, exp(2.0); atol=1e-10)
    end

    @testset "log forward" begin
        @test log(Value(1.0)).data == 0.0
        @test isapprox(log(Value(2.0)).data, log(2.0); atol=1e-10)
    end

    # === Gradients ===

    @testset "addition gradient" begin
        a = Value(3.0); b = Value(5.0)
        out = a + b
        out.grad = 1.0
        out._backward()
        @test a.grad == 1.0
        @test b.grad == 1.0
    end

    @testset "addition gradient (Value + Number)" begin
        a = Value(3.0)
        out = a + 7.0
        out.grad = 1.0
        out._backward()
        @test a.grad == 1.0
    end

    @testset "multiplication gradient" begin
        a = Value(3.0); b = Value(-4.0)
        out = a * b
        out.grad = 1.0
        out._backward()
        @test a.grad == -4.0   # d(a*b)/da = b
        @test b.grad == 3.0    # d(a*b)/db = a
    end

    @testset "multiplication gradient (Value * Number)" begin
        a = Value(3.0)
        out = a * 5.0
        out.grad = 1.0
        out._backward()
        @test a.grad == 5.0
    end

    @testset "tanh gradient" begin
        a = Value(0.5)
        out = tanh(a)
        out.grad = 1.0
        out._backward()
        expected = 1.0 - tanh(0.5)^2
        @test abs(a.grad - expected) < 1e-10
    end

    @testset "power does not accept Value exponent" begin
        @test_throws MethodError Value(3.0)^Value(2.0)
    end

    @testset "power gradient" begin
        # d(x^3)/dx = 3x^2
        a = Value(2.0)
        out = a^3
        out.grad = 1.0
        out._backward()
        @test a.grad == 12.0  # 3 * 2^2

        # d(x^2)/dx = 2x, with non-unit upstream grad
        a = Value(5.0)
        out = a^2
        out.grad = 3.0
        out._backward()
        @test a.grad == 30.0  # 2 * 5 * 3
    end

    @testset "inverse gradient" begin
        # d(1/x)/dx = -1/x^2
        a = Value(2.0)
        out = inv(a)
        out.grad = 1.0
        out._backward()
        @test a.grad == -0.25  # -1 / 2^2

        a = Value(4.0)
        out = inv(a)
        out.grad = 2.0
        out._backward()
        @test abs(a.grad - (-2.0 / 16.0)) < 1e-10  # -1/16 * 2
    end

    @testset "negation gradient" begin
        a = Value(7.0)
        out = -a
        out.grad = 1.0
        out._backward()
        @test a.grad == -1.0
    end

    @testset "relu gradient" begin
        # positive input: gradient passes through
        a = Value(3.0)
        out = relu(a)
        out.grad = 1.0
        out._backward()
        @test a.grad == 1.0

        # negative input: gradient blocked
        a = Value(-2.0)
        out = relu(a)
        out.grad = 1.0
        out._backward()
        @test a.grad == 0.0

        # zero input: gradient blocked
        a = Value(0.0)
        out = relu(a)
        out.grad = 1.0
        out._backward()
        @test a.grad == 0.0
    end

    @testset "exp gradient" begin
        a = Value(2.0)
        out = exp(a)
        out.grad = 3.0
        out._backward()
        @test isapprox(a.grad, 3.0 * exp(2.0); atol=1e-10)
    end

    @testset "log gradient" begin
        a = Value(4.0)
        out = log(a)
        out.grad = 2.0
        out._backward()
        @test a.grad == 0.5
    end

    # === Topological sort ===

    @testset "topo single leaf" begin
        a = Value(3.0)
        order = topo(a)
        @test length(order) == 1
        @test order[1] === a
    end

    @testset "topo linear chain" begin
        # a + b = c, c * d = e → order should end with e
        a = Value(1.0); b = Value(2.0)
        c = a + b
        d = Value(3.0)
        e = c * d
        order = topo(e)
        @test length(order) == 5
        @test order[end] === e
        # all leaves come before their consumers
        @test findfirst(x -> x === a, order) < findfirst(x -> x === c, order)
        @test findfirst(x -> x === b, order) < findfirst(x -> x === c, order)
        @test findfirst(x -> x === c, order) < findfirst(x -> x === e, order)
        @test findfirst(x -> x === d, order) < findfirst(x -> x === e, order)
    end

    @testset "topo shared node" begin
        # a used twice: a + a — should appear only once in order
        a = Value(5.0)
        out = a + a
        order = topo(out)
        @test count(==(a), order) == 1
        @test order[end] === out
    end

    @testset "topo deep chain" begin
        x = Value(1.0)
        out = x
        for _ in 1:2000
            out = out + 1.0
        end
        order = topo(out)
        @test length(order) == 4001
        @test count(==(x), order) == 1
        @test findfirst(==(x), order) < findfirst(==(out), order)
        @test order[end] === out
    end

    # === backward! ===

    @testset "backward! simple chain" begin
        # y = (a + b) * c — verify end-to-end gradients
        a = Value(2.0); b = Value(3.0); c = Value(4.0)
        y = (a + b) * c
        backward!(y)
        @test y.data == 20.0
        @test c.grad == 5.0   # d(y)/dc = a + b = 5
        @test a.grad == 4.0   # d(y)/da = c = 4
        @test b.grad == 4.0   # d(y)/db = c = 4
    end

    @testset "backward! reused node" begin
        # a used twice: y = a * a → dy/da = 2a
        a = Value(3.0)
        y = a * a
        backward!(y)
        @test y.data == 9.0
        @test a.grad == 6.0  # 2 * 3
    end

    @testset "backward! deep chain" begin
        x = Value(1.0)
        out = x
        for _ in 1:2000
            out = out + 1.0
        end
        backward!(out)
        @test x.grad == 1.0
    end

end
