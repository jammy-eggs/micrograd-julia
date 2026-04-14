using Test
using Micrograd

@testset "Micrograd" begin
    include("engine.jl")
    include("nn.jl")
end
