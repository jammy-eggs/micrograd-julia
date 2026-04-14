using CairoMakie
using JSON3

const EXAMPLES_DIR = @__DIR__
const REPO_ROOT = normpath(joinpath(EXAMPLES_DIR, ".."))
const DEFAULT_DATASET = joinpath(EXAMPLES_DIR, "data", "moons_1000.json")
const DEFAULT_OUT = joinpath(REPO_ROOT, "docs", "figures", "moons_dataset.png")

function main()
    data = JSON3.read(read(DEFAULT_DATASET, String))
    X = data["X"]
    y = data["y"]

    xs_neg = Float64[]
    ys_neg = Float64[]
    xs_pos = Float64[]
    ys_pos = Float64[]

    for (point, label) in zip(X, y)
        px = Float64(point[1])
        py = Float64(point[2])
        if Int(label) < 0
            push!(xs_neg, px)
            push!(ys_neg, py)
        else
            push!(xs_pos, px)
            push!(ys_pos, py)
        end
    end

    mkpath(dirname(DEFAULT_OUT))
    set_theme!(theme_minimal())
    fig = Figure(size=(900, 560), fontsize=18, backgroundcolor=:white)
    ax = Axis(
        fig[1, 1],
        title="Shared two-moons dataset",
        xlabel="x1",
        ylabel="x2",
        aspect=DataAspect(),
        titlealign=:left,
        titlesize=28,
        xlabelsize=18,
        ylabelsize=18,
        xticklabelsize=15,
        yticklabelsize=15,
        xgridvisible=false,
        ygridvisible=false,
        topspinevisible=false,
        rightspinevisible=false,
    )
    scatter!(ax, xs_neg, ys_neg; color=(colorant"#4C78A8", 0.82), markersize=10, strokecolor=:white, strokewidth=0.4, label="label -1")
    scatter!(ax, xs_pos, ys_pos; color=(colorant"#F58518", 0.82), markersize=10, strokecolor=:white, strokewidth=0.4, label="label +1")
    axislegend(ax; framevisible=false, position=:rb, labelsize=14)
    save(DEFAULT_OUT, fig, px_per_unit=2)
    println("saved $DEFAULT_OUT")
end

main()
