using Micrograd
using Printf

include("utils.jl")
using .MicrogradExampleUtils

function train(;
    n_samples=200,
    arch=[16, 16, 1],
    n_steps=100,
    lr=1.0,
    activation=:relu,
    loss_fn=hinge_loss,
)
    X, y = load_dataset(n_samples)
    model = MLP(2, arch; activation=activation_symbol(activation))
    load_weights!(model, weights_path(arch))

    elapsed = @elapsed for step in 0:(n_steps - 1)
        total_loss, acc = loss_fn(model, X, y)
        zero_grad!(model)
        backward!(total_loss)
        update_params!(model, step, n_steps, lr)

        if step % 10 == 0 || step == n_steps - 1
            @printf("step %3d  loss %.4f  acc %.1f%%\n", step, total_loss.data, acc * 100)
        end
    end

    @printf("\n%d steps in %.3fs (%.1fms/step)\n", n_steps, elapsed, elapsed / n_steps * 1000)
    return model
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    train()
end
