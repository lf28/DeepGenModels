#=
File: utils.jl
Author: lf28
Description: utilities for training 
License: MIT
=#
using Random

## common fit function
function fit!(model, data_loader, optim=Adam(), max_epochs=10; verbose=true, every_epoch=10, valid_data=nothing, patience=20, device=device)
    opt = Flux.setup(optim, model)
    losses = []
    valid_loss = []
    validation_check = !isnothing(valid_data)
    if validation_check
        best_val_loss = Inf
        epochs_no_improve = 0
    end
    @showprogress for epoch in 1:max_epochs
        epoch_loss = 0.0f0
        for (xs,) in data_loader
            xs = xs |> device
            li, gs = Flux.withgradient(model) do m
                -logpdf(m, xs)
            end
            epoch_loss += li
            Flux.update!(opt, model, gs[1])
            push!(losses, li)
        end
        epoch_loss = epoch_loss/length(data_loader)
        if verbose && (epoch % every_epoch) == 1
            @info "Epoch $(epoch); Loss = $(round(epoch_loss; digits=4))"
        end
        if validation_check
            current_val_loss = -logpdf(model, valid_data |> device)
            push!(valid_loss, current_val_loss)
            if current_val_loss < best_val_loss
                best_val_loss = current_val_loss
                best_model = deepcopy(model |> cpu) # Save snapshot
                epochs_no_improve = 0
            else
                epochs_no_improve += 1
            end

            # Check if patience has run out
            if epochs_no_improve >= patience
                @info "🛑 Early stopping triggered! No improvement for $patience epochs."
                epochs_no_improve = 0
                break
            end
        end
    end
    return losses, valid_loss
end


## set the same random seed for both CPU and GPU
function set_rand_seed(seed)
    Random.seed!(seed)
    if @isdefined(CUDA)
        CUDA.seed!(seed)
    end
end

function randn_like(template_array)
    # Allocates the exact same array type (CPU Vector or CuVector etc.)
    out = similar(template_array)
    Random.randn!(out)
    return out
end


function rand_like(template_array)
    # Allocates the exact same array type (CPU Vector or CuVector etc.)
    out = similar(template_array)
    Random.rand!(out)
    return out
end



WarmupLinear(startlr, initlr, warmup, total_iters, schedule) =
    Sequence(Triangle(l0=startlr, l1=initlr, period=2 * warmup) => warmup,
        schedule => total_iters)


# Intercepts logs and drops anything containing "cuDNN"
cudnn_filter = ActiveFilteredLogger(global_logger()) do log
    !occursin("cuDNN", string(log._module))
end
