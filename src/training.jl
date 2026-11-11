#=
File: training.jl
Author: lf28
Description: maximum-likelihood training loop, test log-likelihood over a data loader,
             random seeds and small random helpers
License: MIT
=#

## the default per-batch callback does nothing; MADE overrides it to switch masks
step_callback!(model, batch_idx) = nothing


## average log-likelihood over a whole data loader;
## each batch is a tuple, e.g. (xs,) or (xs, ys), passed on to the model's own logpdf
function logpdf(model, data_loader::Union{Flux.DataLoader,DeviceIterator}; reduction=mean)
    logLiks = []
    for xs in data_loader
        logL = logpdf(model, xs...; reduction=identity)[:]
        append!(logLiks, logL)
    end
    return logLiks |> reduction
end


## common fit function for general MLE: gradient descent on the negative log-likelihood
function fit!(model, data_loader, optim=Adam(), max_epochs=10;
    save_check_points=false,
    conditional_model=false,   # kept for old calls; the batch tuple decides what logpdf receives
    verbose=true,
    every_epoch=10,
    valid_data=nothing,
    patience=20,
    checkpoint_folder=nothing,  # e.g. "PixelCNN_12_layers"; default: the model's name and settings
    device=device
)
    opt = Flux.setup(optim, model)
    losses = []
    valid_loss = []
    validation_check = !isnothing(valid_data)
    if validation_check
        current_val_loss = (-logpdf(model, valid_data |> device))
        best_val_loss = Inf
        epochs_no_improve = 0
        best_model = deepcopy(model |> cpu)
    end
    if save_check_points
        # folder name: the model's struct name plus its lightweight settings, e.g. "WaveNet_d=784_add_loc=true"
        struct_name = string(nameof(typeof(model)))
        param_strings = String[]
        for field in propertynames(model)
            val = getproperty(model, field)
            # only numbers, booleans, symbols and strings; skips layers and weight arrays
            if val isa Union{Number,Bool,Symbol,AbstractString}
                push!(param_strings, "$(field)=$(val)")
            end
        end
        folder_suffix = join(param_strings, "_")
        full_folder_name = isempty(folder_suffix) ? struct_name : "$(struct_name)_$(folder_suffix)"
        if !isnothing(checkpoint_folder)
            full_folder_name = checkpoint_folder
        end
        output_dir = joinpath("checkpoints", full_folder_name)
        mkpath(output_dir)
        println("Checkpoints will be saved to: $output_dir")
        # epoch 0: the untrained model
        epoch_file = joinpath(output_dir, "model_epoch_$(0).jld2")
        jldsave(epoch_file;
            model=model |> cpu,
            train_loss=(-logpdf(model, data_loader |> device)),
            val_loss=validation_check ? current_val_loss : missing,
            epoch=0
        )
    end
    @showprogress for epoch in 1:max_epochs
        epoch_loss = 0.0f0
        for (batch_idx, xs) in enumerate(data_loader)
            xs = xs |> device
            li, gs = Flux.withgradient(model) do m
                -logpdf(m, xs...)
            end
            epoch_loss += li
            Flux.update!(opt, model, gs[1])
            push!(losses, li)
            step_callback!(model, batch_idx)
        end
        epoch_loss = epoch_loss / length(data_loader)
        if verbose && (epoch % every_epoch) == 1
            @info "Epoch $(epoch); Loss = $(round(epoch_loss; digits=4))"
        end
        if validation_check
            current_val_loss = -logpdf(model, valid_data |> device)
            push!(valid_loss, current_val_loss)
            if current_val_loss < best_val_loss
                best_val_loss = current_val_loss
                best_model = deepcopy(model |> cpu) # save a snapshot
                epochs_no_improve = 0
            else
                epochs_no_improve += 1
            end
            if epochs_no_improve >= patience
                @info "🛑 Early stopping triggered! No improvement for $patience epochs."
                epochs_no_improve = 0
                break
            end
        end

        if save_check_points
            epoch_file = joinpath(output_dir, "model_epoch_$(epoch).jld2")
            jldsave(epoch_file;
                model=model |> cpu,
                train_loss=epoch_loss,
                val_loss=validation_check ? current_val_loss : missing,
                epoch=epoch
            )
        end
    end
    if validation_check
        return losses, valid_loss, best_model
    else
        return losses
    end
end


## learning-rate schedule: linear warm-up from startlr to initlr, then `schedule`
WarmupLinear(startlr, initlr, warmup, total_iters, schedule) =
    Sequence(Triangle(l0=startlr, l1=initlr, period=2 * warmup) => warmup,
        schedule => total_iters)


## set the same random seed for both CPU and GPU
function set_rand_seed(seed)
    Random.seed!(seed)
    if CUDA.functional()
        CUDA.seed!(seed)
    end
end

## random numbers of the same array type as the template (CPU Array or CuArray)
function randn_like(template_array)
    out = similar(template_array)
    Random.randn!(out)
    return out
end

function rand_like(template_array)
    out = similar(template_array)
    Random.rand!(out)
    return out
end


## drops cuDNN's log messages
cudnn_filter = ActiveFilteredLogger(global_logger()) do log
    !occursin("cuDNN", string(log._module))
end
