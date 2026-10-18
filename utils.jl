#=
File: utils.jl
Author: lf28
Description: utilities for training 
License: MIT
=#
using Random
using JLD2



# 1. The default fallback method: Do nothing for normal autoregressive models
# useful for MADE
step_callback!(model, batch_idx) = nothing

## common fit function for general MLE 
function fit!(model, data_loader, optim=Adam(), max_epochs=10;
    save_check_points=false,
    conditional_model=false,
    verbose=true,
    every_epoch=10,
    valid_data=nothing,
    patience=20,
    device=device
)
    opt = Flux.setup(optim, model)
    losses = []
    valid_loss = []
    validation_check = !isnothing(valid_data)
    if validation_check
        current_val_loss = (-logpdf(model, valid_data|>device))
        best_val_loss = Inf
        epochs_no_improve = 0
        best_model = deepcopy(model |> cpu)
    end
    if save_check_points
        # 1. Dynamically extract the struct name (e.g., "ConWaveNet")
        struct_name = string(nameof(typeof(model)))
        # 2. Loop through all fields and append configuration fields to a list
        param_strings = String[]
        for field in propertynames(model)
            val = getproperty(model, field)
            # ONLY capture lightweight metadata fields (Numbers, Booleans, Symbols, Strings)
            # This safely skips massive structural fields like weight matrices, layers, or chains
            if val isa Union{Number,Bool,Symbol,AbstractString}
                push!(param_strings, "$(field)=$(val)")
            end
        end
        folder_suffix = join(param_strings, "_")
        full_folder_name = isempty(folder_suffix) ? struct_name : "$(struct_name)_$(folder_suffix)"

        # 2. Use it to construct your sub-folder path (e.g., "checkpoints/ConWaveNet")
        output_dir = joinpath("checkpoints", full_folder_name)
        # 3. Create the directory path recursively
        mkpath(output_dir)
        println("Checkpoints will be saved to: $output_dir")
        # 4. Generate the epoch filename inside the model-named folder
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
            if !conditional_model ## if p(x) only, ignore the label ys
                xs = xs[1]
            end
            xs = xs |> device
            li, gs = Flux.withgradient(model) do m
                -logpdf(m, xs)
            end
            epoch_loss += li
            Flux.update!(opt, model, gs[1])
            push!(losses, li)
            step_callback!(model, batch_idx)
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
