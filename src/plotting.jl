#=
File: plotting.jl
Author: lf28
Description: image mosaics of samples, loss curves, and sample grids for every saved checkpoint
License: MIT
=#

using MosaicViews
using Images


function tensor2imgs(tensors; nrows=2, flip=false, npad=2, title="")
    tensors = tensors |> cpu
    tensors = dropdims(tensors, dims=3)
    if flip
        tensors = 1 .- tensors
    end
    mosaic = mosaicview(Gray.(permuteddimsview(tensors, [2, 1, 3])), fillvalue=0, nrow=nrows, npad=npad, rowmajor=true)

    # 2. If no title is supplied, return the raw mosaic as you did before
    if isempty(title)
        return mosaic
    end

    # 3. If a title is provided, wrap the mosaic inside a Plots layout canvas
    return plot(mosaic,
        title=title,
        titlefontsize=14,
        titlelocation=:center,
        ticks=false,         # Hide the pixel grid axes
        showaxis=false,      # Hide the surrounding plot borders
        legend=false,        # Remove color mapping keys
        aspect_ratio=:equal) # Maintain the image's original dimensions
end;


function vec2imgs(vecs; nrows=2, flip=false, npad=2, H=28, W=28, title="")
    vecs = vecs |> cpu
    if ndims(vecs) == 1
        vecs = vecs'
    end
    n = size(vecs)[2]

    output = zeros(Float64, H, W, n)
    for i in 1:n
        output[:, :, i] = reshape(vecs[:, i], H, W)'
    end

    if flip
        output = 1 .- output
    end
    # return Gray.(output)
    mosaic = mosaicview(Gray.(output), fillvalue=0.2, nrow=nrows, npad=npad, rowmajor=true)


    # 2. If no title is supplied, return the raw mosaic as you did before
    if isempty(title)
        return mosaic
    end

    # 3. If a title is provided, wrap the mosaic inside a Plots layout canvas
    return plot(mosaic,
        title=title,
        titlefontsize=14,
        titlelocation=:center,
        ticks=false,         # Hide the pixel grid axes
        showaxis=false,      # Hide the surrounding plot borders
        legend=false,        # Remove color mapping keys
        aspect_ratio=:equal) # Maintain the image's original dimensions
end

"""
    plot 2d samples as scatter plot; true data and simulated data are super-imposed; only used in real valued model
"""
function plot_samples(truedata, samples; title="RNADE", ms1=4, ms2=3, alpha2=0.8, fmt=:svg)
    truedata = truedata |> cpu
    samples = samples |> cpu
    plt = scatter(truedata[1, :], truedata[2, :], st=:scatter, markerstrokecolor=:white, alpha=1.0, title=title, ratio=1, label="real data", ms=ms1, framestyle=:none, fmt=fmt)
    plot!(samples[1, :], samples[2, :], st=:scatter, ms=ms2, alpha=alpha2, markerstrokecolor=:white, label="sampled data")
    return plt
end


function plot_loss_history(train_nll_history, val_nll_history, train_loader; xscale=:linear, yscale=:linear)
    # Calculate how many minibatches make up a single full epoch
    batches_per_epoch = length(train_loader)

    # Map each minibatch iteration index to its exact fractional epoch position
    # e.g., if batches_per_epoch = 100: iteration 50 -> epoch 0.5, iteration 150 -> epoch 1.5
    train_epochs = (1:length(train_nll_history)) ./ batches_per_epoch

    # Validation occurs exactly at the end of each integer epoch milestone
    val_epochs = 1:length(val_nll_history)

    # ylims = extrema(train_nll_history)
    ylims = (minimum([train_nll_history..., val_nll_history...]), maximum(val_nll_history) + max(0.5 * std(val_nll_history), 0.25 * std(train_nll_history)))
    # Initialize the plot layout
    plt = plot(
        title="Training Progress: Negative Log-Likelihood",
        xlabel="Epoch",
        ylabel="Average NLL",
        grid=true,
        legend=:topright,
        # size = (800, 500),
        xscale=xscale,
        yscale=yscale,
        ylims=ylims,
        fmt=:svg
    )

    # Plot training loss as a light, thin line (since it fluctuates heavily per batch)
    plot!(
        train_epochs, train_nll_history,
        label="Train Loss (per minibatch)",
        color=1,
        alpha=0.5,   # Makes the noise less distracting
        linewidth=1.0
    )

    # Plot validation loss as a solid line with markers at each epoch milestone
    plot!(
        val_epochs, val_nll_history,
        label="Validation Loss (per epoch)",
        color=2,
        linewidth=2,
        markershape=:circle,
        markersize=3,
        alpha=0.8,
        markerstrokecolor=:white
    )

    # Display the final interactive chart window
    plt
end



function plot_loss(loss, n_batch; xscale=:log10, fmt=:svg, title="")
    if !(n_batch isa Number)
        n_batch = length(n_batch)
    end
    plt = plot(loss, xscale=xscale, xlabel="Iterations", label="batch loss", ylabel="negative log-likelihood", fmt=fmt, title=title)
    plot!(n_batch:n_batch:length(loss), mean.(Iterators.partition(loss, n_batch)),
        label="epoch mean")
    plt
end



# 2. The pipeline that loops through your sub-folder checkpoints
function visualize_all_checkpoints(folder_path::String; num_samples=64, nrows=8, every_epoch=5)
    # Locate and sort all numbered epoch checkpoints chronologically
    all_files = readdir(folder_path)
    epoch_files = filter(f -> occursin(r"model_epoch_\d+\.jld2", f), all_files)
    sort!(epoch_files, by=f -> parse(Int, match(r"\d+", f).match))

    # Create an output directory for your generation results
    plots_out_dir = joinpath(folder_path, "sampled_plots")
    mkpath(plots_out_dir)
    println("Found $(length(epoch_files)) checkpoints. Rendering plots...")

    plts = []
    for (i, file) in enumerate(epoch_files)
        # === FILTER RULE ===
        # i <= 5: Shows indices 1, 2, 3, 4, 5
        # i % every_epoch == 0
        if i <= 5 || (i-1) % every_epoch == 0 || i == length(epoch_files)
            epoch_num = match(r"\d+", file).match
            checkpoint_path = joinpath(folder_path, file)

            # 2. Extract model and validation loss from the JLD2 archive
            cpu_model, val_loss = jldopen(checkpoint_path, "r") do io
                return io["model"], io["val_loss"]
            end

            # # Move the model to the GPU for quick generation
            model = cpu_model |> device

            # # 3. Generate your samples
            raw_samples = sample(model, num_samples)
            #    

            if ndims(raw_samples) == 4
                raw_samples = tensor2vec(raw_samples)
            end
            formatted_loss = round(val_loss, digits=2)
            #         # --- MOSAIC & PLOTTING BLOCK ---
            #         # Generate the 10x10 layout using your custom vec2imgs method
            grid_figure = vec2imgs(
                raw_samples;
                nrows=nrows,
                title="Epoch $epoch_num | Valid Loss: $formatted_loss"
            )

            push!(plts, grid_figure)
            # 6. Force inline display in Jupyter / VS Code
            display(grid_figure)

            # free GPU memory before the next checkpoint
            CUDA.functional() && CUDA.reclaim()

            ## Save the figure to your subfolder
            out_filename = joinpath(plots_out_dir, "epoch_$(epoch_num)_grid.png")
            savefig(grid_figure, out_filename)
        end
    end
    plts
end
