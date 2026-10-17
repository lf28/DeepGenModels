#=
File: io_utils.jl
Author: lf28
Description: io utilities
License: MIT
=#

using MLDatasets
using NPZ
using Flux: DataLoader
using MosaicViews
using Images
using Random
using Plots;
default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style


"""
    Transform matrix to tensor
"""
vec2tensor(s; H=28, W=28) = reshape(s, (H, W, 1, size(s)[end]))
"""
    Transform a tensor to matrix
"""
tensor2vec(t; H=28, W=28) = reshape(t, (H * W, size(t)[end]));

# function get_data_binary(batch_size=2000; tensor_shape=(28^2, :), stochastic=true, train_size=50000)
#     ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
#     reshape_tf = x -> reshape(x, tensor_shape)
#     xtrain, ytrain = MLDatasets.MNIST(split=:train)[:]
#     xtest, ytest = MLDatasets.MNIST(split=:test)[:]
#     xvalid, yvalid = MLDatasets.MNIST(split=:test)[:]
#     mnist_binary_test = xtest |> reshape_tf
#     mnist_binary_train = xtrain |> reshape_tf
#     mnist_binary_valid = xvalid |> reshape_tf

#     if stochastic
#         xtrain = rand(size(xtrain)...) .< xtrain
#         xtest = rand(size(xtest)...) .< xtest
#     else
#         xtrain = xtrain .> 0.5
#         xtest = xtest .> 0.5
#     end
#     Flux.DataLoader((xtrain, ytrain), batchsize=batch_size, shuffle=true), xtest, ytest
# end



# function load_binary_mnist_data(batch_size::Int=2000; tensor_shape=(28^2, :); seed=2345, train_size=50000)
#     # bmnist = npzread(download(url))
#     # bmnist = npzread("./datasets/binarized_mnist.npz")
#     # tensor_shape = tensor ? () : identity
#     # mnist_binary_test = BitMatrix(bmnist["test_data"]') |> v2t_transform
#     # mnist_binary_train = BitMatrix(bmnist["train_data"]') |> v2t_transform
#     # mnist_binary_valid = BitMatrix(bmnist["valid_data"]') |> v2t_transform
#     tensor_shape = tensor ? (28, 28, 1, :) : (28^2, :)
#     return load_binary_mnist_data(batch_size=batch_size; tensor_shape=tensor_shape, seed=seed, train_size=train_size)

#     # train_size = max(train_size, size(mnist_binary_train)[end])
#     # train_data = selectdim(mnist_binary_train, ndims(mnist_binary_train), 1:train_size)
#     # mnist_binary_loader = Flux.DataLoader((train_data,), batchsize=2000, shuffle=true)
#     # return mnist_binary_loader, mnist_binary_test, mnist_binary_valid
# end

function load_mnist_data(batch_size::Int=2000; tensor_shape=(28^2, :), seed=2345, dtype=Bool, train_size=50000)
    Random.seed!(seed)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    reshape_tf = x -> reshape(x, tensor_shape...)
    Tx = dtype == Bool ? Float32 : dtype
    xtrain, ytrain = MLDatasets.MNIST(Tx, split=:train)[:]
    xtest, ytest = MLDatasets.MNIST(Tx, split=:test)[:]
    xvalid, yvalid = MLDatasets.MNIST(Tx, split=:test)[:]
    if dtype == Bool
        xtrain = (rand_like(xtrain) .< xtrain)
        xtest = (rand_like(xtest) .< xtest)
        xvalid = (rand_like(xvalid) .< xvalid)
    end
    xtrain = xtrain |> reshape_tf
    xtest = xtest |> reshape_tf
    xvalid = xvalid |> reshape_tf
    train_size = max(train_size, size(xtrain)[end])
    xtrain = selectdim(xtrain, ndims(xtrain), 1:train_size)
    ytrain = ytrain[1:train_size]
    train_loader = Flux.DataLoader((Float32.(xtrain), ytrain), batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((Float32.(xtest), ytest), batchsize=batch_size, shuffle=true)
    valid_loader = Flux.DataLoader((Float32.(xvalid), yvalid), batchsize=batch_size, shuffle=true)
    return train_loader, test_loader, valid_loader
end



function load_binary_mnist_data(batch_size::Int=2000; tensor_shape=(28^2, :), seed=2345, train_size=50000, url="https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz")
    Random.seed!(seed)
    # bmnist = npzread(download(url))
    bmnist = npzread("./datasets/binarized_mnist.npz")
    reshape_tf = x -> reshape(x, tensor_shape)
    mnist_binary_test = BitMatrix(bmnist["test_data"]') |> reshape_tf
    mnist_binary_train = BitMatrix(bmnist["train_data"]') |> reshape_tf
    mnist_binary_valid = BitMatrix(bmnist["valid_data"]') |> reshape_tf

    train_size = max(train_size, size(mnist_binary_train)[end])
    train_data = selectdim(mnist_binary_train, ndims(mnist_binary_train), 1:train_size)
    train_loader = Flux.DataLoader((Float32.(train_data),), batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((Float32.(mnist_binary_test),), batchsize=batch_size, shuffle=true)
    valid_loader = Flux.DataLoader((Float32.(mnist_binary_valid),), batchsize=batch_size, shuffle=true)
    return train_loader, test_loader, valid_loader
end

function tensor2imgs(tensors; nrows=2, flip=false, npad=2)
    tensors = tensors |> cpu
    tensors = dropdims(tensors, dims=3)
    if flip
        tensors = 1 .- tensors
    end
    mosaicview(Gray.(permuteddimsview(tensors, [2, 1, 3])), fillvalue=0, nrow=nrows, npad=npad, rowmajor=true)
end;


function vec2imgs(vecs; nrows=2, flip=false, npad=2, H=28, W=28)
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
    mosaicview(Gray.(output), fillvalue=0.2, nrow=nrows, npad=npad, rowmajor=true)
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
    ylims = (minimum([train_nll_history..., val_nll_history...]), maximum(val_nll_history) + 0.5 * std(val_nll_history))
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


