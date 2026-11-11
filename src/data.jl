#=
File: data.jl
Author: lf28
Description: MNIST data loaders: the fixed binarised MNIST benchmark (binarized_mnist.npz)
             and MLDatasets' MNIST with labels (stochastic binarisation, optionally afresh every epoch)
License: MIT
=#

using MLDatasets
using NPZ


## matrix (784 × n) to image tensor (28 × 28 × 1 × n), and back
vec2tensor(s; H=28, W=28) = reshape(s, (H, W, 1, size(s)[end]))
tensor2vec(t; H=28, W=28) = reshape(t, (H * W, size(t)[end]))


## the standard binarised MNIST benchmark (Larochelle & Murray, 2011): 50,000 / 10,000 / 10,000 images,
## binarised once; the file is at ./datasets/binarized_mnist.npz, downloadable from `url`
function load_binary_mnist_data(batch_size::Int=2000; tensor_shape=(28^2, :), seed=2345, train_size=50000,
    url="https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz")
    Random.seed!(seed)
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


## grey-level MNIST, binarised afresh every time a batch is drawn:
## pixel d of image n is 1 with probability intensity[d, n]
struct DynamicBinaryMNIST{A}
    intensity::A           # e.g. 28 × 28 × 1 × N or 784 × N, values in [0, 1]
    labels::Vector{Int}
end

Base.length(data::DynamicBinaryMNIST) = length(data.labels)

function Base.getindex(data::DynamicBinaryMNIST, idx)
    xs = collect(selectdim(data.intensity, ndims(data.intensity), idx))
    xs_binary = Float32.(rand(Float32, size(xs)) .< xs)
    return (xs_binary, data.labels[idx])
end


## MNIST with labels (for the conditional models).
## dtype=Bool: binarise stochastically with `seed`; dynamic=true: the training set is binarised again every epoch,
## while the test and validation sets stay fixed.
## NB: as before, the validation set is the TEST split binarised separately, and all 60,000 training images are used.
function load_mnist_data(batch_size::Int=2000; tensor_shape=(28^2, :), seed=2345, dtype=Bool, train_size=50000, dynamic=false)
    Random.seed!(seed)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    reshape_tf = x -> reshape(x, tensor_shape...)
    Tx = dtype == Bool ? Float32 : dtype
    xtrain, ytrain = MLDatasets.MNIST(Tx, split=:train)[:]
    xtest, ytest = MLDatasets.MNIST(Tx, split=:test)[:]
    xvalid, yvalid = MLDatasets.MNIST(Tx, split=:test)[:]
    xtrain_intensity = xtrain
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
    if dynamic
        intensity = selectdim(reshape_tf(xtrain_intensity), ndims(xtrain), 1:train_size)
        train_loader = Flux.DataLoader(DynamicBinaryMNIST(Float32.(intensity), ytrain), batchsize=batch_size, shuffle=true)
    else
        train_loader = Flux.DataLoader((Float32.(xtrain), ytrain), batchsize=batch_size, shuffle=true)
    end
    test_loader = Flux.DataLoader((Float32.(xtest), ytest), batchsize=batch_size, shuffle=true)
    valid_loader = Flux.DataLoader((Float32.(xvalid), yvalid), batchsize=batch_size, shuffle=true)
    return train_loader, test_loader, valid_loader
end
