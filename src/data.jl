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

    train_size = min(train_size, size(mnist_binary_train)[end])   # at most the 50,000 in the file
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
## The 60,000 training images are split: the first `train_size` for training, the rest for validation;
## the 10,000 test images are the test set.
## dtype=Bool: binarise stochastically with `seed`; dynamic=true: the training set is binarised again every epoch,
## while the validation and test sets stay fixed.
function load_mnist_data(batch_size::Int=2000; tensor_shape=(28^2, :), seed=2345, dtype=Bool, train_size=50000, dynamic=false)
    Random.seed!(seed)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"
    reshape_tf = x -> reshape(x, tensor_shape...)
    Tx = dtype == Bool ? Float32 : dtype
    xall, yall = MLDatasets.MNIST(Tx, split=:train)[:]
    xtest, ytest = MLDatasets.MNIST(Tx, split=:test)[:]
    n_all = size(xall)[end]
    train_size = min(train_size, n_all - 1)       # leave at least one image for validation
    xtrain_intensity = xall[:, :, 1:train_size]
    xvalid = xall[:, :, (train_size+1):n_all]
    ytrain = yall[1:train_size]
    yvalid = yall[(train_size+1):n_all]
    xtrain = xtrain_intensity
    if dtype == Bool
        xtrain = (rand_like(xtrain) .< xtrain)
        xvalid = (rand_like(xvalid) .< xvalid)
        xtest = (rand_like(xtest) .< xtest)
    end
    if dynamic
        train_loader = Flux.DataLoader(DynamicBinaryMNIST(Float32.(reshape_tf(xtrain_intensity)), ytrain), batchsize=batch_size, shuffle=true)
    else
        train_loader = Flux.DataLoader((Float32.(reshape_tf(xtrain)), ytrain), batchsize=batch_size, shuffle=true)
    end
    valid_loader = Flux.DataLoader((Float32.(reshape_tf(xvalid)), yvalid), batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((Float32.(reshape_tf(xtest)), ytest), batchsize=batch_size, shuffle=true)
    return train_loader, test_loader, valid_loader
end
