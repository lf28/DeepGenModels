#=
File: experiments/convnade.jl
Description: ConvNADE on binarised MNIST (order-agnostic training with a convolutional network).
             Run cell by cell: a cell starts at a line beginning with `##`; Alt+Enter runs the current cell.
=#

## setup
cd(joinpath(@__DIR__, ".."))     # the repository root: data in ./datasets
include("setup.jl")
include("src/convnade.jl")

## data: the fixed binarised MNIST benchmark, as images (28, 28, 1, n); the default batch size is 2000
mnist_binary_loader, mnist_binary_test, mnist_binary_valid = load_binary_mnist_data(; tensor_shape=(28, 28, 1, :));
test_data = mnist_binary_test.data[1]

## a few training images
xs, = mnist_binary_loader |> first;
tensor2imgs(xs[:, :, :, 1:64]; nrows=8, flip=false)

## the larger network (8 convolutions, 64 channels); 200 epochs with warm-up and cosine decay
Random.seed!(123)
conv_nade = ConvNADE(28^2, true; smaller_net=false) |> device
losses_connet, best_model = fit!(conv_nade, mnist_binary_loader, Adam(3e-4), 200; verbose=true, learning_rate_decay=true, validation_data=nothing, patience=Inf)

##
plot_loss(losses_connet, length(mnist_binary_loader))

## samples, and the probabilities they were drawn from
set_rand_seed(112233)
tensor2imgs(sample(conv_nade, 100); nrows=10, flip=false)

##
set_rand_seed(112233)
tensor2imgs(sample(conv_nade, 100; sample_data=false); nrows=10, flip=false)

## exact test log-likelihood, one random order
logpdf(conv_nade, test_data |> device)

## the ensemble of 20 orders on every 10th test image; slow: 20 × 784 forward passes
mnist_binary_test_loader = Flux.DataLoader((test_data[:, :, :, 1:10:end],), batchsize=2000)
logpdf(conv_nade, mnist_binary_test_loader; ensemble_size=20, one_order=false)
