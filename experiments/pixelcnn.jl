#=
File: experiments/pixelcnn.jl
Description: PixelCNN on binarised MNIST: single stack (blind spot), two stacks, gated vs ReLU,
             and the class-conditional two-stack PixelCNN.
             Run cell by cell: a cell starts at a line beginning with `##`; Alt+Enter runs the current cell.
=#

## setup
cd(joinpath(@__DIR__, ".."))     # the repository root: data in ./datasets, checkpoints in ./checkpoints
include("../setup.jl")             # paths in include() are relative to THIS file
include("../src/pixelcnn.jl")

## data: the fixed binarised MNIST benchmark, as images (28, 28, 1, n)
batch_size = 128
mnist_binary_loader, mnist_binary_test, mnist_binary_valid = load_binary_mnist_data(batch_size; tensor_shape=(28, 28, 1, :));


## ============================================================================================
## Single-stack PixelCNN (masked convolutions)
## ============================================================================================

## train: 3 layers, 32 channels, 5 × 5 kernels
Random.seed!(123)
ch_size = 32
kernel_size = (5, 5)
num_layers = 3
pixcnn = PixelCNN2D((28, 28), ch_size, kernel_size, num_layers; act=relu, bias=true, init=Flux.kaiming_uniform) |> device
losses_pixcnn, valid_loss, best_pixcnn = fit!(pixcnn, mnist_binary_loader, Adam(), 10; every_epoch=2, valid_data=mnist_binary_valid)

## loss curves, samples and the test log-likelihood
plot_loss_history(losses_pixcnn, valid_loss, mnist_binary_loader; xscale=:linear)

##
tensor2imgs(sample(pixcnn, 200); nrows=10)

##
logpdf(pixcnn, mnist_binary_test |> device)

## with the location embedding (binary codes of the row and column), GELU, 20 epochs
Random.seed!(1234)
pixcnn_loc = PixelCNN2D((28, 28), ch_size, kernel_size, num_layers; add_loc=true, act=gelu, bias=true, init=Flux.kaiming_uniform) |> device
losses_pixcnn_loc, valid_loss_loc, best_pixcnn_loc = fit!(pixcnn_loc, mnist_binary_loader, Adam(), 20; every_epoch=5, valid_data=mnist_binary_valid)

##
plot_loss_history(losses_pixcnn_loc, valid_loss_loc, mnist_binary_loader; xscale=:linear)

##
logpdf(pixcnn_loc, mnist_binary_test |> device)

## samples, and the probabilities they were drawn from
set_rand_seed(1234)
xs_samples, probs_samples = sample(pixcnn_loc, 200; samples_only=false)
tensor2imgs(xs_samples; nrows=10)

##
tensor2imgs(probs_samples; nrows=10)


## ============================================================================================
## Receptive fields: the single stack has a blind spot, the two stacks do not
## ============================================================================================

## single stack, shifted convolutions, 1 to 4 layers
plts = []
for layer in 1:4
    toy_pixelcnn = PixelCNN2D((28, 28), 1, (3, 3), layer; act=identity, causal_style=:shift)
    push!(plts, plot_receptive_field(toy_pixelcnn, randn32(10, 10, 1, 1)))
end
plot(plts..., layout=(4, 1), size=(500, 800), titlefontsize=10)

## single stack, masked convolutions, 1 to 4 layers
plts = []
for layer in 1:4
    toy_pixelcnn = PixelCNN2D((28, 28), 1, (3, 3), layer; act=identity, causal_style=:mask)
    push!(plts, plot_receptive_field(toy_pixelcnn, randn32(10, 10, 1, 1)))
end
plot(plts..., layout=(4, 1), size=(500, 800), titlefontsize=10)

## two stacks, shifted convolutions, 4 layers
xs = randn32(10, 10, 1, 1)
pxnnt = PixelCNNTwoStack((28, 28), 1, 3, 4; act=identity, weight_norm=true, gated=true, bias=false, init=Flux.kaiming_uniform)
plot_receptive_field(pxnnt, xs)

## two stacks, masked convolutions, 4 layers
pxnnt = PixelCNNTwoStack((28, 28), 1, 3, 4; vert_causal_style=:masked, hori_causal_style=:masked, act=identity, weight_norm=true, gated=true, bias=false, init=Flux.kaiming_uniform)
plot_receptive_field(pxnnt, xs)

## the vertical convolution on its own (shifted and masked)
Random.seed!(1234)
plot_receptive_field(CausalConvVert((3, 3), (1 => 1)), randn32(20, 20, 1, 1))

##
plot_receptive_field(CausalConvVertMasked((3, 3), (1 => 1)), xs)


## ============================================================================================
## Two-stack gated PixelCNN
## ============================================================================================

## 6 layers, 32 channels, 3 × 3 kernels; a checkpoint every epoch (epoch 0 = untrained)
Random.seed!(1234)
pixcnn2stacks = PixelCNNTwoStack((28, 28), 32, 3, 6; add_loc=true, weight_norm=true, init=Flux.kaiming_uniform) |> device
losses_pixcnn2stacks, valid_loss_pixcnn2stacks, best_pixcnn2stacks = fit!(pixcnn2stacks, mnist_binary_loader,
    Adam(0.001), 20;
    save_check_points=true,
    every_epoch=2,
    valid_data=mnist_binary_valid
)

## sample grids for the saved checkpoints (saved as PNGs in the checkpoint folder's sampled_plots/)
folder_path = "./checkpoints/PixelCNNTwoStack_in_ch=1"
sample_figs = visualize_all_checkpoints(folder_path; num_samples=64, nrows=8, every_epoch=2);

##
plot_loss_history(losses_pixcnn2stacks, valid_loss_pixcnn2stacks, mnist_binary_loader; xscale=:linear)

##
tensor2imgs(sample(pixcnn2stacks, 200); nrows=10)

##
logpdf(pixcnn2stacks, mnist_binary_test |> device)

## the same with masked instead of shifted convolutions
Random.seed!(1234)
pixcnn2stacks_masked = PixelCNNTwoStack((28, 28), 32, 3, 6;
    vert_causal_style=:masked, hori_causal_style=:masked,
    add_loc=true, weight_norm=true, init=Flux.kaiming_uniform) |> device
losses_pixcnn2stacks_masked, valid_loss_pixcnn2stacks_masked, best_pixcnn2stacks_masked = fit!(
    pixcnn2stacks_masked, mnist_binary_loader,
    Adam(0.001), 25;
    every_epoch=2,
    valid_data=mnist_binary_valid
)

##
plot_loss_history(losses_pixcnn2stacks_masked, valid_loss_pixcnn2stacks_masked, mnist_binary_loader; xscale=:linear)

##
tensor2imgs(sample(pixcnn2stacks_masked, 200); nrows=10)

##
logpdf(pixcnn2stacks_masked, mnist_binary_test |> device)

## ReLU instead of the gated activation (3 layers, 5 × 5 kernels, as the single stack above)
Random.seed!(1234)
pixcnn2stacks_relu = PixelCNNTwoStack((28, 28), 32, kernel_size[1], num_layers;
    add_loc=true, weight_norm=true, gated=false, init=Flux.kaiming_uniform) |> device
losses_pixcnn2stacks_relu = fit!(pixcnn2stacks_relu, mnist_binary_loader, Adam(), 30; every_epoch=2, valid_data=nothing)

##
logpdf(pixcnn2stacks_relu, mnist_binary_test |> device)

##
tensor2imgs(sample(pixcnn2stacks_relu, 200); nrows=10)

## NEW: a deeper two-stack PixelCNN, 12 layers (a larger receptive field); 40 epochs, keep the best checkpoint
Random.seed!(1234)
pixcnn2stacks_12 = PixelCNNTwoStack((28, 28), 32, 3, 12; add_loc=true, weight_norm=true, init=Flux.kaiming_uniform) |> device
losses_pixcnn2stacks_12, valid_loss_pixcnn2stacks_12, best_pixcnn2stacks_12 = fit!(pixcnn2stacks_12, mnist_binary_loader,
    Adam(0.001), 40;
    save_check_points=true,
    checkpoint_folder="PixelCNNTwoStack_12_layers",   # so the 6-layer checkpoints are not overwritten
    every_epoch=2,
    valid_data=mnist_binary_valid,
    patience=5
)

##
logpdf(best_pixcnn2stacks_12 |> device, mnist_binary_test |> device)



logpdf(pixcnn2stacks_12 |> device, mnist_binary_test |> device)

##

folder_path = "./checkpoints/PixelCNNTwoStack_12_layers"
set_rand_seed(1234)
sample_figs = visualize_all_checkpoints(folder_path; num_samples=64, nrows=8, every_epoch=2);
## ============================================================================================
## Class-conditional two-stack PixelCNN: p(x | y)
## ============================================================================================

## labelled MNIST, binarised stochastically; dynamic=true binarises the training images again every epoch
train_loader, test_loader, valid_loader = load_mnist_data(batch_size; dtype=Bool, tensor_shape=(28, 28, 1, :), dynamic=true)

## 6 layers, 32 channels
Random.seed!(123)
con_pix = ConPixelCNN() |> device
losses_con_pix = fit!(con_pix, train_loader, Adam(), 10; verbose=true, every_epoch=2)

## log p(x | y) on the test set
logpdf(con_pix, test_loader |> device)

## 20 samples of each digit, one digit per row
tensor2imgs(sample(con_pix, repeat(0:9, inner=20), 200); nrows=10)

## 200 samples of the digit 0
tensor2imgs(sample(con_pix, [0], 200); nrows=10)
