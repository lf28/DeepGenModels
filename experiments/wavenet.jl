#=
File: experiments/wavenet.jl
Description: binarised MNIST as a sequence of 784 pixels: PixelCNN1D with and without dilation,
             WaveNet (chain and skip outputs), and the class-conditional WaveNet.
             Run cell by cell: a cell starts at a line beginning with `##`; Alt+Enter runs the current cell.
=#

## setup
cd(joinpath(@__DIR__, ".."))     # the repository root: data in ./datasets, checkpoints in ./checkpoints
include("setup.jl")
include("src/wavenet.jl")

## data: 1D convolutions expect (784, 1, batch), so a dummy channel dimension is added
batch_size = 64
mnist_binary_loader, mnist_binary_test, mnist_binary_valid = load_binary_mnist_data(batch_size; tensor_shape=(28^2, 1, :));

## settings shared by all models below
ks = 2        # kernel size
nc = 32       # channels
n_layers = 9


## ============================================================================================
## PixelCNN1D: plain causal convolutions
## ============================================================================================

## no dilation: a short memory (receptive field n_layers + 1)
Random.seed!(123)
pix_net_no_dil = PixelCNN1D(28^2, n_layers, ks, nc; dilation=false, built_in_padding=true) |> device
losses_no_dil = fit!(pix_net_no_dil, mnist_binary_loader, Adam(), 5; verbose=true, every_epoch=2, valid_data=nothing)

##
plot_loss(losses_no_dil, length(mnist_binary_loader))

##
vec2imgs(sample(pix_net_no_dil, 100); nrows=10)

##
logpdf(pix_net_no_dil, mnist_binary_test |> device; reduction=mean)

## dilations 1, 2, 4, …, 256: receptive field 512
Random.seed!(123)
pix_net_dil = PixelCNN1D(28^2, n_layers, ks, nc; dilation=true, built_in_padding=false) |> device
losses_dil, valid_losses_dil, best_pix_net_dil = fit!(pix_net_dil, mnist_binary_loader, Adam(), 8; verbose=true, every_epoch=2, valid_data=mnist_binary_valid);

##
plot_loss_history(losses_dil, valid_losses_dil, mnist_binary_loader)

##
set_rand_seed(1234)
vec2imgs(sample(pix_net_dil, 100); nrows=10)

##
logpdf(pix_net_dil, mnist_binary_test |> device; reduction=mean)

## dilation + location feature + residual connections
Random.seed!(123)
pix_net_dil_loc_res = PixelCNN1D(28^2, n_layers, ks, nc; res_con=true, dilation=true, add_loc=true, built_in_padding=true) |> device
losses_dil_loc_res, valid_losses_dil_loc_res, best_pix_net_dil_loc_res = fit!(pix_net_dil_loc_res, mnist_binary_loader,
    Adam(1e-3), 10;
    verbose=true,
    every_epoch=2,
    valid_data=mnist_binary_valid
);

##
plot_loss_history(losses_dil_loc_res, valid_losses_dil_loc_res, mnist_binary_loader)

##
logpdf(pix_net_dil_loc_res, mnist_binary_test |> device; reduction=mean)

## dilation + location feature
Random.seed!(123)
pix_net_dil_loc = PixelCNN1D(28^2, n_layers, ks, nc; dilation=true, add_loc=true) |> device
losses_dil_loc, valid_losses_dil_loc, best_pix_net_dil_loc = fit!(pix_net_dil_loc, mnist_binary_loader,
    Adam(), 10;
    verbose=true,
    every_epoch=2,
    valid_data=mnist_binary_valid
);

##
set_rand_seed(1234)
vec2imgs(sample(pix_net_dil_loc, 200); nrows=10)

##
logpdf(pix_net_dil_loc, mnist_binary_test |> device; reduction=mean)


## ============================================================================================
## WaveNet: gated residual blocks
## ============================================================================================

## output from the last block (:chain)
Random.seed!(123)
wave_net_dil = WaveNet(28^2, n_layers, ks, nc; add_loc=false, dilation=true) |> device
losses_dil_wn, valid_losses_dil_wn, best_wave_net_dil = fit!(wave_net_dil, mnist_binary_loader,
    Adam(), 10;
    verbose=true, every_epoch=2,
    valid_data=mnist_binary_valid
)

##
set_rand_seed(1234)
vec2imgs(sample(wave_net_dil, 200); nrows=10)

##
logpdf(wave_net_dil, mnist_binary_test |> device)

## with the location feature
Random.seed!(123)
wave_net_dil_loc = WaveNet(28^2, n_layers, ks, nc; add_loc=true, dilation=true) |> device
losses_dil_loc_wn, valid_losses_dil_loc_wn, best_wave_net_dil_loc = fit!(
    wave_net_dil_loc, mnist_binary_loader,
    Adam(), 10;
    verbose=true,
    every_epoch=2,
    valid_data=mnist_binary_valid
)

##
set_rand_seed(1234)
vec2imgs(sample(wave_net_dil_loc, 200); nrows=10)

##
logpdf(wave_net_dil_loc, mnist_binary_test |> device)

## check causality with autograd: no output may depend on a later input
test_autoregressive_property_comprehensive(wave_net_dil_loc |> cpu; input_shape=(28^2, 1, 1))

## output from the sum of the skip connections (:skip); a checkpoint every epoch
Random.seed!(123)
wave_net_dil_loc_skip_fwd = WaveNet(28^2, n_layers, ks, nc; forward_mode=:skip, add_loc=true, dilation=true) |> device
losses_dil_loc_skipfwd_wn, valid_losses_dil_loc_skipfwd_wn, best_wave_net_dil_loc_skip_fwd = fit!(
    wave_net_dil_loc_skip_fwd, mnist_binary_loader,
    Adam(), 20;
    save_check_points=true,
    verbose=true, every_epoch=4,
    valid_data=mnist_binary_valid
)

##
plot_loss_history(losses_dil_loc_skipfwd_wn, valid_losses_dil_loc_skipfwd_wn, mnist_binary_loader)

##
logpdf(wave_net_dil_loc_skip_fwd, mnist_binary_test |> device)

##
set_rand_seed(1234)
vec2imgs(sample(wave_net_dil_loc_skip_fwd, 200); nrows=10)

## sample grids for the saved checkpoints (saved as PNGs in the checkpoint folder's sampled_plots/)
folder_path = "./checkpoints/WaveNet_d=784_add_loc=true_dilation=true_forward_mode=skip"
sample_figs = visualize_all_checkpoints(folder_path; num_samples=36, nrows=6);

##
plot(sample_figs..., layout=(3, :), size=(800, 900), titlefontsize=8)


## ============================================================================================
## Labelled MNIST, binarised stochastically (a different data set from the benchmark above)
## ============================================================================================

##
train_loader, test_loader, valid_loader = load_mnist_data(batch_size; dtype=Bool, tensor_shape=(28 * 28, 1, :))

## the unconditional WaveNet on this data, for comparison
Random.seed!(123)
wave_net_dil_loc_new_data = WaveNet(28^2, n_layers, ks, nc; add_loc=true, dilation=true) |> device
losses_dil_loc_wn_new_data, valid_losses_dil_loc_wn_new_data, best_wave_net_dil_loc_new_data = fit!(
    wave_net_dil_loc_new_data, train_loader,
    Adam(), 10;
    verbose=true, every_epoch=2,
    valid_data=valid_loader
)

##
plot_loss_history(losses_dil_loc_wn_new_data, valid_losses_dil_loc_wn_new_data, train_loader)

## on both test sets: the benchmark's and this one's
logpdf(wave_net_dil_loc_new_data, mnist_binary_test |> device), logpdf(wave_net_dil_loc_new_data, test_loader |> device)

##
set_rand_seed(1234)
vec2imgs(sample(wave_net_dil_loc_new_data, 200), nrows=10)


## ============================================================================================
## Class-conditional WaveNet: p(x | y)
## ============================================================================================

## output from the last block
Random.seed!(123)
con_wnet = ConWaveNet(28^2, n_layers, 2, nc, 10; add_loc=true, dilation=true, act=relu, init=Flux.glorot_uniform) |> device
losses_con_wnet = fit!(con_wnet, train_loader, Adam(), 10; verbose=true, every_epoch=2);

## output from the skip connections
Random.seed!(123)
con_wnet_skip = ConWaveNet(28^2, n_layers, 2, nc, 10; forward_mode=:skip, add_loc=true, dilation=true, act=relu, init=Flux.glorot_uniform) |> device
losses_con_wnet_skip = fit!(con_wnet_skip, train_loader, Adam(), 10; verbose=true, every_epoch=2);

## 20 samples of each digit, one digit per row
set_rand_seed(1234)
vec2imgs(sample(con_wnet, repeat(0:9, inner=20), 200), nrows=10)

##
set_rand_seed(1234)
vec2imgs(sample(con_wnet_skip, repeat(0:9, inner=20), 200), nrows=10)

## log p(x | y) on the test set
logpdf(con_wnet, test_loader |> device)

##
logpdf(con_wnet_skip, test_loader |> device)

## samples of a single digit
set_rand_seed(1234)
vec2imgs(sample(con_wnet, 2, 100), nrows=10)

##
set_rand_seed(1234)
vec2imgs(sample(con_wnet_skip, 0, 200), nrows=10)

## the conditional model as a classifier: p(y | x) ∝ p(x | y); test accuracy
pps, targets = predict_y(con_wnet, test_loader |> device)
mean(argmax.(eachcol(pps)) .== vcat(targets...) .+ 1)

##
pps, targets = predict_y(con_wnet_skip, test_loader |> device)
mean(argmax.(eachcol(pps)) .== vcat(targets...) .+ 1)
