#=
File: experiments/made.jl
Description: MADE on binarised MNIST: one fixed order with a direct connection; an ensemble of 20 orders
             (order-agnostic training); and two small MADEs written out layer by layer.
             Run cell by cell: a cell starts at a line beginning with `##`; Alt+Enter runs the current cell.
=#

## setup
cd(joinpath(@__DIR__, ".."))     # the repository root: data in ./datasets, checkpoints in ./checkpoints
include("setup.jl")
include("src/made.jl")

## data: the fixed binarised MNIST benchmark, as vectors (784, n)
batch_size = 128
mnist_binary_loader, mnist_binary_test, mnist_binary_valid = load_binary_mnist_data(batch_size; tensor_shape=(28^2, :));


## ============================================================================================
## One mask: the raster order, 8000 hidden units, a direct input-to-output connection
## ============================================================================================

## train with early stopping; a checkpoint every epoch (epoch 0 = untrained)
Random.seed!(123)
input_order = [1:(28^2);]
hidden_size = 8000
made_model_one_mask = MADE(28^2, [hidden_size]; input_order=input_order, res_con=true, num_masks=1) |> device
losses_made_one_mask, valid_loss_one_mask, best_made_one_mask = fit!(made_model_one_mask, mnist_binary_loader,
    Adam(), 40;
    save_check_points=true,
    verbose=true,
    valid_data=mnist_binary_valid,
    patience=5
)

## sample grids for the saved checkpoints (saved as PNGs in the checkpoint folder's sampled_plots/)
folder_path = "./checkpoints/MADE_d=784_res_con=true_num_masks=1_cycle_mask=true"
sample_figs = visualize_all_checkpoints(folder_path; num_samples=64, nrows=8);

## samples from the final and from the best model
set_rand_seed(1234)
vec2imgs(sample(made_model_one_mask, 100); nrows=10)

##
vec2imgs(sample(best_made_one_mask |> device, 100); nrows=10)

## the masks satisfy the autoregressive property
check_causal_mask(made_model_one_mask)

##
plot_loss_history(losses_made_one_mask, valid_loss_one_mask, mnist_binary_loader)

## test log-likelihood of the final and of the best model
logpdf(made_model_one_mask |> device, mnist_binary_test |> device; train=false)

##
logpdf(best_made_one_mask |> device, mnist_binary_test |> device; train=false)


## ============================================================================================
## Order-agnostic training: 20 random orders, two hidden layers of 8000 units
## ============================================================================================

## the mask changes after every batch (step_callback!); early stopping with patience 30
Random.seed!(1234)
made_model_en = MADE(28^2, [8000, 8000]; act=relu, input_order=nothing, res_con=true, num_masks=20, init=Flux.kaiming_normal) |> device
losses_made_en, valid_loss_en, best_made_en = fit!(made_model_en, mnist_binary_loader, Adam(), 300;
    valid_data=mnist_binary_valid,
    patience=30,
    every_epoch=5
)

##
check_causal_mask(made_model_en)

##
plot_loss_history(losses_made_en, valid_loss_en, mnist_binary_loader)

##
plot_loss(losses_made_en, mnist_binary_loader)

## one order at a time (log p of a single mask) versus the ensemble of all 20
update_maskset!(best_made_en)
logpdf(best_made_en |> device, mnist_binary_test |> device)

##
logpdf(best_made_en |> device, mnist_binary_test |> device; train=false)

## the biased ensemble (averaging per-pixel probabilities), for comparison
logpdf_test_karpathy(made_model_en |> device, mnist_binary_test.data[1] |> device)

##
vec2imgs(sample(made_model_en |> device, 200); nrows=10)


## ============================================================================================
## Labelled MNIST, binarised stochastically (a different data set from the benchmark above)
## ============================================================================================

##
train_loader, test_loader, valid_loader = load_mnist_data(batch_size; dtype=Bool, tensor_shape=(28 * 28, :))

## 20 orders on the new data (the labels are ignored)
Random.seed!(1234)
made_model_en_new_data = MADE(28^2, [8000, 8000]; act=relu, input_order=nothing, res_con=true, num_masks=20, init=Flux.kaiming_uniform) |> device
losses_made_en_new_data, valid_losses_en_new_data, best_made_en_new_data = fit!(made_model_en_new_data, train_loader, Adam(), 60; valid_data=valid_loader, patience=20, every_epoch=15)

##
logpdf(made_model_en_new_data |> device, test_loader.data[1] |> device; train=false)

## samples and the probabilities they were drawn from
xsamples, psamples = sample(made_model_en_new_data |> device, 200; samples_only=false)
vec2imgs(xsamples; nrows=10)

##
vec2imgs(psamples; nrows=10)


## ============================================================================================
## MADE written out by hand: one hidden layer, one random order
## ============================================================================================

## sampling in a given order, and log p(x) for a plain Chain
function _sample(model, n=1; input_order=1:(28^2))
    xs = zeros(Bool, 28^2, n) |> device
    for d in sortperm(input_order)
        logits_d = model(xs)[d, :]       # the full forward pass, then pixel d only
        ps = σ.(logits_d)
        xs[d, :] = rand_like(ps) .< ps
    end
    return xs
end

function logpdf(model::Chain, xs; reduction=mean)
    xs_preds = model(xs)
    logL = -Flux.logitbinarycrossentropy(xs_preds, xs; agg=identity)
    logL = sum(logL, dims=1)
    logL |> reduction
end

## 784 hidden units with sorted degrees, a random input order; 20 epochs with a plain training loop
Random.seed!(1234)
hidden_size = 28^2
hidden_order = sort(rand(1:(28^2-1), hidden_size))
input_order = shuffle([1:(28^2);])
md = MaskedDense(28^2, hidden_size, input_order, hidden_order; act=relu, output_layer=false)
out_layer = MaskedDense(hidden_size, 28^2, hidden_order, input_order; act=identity, output_layer=true)
made1 = Chain(md, out_layer) |> device
opt = Flux.setup(Adam(0.001), made1)
losses = []
@showprogress for epoch in 1:20
    for (xs,) in mnist_binary_loader
        xs = xs |> device
        li, gs = Flux.withgradient(made1) do m
            preds = m(xs)
            sum(Flux.logitbinarycrossentropy(preds, xs; agg=identity), dims=1) |> mean
        end
        Flux.update!(opt, made1, gs[1])
        push!(losses, li)
    end
end

##
plot_loss(losses, length(mnist_binary_loader))

##
vec2imgs(_sample(made1, 200; input_order=input_order); nrows=10)

##
logpdf(made1, mnist_binary_test.data[1] |> device)

## 500 tanh units plus a direct connection (two masked paths added together); AdamW
Random.seed!(1234)
hidden_size = 500
hidden_order = rand(1:28^2-1, hidden_size)
input_order = shuffle([1:28^2;])
path1 = Chain(MaskedDense(28^2, hidden_size, input_order, hidden_order; act=tanh, output_layer=false),
    MaskedDense(hidden_size, 28^2, hidden_order, input_order; act=identity, output_layer=true))
path2 = MaskedDense(28^2, 28^2, input_order, input_order; act=identity, output_layer=true)
made2_res = Parallel(+; path1, path2) |> device
opt = Flux.setup(AdamW(0.001, (0.9, 0.999), 3e-4, couple=false), made2_res)
losses2 = []
@showprogress for epoch in 1:20
    for (xs,) in mnist_binary_loader
        xs = xs |> device
        li, gs = Flux.withgradient(made2_res) do m
            preds = m(xs)
            sum(Flux.logitbinarycrossentropy(preds, xs; agg=identity), dims=1) |> mean
        end
        Flux.update!(opt, made2_res, gs[1])
        push!(losses2, li)
    end
end

##
plot_loss(losses2, length(mnist_binary_loader))

##
vec2imgs(_sample(made2_res, 100; input_order=input_order); nrows=10)
