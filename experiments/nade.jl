#=
File: experiments/nade.jl
Description: DeepNADE on binarised MNIST (order-agnostic training), and RNADE (a mixture of
             Gaussians per dimension) on five 2D toy data sets, with and without the mask input.
             Run cell by cell: a cell starts at a line beginning with `##`; Alt+Enter runs the current cell.
=#

## setup
cd(joinpath(@__DIR__, ".."))     # the repository root: data in ./datasets
include("../setup.jl")             # paths in include() are relative to THIS file
include("../src/nade.jl")
include("../src/toy_data.jl")

## data: the fixed binarised MNIST benchmark, as vectors (784, n); the arrays behind the test and validation loaders
mnist_binary_loader, mnist_binary_test, mnist_binary_valid = load_binary_mnist_data(1000; tensor_shape=(28^2, :));
test_data = mnist_binary_test.data[1]
valid_data = mnist_binary_valid.data[1]


## ============================================================================================
## DeepNADE on MNIST: 2 × 500 ReLU units, the mask as extra input
## ============================================================================================

## 500 epochs, warm-up then cosine learning-rate decay
Random.seed!(2345)
dpnade_model = DeepNADE(28^2, true) |> device
losses_dnade, best_dpnade_model = fit!(dpnade_model, mnist_binary_loader, Adam(0.001), 500; learning_rate_decay=true, validation_data=valid_data, patience=Inf, every_epoch=20)

## the one-pass estimate, then the exact log-likelihood for one order, and the ensemble of 20 orders per digit
logpdf_val(dpnade_model, test_data |> device)

##
logpdf(dpnade_model, test_data |> device; ensemble_size=1, one_order=true)

##
logpdf(dpnade_model, test_data |> device; ensemble_size=20, one_order=false)

##
plot_loss(losses_dnade, length(mnist_binary_loader))

## samples, and the probabilities they were drawn from
set_rand_seed(1234)
vec2imgs(sample(dpnade_model, 100); nrows=10)

##
set_rand_seed(1234)
vec2imgs(sample(dpnade_model, 100; sample_data=false); nrows=10)


## ============================================================================================
## RNADE on 2D toy data
## ============================================================================================

## five data sets, 1000 training and 1000 test points each
Random.seed!(1234)
noise_level = 0.06
n_points = 1000
data_moons = generate_moons(n_points * 2; noise=noise_level)
data_circles = generate_circles(n_points * 2; noise=noise_level)
data_ring = generate_gaussian_ring(n_points * 2; noise=noise_level * 3)
data_spiral = generate_spiral(n_points * 2; noise=noise_level)
data_ribbon = generate_toy_dataset_1(n_points * 2);

## standardise with the training half; a data loader per data set
data_2d_raw = Dict("moon" => data_moons, "circles" => data_circles,
    "ring" => data_ring, "spiral" => data_spiral, "ribbon" => data_ribbon)
tf_2d = Dict()
data_2d_test = Dict()
data_2d_train_loaders = Dict()
batch_size = 200
for (name, data) in data_2d_raw
    data_train = data[:, 1:n_points]
    data_test = data[:, (n_points+1):end]
    tf = StatsBase.fit(ZScoreTransform, data_train, dims=2)
    data_train_std = StatsBase.transform(tf, data_train)
    data_test_std = StatsBase.transform(tf, data_test)
    tf_2d[name] = tf
    data_2d_train_loaders[name] = Flux.DataLoader((data_train_std,), batchsize=batch_size, shuffle=true)
    data_2d_test[name] = data_test_std
end

## K = 8 components, 2 × 500 ReLU units; with and without the mask input
Random.seed!(12345)
K = 8
input_d = 2
hidden_layers = [500, 500]
act_fun = relu
max_epochs = 500
optimiser = Adam(0.001)
rnade_models_with_masks = Dict()
losses_rnade = Dict()
for (name, data_loader) in data_2d_train_loaders
    rnade_model = DeepNADE(input_d, true, hidden_layers, act_fun; rnade=true, K=K) |> device
    losses, _ = fit!(rnade_model, data_loader, optimiser, max_epochs; verbose=false)
    rnade_models_with_masks[name] = rnade_model
    losses_rnade[name] = losses
end
rnade_models_without_masks = Dict()
for (name, data_loader) in data_2d_train_loaders
    rnade_model = DeepNADE(input_d, false, hidden_layers, act_fun; rnade=true, K=K) |> device
    losses, _ = fit!(rnade_model, data_loader, optimiser, max_epochs; verbose=false)
    rnade_models_without_masks[name] = rnade_model
end

##
plot_loss(losses_rnade["circles"], length(data_2d_train_loaders["circles"]))

## samples over the test data, with the test log-likelihood in the title: with the mask input
Random.seed!(1234)
plts_with_masks = Dict()
sample_size = 800
for (name, model) in rnade_models_with_masks
    test_ll = logpdf(model, data_2d_test[name] |> device)
    plts_with_masks[name] = plot_samples(data_2d_test[name], sample(model, sample_size); alpha2=0.5, title="RNADE log-lik: " * L"%$(round(test_ll; digits=2))")
end
plot(values(plts_with_masks)..., layout=(5, 1), size=(580, 1200), titlefontsize=10, dpi=400, legendfontsize=6, legend=:bottomleft, fmt=:svg)

## without the mask input
Random.seed!(1234)
plts_without_masks = Dict()
for (name, model) in rnade_models_without_masks
    test_ll = logpdf(model, data_2d_test[name] |> device)
    plts_without_masks[name] = plot_samples(data_2d_test[name], sample(model, sample_size); alpha2=0.5, title="RNADE log-lik: " * L"%$(round(test_ll; digits=2))")
end
plot(values(plts_without_masks)..., layout=(5, 1), size=(580, 1200), titlefontsize=10, dpi=400, legendfontsize=6, legend=:bottomleft, fmt=:svg)
