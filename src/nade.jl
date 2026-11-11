#=
File: nade.jl
Author: lf28
Description: DeepNADE (Uria et al., 2014): one MLP for every conditional, trained order-agnostically
             with random masks; Bernoulli outputs for binary images, or a mixture of Gaussians per
             dimension (RNADE) for real-valued data
License: MIT
Layout: data are (d, batch); masks are Bool arrays of the same size (true = observed)
=#


## the network input: [x ⊙ mask; mask] (with_mask=true) or x ⊙ mask
struct DeepNADEInputWrapper
    with_mask::Bool
end

Flux.@layer DeepNADEInputWrapper

function (m::DeepNADEInputWrapper)((x, mask))
    if m.with_mask
        return vcat(x .* mask, mask)
    else
        return x .* mask
    end
end


struct DeepNADE{T<:Flux.Chain}
    d::Int
    with_mask::Bool
    net::T
    rnade::Bool ## real-valued data: K-component mixture of Gaussians per dimension
    K::Int
end

Flux.@layer DeepNADE trainable = (net,)

function DeepNADE(d::Int=28^2, with_mask::Bool=true, hidden_sizes::Vector{Int}=[500, 500], act=relu; rnade::Bool=false, K::Int=3)
    input_size = (with_mask ? 2 : 1) * d
    output_size = rnade ? (3 * K * d) : d ## RNADE: (π, μ, σ) for each of K components per dimension
    hidden_sizes = [input_size, hidden_sizes..., output_size]
    net = []
    for l in 1:(length(hidden_sizes)-1)
        if l == length(hidden_sizes) - 1
            push!(net, Dense(hidden_sizes[l] => hidden_sizes[l+1]))
        else
            push!(net, Dense(hidden_sizes[l] => hidden_sizes[l+1], act))
        end
    end
    input_wrapper = DeepNADEInputWrapper(with_mask)
    network = Chain(input_wrapper, net...)
    return DeepNADE(d, with_mask, network, rnade, K)
end

(m::DeepNADE)((xs, masks)) = m.net((xs, masks))


## ---------------------------------------------------------------------------------------------
## Mixture of Gaussians log-density (RNADE)
## ---------------------------------------------------------------------------------------------

const gaussian_nconst = 1.0f0 / sqrt(2.0f0 * π)
const log_gaussian_nconst = log(1.0f0 / sqrt(2.0f0 * π))

function logpdf_gaussians(μs, σs, xs)
    diff = (μs .- xs) ./ σs
    sqerr = -0.5f0 * diff .^ 2
    return log_gaussian_nconst .+ sqerr .- log.(σs)
end

function logpdf_MoG(logπs, μs, σs, xs)
    result = logpdf_gaussians(μs, σs, xs) .+ logπs
    result = logsumexp(result, dims=1)
    return result
end

## network outputs → log-likelihood of every dimension of every sample (a d × batch matrix)
function loglik_MoG_from_net(outputs, xs; K=3)
    d, batch_size = size(xs)
    pars_per_d = 3
    outputs = reshape(outputs, K, d, pars_per_d, batch_size)
    logπs = logsoftmax(outputs[:, :, 1, :], dims=1)
    μs = outputs[:, :, 2, :]
    lnσs = outputs[:, :, end, :]
    σs = softplus.(lnσs)
    logLiks_d = logpdf_MoG(logπs, μs, σs, reshape(xs, 1, size(xs)...))[1, :, :]
end


## ---------------------------------------------------------------------------------------------
## log-likelihood: exact (d forward passes per order) and the cheap one-pass estimate used for validation
## ---------------------------------------------------------------------------------------------

## exact log p(x) under random orders; ensemble_size > 1 averages the densities of several orders;
## one_order=true: the same order for every sample
function logpdf(model::DeepNADE, xs::AbstractArray; reduction=mean, ensemble_size=1, one_order=true)
    n = size(xs)[2]
    total_d = model.d
    logliks = zeros32(ensemble_size, n) |> device
    for e in 1:ensemble_size
        masks = fill(false, size(xs)...) |> device
        if one_order
            orders = repeat(shuffle(1:total_d)', n)'
        else
            orders = hcat([shuffle(1:total_d) for _ in 1:n]...)
        end
        ll = zeros(n) |> device
        for d in 1:total_d
            preds = model((xs, masks))
            if model.rnade
                ll += loglik_MoG_from_net(preds, xs; K=model.K)[CartesianIndex.(orders[d, :], 1:n)]
            else
                ll += -Flux.logitbinarycrossentropy(preds[CartesianIndex.(orders[d, :], 1:n)], xs[CartesianIndex.(orders[d, :], 1:n)]; agg=identity)
            end
            masks[CartesianIndex.(orders[d, :], 1:n)] .= true
        end
        logliks[e, :] = ll
    end
    (logsumexp(logliks, dims=1) .- log(Float32(ensemble_size))) |> reduction
end

## one-pass estimate: a random mask per sample, the average log-likelihood of the hidden dimensions × d
function logpdf_val(model::DeepNADE, xs::AbstractArray; reduction=mean)
    n = size(xs)[2]
    total_d = model.d
    orders = hcat([shuffle(1:total_d) for _ in 1:n]...)
    ds = rand(1:total_d, n)
    masks = (orders .< ds') |> device
    preds = model((xs, masks))
    negative_masks = .!masks
    if model.rnade
        logLiks = loglik_MoG_from_net(preds, xs; K=model.K) .* negative_masks
    else
        logLiks = -sum(Flux.logitbinarycrossentropy(preds, xs; agg=identity) .* negative_masks, dims=1)
    end
    num_of_pixels = sum(negative_masks, dims=1)
    (logLiks ./ num_of_pixels) * total_d |> mean
end


## sample in a random order per sample; sample_data=false returns the probabilities (or mixture means)
function sample(model::DeepNADE, n=1; sample_data=true)
    total_d = model.d
    xs_samples = zeros(Float32, total_d, n) |> device
    ps_samples = similar(xs_samples, Float32, total_d, n)
    masks = fill(false, size(xs_samples)...) |> device
    orders = hcat([shuffle(1:total_d) for _ in 1:n]...)
    for d = 1:total_d
        outputs = model((xs_samples, masks))
        if model.rnade
            outputs = reshape(outputs, model.K, total_d, 3, n)
            πs = softmax(outputs[:, :, 1, :], dims=1)[:, CartesianIndex.(orders[d, :], 1:n)]
            μs = outputs[:, :, 2, :][:, CartesianIndex.(orders[d, :], 1:n)]
            lnσs = outputs[:, :, end, :][:, CartesianIndex.(orders[d, :], 1:n)]
            σs = softplus.(lnσs)
            assignments = 1 .+ sum(rand_like(πs[1:1, :]) .> cumsum(πs, dims=1), dims=1) ## a component per column
            μzs = μs[CartesianIndex.(assignments[:], 1:n)]
            σzs = σs[CartesianIndex.(assignments[:], 1:n)]
            xs_samples[CartesianIndex.(orders[d, :], 1:n)] = (randn_like(σzs) .* σzs) .+ μzs
            ps_samples[CartesianIndex.(orders[d, :], 1:n)] = sum(μs .* πs, dims=1)[:]
        else
            ps = outputs[CartesianIndex.(orders[d, :], 1:n)] .|> sigmoid
            x_pixels = rand_like(ps) .< ps
            xs_samples[CartesianIndex.(orders[d, :], 1:n)] = x_pixels
            ps_samples[CartesianIndex.(orders[d, :], 1:n)] = ps
        end
        masks[CartesianIndex.(orders[d, :], 1:n)] .= true
    end
    if sample_data
        return xs_samples
    else
        return ps_samples
    end
end


## ---------------------------------------------------------------------------------------------
## Order-agnostic training: for every sample, observe a random subset of c dimensions (c uniform in 0..d-1)
## and maximise the likelihood of the rest, rescaled by d / (d - c)
## ---------------------------------------------------------------------------------------------

function fit!(model::DeepNADE, data_loader, optim=Adam(3f-4), max_epochs=50; verbose=true, learning_rate_decay=false, validation_data=nothing, patience=Inf, every_epoch=10)
    total_d = model.d
    batch_size = data_loader.batchsize
    initial_lr = optim.eta ## initial learning rate
    if learning_rate_decay
        total_batches_per_epoch = length(data_loader)
        total_steps = max_epochs * total_batches_per_epoch
        warmup_steps = Int(ceil(0.1 * total_steps))   # 10% of the steps
        decay_steps = total_steps - warmup_steps
        ss = WarmupLinear(1e-6, Float64(initial_lr), warmup_steps, total_steps, CosAnneal(Float64(initial_lr), 1e-6, decay_steps))
        schedule = ParameterSchedulers.Stateful(ss)
    end
    opt = Flux.setup(optim, model)
    losses = []
    mask_buffer = zeros(Bool, total_d, batch_size)
    validation_check = false
    best_model = deepcopy(model) # the best model so far
    if !isnothing(validation_data)
        validation_check = true
        best_val_loss = Inf
        epochs_no_improve = 0
    end
    @showprogress for epoch in 1:max_epochs
        for (xs,) in data_loader
            batch_size_real = size(xs)[2] ## the last batch may be smaller
            for i in 1:batch_size_real
                ## 1. how many dimensions this sample observes: c in 0..d-1
                c = rand(0:(total_d-1))
                ## 2. c ones then zeros, 3. shuffled in place
                @views mask_buffer[1:c, i] .= true
                @views mask_buffer[(c+1):end, i] .= false
                @views shuffle!(mask_buffer[:, i])
            end
            mask_buffer_gpu = mask_buffer[:, 1:batch_size_real] |> device
            negative_masks = .!mask_buffer_gpu
            xs_gpu = xs |> device
            li, gs = Flux.withgradient(model) do m
                preds = m((xs_gpu, mask_buffer_gpu))
                if m.rnade
                    logLiks_d = -loglik_MoG_from_net(preds, xs_gpu; K=m.K) ## d × batch
                else
                    logLiks_d = Flux.logitbinarycrossentropy(preds, xs_gpu; agg=identity)
                end
                logLiks = sum(logLiks_d .* negative_masks, dims=1)
                num_of_pixels = sum(negative_masks, dims=1)
                (logLiks ./ num_of_pixels) * total_d |> mean
            end
            Flux.update!(opt, model, gs[1])
            push!(losses, li)
            if learning_rate_decay
                next_lr = ParameterSchedulers.next!(schedule)
                Flux.adjust!(opt, next_lr)
            end
        end

        if validation_check
            current_val_loss = -logpdf_val(model, validation_data |> device)
            if current_val_loss < best_val_loss
                best_val_loss = current_val_loss
                best_model = deepcopy(model |> cpu) # save a snapshot
                epochs_no_improve = 0
            else
                epochs_no_improve += 1
            end
            if epochs_no_improve >= patience
                @info "🛑 Early stopping triggered! No improvement for $patience epochs."
                break
            end
        end

        if verbose && (epoch % every_epoch) == 1
            @info "Epoch $(epoch); Loss = $(round(losses[end]; digits=4))"
        end
    end
    return losses, best_model
end
