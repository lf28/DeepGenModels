#=
File: convnade.jl
Author: lf28
Description: ConvNADE: an order-agnostic NADE whose network is convolutional
             (valid convolutions shrink the image, full convolutions grow it back)
License: MIT
Layout: images are (28, 28, 1, batch); masks have the same size (true = observed)
=#


## the network input: x ⊙ mask and the mask as two channels (with_mask=true), or x ⊙ mask
struct ConvNADEInputWrapper
    with_mask::Bool
end

Flux.@layer ConvNADEInputWrapper

function (m::ConvNADEInputWrapper)((x, mask))
    if m.with_mask
        return cat(x .* mask, mask, dims=3)
    else
        return x .* mask
    end
end

## valid_full[l] = true: a valid convolution (no padding); false: a full convolution (padding k - 1)
function create_con_net(with_mask, filter_sizes, filters, valid_full, act=relu)
    net = []
    input_size = with_mask ? 2 : 1
    for l in 1:length(filters)
        padding = valid_full[l] ? 0 : filter_sizes[l] - 1
        if l == 1
            push!(net, Conv((filter_sizes[l], filter_sizes[l]), input_size => filters[l], act, pad=padding))
        elseif l == length(filters)
            push!(net, Conv((filter_sizes[l], filter_sizes[l]), filters[l-1] => filters[l], pad=padding))
        else
            push!(net, Conv((filter_sizes[l], filter_sizes[l]), filters[l-1] => filters[l], act, pad=padding))
        end
    end
    return Chain(net...)
end


struct ConvNADE{T<:Flux.Chain}
    d::Int
    with_mask::Bool
    net::T
end

Flux.@layer ConvNADE trainable = (net,)

function ConvNADE(d::Int=28^2, with_mask::Bool=true; smaller_net=true)
    if smaller_net
        net = create_con_net(with_mask, [5, 7, 7, 5], [4, 8, 4, 1], [true, true, false, false])
    else
        net = create_con_net(with_mask, [8, 4, 6, 7, 7, 6, 4, 8], [64, 64, 64, 64, 64, 64, 64, 1], [true, true, true, true, false, false, false, false])
    end
    input_wrapper = ConvNADEInputWrapper(with_mask)
    net = Chain(input_wrapper, net...)
    return ConvNADE(d, with_mask, net)
end

(m::ConvNADE)((xs, masks)) = m.net((xs, masks))


## order-agnostic training, as for DeepNADE: observe c random pixels per image, predict the rest
function fit!(model::ConvNADE, data_loader, optim=Adam(3e-4), max_epochs=50; verbose=true, learning_rate_decay=true, validation_data=nothing, patience=Inf, every_epoch=10)
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
    best_model = deepcopy(model |> cpu)
    if !isnothing(validation_data)
        validation_check = true
        best_val_loss = Inf
        epochs_no_improve = 0
    end
    @showprogress for epoch in 1:max_epochs
        for (xs,) in data_loader
            batch_size_real = size(xs)[end] ## the last batch may be smaller
            for i in 1:batch_size_real
                c = rand(0:(total_d-1))
                @views mask_buffer[1:c, i] .= true
                @views mask_buffer[(c+1):end, i] .= false
                @views shuffle!(mask_buffer[:, i])
            end
            mask_buffer_gpu = (mask_buffer[:, 1:batch_size_real]) |> device
            xs_gpu = xs |> device
            masks = vec2tensor(mask_buffer_gpu)
            li, gs = Flux.withgradient(model) do m
                preds = m((xs_gpu, masks))
                preds_vecs = tensor2vec(preds)
                xs_vecs = tensor2vec(xs_gpu)
                negative_masks = .!mask_buffer_gpu
                logLiks = sum(Flux.logitbinarycrossentropy(preds_vecs, xs_vecs; agg=identity) .* negative_masks, dims=1)
                num_of_pixels = sum(negative_masks, dims=1)
                (logLiks ./ num_of_pixels) * total_d |> mean
            end
            Flux.update!(opt, model, gs[1])
            push!(losses, cpu(li))
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
                println("🛑 Early stopping triggered! No improvement for $patience epochs.")
                println("🏆 Best Validation Loss achieved: $(round(best_val_loss, digits=5))")
                break
            end
        end

        if verbose && (epoch % every_epoch) == 1
            @info "Epoch $(epoch); Loss = $(round(losses[end]; digits=4))"
        end
    end
    return losses, best_model
end


## exact log p(x) under random orders (784 forward passes per order); ensemble_size > 1 averages the densities
function logpdf(model::ConvNADE, xs::AbstractArray; reduction=mean, ensemble_size=1, one_order=true)
    n = size(xs)[end]
    total_d = model.d
    logliks = zeros32(ensemble_size, n) |> device
    xs_vec = tensor2vec(xs)
    for e in 1:ensemble_size
        masks = fill(false, size(xs)...) |> device
        masks_vec = tensor2vec(masks)
        if one_order
            orders = repeat(shuffle(1:total_d)', n)'
        else
            orders = hcat([shuffle(1:total_d) for _ in 1:n]...)
        end
        ll = zeros(n) |> device
        for d in 1:total_d
            preds_vec = tensor2vec(model((xs, masks)))
            ll += -Flux.logitbinarycrossentropy(preds_vec[CartesianIndex.(orders[d, :], 1:n)], xs_vec[CartesianIndex.(orders[d, :], 1:n)]; agg=identity)
            masks_vec[CartesianIndex.(orders[d, :], 1:n)] .= true
            masks = vec2tensor(masks_vec)
        end
        logliks[e, :] = ll
    end
    (logsumexp(logliks, dims=1) .- log(Float32(ensemble_size))) |> reduction
end

function logpdf(model::ConvNADE, data_loader::Flux.DataLoader; reduction=mean, ensemble_size=1, one_order=true)
    logLiks = []
    for (xs,) in data_loader
        xs = xs |> device
        logL = logpdf(model, xs; reduction=identity, ensemble_size=ensemble_size, one_order=one_order)[:]
        append!(logLiks, logL)
    end
    return logLiks |> reduction
end

## one-pass estimate, used for validation
function logpdf_val(model::ConvNADE, xs::AbstractArray; reduction=mean, device=device)
    n = size(xs)[end]
    total_d = model.d
    orders = hcat([shuffle(1:total_d) for _ in 1:n]...)
    ds = rand(1:total_d, n)
    masks = (orders .< ds') |> device
    negative_masks = .!masks
    preds = model((xs, vec2tensor(masks)))
    logLiks = -sum(Flux.logitbinarycrossentropy(tensor2vec(preds), tensor2vec(xs); agg=identity) .* negative_masks, dims=1)
    num_of_pixels = sum(negative_masks, dims=1)
    ((logLiks ./ num_of_pixels) * total_d |> reduction)
end

## sample in a random order per image; sample_data=false returns the probabilities
function sample(model::ConvNADE, n=1; sample_data=true)
    total_d = model.d
    xs_samples = zeros(Bool, total_d, n) |> device
    ps_samples = similar(xs_samples, Float32, total_d, n)
    masks = fill(false, size(xs_samples)...) |> device
    orders = hcat([shuffle(1:total_d) for _ in 1:n]...)
    for d = 1:total_d
        xs_tensor = vec2tensor(xs_samples)
        masks_tensor = vec2tensor(masks)
        ps_vec = tensor2vec(model((xs_tensor, masks_tensor)))
        ps = ps_vec[CartesianIndex.(orders[d, :], 1:n)] .|> sigmoid
        x_pixels = rand_like(ps) .< ps
        xs_samples[CartesianIndex.(orders[d, :], 1:n)] = x_pixels
        ps_samples[CartesianIndex.(orders[d, :], 1:n)] = ps
        masks[CartesianIndex.(orders[d, :], 1:n)] .= true
    end
    if sample_data
        return vec2tensor(xs_samples)
    else
        return vec2tensor(ps_samples)
    end
end
