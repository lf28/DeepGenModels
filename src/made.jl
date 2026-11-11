#=
File: made.jl
Author: lf28
Description: MADE (Germain et al., 2015): masked dense layers, mask construction from degrees,
             order-agnostic training with a pool of masks, and the test log-likelihood of the ensemble
License: MIT
Layout: data are (784, batch)
=#


## ---------------------------------------------------------------------------------------------
## Masked dense layer: W ⊙ M
## ---------------------------------------------------------------------------------------------

struct MaskedDense{T<:Real,B}
    weight::AbstractMatrix{T}
    bias::B
    act::Function
    mask::AbstractMatrix{Bool}
end

Flux.@layer MaskedDense trainable = (weight, bias)

MaskedDense(in::Int, out::Int, mask; act=identity, bias=true, init=Flux.glorot_uniform) = MaskedDense(
    init(out, in),
    bias ? zeros32(out) : false,
    act,
    mask
)

MaskedDense(in::Int, out::Int; act=identity, bias=true, init=Flux.glorot_uniform) = MaskedDense(in, out, fill(true, out, in); act=act, bias=bias, init=init)

## build the mask from the degrees of the inputs (ms_in) and of the outputs (ms_out)
function MaskedDense(in::Int, out::Int, ms_in::AbstractVector{<:Integer}, ms_out::AbstractVector{<:Integer};
    output_layer=false,
    act=identity,
    bias=true,
    init=Flux.glorot_uniform
)
    md = MaskedDense(in, out; act=act, bias=bias, init=init)
    set_mask!(md, ms_in, ms_out; output_layer=output_layer)
    return md
end

function (m::MaskedDense)(xs)
    W = m.weight .* m.mask
    zs = W * xs
    if !(m.bias === false)
        zs = zs .+ m.bias
    end
    m.act.(zs)
end

function register_mask!(m::MaskedDense, mask)
    m.mask .= mask
end

function set_mask!(m::MaskedDense, ms_in, ms_out; output_layer=false)
    set_mask!(m.mask, ms_in, ms_out; output_layer=output_layer)
end


## ---------------------------------------------------------------------------------------------
## Masks from degrees: hidden unit k may see input d if m(k) ≥ m(d);
## output d may see hidden unit k only if m(d) > m(k)
## ---------------------------------------------------------------------------------------------

function set_mask!(mask::AbstractMatrix{Bool}, ms_in, ms_out; output_layer=false)
    for idx in CartesianIndices(mask)
        mask[idx] = output_layer ? ms_out[idx[1]] > ms_in[idx[2]] : ms_out[idx[1]] >= ms_in[idx[2]]
    end
end

## one set of masks for all layers, for a random (or a given) input order
function produce_mask_set(total_d::Int, hidden_sizes::Vector{Int}; fixed_input_order=nothing, res_con::Bool=false)
    masks = Vector{AbstractMatrix{Bool}}(undef, length(hidden_sizes) + 1 + res_con) ## res_con needs one more mask
    input_order = isnothing(fixed_input_order) ? shuffle(1:total_d) : fixed_input_order
    m_in = input_order
    h_in = total_d
    for (l, h_out) in enumerate(hidden_sizes)
        m_out = rand(minimum(m_in):(total_d-1), h_out)
        mask_layer = fill(false, h_out, h_in)
        set_mask!(mask_layer, m_in, m_out)
        masks[l] = mask_layer
        m_in = m_out
        h_in = h_out
    end
    mask_out_layer = fill(false, total_d, hidden_sizes[end])
    set_mask!(mask_out_layer, m_in, input_order; output_layer=true)
    masks[end-res_con] = mask_out_layer
    if res_con
        mask_res_con = fill(false, total_d, total_d)
        set_mask!(mask_res_con, input_order, input_order; output_layer=true)
        masks[end] = mask_res_con
    end
    return masks, input_order
end

function create_network(d::Int, hidden_sizes::Vector{Int}; act=relu, bias=true, res_con=false, init=Flux.glorot_uniform)
    hs = [d, hidden_sizes...]
    mlp_net = []
    for l in 1:(length(hs)-1)
        layer = MaskedDense(hs[l], hs[l+1]; act=act, bias=bias, init=init)
        push!(mlp_net, layer)
    end
    push!(mlp_net, MaskedDense(hs[end], d; act=identity, bias=bias, init=init))
    if res_con
        res_net = MaskedDense(d, d; act=identity, bias=bias, init=init)
        return Parallel(+, Chain(mlp_net...), res_net)
    else
        return Chain(mlp_net...)
    end
end


## ---------------------------------------------------------------------------------------------
## Checking the autoregressive property: the input-to-output connectivity, sorted by the input
## order, must be strictly upper triangular
## ---------------------------------------------------------------------------------------------

function compute_con_matrix(mask_set, input_order, res_con::Bool)
    con = mask_set[1]'
    for m in mask_set[2:(end-res_con)]
        con = con * m'
    end
    con = res_con ? con + mask_set[end]' : con
    i_sp = sortperm(input_order)
    sorted_con = con[i_sp, i_sp]
    return sorted_con
end

function check_causal_mask(mask_set, input_order, res_con::Bool)
    sorted_con = compute_con_matrix(mask_set, input_order, res_con)
    istriu(sorted_con) && all(diag(sorted_con) .== 0)
end


## ---------------------------------------------------------------------------------------------
## MADE with a pool of masks (num_masks > 1: order-agnostic training)
## ---------------------------------------------------------------------------------------------

struct MADE{T<:Union{Chain,Parallel}}
    d::Integer ## input dimension
    net::T ## trainable network
    res_con::Bool ## direct input-to-output connection or not
    num_masks::Int ## number of masks used in training
    mask_pool
    input_orders
    current_mask_idx::Base.RefValue{Int} # a mutable container for the integer
    cycle_mask::Bool
end

Flux.@layer MADE trainable = (net,)

function MADE(d::Int=28^2, hidden_sizes::Vector{Int}=[8000];
    act=relu,
    cycle_mask=true,
    bias=true,
    input_order=nothing,
    res_con=true,
    num_masks=1,
    init=Flux.glorot_uniform
)
    mask_pool_with_input_orders = [produce_mask_set(d, hidden_sizes; fixed_input_order=input_order, res_con=res_con) for _ in 1:num_masks]
    mask_pool = [mask_pool_with_input_orders[i][1] for i in 1:num_masks]
    input_orders = [mask_pool_with_input_orders[i][2] for i in 1:num_masks] ## needed for sampling
    current_mask_idx = 1
    net = create_network(d, hidden_sizes; act=act, bias=bias, res_con=res_con, init=init)
    set_maskset!(net, mask_pool[current_mask_idx], res_con)
    MADE(d, net, res_con, num_masks, mask_pool, input_orders, Ref(current_mask_idx), cycle_mask)
end

function set_maskset!(net::Union{Chain,Parallel}, mask_set::Vector{Y}, res_con::Bool) where {Y<:AbstractMatrix{Bool}}
    if res_con
        mlp = net[1]
        res_net = net[2]
        for (li, masked_layer) in enumerate(mlp)
            register_mask!(masked_layer, mask_set[li])
        end
        register_mask!(res_net, mask_set[end])
    else
        for (li, masked_layer) in enumerate(net)
            register_mask!(masked_layer, mask_set[li])
        end
    end
end

## move to the next mask in the pool (in turn, or at random)
function update_maskset!(model::MADE)
    if model.num_masks > 1
        if model.cycle_mask
            mask_idx = (model.current_mask_idx[]) % model.num_masks + 1
        else
            mask_idx = rand(1:model.num_masks)
        end
        model.current_mask_idx[] = mask_idx
        set_maskset!(model.net, model.mask_pool[mask_idx], model.res_con)
    end
end

## called by fit! after every batch: train with the next mask
function step_callback!(model::MADE, batch_idx)
    if model.num_masks > 1
        update_maskset!(model)
    end
end

function (m::MADE)(x)
    m.net(x)
end


## ---------------------------------------------------------------------------------------------
## log-likelihood and sampling
## ---------------------------------------------------------------------------------------------

## train=true: log p(x) under the current mask; train=false: the ensemble over all masks
function logpdf(model::MADE, xs::AbstractArray; reduction=mean, train=true)
    if train
        xs_preds = model(xs)
        logL = -Flux.logitbinarycrossentropy(xs_preds, xs; agg=identity)
        logL = sum(logL, dims=1)
        logL |> reduction
    else
        logpdf_test(model, xs; reduction=reduction)
    end
end

## a labelled batch (xs, ys): ignore the labels
logpdf(model::MADE, xs::AbstractArray, ys; reduction=mean, train=true) = logpdf(model, xs; reduction=reduction, train=train)

function logpdf(model::MADE, data_loader::Union{Flux.DataLoader,DeviceIterator}; reduction=mean, train=true)
    logLiks = []
    for (xs,) in data_loader
        if train
            logL = logpdf(model, xs; reduction=identity)[:]
        else
            logL = logpdf_test(model, xs; reduction=identity)[:]
        end
        append!(logLiks, logL)
    end
    return logLiks |> reduction
end

## the ensemble: log (1/M) Σ_m p_m(x), the average of the densities, not of the log-densities
function logpdf_test(model::MADE, xs; reduction=mean)
    num_masks = model.num_masks
    logLiks = similar(xs, Float32, num_masks, size(xs)[2])
    logLiks .= 0
    for m in 1:num_masks
        logL = logpdf(model, xs; reduction=identity, train=true)
        logLiks[m, :] = logL[:]
        update_maskset!(model)
    end
    logsumexp(logLiks .+ log(1.0f0 / num_masks), dims=1) |> reduction
end

## Andrej Karpathy's version averages the per-pixel probabilities over masks: biased (not a valid density)
function logpdf_test_karpathy(model::MADE, xs; reduction=mean)
    num_masks = model.num_masks
    liks = zeros32(size(xs)[1], size(xs)[2])
    for m in 1:num_masks
        xs_preds = model(xs)
        probs = sigmoid(xs_preds)
        pixel_likelihood = xs .* probs + (1 .- xs) .* (1 .- probs)
        liks += pixel_likelihood |> cpu
        update_maskset!(model)
    end
    sum(log.(liks / num_masks .+ 1.0f-10), dims=1) |> mean
end

## n samples, split evenly between the masks; each mask samples in its own input order
function sample(model::MADE, n=1; samples_only=true)
    total_d = model.d
    n_each_mask = Int(ceil(n / model.num_masks))
    xs_all = []
    ps_all = []
    for _ in 1:model.num_masks
        xs = zeros(Float32, total_d, n_each_mask) |> device
        ps_mat = similar(xs, Float32)
        mask_idx = model.current_mask_idx[]
        d_order = model.input_orders[mask_idx] |> cpu
        for d in sortperm(d_order)
            ps = σ(model(xs)[d, :])
            xs[d, :] = rand_like(ps) .< ps
            ps_mat[d, :] = ps
        end
        update_maskset!(model)
        push!(xs_all, xs)
        push!(ps_all, ps_mat)
    end
    xs_samples = hcat(xs_all...)[:, 1:n]
    if samples_only
        return xs_samples
    else
        return xs_samples, hcat(ps_all...)[:, 1:n]
    end
end

## connectivity matrices and the causality check for every mask in the pool
function compute_con_matrix(model::MADE)
    con_mats = []
    for i in 1:model.num_masks
        mat = compute_con_matrix(model.mask_pool[i], model.input_orders[i], model.res_con)
        push!(con_mats, mat)
    end
    return con_mats
end

function check_causal_mask(model::MADE)
    for i in 1:model.num_masks
        if !check_causal_mask(model.mask_pool[i], model.input_orders[i], model.res_con)
            return false
        end
    end
    return true
end
