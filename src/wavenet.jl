#=
File: wavenet.jl
Author: lf28
Description: autoregressive models of a sequence of 784 binary pixels (raster order):
             causal 1D convolutions, PixelCNN1D, WaveNet and the class-conditional WaveNet
License: MIT
Layout: sequences are (length, channels, batch)
=#


## ---------------------------------------------------------------------------------------------
## Causal 1D convolution: mask type A (input layer) hides x_t, type B (later layers) keeps it
## ---------------------------------------------------------------------------------------------

abstract type AbstractCausalConv1D end

## version 1: pad the input by hand on the left
struct CausalConv1D{C<:Conv,R} <: AbstractCausalConv1D
    conv::C
    res_con::Bool
    res_conv::R
    mask_type::Symbol
end

Flux.@layer CausalConv1D trainable = (conv,)

function CausalConv1D(kernel_size::Int, ch::Pair{<:Integer,<:Integer}, act=identity; mask_type::Symbol=:B, res_con=false, dilation=1, kwargs...)
    standard_conv = Conv((kernel_size,), ch, act; dilation=dilation, pad=0, kwargs...)
    res_con = res_con && (ch[1] == ch[2]) ## residual connection only if in and out channels match
    res_conv = Conv((1,), ch, identity)
    @assert mask_type in (:A, :B)
    return CausalConv1D(standard_conv, res_con, res_conv, mask_type)
end

function (m::CausalConv1D)(x::AbstractArray)
    k = size(m.conv.weight, 1)
    d = m.conv.dilation[1]
    left_pad = (k - 1) * d + (m.mask_type == :A)   # type A: one extra step, so x_t itself is never seen
    x_padded = Flux.pad_zeros(x, (left_pad, 0); dims=1)
    out = m.conv(x_padded)[1:size(x, 1), :, :]
    if m.res_con
        return m.res_conv(out) + x
    else
        return out
    end
end

## version 2: the same causal convolution, using Conv's own (left) padding
struct CausalConv1D2{C<:Conv,R} <: AbstractCausalConv1D
    conv::C
    res_con::Bool
    res_conv::R
    mask_type::Symbol
end

Flux.@layer CausalConv1D2 trainable = (conv,)

function CausalConv1D2(kernel_size::Int, ch::Pair{<:Int,<:Int}, act=identity; mask_type::Symbol=:B, res_con=false, dilation=1, kwargs...)
    @assert mask_type in (:A, :B)
    left_pad = (kernel_size - 1) * dilation + (mask_type == :A)
    standard_conv = Conv((kernel_size,), ch, act; dilation=dilation, pad=(left_pad, 0), kwargs...)
    res_con = res_con && (ch[1] == ch[2])
    res_conv = Conv((1,), ch, identity)
    return CausalConv1D2(standard_conv, res_con, res_conv, mask_type)
end

function (m::CausalConv1D2)(x::AbstractArray)
    out = m.conv(x)[1:size(x, 1), :, :]
    if m.res_con
        out = m.res_conv(out) + x
    end
    return out
end


## ---------------------------------------------------------------------------------------------
## Location feature: the position t / (T - 1) in [0, 1], appended as an extra input channel
## ---------------------------------------------------------------------------------------------

struct LocationInputWrapper
    add_loc::Bool
end

Flux.@layer LocationInputWrapper

function (m::LocationInputWrapper)(xs)
    if m.add_loc
        loc = Flux.ignore_derivatives() do
            location_feature(xs)
        end
        xs = cat(xs, loc, dims=2)
    else
        xs
    end
end

function location_feature(xs)
    d = size(xs, 1)
    if eltype(xs) == Bool
        xs = Float32.(xs)
    end
    loc_mat = similar(xs)
    loc_mat[:, :, 1:end] .= range(0.0f0, 1.0f0, d)
    return loc_mat
end


## ---------------------------------------------------------------------------------------------
## PixelCNN1D: a plain stack of causal convolutions, optionally dilated
## ---------------------------------------------------------------------------------------------

struct PixelCNN1D{T<:Chain}
    d::Int ## sequence length
    net::T ## trainable network
    kernel_size::Int
    dilation::Bool
    add_loc::Bool
    res_con::Bool
end

Flux.@layer PixelCNN1D trainable = (net,)

function PixelCNN1D(d::Int=28^2, num_layers=5, kernel_size::Int=3, nc::Int=32;
    input_ch::Int=1,
    res_con=false,
    add_loc=false,
    dilation=false,
    act=relu,
    built_in_padding=false,
    max_dilation::Int=512,
    kwargs...
)
    loc_input_wrapper = LocationInputWrapper(add_loc)
    net = Any[loc_input_wrapper]
    cap_l = floor(Int, log(kernel_size, max_dilation))
    dils = [2^(l - 1) for l in 1:(cap_l+1)]
    for l in 1:num_layers
        dil = dilation ? dils[((l-1)%(length(dils))+1)] : 1 ## dilations 1, 2, …, 512, then back to 1, 2, …
        dil = min(dil, 512)
        in_nc = l == 1 ? (add_loc ? input_ch + 1 : input_ch) : nc
        res_connection = l == 1 ? false : res_con ## no residual connection in the input layer
        mask_type = l == 1 ? :A : :B ## type A mask in the input layer
        if built_in_padding
            push!(net, CausalConv1D2(kernel_size, in_nc => nc, act; mask_type=mask_type, res_con=res_connection, dilation=dil, kwargs...))
        else
            push!(net, CausalConv1D(kernel_size, in_nc => nc, act; mask_type=mask_type, res_con=res_connection, dilation=dil, kwargs...))
        end
    end
    push!(net, Conv((1,), nc => 1; kwargs...)) ## output layer
    net = Chain(net...)
    return PixelCNN1D(d, net, kernel_size, dilation, add_loc, res_con)
end

function (m::PixelCNN1D)(xs)
    return m.net(xs)
end


## ---------------------------------------------------------------------------------------------
## WaveNet: gated residual blocks with dilated causal convolutions, and skip connections
## ---------------------------------------------------------------------------------------------

struct ResWaveNetBlock{T<:AbstractCausalConv1D,Y<:Conv}
    causal_con::T
    res_con::Y
end

Flux.@layer ResWaveNetBlock trainable = (causal_con, res_con)

function ResWaveNetBlock(kernel_size::Int=2, dilation::Int=1, ch::Int=32; kwargs...)
    causal_con = CausalConv1D(kernel_size, ch => 2 * ch, identity; mask_type=:B, res_con=false, dilation=dilation, kwargs...)
    res_con = Conv((1,), ch => ch, identity)
    ResWaveNetBlock(causal_con, res_con)
end

## returns (input + block output, block output); the second goes to the skip sum
function (m::ResWaveNetBlock)(xs)
    output = m.causal_con(xs)
    out_f, out_g = Flux.chunk(output, 2, dims=2)
    output = tanh.(out_f) .* σ(out_g)
    output = m.res_con(output)
    return xs + output, output
end

struct WaveNet{T,Y,K}
    d::Int ## sequence length
    input_layer::T
    res_blocks::Y
    output_layer::K
    add_loc::Bool
    dilation::Bool
    forward_mode::Symbol ## :chain (output from the last block) or :skip (output from the sum of skips)
end

Flux.@layer WaveNet trainable = (input_layer, res_blocks, output_layer)

function WaveNet(d::Int=28^2, num_layers=5, kernel_size::Int=2, nc::Int=32;
    forward_mode=:chain,
    add_loc=false,
    dilation=false,
    act=relu,
    kwargs...
)
    input_layer = Chain(
        LocationInputWrapper(add_loc),
        CausalConv1D(kernel_size, (add_loc ? 2 : 1) => nc, identity;
            mask_type=:A, ## causal mask A
            res_con=false,
            dilation=1,
            kwargs...
        )
    )
    res_blocks = []
    cap_l = floor(Int, log(kernel_size, 512))
    dils = [2^(l - 1) for l in 1:(cap_l+1)]
    for l in 1:num_layers
        dil = dilation ? dils[((l-1)%(length(dils))+1)] : 1 ## dilations 1, 2, …, 512, then back to 1, 2, …
        dil = min(dil, 512)
        push!(res_blocks, ResWaveNetBlock(kernel_size, dil, nc; kwargs...))
    end
    if forward_mode == :chain
        output_layer = Conv((1,), nc => 1; kwargs...)
    else
        out1 = Conv((1,), nc => nc, act; kwargs...) ## a 1 × 1 kernel needs no mask
        out2 = Conv((1,), nc => 1, identity; kwargs...)
        output_layer = Chain(act, out1, out2)
    end
    WaveNet(d, input_layer, Chain(res_blocks...), output_layer, add_loc, dilation, forward_mode)
end

function (m::WaveNet)(xs)
    hs = m.input_layer(xs) ## location feature and the type A mask
    out1, out2 = m.res_blocks[1](hs)
    total_skip = out2
    for layer in m.res_blocks[2:end]
        out1, out2 = layer(out1)
        total_skip += out2
    end
    if m.forward_mode == :chain
        output = m.output_layer(out1)
    else
        output = m.output_layer(total_skip)
    end
    return output
end


## ---------------------------------------------------------------------------------------------
## Class-conditional WaveNet: the one-hot label adds a learnt bias inside every gate
## ---------------------------------------------------------------------------------------------

struct ConResWaveNetBlock{T<:ResWaveNetBlock}
    res_block::T
    con_f::Dense
    con_g::Dense
end

Flux.@layer ConResWaveNetBlock trainable = (res_block, con_f, con_g)

function ConResWaveNetBlock(kernel_size::Int=2, dilation::Int=1, ch::Int=32, h_size::Int=5; kwargs...)
    res_block = ResWaveNetBlock(kernel_size, dilation, ch; kwargs...)
    con_f = Dense(h_size => ch; kwargs...)
    con_g = Dense(h_size => ch; kwargs...)
    ConResWaveNetBlock(res_block, con_f, con_g)
end

function (m::ConResWaveNetBlock)((xs, hs))
    output = m.res_block.causal_con(xs)
    out_f, out_g = Flux.chunk(output, 2, dims=2)
    hf = m.con_f(hs)
    hg = m.con_g(hs)
    output = tanh.(out_f .+ reshape(hf, 1, size(hf)...)) .* σ(out_g .+ reshape(hg, 1, size(hg)...))
    output = m.res_block.res_con(output)
    return xs + output, output
end

struct ConWaveNet{T,Y,K}
    d::Int ## sequence length
    num_classes::Int
    input_layer::T
    res_blocks::Y
    output_layer::K
    add_loc::Bool
    dilation::Bool
    forward_mode::Symbol ## :chain or :skip
end

Flux.@layer ConWaveNet trainable = (input_layer, res_blocks, output_layer)

function ConWaveNet(d::Int=28^2, num_layers=5, kernel_size::Int=2, nc::Int=32, num_classes::Int=10;
    forward_mode=:chain,
    add_loc=true,
    dilation=true,
    act=relu,
    max_dilation::Int=512,
    kwargs...
)
    input_layer = Chain(
        LocationInputWrapper(add_loc),
        CausalConv1D(kernel_size, (add_loc ? 2 : 1) => nc,
            identity;
            mask_type=:A, ## causal mask A
            res_con=false,
            dilation=1,
            kwargs...)
    )
    res_blocks = []
    cap_l = floor(Int, log(kernel_size, max_dilation))
    dils = [2^(l - 1) for l in 1:(cap_l+1)]
    for l in 1:num_layers
        dil = dilation ? dils[((l-1)%(length(dils))+1)] : 1
        dil = min(dil, max_dilation)
        push!(res_blocks, ConResWaveNetBlock(kernel_size, dil, nc, num_classes; kwargs...))
    end
    if forward_mode == :chain
        output_layer = Conv((1,), nc => 1; kwargs...)
    else
        out1 = Conv((1,), nc => nc, act; kwargs...)
        out2 = Conv((1,), nc => 1, identity; kwargs...)
        output_layer = Chain(out1, out2)
    end
    ConWaveNet(d, num_classes, input_layer, Chain(res_blocks...), output_layer, add_loc, dilation, forward_mode)
end

function (m::ConWaveNet)(xs, ys)
    outs = m.input_layer(xs)
    hs = Flux.onehotbatch(ys, 0:(m.num_classes-1))
    out1, out2 = m.res_blocks[1]((outs, hs)) ## both the sequence and the label
    total_skip = out2
    for layer in m.res_blocks[2:end]
        out1, out2 = layer((out1, hs)) ## the label enters every block
        total_skip += out2
    end
    if m.forward_mode == :chain
        output = m.output_layer(out1)
    else
        output = m.output_layer(total_skip)
    end
    return output
end


## ---------------------------------------------------------------------------------------------
## log-likelihood, sampling and classification by Bayes' rule
## ---------------------------------------------------------------------------------------------

const WaveNetModel = Union{PixelCNN1D,WaveNet}

## log p(x) per sequence
function logpdf(model::WaveNetModel, xs::AbstractArray; reduction=mean)
    xs_preds = model(xs)
    logL = -Flux.logitbinarycrossentropy(xs_preds[:, 1, :], xs[:, 1, :]; agg=identity)
    logL = sum(logL, dims=1)
    logL |> reduction
end

## unconditional model on a labelled batch: ignore the labels
logpdf(model::WaveNetModel, xs::AbstractArray, ys; reduction=mean) = logpdf(model, xs; reduction=reduction)

## log p(x | y)
function logpdf(model::ConWaveNet, xs::AbstractArray, ys::AbstractArray; reduction=mean)
    xs_preds = model(xs, ys)
    logL = -Flux.logitbinarycrossentropy(xs_preds[:, 1, :], xs[:, 1, :]; agg=identity)
    logL = sum(logL, dims=1)
    logL |> reduction
end

## one step per position; returns a 784 × n matrix
function sample(model::WaveNetModel, n=1)
    xs = Array{Float32}(undef, model.d, 1, n) |> device
    for d in 1:model.d
        ps = σ(model(xs)[d, :, :])
        xs[d, 1, :] = rand_like(ps)[:] .< ps[:]
    end
    return xs[:, 1, :]
end

## ys: one label for all n samples, or a vector of n labels
function sample(model::ConWaveNet, ys, n=1; samples_only=true)
    xs = Array{Float32}(undef, model.d, 1, n) |> device
    probs = similar(xs)
    if ys isa Number
        ys = repeat([ys], n)
    end
    for d in 1:model.d
        ps = σ(model(xs, ys)[d, :, :])
        xs[d, 1, :] = rand_like(ps)[:] .< ps[:]
        probs[d, 1, :] = ps
    end
    if samples_only
        return xs[:, 1, :]
    else
        return xs[:, 1, :], probs[:, 1, :]
    end
end

## p(y | x) ∝ p(x | y) for each class y (uniform prior)
function predict_y(model::ConWaveNet, xs::AbstractArray)
    logliks = []
    for y in 0:(model.num_classes-1)
        ys = fill!(similar(xs, typeof(y), (size(xs)[end],)), y)
        ll = logpdf(model, xs, ys; reduction=identity)
        push!(logliks, ll)
    end
    logliks = vcat(logliks...)
    postp = exp.(logliks .- logsumexp(logliks, dims=1))
    return postp
end

function predict_y(model::ConWaveNet, data_loader::Union{Flux.DataLoader,DeviceIterator})
    post_prob = []
    targets = []
    for (xs, ys) in data_loader
        postp = predict_y(model, xs)
        # copy to the CPU straight away, before the device iterator frees its buffers
        push!(post_prob, cpu(postp))
        push!(targets, cpu(ys))
    end
    return hcat(post_prob...), targets
end
