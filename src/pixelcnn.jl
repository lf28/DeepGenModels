#=
File: pixelcnn.jl
Author: lf28
Description: PixelCNN for 28 × 28 binary images:
             masked / shifted causal convolutions, the single-stack PixelCNN2D,
             the gated two-stack PixelCNN (vertical + horizontal stacks) and its class-conditional version
License: MIT
Layout: images are (H, W, C, batch); raster-scan order, row by row
=#


## ---------------------------------------------------------------------------------------------
## Masked convolution: a standard Conv whose weights are multiplied by a fixed 0/1 mask
## ---------------------------------------------------------------------------------------------

struct MaskedConv{C,M}
    conv::C
    mask::M
end

Flux.@layer MaskedConv trainable = (conv,)

function MaskedConv(k_size::Tuple{Int,Int}, ch::Pair{Int,Int}, act::Function=identity; weight_norm=false, kwargs...)
    conv = Conv(k_size, ch, act; pad=:same, kwargs...)
    if weight_norm
        conv = WeightNorm(conv, :weight)
    end
    mask = similar(get_base_conv(conv).weight)
    mask .= 1 ## initialise all to 1
    MaskedConv(conv, mask)
end

function set_mask!(m::MaskedConv, mask)
    @assert size(m.mask) == size(mask)
    m.mask .= mask
end

function (m::MaskedConv)(x)
    base_conv = get_base_conv(m.conv)
    masked_weight = base_conv.weight .* m.mask
    (; pad, stride, dilation, groups) = base_conv
    ## flipped = true: PyTorch convolution (cross-correlation), so the mask is in reading order
    out = Flux.conv(x, masked_weight; flipped=true, pad, stride, dilation, groups)
    if !(typeof(base_conv.bias) == Bool && base_conv.bias == false)
        out = out .+ reshape(base_conv.bias, 1, 1, :, 1)
    end
    out = base_conv.σ.(out)
    return out
end

## the PixelCNN mask: rows above the centre, and the centre row up to the centre;
## type A also hides the centre pixel itself (input layer), type B keeps it (later layers)
function create_mask(conv, type::Symbol=:B)
    conv = get_base_conv(conv)
    mask = similar(conv.weight)
    mask .= 0 ## initialise all to zero
    kh, kw = size(mask)[1:2]
    @assert isodd(kw) && isodd(kh)
    central_idx1 = kh ÷ 2 + 1
    central_idx2 = kw ÷ 2 + 1
    mask[1:(central_idx1-1), :, :, :] .= 1
    mask[central_idx1, 1:central_idx2, :, :] .= 1
    if type == :A
        mask[central_idx1, central_idx2, :, :] .= 0
    end
    return mask
end

## the plain Conv inside any of the wrapper layers in this file
get_base_conv(layer::Conv) = layer
get_base_conv(layer::WeightNorm) = layer.layer
get_base_conv(layer::MaskedConv) = get_base_conv(layer.conv)


## ---------------------------------------------------------------------------------------------
## Location embedding: binary codes of the row and column index, appended as extra input channels
## ---------------------------------------------------------------------------------------------

function encode_pix_loc(n::Int, length::Int=5)
    @assert 0 <= n <= 2^length - 1 "Number must be between 0 and $(2^length - 1)"
    # digits(n, base=2, pad=5) returns the least significant bit first
    return digits(n, base=2, pad=length)
end

function create_loc_embedding(d::Int=28)
    emb_len = ceil(Int, log2(d + 1))   # enough bits for the indices 1, …, d
    loc_mat = zeros32(d, d, emb_len * 2)
    for i in 1:d
        loc_mat[i, :, 1:emb_len] .= encode_pix_loc(i, emb_len)'
        loc_mat[:, i, (emb_len+1):end] .= encode_pix_loc(i, emb_len)'
    end
    return loc_mat
end

struct LocationEmbInputWrapper{A}
    add_loc::Bool
    embedding::A
end

Flux.@layer LocationEmbInputWrapper trainable = (embedding,)

function LocationEmbInputWrapper(add_loc::Bool, d::Int)
    embedding = create_loc_embedding(d)
    LocationEmbInputWrapper(add_loc, embedding)
end

## (xs, ys_bias): pass the class bias through untouched
function (m::LocationEmbInputWrapper)((xs, ys_bias)::Tuple)
    xs = m(xs)
    if isnothing(ys_bias) ## no condition injected
        return xs
    else
        return xs, ys_bias
    end
end

function (m::LocationEmbInputWrapper)(xs::AbstractArray)
    if m.add_loc
        batch_size = size(xs)[end]
        emb = repeat(m.embedding, 1, 1, 1, batch_size)
        xs = cat(xs, emb, dims=3)
    end
    return xs
end


## ---------------------------------------------------------------------------------------------
## Single-stack causal 2D convolution: by shifting (padding) or by masking the kernel
## ---------------------------------------------------------------------------------------------

abstract type AbstractCausalConv2D end

## shift version: pad the top and the left, so the output at (i, j) only sees rows above and pixels to its left
struct CausalConv2D{C} <: AbstractCausalConv2D
    conv::C
    mask_type::Symbol
end

Flux.@layer CausalConv2D trainable = (conv,)

function CausalConv2D(k_size::Tuple{Int,Int}, ch::Pair{Int,Int}, act::Function=identity; weight_norm=false, mask_type::Symbol=:B, kwargs...)
    conv = Conv(k_size, ch, act; pad=0, kwargs...)
    if weight_norm
        conv = WeightNorm(conv, :weight)
    end
    CausalConv2D(conv, mask_type)
end

function (m::CausalConv2D)(xs)
    base_conv = get_base_conv(m.conv)
    kv, kh = size(base_conv.weight)[1:2]
    dv, dh = base_conv.dilation
    top_pad = (kv - 1) * dv
    left_pad = (kh - 1) * dh
    x_padded = Flux.pad_zeros(xs, (top_pad, 0, left_pad, 0); dims=(1, 2))
    out = m.conv(x_padded)
    if m.mask_type == :A
        out = Flux.pad_zeros(out, (1, 0); dims=2)[:, 1:(end-1), :, :]
    end
    return out
end

get_base_conv(layer::CausalConv2D) = get_base_conv(layer.conv)

## mask version: a `same` convolution with the PixelCNN mask on the kernel
struct CausalConv2DMasked{C} <: AbstractCausalConv2D
    conv::C
    mask_type::Symbol
end

Flux.@layer CausalConv2DMasked trainable = (conv,)

function CausalConv2DMasked(k_size::Tuple{Int,Int}, ch::Pair{Int,Int}, act::Function=identity; weight_norm=false, mask_type::Symbol=:B, kwargs...)
    conv = MaskedConv(k_size, ch, act; weight_norm=weight_norm, kwargs...)
    mask = create_mask(conv, mask_type)
    set_mask!(conv, mask)
    CausalConv2DMasked(conv, mask_type)
end

function (m::CausalConv2DMasked)(x)
    return m.conv(x)
end

get_base_conv(layer::CausalConv2DMasked) = get_base_conv(layer.conv)


## ---------------------------------------------------------------------------------------------
## PixelCNN2D: one stack of causal convolutions (has a blind spot)
## ---------------------------------------------------------------------------------------------

struct PixelCNN2D{T}
    d::Tuple{Int,Int} ## image size
    in_ch::Int
    net::T ## trainable network
    add_loc::Bool
end

Flux.@layer PixelCNN2D trainable = (net,)

function PixelCNN2D(d::Tuple{Int,Int}=(28, 28), ch_size::Int=64, k_size::Tuple{Int,Int}=(7, 7), num_layers::Int=4;
    causal_style::Symbol=:masked, # :masked or :shift
    input_ch::Int=1,
    output_ch::Int=1,
    add_loc=false,
    act=relu, kwargs...
)
    in_ch_with_loc = input_ch + add_loc * (2 * ceil(Int, log2(max(d...) + 1)))
    loc_wrapper = LocationEmbInputWrapper(add_loc, max(d...))
    net = Any[loc_wrapper]
    input_layer = if causal_style == :shift
        CausalConv2D(k_size, in_ch_with_loc => ch_size, act; mask_type=:A, kwargs...)
    else
        CausalConv2DMasked(k_size, in_ch_with_loc => ch_size, act; mask_type=:A, kwargs...)
    end
    push!(net, input_layer)
    for _ in 1:(num_layers-1)
        layer = if causal_style == :shift
            CausalConv2D(k_size, ch_size => ch_size, act; mask_type=:B, kwargs...)
        else
            CausalConv2DMasked(k_size, ch_size => ch_size, act; mask_type=:B, kwargs...)
        end
        push!(net, layer)
    end
    output_layer = if causal_style == :shift
        CausalConv2D(k_size, ch_size => output_ch, identity; mask_type=:B, kwargs...)
    else
        CausalConv2DMasked(k_size, ch_size => output_ch, identity; mask_type=:B, kwargs...)
    end
    push!(net, output_layer)
    net = Chain(net...)
    return PixelCNN2D(d, input_ch, net, add_loc)
end

function (m::PixelCNN2D)(xs)
    m.net(xs)
end


## ---------------------------------------------------------------------------------------------
## Two-stack PixelCNN (van den Oord et al., 2016): a vertical stack (rows above)
## and a horizontal stack (pixels to the left in the current row); no blind spot
## ---------------------------------------------------------------------------------------------

abstract type AbstractCausalConvVert end
abstract type AbstractCausalConvHori end

"""
Vertical causal convolution by masking: the kernel rows up to (and optionally including) the centre row
- include_mid_row: whether the centre row is kept
"""
struct CausalConvVertMasked{C} <: AbstractCausalConvVert
    conv::C
    include_mid_row::Bool
end

Flux.@layer CausalConvVertMasked trainable = (conv,)

function CausalConvVertMasked(k_size::Tuple{Int,Int}, ch::Pair{Int,Int}, act::Function=identity;
    weight_norm::Bool=false, include_mid_row::Bool=true, kwargs...)
    conv = MaskedConv(k_size, ch, act; weight_norm=weight_norm, kwargs...)
    kh, kw = k_size
    @assert isodd(kw) && isodd(kh)
    central_idx1 = kh ÷ 2 + 1
    conv.mask .= 0 ## reset the mask to 0s
    if include_mid_row
        conv.mask[1:central_idx1, :, :, :] .= 1
    else
        conv.mask[1:(central_idx1-1), :, :, :] .= 1
    end
    CausalConvVertMasked(conv, include_mid_row)
end

function (m::CausalConvVertMasked)(xs)
    return m.conv(xs)
end

get_base_conv(layer::CausalConvVertMasked) = get_base_conv(layer.conv)

"""
Horizontal causal convolution by masking: a 1 × k kernel over the current row, left of the pixel
- mask_type: :A excludes the current pixel, :B includes it
"""
struct CausalConvHoriMasked{C} <: AbstractCausalConvHori
    conv::C
    mask_type::Symbol
end

Flux.@layer CausalConvHoriMasked trainable = (conv,)

function CausalConvHoriMasked(kernel_size::Int, ch::Pair{<:Integer,<:Integer}, act=identity;
    mask_type::Symbol=:A, weight_norm::Bool=false, kwargs...)
    conv = CausalConv2DMasked((1, kernel_size), ch, act; mask_type=mask_type, weight_norm=weight_norm, kwargs...)
    CausalConvHoriMasked(conv, mask_type)
end

function (m::CausalConvHoriMasked)(x::AbstractArray)
    return m.conv(x)
end

get_base_conv(layer::CausalConvHoriMasked) = get_base_conv(layer.conv)

"""
Vertical causal convolution by shifting: pad the top by (k - 1) rows and both sides by half the width
- include_mid_row: whether the centre row is kept; if not, the output is shifted down one row
"""
struct CausalConvVert{C} <: AbstractCausalConvVert
    conv::C
    include_mid_row::Bool
end

Flux.@layer CausalConvVert trainable = (conv,)

function CausalConvVert(k_size::Tuple{Int,Int}, ch::Pair{Int,Int}, act::Function=identity;
    weight_norm::Bool=false, include_mid_row::Bool=true, kwargs...)
    conv = Conv(k_size, ch, act; pad=0, kwargs...) ## valid convolution; the input is padded by hand
    @assert isodd(k_size[1]) && isodd(k_size[2])
    if weight_norm
        conv = WeightNorm(conv, :weight)
        conv.g .= 1.0
    end
    CausalConvVert(conv, include_mid_row)
end

function (m::CausalConvVert)(xs)
    base_conv = get_base_conv(m.conv)
    kv, kh = size(base_conv.weight)[1:2]
    dv, dh = base_conv.dilation
    left_right_pad = ((kh - 1) * dh) ÷ 2
    top_pad = (kv - 1) * dv
    x_padded = Flux.pad_zeros(xs, (top_pad, 0, left_right_pad, left_right_pad); dims=(1, 2)) ## keeps the width
    out = m.conv(x_padded)
    if !m.include_mid_row
        out = Flux.pad_zeros(out, (1, 0); dims=1)[1:(end-1), :, :, :]
    end
    return out
end

get_base_conv(layer::CausalConvVert) = get_base_conv(layer.conv)

"""
Horizontal causal convolution by shifting: pad the left of the row
- mask_type: :A excludes the current pixel (one extra column of padding), :B includes it
"""
struct CausalConvHori{C} <: AbstractCausalConvHori
    conv::C
    mask_type::Symbol
end

Flux.@layer CausalConvHori trainable = (conv,)

function CausalConvHori(kernel_size::Int, ch::Pair{<:Integer,<:Integer}, act=identity;
    mask_type::Symbol=:A, weight_norm::Bool=false, kwargs...)
    conv = Conv((1, kernel_size), ch, act; pad=0, kwargs...)
    if weight_norm
        conv = WeightNorm(conv, :weight)
        conv.g .= 1.0
    end
    return CausalConvHori(conv, mask_type)
end

function (m::CausalConvHori)(x::AbstractArray)
    base_conv = get_base_conv(m.conv)
    k = size(base_conv.weight, 2)
    d = base_conv.dilation[2]
    left_pad = (k - 1) * d + (m.mask_type == :A)
    x_padded = Flux.pad_zeros(x, (left_pad, 0); dims=2)
    out = m.conv(x_padded)
    return out[:, 1:size(x, 2), :, :]
end

get_base_conv(layer::CausalConvHori) = get_base_conv(layer.conv)


## gated activation: tanh(f) ⊙ σ(g), the channels split in half; optional class bias added to f and g
struct GatedActivation
    channel_dim::Int ## which dimension to split into two
end

Flux.@layer GatedActivation trainable = ()

GatedActivation(; d::Int=3) = GatedActivation(d)

function (g::GatedActivation)(x, y_bias=nothing)
    total_channels = size(x, g.channel_dim)
    out_ch = total_channels ÷ 2
    f_path = x[:, :, 1:out_ch, :]
    g_path = x[:, :, (out_ch+1):end, :]
    if !isnothing(y_bias)
        bias_f, bias_g = Flux.chunk(y_bias, 2, dims=1)
        f_path = f_path .+ reshape(bias_f, 1, 1, out_ch, :)
        g_path = g_path .+ reshape(bias_g, 1, 1, out_ch, :)
    end
    return tanh.(f_path) .* σ.(g_path)
end


## one two-stack layer: vertical and horizontal convolutions, the vertical-to-horizontal link,
## the activation, and a residual connection on the horizontal stack
struct CausalPixCNNConv2D{V<:AbstractCausalConvVert,H<:AbstractCausalConvHori,C<:Conv}
    vert_conv::V
    hori_conv::H
    v2hori::C
    res_conv::C
    gated::Bool
    mask_type::Symbol
    act::Union{Function,GatedActivation}
end

Flux.@layer CausalPixCNNConv2D trainable = (vert_conv, hori_conv, v2hori, res_conv)

function CausalPixCNNConv2D(k_size::Int, ch::Pair{Int,Int}, act::Function=relu;
    vert_causal_style::Symbol=:shift, # :masked or :shift
    hori_causal_style::Symbol=:shift, # :masked or :shift
    weight_norm=true,
    gated::Bool=true,
    mask_type::Symbol=:B,
    kwargs...
)
    in_ch, out_ch = ch
    out_ch_inner = gated ? 2 * out_ch : out_ch
    vert_conv = if vert_causal_style == :shift
        CausalConvVert((k_size, k_size), in_ch => out_ch_inner, identity;
            weight_norm=weight_norm, include_mid_row=true, kwargs...)
    else
        CausalConvVertMasked((k_size, k_size), in_ch => out_ch_inner, identity;
            weight_norm=weight_norm, include_mid_row=true, kwargs...)
    end
    hori_conv = if hori_causal_style == :shift
        CausalConvHori(k_size, in_ch => out_ch_inner, identity;
            weight_norm=weight_norm, mask_type=mask_type, kwargs...)
    else
        CausalConvHoriMasked(k_size, in_ch => out_ch_inner, identity;
            weight_norm=weight_norm, mask_type=mask_type, kwargs...)
    end
    v2hori = Conv((1, 1), out_ch_inner => out_ch_inner, identity; kwargs...)
    res_conv = Conv((1, 1), out_ch => out_ch, identity; kwargs...)
    if gated ## replace the activation by the gated activation
        act = GatedActivation(3)
    end
    return CausalPixCNNConv2D(vert_conv, hori_conv, v2hori, res_conv, gated, mask_type, act)
end

## shift down one row, so the horizontal stack only receives rows strictly above
function down_shift(xs::AbstractArray)
    xs = xs[1:(end-1), :, :, :]
    return Flux.pad_zeros(xs, (1, 0), dims=1)
end

function (m::CausalPixCNNConv2D)((xh, xv)::Tuple, y_bias=nothing)
    out_h = m.hori_conv(xh)
    out_v = m.vert_conv(xv)
    out_h = out_h + m.v2hori(down_shift(out_v))
    if isnothing(y_bias)
        out_h = m.act(out_h)
        out_v = m.act(out_v)
    else
        # split the class bias between the two stacks
        bias_h, bias_v = Flux.chunk(y_bias, 2, dims=1)
        out_h = m.act(out_h, bias_h)
        out_v = m.act(out_v, bias_v)
    end
    out_h = m.res_conv(out_h)
    if m.mask_type == :A ## input layer: no residual connection, it would leak the current pixel
        out_h = out_h
    else
        out_h = out_h + xh
    end
    return out_h, out_v
end

function (m::CausalPixCNNConv2D)(xs::AbstractArray, y_bias=nothing)
    return m((xs, xs), y_bias)
end


struct PixelCNNTwoStack{Y<:Chain}
    d::Tuple{Int,Int} ## image size
    in_ch::Int
    net::Y ## trainable network
    output_layer::Conv
end

Flux.@layer PixelCNNTwoStack trainable = (net, output_layer)

function PixelCNNTwoStack(d::Tuple{Int,Int}=(28, 28), ch_size::Int=64, k_size::Int=3, num_layers::Int=4;
    input_ch::Int=1, output_ch::Int=1,
    vert_causal_style::Symbol=:shift, # :masked or :shift
    hori_causal_style::Symbol=:shift, # :masked or :shift
    weight_norm=false,
    gated=true,
    add_loc=false,
    act=relu,
    kwargs...
)
    in_ch_with_loc = input_ch + add_loc * (2 * ceil(Int, log2(max(d...) + 1)))
    loc_wrapper = LocationEmbInputWrapper(add_loc, max(d...))
    input_layer = Chain(loc=loc_wrapper, inpt_layer=CausalPixCNNConv2D(k_size, in_ch_with_loc => ch_size, act;
        vert_causal_style=vert_causal_style, hori_causal_style=hori_causal_style,
        weight_norm=weight_norm, gated=gated, mask_type=:A, kwargs...))
    net = Any[]
    for _ in 1:(num_layers-1)
        push!(net, CausalPixCNNConv2D(k_size, ch_size => ch_size, act;
            vert_causal_style=vert_causal_style, hori_causal_style=hori_causal_style,
            weight_norm=weight_norm, gated=gated, mask_type=:B, kwargs...))
    end
    net = Chain(input_layer, Chain(net...))
    output_layer = Conv((1, 1), ch_size => 1)
    return PixelCNNTwoStack(d, input_ch, net, output_layer)
end

function (m::PixelCNNTwoStack)(xs)
    outh, outv = m.net(xs)
    logits = m.output_layer(outh)
end


## ---------------------------------------------------------------------------------------------
## Class-conditional two-stack PixelCNN: one-hot label → a learnt bias for every gate of every layer
## ---------------------------------------------------------------------------------------------

struct ConPixelCNN{T,D}
    d::Tuple{Int,Int}
    con_blocks::T
    proj_con::D
    num_classes::Int
    num_layers::Int
end

Flux.@layer ConPixelCNN trainable = (con_blocks, proj_con)

function ConPixelCNN(num_classes::Int=10, d::Tuple{Int,Int}=(28, 28), ch_size::Int=32, k_size::Int=3, num_layers::Int=6;
    weight_norm::Bool=false,
    add_loc::Bool=true,
    kwargs...
)
    con_blocks = PixelCNNTwoStack(d, ch_size, k_size, num_layers;
        weight_norm=weight_norm,
        gated=true,
        add_loc=add_loc,
        kwargs...
    )
    ## per layer: 2 stacks × 2 gate halves × ch_size biases
    proj_con = Dense(num_classes => 4 * ch_size * num_layers; bias=false, kwargs...)
    return ConPixelCNN(d, con_blocks, proj_con, num_classes, num_layers)
end

function (m::ConPixelCNN)(xs, ys)
    hs = Flux.onehotbatch(ys, 0:(m.num_classes-1))
    bias_global = m.proj_con(hs) ## 4 * ch * layers rows
    bias_per_layers = Flux.chunk(bias_global, m.num_layers, dims=1)
    net = m.con_blocks.net
    output_layer = m.con_blocks.output_layer
    input_layer = net[1]
    hidden_layers = net[2]
    xs = input_layer[1](xs) ## location embedding
    outh, outv = input_layer[2](xs, bias_per_layers[1])
    for (i, layer) in enumerate(hidden_layers)
        (outh, outv) = layer((outh, outv), bias_per_layers[i+1])
    end
    logits = output_layer(outh)
end


## ---------------------------------------------------------------------------------------------
## log-likelihood and sampling
## ---------------------------------------------------------------------------------------------

const PixelCNNModel = Union{PixelCNN2D,PixelCNNTwoStack}

## log p(x) per image: Bernoulli log-likelihood summed over all pixels
function logpdf(model::PixelCNNModel, xs::AbstractArray; reduction=mean)
    xs_preds = model(xs)
    loss = Flux.logitbinarycrossentropy(xs_preds, xs; agg=identity)
    -sum(loss, dims=1:3) |> reduction
end

## unconditional model on a labelled batch: ignore the labels
logpdf(model::PixelCNNModel, xs::AbstractArray, ys; reduction=mean) = logpdf(model, xs; reduction=reduction)

## log p(x | y)
function logpdf(model::ConPixelCNN, xs::AbstractArray, ys; reduction=mean)
    xs_preds = model(xs, ys)
    loss = Flux.logitbinarycrossentropy(xs_preds, xs; agg=identity)
    -sum(loss, dims=1:3) |> reduction
end

## pixel by pixel in raster-scan order (raster_scan=false: column by column)
function sample(model::PixelCNNModel, n=1; raster_scan=true, samples_only=true)
    xs = Array{Float32}(undef, model.d..., model.in_ch, n)
    xs .= 0
    xs = xs |> device
    probs = similar(xs)
    for i in 1:model.d[1]
        for j in 1:model.d[2]
            if raster_scan
                ps = σ.(model(xs))[i, j, :, :]
                xs[i, j, :, :] = (rand_like(ps) .< ps)
                probs[i, j, :, :] = ps
            else
                ps = σ.(model(xs))[j, i, :, :]
                xs[j, i, :, :] = (rand_like(ps) .< ps)
                probs[j, i, :, :] = ps
            end
        end
    end
    if samples_only
        return xs
    else
        return xs, probs
    end
end

## ys: one label for all n samples, or a vector of n labels
function sample(model::ConPixelCNN, ys, n=1; raster_scan=true, samples_only=true)
    d = model.d
    xs = Array{Float32}(undef, d..., 1, n) |> device
    xs .= 0
    if ys isa Number
        ys = repeat([ys], n)
    end
    probs = similar(xs)
    for i in 1:d[1]
        for j in 1:d[2]
            if raster_scan
                ps = σ.(model(xs, ys))[i, j, :, :]
                xs[i, j, :, :] = (rand_like(ps) .< ps)
                probs[i, j, :, :] = ps
            else
                ps = σ.(model(xs, ys))[j, i, :, :]
                xs[j, i, :, :] = (rand_like(ps) .< ps)
                probs[j, i, :, :] = ps
            end
        end
    end
    if samples_only
        return xs
    else
        return xs, probs
    end
end
