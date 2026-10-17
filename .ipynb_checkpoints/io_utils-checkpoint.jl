#=
File: io_utils.jl
Author: lf28
Description: io utilities
License: MIT
=#



"""
    Transform matrix to tensor
"""
vec2tensor(s; H = 28, W = 28) = reshape(s, (H, W, 1, size(s)[end]))
"""
    Transform a tensor to matrix
"""
tensor2vec(t; H = 28, W = 28) = reshape(t, (H * W, size(t)[end]));




function load_binary_mnist_data(batch_size = 2000; tensor =false, seed =2345, train_size = 50000, url = "https://github.com/mgermain/MADE/releases/download/ICML2015/binarized_mnist.npz")
    Random.seed!(seed)
    # bmnist = npzread(download(url))
    bmnist = npzread("./datasets/binarized_mnist.npz")
    v2t_transform = tensor ? vec2tensor : identity
    mnist_binary_test = BitMatrix(bmnist["test_data"]') |> v2t_transform
    mnist_binary_train = BitMatrix(bmnist["train_data"]') |> v2t_transform
    mnist_binary_valid = BitMatrix(bmnist["valid_data"]') |> v2t_transform

    train_size = max(train_size, size(mnist_binary_train)[end])
    train_data = selectdim(mnist_binary_train, ndims(mnist_binary_train), 1:train_size) 
    mnist_binary_loader = Flux.DataLoader((train_data,), batchsize=2000, shuffle=true)
    return mnist_binary_loader, mnist_binary_test, mnist_binary_valid
end

function tensor2imgs(tensors; nrows=2, flip = false, npad = 2)
	tensors = tensors |> cpu
	tensors = dropdims(tensors, dims=3)
	if flip
		tensors = 1 .- tensors
	end
	mosaicview(Gray.(permuteddimsview(tensors, [2,1,3])), fillvalue=0, nrow = nrows, npad = npad, rowmajor=true)
end;


function vec2imgs(vecs; nrows=2, flip=false, npad=2, H=28, W=28)
    vecs = vecs |> cpu
    if ndims(vecs) == 1
        vecs = vecs'
    end
    n = size(vecs)[2]

    output = zeros(Float64, H, W, n)
    for i in 1:n
        output[:, :, i] = reshape(vecs[:, i], H, W)'
    end

    if flip
        output = 1 .- output
    end
    # return Gray.(output)
    mosaicview(Gray.(output), fillvalue=0.2, nrow=nrows, npad=npad, rowmajor=true)
end

"""
    plot 2d samples as scatter plot; true data and simulated data are super-imposed 
"""
function plot_samples(truedata, samples; title= "RNADE", ms1 = 4, ms2=3, alpha2 =0.8, fmt =:svg)
    truedata = truedata |> cpu
    samples = samples |> cpu
	plt = scatter(truedata[1, :], truedata[2, :], st=:scatter, markerstrokecolor=:white, alpha=1.0, title=title, ratio=1, label="real data", ms=ms1, framestyle =:none, fmt=fmt)
	plot!(samples[1,:], samples[2,:], st=:scatter, ms=ms2, alpha=alpha2, markerstrokecolor=:white, label="sampled data")
	return plt
end



