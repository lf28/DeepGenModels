#=
File: test/smoke_test.jl
Description: a quick check after changing the code: every model builds, trains for one short epoch on
             tiny FAKE data, gives a finite log-likelihood, samples, and is causal. About 5 minutes on a CPU.
             Runs on the GPU when one is available: every model is moved with `|> device`, like the data.
             Run from the repository root:  julia --project test/smoke_test.jl
             It works in a temporary folder, so ./datasets and ./checkpoints are not touched.
=#
root = dirname(@__DIR__)
include(joinpath(root, "setup.jl"))
for f in ["src/pixelcnn.jl", "src/wavenet.jl", "src/made.jl", "src/nade.jl", "src/convnade.jl", "src/toy_data.jl"]
    include(joinpath(root, f))
end
cd(mktempdir())
using Test

# fake binarised MNIST, in the temporary folder
mkpath("datasets")
using NPZ
Random.seed!(0)
npzwrite("datasets/binarized_mnist.npz", Dict("train_data" => UInt8.(rand(50000, 784) .< 0.2),
    "valid_data" => Float64.(rand(64, 784) .< 0.2), "test_data" => Float64.(rand(64, 784) .< 0.2)))

ok(x) = isfinite(x) && x < 0

# load_binary_mnist_data always uses 50,000 training images: keep only the first 256 for speed
function small_data(batch_size; tensor_shape)
    tr, te, va = load_binary_mnist_data(batch_size; tensor_shape=tensor_shape)
    xs = tr.data[1]
    small = collect(selectdim(xs, ndims(xs), 1:256))
    return Flux.DataLoader((small,), batchsize=batch_size, shuffle=true), te, va
end

@testset "refactor smoke test" begin
    @testset "PixelCNN" begin
        tr, te, va = small_data(32; tensor_shape=(28, 28, 1, :))
        m = PixelCNN2D((28, 28), 8, (3, 3), 2) |> device
        l, v, b = fit!(m, tr, Adam(), 1; valid_data=va, verbose=false)
        @test ok(logpdf(m, te |> device))
        m2 = PixelCNNTwoStack((28, 28), 8, 3, 3; add_loc=true, weight_norm=true) |> device
        l, v, b = fit!(m2, tr, Adam(), 1; valid_data=va, verbose=false, save_check_points=true, checkpoint_folder="smoke_pixelcnn")
        @test isfile("checkpoints/smoke_pixelcnn/model_epoch_0.jld2") && isfile("checkpoints/smoke_pixelcnn/model_epoch_1.jld2")
        @test ok(logpdf(m2, te |> device))
        m3 = PixelCNNTwoStack((28, 28), 8, 3, 3; vert_causal_style=:masked, hori_causal_style=:masked, add_loc=true) |> device
        @test ok(logpdf(m3, te |> device))
        m4 = PixelCNNTwoStack((28, 28), 8, 3, 3; gated=false) |> device
        @test ok(logpdf(m4, te |> device))
        # causality of the two-stack model: the output at (i, j) depends only on earlier pixels
        x = randn32(8, 8, 1, 1)
        net = PixelCNNTwoStack((8, 8), 4, 3, 3; act=identity, gated=true)
        for (i, j) in [(4, 4), (1, 5), (8, 1)]
            g = Flux.gradient(z -> net(z)[i, j, 1, 1], x)[1][:, :, 1, 1]
            for r in 1:8, c in 1:8
                if (r > i) || (r == i && c >= j)
                    @test g[r, c] == 0
                end
            end
        end
        s = sample(PixelCNNTwoStack((6, 6), 4, 3, 2) |> device, 3) |> cpu
        @test size(s) == (6, 6, 1, 3) && all(in((0, 1)), s)
        # conditional model on labelled batches
        xs = Float32.(rand(28, 28, 1, 64) .< 0.2)
        ys = rand(0:9, 64)
        lab = Flux.DataLoader((xs, ys), batchsize=32, shuffle=true)
        cm = ConPixelCNN(10, (28, 28), 8, 3, 3) |> device
        l = fit!(cm, lab, Adam(), 1; verbose=false)
        @test ok(logpdf(cm, lab |> device))
        @test ok(logpdf(m2, lab |> device))           # unconditional model ignores the labels
        cs = sample(ConPixelCNN(10, (6, 6), 4, 3, 2) |> device, [3, 4], 2)
        @test size(cs) == (6, 6, 1, 2)
        @test plot_receptive_field(net, x) isa Plots.Plot
    end

    @testset "dynamic binarisation" begin
        intensity = rand(Float32, 28, 28, 1, 64)
        d = DynamicBinaryMNIST(intensity, rand(0:9, 64))
        loader = Flux.DataLoader(d, batchsize=16, shuffle=true)
        (b1, y1) = first(loader)
        @test size(b1) == (28, 28, 1, 16) && eltype(b1) == Float32 && length(y1) == 16
        @test d[1:4][1] != d[1:4][1]                   # a fresh draw every time
        cm = ConPixelCNN(10, (28, 28), 4, 3, 2) |> device
        l = fit!(cm, loader, Adam(), 1; verbose=false)
        @test length(l) == 4
    end

    @testset "WaveNet" begin
        tr, te, va = small_data(32; tensor_shape=(28^2, 1, :))
        for m in [PixelCNN1D(28^2, 3, 2, 8; dilation=false, built_in_padding=true),
                  PixelCNN1D(28^2, 3, 2, 8; res_con=true, dilation=true, add_loc=true),
                  WaveNet(28^2, 4, 2, 8; add_loc=true, dilation=true),
                  WaveNet(28^2, 4, 2, 8; forward_mode=:skip, add_loc=true, dilation=true)]
            m = m |> device
            l, v, b = fit!(m, tr, Adam(), 1; valid_data=va, verbose=false)
            @test ok(logpdf(m, te |> device))
        end
        @test test_autoregressive_property_comprehensive(WaveNet(20, 4, 2, 4; add_loc=true, dilation=true); input_shape=(20, 1, 1))
        @test size(sample(WaveNet(30, 2, 2, 4) |> device, 3)) == (30, 3)
        xs = Float32.(rand(28^2, 1, 64) .< 0.2)
        ys = rand(0:9, 64)
        lab = Flux.DataLoader((xs, ys), batchsize=32)
        for fm in (:chain, :skip)
            cw = ConWaveNet(28^2, 3, 2, 8, 10; forward_mode=fm) |> device
            l = fit!(cw, lab, Adam(), 1; verbose=false)
            @test ok(logpdf(cw, lab |> device))
            pps, targets = predict_y(cw, lab |> device)
            @test size(pps) == (10, 64) && all(isapprox.(sum(pps, dims=1), 1; atol=1e-4))
        end
        @test size(sample(ConWaveNet(30, 2, 2, 4, 10) |> device, 3, 2)) == (30, 2)
    end

    @testset "MADE" begin
        tr, te, va = small_data(32; tensor_shape=(28^2, :))
        m = MADE(28^2, [64]; input_order=[1:784;], res_con=true, num_masks=1) |> device
        l, v, b = fit!(m, tr, Adam(), 1; valid_data=va, verbose=false)
        @test check_causal_mask(m)
        @test ok(logpdf(m, te |> device; train=false))
        me = MADE(28^2, [64, 64]; res_con=true, num_masks=4) |> device
        l, v, b = fit!(me, tr, Adam(), 1; valid_data=va, verbose=false)
        @test check_causal_mask(me)
        @test ok(logpdf(me, te.data[1] |> device; train=false))
        @test size(sample(me, 10)) == (784, 10)
        xs = Float32.(rand(784, 64) .< 0.2)
        lab = Flux.DataLoader((xs, rand(0:9, 64)), batchsize=32)
        l, v, b = fit!(me, lab, Adam(), 1; valid_data=lab, verbose=false)   # labelled batches: labels ignored
        @test length(l) == 2
    end

    @testset "DeepNADE / RNADE" begin
        tr, te, va = small_data(64; tensor_shape=(28^2, :))
        m = DeepNADE(28^2, true, [32]) |> device
        l, b = fit!(m, tr, Adam(0.001), 1; learning_rate_decay=true, validation_data=va.data[1], verbose=false)
        @test isfinite(logpdf_val(m, te.data[1][:, 1:8] |> device))
        @test ok(logpdf(m, te.data[1][:, 1:8] |> device; ensemble_size=2, one_order=false))
        @test size(sample(m, 2)) == (784, 2)
        Random.seed!(1)
        data = generate_moons(400; noise=0.06)
        loader = Flux.DataLoader((data,), batchsize=100, shuffle=true)
        r = DeepNADE(2, true, [32], relu; rnade=true, K=3) |> device
        l, _ = fit!(r, loader, Adam(0.001), 2; verbose=false)
        @test isfinite(logpdf(r, data |> device))
        @test size(sample(r, 5)) == (2, 5)
    end

    @testset "ConvNADE" begin
        tr, te, va = small_data(64; tensor_shape=(28, 28, 1, :))
        m = ConvNADE(28^2, true; smaller_net=true) |> device
        l, b = fit!(m, tr, Adam(3e-4), 1; verbose=false, learning_rate_decay=true)
        @test ok(logpdf(m, te.data[1][:, :, :, 1:4] |> device))
        @test ok(logpdf(m, Flux.DataLoader((te.data[1][:, :, :, 1:4],), batchsize=2); ensemble_size=2, one_order=false))
        @test size(sample(m, 2)) == (28, 28, 1, 2)
    end

    @testset "plotting" begin
        tr, te, va = small_data(32; tensor_shape=(28^2, :))
        @test vec2imgs(te.data[1][:, 1:4]; nrows=2, title="x") isa Plots.Plot
        @test plot_loss(rand(20), 5) isa Plots.Plot
        @test plot_loss_history(rand(16), rand(2), Flux.DataLoader(rand(2, 64), batchsize=8)) isa Plots.Plot
        figs = visualize_all_checkpoints("checkpoints/smoke_pixelcnn"; num_samples=4, nrows=2)
        @test length(figs) == 2 && isfile("checkpoints/smoke_pixelcnn/sampled_plots/epoch_0_grid.png")
    end
end
