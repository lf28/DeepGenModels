#=
File: generate_data.jl
Author: lf28
Description: simulate toy datasets mostly in 2D
License: MIT
=#

function shuffle_columns(M)
    shuffled_cols = shuffle(1:size(M, 2))  # e.g., [3, 1, 4, 2]
    # 2. Slice the matrix using the shuffled index order
    M_shuffled_cols = M[:, shuffled_cols]
end

"""
    generate_moons(n_samples; noise=0.05)
Generates two interlocking half-moons.
"""
function generate_moons(n_samples; noise=0.05)
    n_samples_out = div(n_samples, 2)
    n_samples_in = n_samples - n_samples_out
    
    # Outer moon
    t_out = range(0, pi, length=n_samples_out)
    x_out = cos.(t_out)
    y_out = sin.(t_out)
    
    # Inner moon
    t_in = range(0, pi, length=n_samples_in)
    x_in = 1.0 .- cos.(t_in)
    y_in = 0.5 .- sin.(t_in)
    
    X = vcat(hcat(x_out, y_out), hcat(x_in, y_in))
    X .+= noise .* randn(size(X))
    return Matrix(Float32.(X)') |> shuffle_columns
end

"""
    generate_circles(n_samples; noise=0.05, factor=0.5)
Generates concentric circles (outer and inner rings).
"""
function generate_circles(n_samples; noise=0.05, factor=0.5)
    n_samples_out = div(n_samples, 2)
    n_samples_in = n_samples - n_samples_out
    
    # Outer circle
    t_out = range(0, 2pi, length=n_samples_out)
    x_out = cos.(t_out)
    y_out = sin.(t_out)
    
    # Inner circle
    t_in = range(0, 2pi, length=n_samples_in)
    x_in = factor .* cos.(t_in)
    y_in = factor .* sin.(t_in)
    
    X = vcat(hcat(x_out, y_out), hcat(x_in, y_in))
    X .+= noise .* randn(size(X))
    return Matrix(Float32.(X)') |> shuffle_columns
end

"""
    generate_gaussian_ring(n_samples; n_modes=8, radius=2.0, std=0.1)
Generates a discrete ring of 8 isotropic Gaussian clusters.
"""
function generate_gaussian_ring(n_samples; n_modes=8, radius=2.0, noise=0.1)
    X = zeros(n_samples, 2)
    for i in 1:n_samples
        # Pick one of the 8 modes at random
        mode = rand(0:(n_modes - 1))
        angle = 2pi * mode / n_modes
        
        # Center of the selected Gaussian mode
        mean_x = radius * cos(angle)
        mean_y = radius * sin(angle)
        
        # Sample with standard deviation
        X[i, 1] = mean_x + noise * randn()
        X[i, 2] = mean_y + noise * randn()
    end
    return Matrix(Float32.(X)') |> shuffle_columns
end

"""
    generate_spiral(n_samples; noise=0.05)
Generates an Archimedean spiral dataset.
"""
function generate_spiral(n_samples; noise=0.05)
    t = range(0.5, 3pi, length=n_samples)
    x = 0.2 .* t .* cos.(t)
    y = 0.2 .* t .* sin.(t)
    
    X = hcat(x, y)
    X .+= noise .* randn(size(X))
    return Matrix(Float32.(X)') |> shuffle_columns
end


function generate_toy_dataset_1(n_samples=2_000)
    f(t) = t + 0.6f0 * sin(2.0f0π * t)
    ys = Float32[range(0, 1, length=n_samples);]
    # xs = Float32.(f.(ys) + rand(Uniform{Float32}(-0.1f0, 0.1f0), N))
    xs = Float32.(f.(ys) + (rand32(n_samples) * 0.2f0 .- 0.1f0))
    # xs = xs' |> Matrix
    # ys = ys' |> Matrix
    # train, test = splitobs((xs, ys), at=0.75, shuffle=true)
    # return train..., test...
    return [xs'; ys'] |> shuffle_columns
end


function generate_toy_dataset_2(n_samples=1_000)
    f₂(t) = 7.0f0 * sin(0.75f0 * t) + 0.5f0 * t
    # ys = Float32.(rand(Uniform(-10.5f0, 10.5f0), N))
    ys = Float32.(rand32(n_samples) * 10.5f0 * 2 .- 10.5f0)
    xs = Float32.(f₂.(ys) + randn(n_samples))
    # xs = xs' |> Matrix
    # ys = ys' |> Matrix
    # train, test = splitobs((xs, ys), at=0.75, shuffle=true)
    # return train..., test...
    return [xs'; ys'] |> shuffle_columns
end

