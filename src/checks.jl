#=
File: checks.jl
Author: lf28
Description: checking the autoregressive property with autograd:
             output t may only depend on inputs before t
License: MIT
=#

"""
    test_autoregressive_property_comprehensive(model; input_shape=(T, C, N), time_dim=1)

Vigorously tests the autoregressive property across EVERY time step in a sequence.
Returns `true` if the model is perfectly causal across all points, or `false` if a leak occurs.
"""
function test_autoregressive_property_comprehensive(model; input_shape=(10, 1, 1), time_dim=1)
    # 1. Generate a dummy input tensor with random floats
    x = rand(Float32, input_shape...)
    T = size(x, time_dim) # Number of time steps

    # 2. Iterate through every single output time step 't_out'
    for t_out in 1:T

        # 3. Compute gradients of the output at step t_out with respect to ALL inputs
        grads = Flux.gradient(x) do input_tensor
            out = model(input_tensor)

            # Isolate the output slice at this specific time step
            out_at_t = selectdim(out, time_dim, t_out)

            # Sum to scalar objective for AD tracking
            return sum(out_at_t)
        end

        # Extract the gradient array back from the gradients tuple
        grad_x = grads[1]

        # 4. Check every input time step 't_in' against our current 't_out'
        for t_in in 1:T
            grad_slice = selectdim(grad_x, time_dim, t_in)
            max_grad = maximum(abs, grad_slice)

            # CRITICAL PROPERTY: If t_in occurs AFTER t_out, the gradient MUST be 0.
            if t_in > t_out
                if max_grad > 1e-7
                    @error "🚨 Future Leakage Found!" t_out t_in max_grad
                    return false
                end
            end
        end
    end

    println("✅ Perfect Causality: Verified across all time steps.")
    return true
end


## the pixels that the output at the centre pixel depends on (non-zero gradient)
function plot_receptive_field(model, xs)
    H = max(size(xs)[1:2]...)
    central_pixel = H ÷ 2 + 1
    gx = Flux.gradient(xs) do xs
        out = model(xs)
        out[central_pixel, central_pixel, 1, 1]
    end

    grad_img = Float32.((abs.(gx[1]) .> 1f-8)[:, :, 1, 1])
    ## the centre pixel: 1 if it influences its own output (a leak), 0.5 otherwise

    if grad_img[central_pixel, central_pixel] > 0
        grad_img[central_pixel, central_pixel] = 1
    else
        grad_img[central_pixel, central_pixel] = 0.5
    end
    layer_len = 1

    if hasproperty(model, :net)
        layer_len = length(model.net)-1
    end
    heatmap(grad_img, yflip=true, ratio=1, xlims=[1, H], ylims=[1, H], legend=false, c=:viridis, title="Receptive field from the centre with $(layer_len) layers")
end
