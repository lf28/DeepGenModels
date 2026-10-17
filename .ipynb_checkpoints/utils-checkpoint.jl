#=
File: utils.jl
Author: lf28
Description: utilities for training 
License: MIT
=#

function randn_like(template_array)
    # Allocates the exact same array type (CPU Vector or CuVector etc.)
    out = similar(template_array) 
    Random.randn!(out)
    return out
end


function rand_like(template_array)
    # Allocates the exact same array type (CPU Vector or CuVector etc.)
    out = similar(template_array) 
    Random.rand!(out)
    return out
end



WarmupLinear(startlr, initlr, warmup, total_iters, schedule) =
    Sequence(Triangle(l0=startlr, l1=initlr, period=2 * warmup) => warmup,
        schedule => total_iters)


# Intercepts logs and drops anything containing "cuDNN"
cudnn_filter = ActiveFilteredLogger(global_logger()) do log
    !occursin("cuDNN", string(log._module))
end
