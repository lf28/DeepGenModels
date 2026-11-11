#=
File: setup.jl
Author: lf28
Description: packages, the compute device and the shared code.
             Every experiment script starts with `include("setup.jl")`;
             each script then includes the model file(s) it needs from src/.
License: MIT
=#

using Flux
using CUDA, cuDNN
using StatsPlots
using Plots
default(fontfamily="Computer Modern", framestyle=:box) # LaTeX-style
using Random
using LaTeXStrings
using Logging
using LoggingExtras
using ProgressMeter
using StatsBase
using LinearAlgebra
using ParameterSchedulers
using JLD2

## which GPU to use (1 = the first card); falls back to the CPU when CUDA is not available
gpu_id = 1

## `device` moves data and models: x |> device (shadows CUDA's own `device` function)
if CUDA.functional()
    CUDA.device!(gpu_id - 1)                  # CUDA counts cards from 0
    const device = Flux.gpu_device(gpu_id)    # Flux counts cards from 1
    CUDA.allowscalar(false)
    @info "🚀 Using GPU $(gpu_id): $(CUDA.name(CUDA.device()))"
else
    const device = Flux.cpu_device()
    @warn "💻 No GPU found. Falling back to host CPU."
end

include("src/training.jl")
include("src/data.jl")
include("src/plotting.jl")
include("src/checks.jl")

## hide cuDNN's repeated log messages
global_logger(cudnn_filter)
