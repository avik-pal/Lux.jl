module Lux

# Accelerator Support
using CUDA
using CUDA.CUDNN
# Neural Network Backend
using NNlib
import NNlibCUDA: batchnorm, ∇batchnorm, CUDNNFloat
# Julia StdLibs
using Random, Statistics, LinearAlgebra, SparseArrays
# Parameter Manipulation
using Functors, Setfield
import Adapt: adapt, adapt_storage
# Arrays
using FillArrays, ComponentArrays
# Automatic Differentiation
using ChainRulesCore, Zygote
# Optional Dependency
using Requires
# Docstrings
using Markdown
# Optimisers + ComponentArrays
using Optimisers

const use_cuda = Ref{Union{Nothing, Bool}}(nothing)

# Utilities
include("utils.jl")
# Core
include("core.jl")
# Data Transfer Utilities
include("adapt.jl")
# Layer Implementations
include("layers/basic.jl")
include("layers/normalize.jl")
include("layers/conv.jl")
include("layers/dropout.jl")
include("layers/recurrent.jl")
# Neural Network Backend
include("nnlib.jl")
# Pretty Printing
include("layers/display.jl")
# AutoDiff
include("autodiff.jl")
# Flux to Lux
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" include("transform.jl")
end
# Contrib
include("contrib/emptyarrays.jl")

# Deprecations
include("deprecated.jl")

# Data Transfer
export cpu, gpu
# Layers
export Chain, Parallel, SkipConnection, PairwiseFusion, BranchLayer
export Dense, Scale
export Conv, MaxPool, MeanPool, GlobalMaxPool, GlobalMeanPool, AdaptiveMaxPool,
       AdaptiveMeanPool, Upsample
export Dropout, VariationalHiddenDropout
export BatchNorm, GroupNorm
export WeightNorm
export NoOpLayer, ReshapeLayer, SelectDim, FlattenLayer, WrappedFunction, ActivationFunction
export RNNCell, LSTMCell, GRUCell
export SamePad

end
