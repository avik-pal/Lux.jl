# TODO(@avik-pal): Eventually we want to move all these functions and their adjoints to NNlib.jl

# Normalization Implementation
@inline function update_statistics(x::AbstractArray{T, N},
                                   running_mean::AbstractArray{T, N},
                                   running_var::AbstractArray{T, N},
                                   batchmean::AbstractArray{T, N},
                                   batchvar::AbstractArray{T, N}, momentum::T,
                                   ::Val{reduce_dims}) where {T, N, reduce_dims}
    sx = size(x)
    m = T(prod((sx[i] for i in reduce_dims)))
    if reduce_dims[end] != N
        batchmean = mean(batchmean; dims=N)
        batchvar = mean(batchvar; dims=N)
    end
    running_mean = @. (1 - momentum) * running_mean + momentum * batchmean
    running_var = @. (1 - momentum) * running_var + momentum * batchvar * (m / (m - one(m)))
    return (running_mean, running_var)
end

"""
    normalization(x, running_mean, running_var, scale, bias, activation, ::Val{reduce_dims},
                  ::Val{training}, momentum, epsilon)

Performs BatchNorm/GroupNorm/InstanceNorm based on input configuration

!!! note
    
    Detailed docs are WIP
"""
@inline function normalization(x::AbstractArray{T, N},
                               running_mean::Union{Nothing, AbstractVector{T}},
                               running_var::Union{Nothing, AbstractVector{T}},
                               scale::Union{Nothing, AbstractVector{T}},
                               bias::Union{Nothing, AbstractVector{T}}, activation,
                               reduce_dims, t::Val, momentum::T=T(0.1),
                               epsilon::T=T(1e-5)) where {T, N}
    running_mean_reshaped = _reshape_into_proper_shape(running_mean, x)
    running_var_reshaped = _reshape_into_proper_shape(running_var, x)
    scale_reshaped = _reshape_into_proper_shape(scale, x)
    bias_reshaped = _reshape_into_proper_shape(bias, x)
    x_norm, running_mean_, running_var_ = normalization_forward(x, running_mean_reshaped,
                                                                running_var_reshaped,
                                                                scale_reshaped,
                                                                bias_reshaped, activation,
                                                                reduce_dims, t, momentum,
                                                                epsilon)
    return x_norm, _safe_vec(running_mean_), _safe_vec(running_var_)
end

@generated function normalization_forward(x::AbstractArray{T, N}, running_mean::RM,
                                          running_var::RV, scale::S, bias::B, activation::A,
                                          r::Val{reduce_dims}, ::Val{training},
                                          momentum::T=T(0.1f0),
                                          epsilon::T=T(1.0f-5)) where {RM, RV, S, B, T, N,
                                                                       A, training,
                                                                       reduce_dims}
    calls = []
    if !training
        if RM == Nothing
            expr = :(batchmean = mean(x; dims=reduce_dims);
                     batchvar = var(x; mean=batchmean, dims=reduce_dims, corrected=false))
        else
            expr = :(batchmean = running_mean;
                     batchvar = running_var)
        end
        push!(calls, expr)
    else
        expr = :(batchmean = mean(x; dims=reduce_dims);
                 batchvar = var(x; mean=batchmean, dims=reduce_dims, corrected=false))
        push!(calls, expr)

        if RM != Nothing
            push!(calls,
                  :((running_mean, running_var) = update_statistics(x, running_mean,
                                                                    running_var, batchmean,
                                                                    batchvar, momentum, r)))
        end
    end

    expr = if S != Nothing
        if A == typeof(identity)
            :(result = (scale .* (x .- batchmean) ./ sqrt.(batchvar .+ epsilon) .+ bias))
        else
            :(result = activation.(scale .* (x .- batchmean) ./
                                   sqrt.(batchvar .+ epsilon) .+ bias))
        end
    else
        if A == typeof(identity)
            :(result = ((x .- batchmean) ./ sqrt.(batchvar .+ epsilon)))
        else
            :(result = activation.((x .- batchmean) ./ sqrt.(batchvar .+ epsilon)))
        end
    end
    push!(calls, expr)
    push!(calls, :(return (result, running_mean, running_var)))

    return Expr(:block, calls...)
end

# Convolution
@inline conv_wrapper(x, weight, cdims) = conv(x, weight, cdims)

@inline function conv_wrapper(x::SubArray{T, N, <:CuArray}, weight, cdims) where {T, N}
    return conv(copy(x), weight, cdims)
end

# Dropout
@inline _dropout_shape(s, ::Colon) = size(s)
@inline function _dropout_shape(s, dims)
    return tuple((i ∉ dims ? 1 : si for (i, si) in enumerate(size(s)))...)
end

@inline _dropout_kernel(y::T, p, q) where {T} = y > p ? q : zero(T)

@inline function generate_dropout_mask(rng::AbstractRNG, x, p, q; dims=:)
    realfptype = float(real(eltype(x)))
    y = rand!(rng, similar(x, realfptype, _dropout_shape(x, dims)))
    y .= _dropout_kernel.(y, p, q)
    return y
end

"""
    dropout(rng::AbstractRNG, x, p, q, dims, ::Val{training})
    dropout(rng::AbstractRNG, x, mask, p, q, dims, t::Val{training}, ::Val{update_mask})

If `training` then dropout is applied on `x` with probability `p` along `dims`. If `mask` is
passed it is used if `update_mask` is false. If `update_mask` is true then the mask is
generated and used.
"""
@inline @generated function dropout(rng::AbstractRNG, x, p, q, dims,
                                    ::Val{training}) where {training}
    if training
        return :(rng = replicate(rng);
                 mask = generate_dropout_mask(rng, x, p, q; dims);
                 return (elementwise_mul(x, ignore_derivatives(mask)), mask, rng))
    else
        return :(return (x, x, rng))
    end
end

@inline @generated function dropout(rng::AbstractRNG, x, mask, p, q, dims, t::Val{training},
                                    ::Val{update_mask}) where {training, update_mask}
    if update_mask
        return :((y, mask, rng) = dropout(rng, x, p, q, dims, t);
                 return (y, mask, rng, Val(false)))
    else
        if training
            return :(size(x, ndims(x)) != size(mask, ndims(x)) &&
                         return (dropout(rng, x, p, q, dims, t)..., Val(false));
                     return (elementwise_mul(x, ignore_derivatives(mask)), mask, rng,
                             Val(false)))
        else
            return :(return (x, mask, rng, Val(false)))
        end
    end
end

# Adaptive Pooling
@inline function compute_adaptive_pooling_dims(x::AbstractArray, outsize)
    insize = size(x)[1:(end - 2)]
    stride = insize .÷ outsize
    k = insize .- (outsize .- 1) .* stride
    pad = 0
    return PoolDims(x, k; padding=pad, stride=stride)
end

# CUDNN Constants
const cudnnValidActivationTypes = Union{typeof(tanh), typeof(sigmoid), typeof(relu),
                                        typeof(elu), typeof(tanh_fast), typeof(sigmoid_fast)
                                        }

# Activation Functions
## I think this is handled by NNlibCUDA. But currently leaving here for
## benchmarking larger models
function getCUDNNActivationMode(::Union{typeof(tanh), typeof(tanh_fast)})
    return CUDNN.CUDNN_ACTIVATION_TANH
end
function getCUDNNActivationMode(::Union{typeof(sigmoid), typeof(sigmoid_fast)})
    return CUDNN.CUDNN_ACTIVATION_SIGMOID
end
getCUDNNActivationMode(::Union{typeof(relu)}) = CUDNN.CUDNN_ACTIVATION_RELU
getCUDNNActivationMode(::Union{typeof(elu)}) = CUDNN.CUDNN_ACTIVATION_ELU

"""
    applyactivation(f::Function, x::AbstractArray)

Apply the function `f` on `x` elementwise, i.e. `f.(x)`. Dispatches to CUDNN if possible.
"""
@inline applyactivation(f::Function, x::AbstractArray) = f.(x)
@inline function applyactivation(f::cudnnValidActivationTypes, x::CuArray{<:CUDNNFloat})
    return CUDNN.cudnnActivationForward(x; mode=getCUDNNActivationMode(f))
end
@inline applyactivation(::typeof(identity), x::AbstractArray) = x

# Dispatch Certain Broadcasted Functions to CUDNN
@inline function broadcast_shape_pullback(x, Δ)
    sx = size(x)
    sΔ = size(Δ)
    sx == sΔ && return Δ
    return sum(Δ; dims=findall(sx .!= sΔ))
end

@inline isvalidtensorop(x1, x2) = false
@inline function isvalidtensorop(x1::CuArray{N, T},
                                 x2::CuArray{N, T}) where {N, T <: CUDNNFloat}
    return ndims(x1) <= 5 &&
           (all(size(x2, i) == size(x1, i) || size(x2, i) == 1 for i in 1:ndims(x2)))
end

"""
    elementwise_add(x, y)

Computes `x .+ y`. Dispatches to CUDNN if possible
"""
@inline elementwise_add(x, y) = x .+ y
@inline function elementwise_add(x::CuArray, y::CuArray)
    !isvalidtensorop(x, y) && return x .+ y
    return cudnnOpTensorWithDefaults(x, y; op=CUDNN.CUDNN_OP_TENSOR_ADD)
end

@inline function elementwise_add_pullback(x, y, Δ)
    return broadcast_shape_pullback(x, Δ), broadcast_shape_pullback(y, Δ)
end

"""
    elementwise_mul(x, y)

Computes `x .* y`. Dispatches to CUDNN if possible
"""
@inline elementwise_mul(x, y) = x .* y
@inline function elementwise_mul(x::CuArray, y::CuArray)
    !isvalidtensorop(x, y) && return x .* y
    return cudnnOpTensorWithDefaults(x, y; op=CUDNN.CUDNN_OP_TENSOR_MUL)
end

@inline function elementwise_mul_pullback(x, y, Δ)
    return broadcast_shape_pullback(x, elementwise_mul(Δ, y)),
           broadcast_shape_pullback(y, elementwise_mul(Δ, x))
end

# CUDNN Helpers
function cudnnOpTensorWithDefaults(x1, x2; y=similar(x1),
                                   op::CUDNN.cudnnOpTensorOp_t=CUDNN.CUDNN_OP_TENSOR_ADD,
                                   compType::DataType=(eltype(x1) <: Float64 ? Float64 :
                                                       Float32),
                                   nanOpt::CUDNN.cudnnNanPropagation_t=CUDNN.CUDNN_NOT_PROPAGATE_NAN,
                                   opTensorDesc::CUDNN.cudnnOpTensorDescriptor=CUDNN.cudnnOpTensorDescriptor(op,
                                                                                                             CUDNN.cudnnDataType(compType),
                                                                                                             nanOpt),
                                   alpha1::Real=1, alpha2::Real=1, beta::Real=0,
                                   x1Desc::CUDNN.cudnnTensorDescriptor=CUDNN.cudnnTensorDescriptor(x1),
                                   x2Desc::CUDNN.cudnnTensorDescriptor=CUDNN.cudnnTensorDescriptor(x2),
                                   yDesc::CUDNN.cudnnTensorDescriptor=CUDNN.cudnnTensorDescriptor(y))
    T = eltype(x1)
    alpha1, alpha2, beta = CUDNN.scalingParameter.((T,), (alpha1, alpha2, beta))
    return CUDNN.cudnnOpTensorAD(x1, x2; opTensorDesc, alpha1, x1Desc, alpha2, x2Desc, beta,
                                 yDesc, y)
end

function cudnnActivationBackward(y::CuArray{T}, Δ::CuArray{T}, x::CuArray{T};
                                 mode) where {T}
    Δx = similar(x)
    desc = CUDNN.cudnnActivationDescriptor(mode, CUDNN.CUDNN_NOT_PROPAGATE_NAN, Cdouble(1))
    CUDNN.cudnnActivationBackward(CUDNN.handle(), desc, CUDNN.scalingParameter(T, 1),
                                  CUDNN.cudnnTensorDescriptor(y), y,
                                  CUDNN.cudnnTensorDescriptor(Δ), Δ,
                                  CUDNN.cudnnTensorDescriptor(x), x,
                                  CUDNN.scalingParameter(T, 0),
                                  CUDNN.cudnnTensorDescriptor(Δx), Δx)
    return Δx
end
