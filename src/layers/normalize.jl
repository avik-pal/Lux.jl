abstract type AbstractNormalizationLayer{affine, track_stats} <: AbstractExplicitLayer end

"""
    BatchNorm(chs::Integer, activation=identity; init_bias=zeros32, init_scale=ones32, affine=true, track_stats=true, epsilon=1f-5, momentum=0.1f0)

[Batch Normalization](https://arxiv.org/abs/1502.03167) layer.

`BatchNorm` computes the mean and variance for each `D_1×...×D_{N-2}×1×D_N` input slice and normalises the input accordingly.

## Arguments

* `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions, call the `N-1`th the channel dimension. For a batch of feature vectors this is just the data dimension, for `WHCN` images it's the usual channel dimension.
* `activation`: After normalisation, elementwise activation `activation` is applied.

## Keyword Arguments

* If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias bias and scale scale parameters.
    - `init_bias`: Controls how the `bias` is initiliazed
    - `init_scale`: Controls how the `scale` is initiliazed
* If `track_stats=true`, accumulates mean and variance statistics in training phase that will be used to renormalize the input in test phase.
* `epsilon`: a value added to the denominator for numerical stability
* `momentum`:  the value used for the `running_mean` and `running_var` computation

## Inputs

* `x`: Array where `size(x, N - 1) = chs` and `ndims(x) > 2`

## Returns

* `y`: Normalized Array
* Update model state

## Parameters

* `affine=true`
    * `bias`: Bias of shape `(chs,)` 
    * `scale`: Scale of shape `(chs,)`
* `affine=false` - Empty `NamedTuple()`

## States

* Statistics if `track_stats=true`
    - `running_mean`: Running mean of shape `(chs,)`
    - `running_var`: Running variance of shape `(chs,)`
* Statistics if `track_stats=false`
    - `running_mean`: nothing
    - `running_var`: nothing
* `training`: Used to check if training/inference mode

Use [`Lux.testmode`](@ref) during inference.

## Example

```julia
m = Chain(
    Dense(784 => 64),
    BatchNorm(64, relu),
    Dense(64 => 10),
    BatchNorm(10)
)
```

See also [`GroupNorm`](@ref)
"""
struct BatchNorm{affine, track_stats, F1, F2, F3, N} <:
       AbstractNormalizationLayer{affine, track_stats}
    activation::F1
    epsilon::N
    momentum::N
    chs::Int
    init_bias::F2
    init_scale::F3
end

function BatchNorm(chs::Int,
                   activation=identity;
                   init_bias=zeros32,
                   init_scale=ones32,
                   affine::Bool=true,
                   track_stats::Bool=true,
                   epsilon=1.0f-5,
                   momentum=0.1f0)
    activation = NNlib.fast_act(activation)
    return BatchNorm{affine, track_stats, typeof(activation), typeof(init_bias),
                     typeof(init_scale), typeof(epsilon)}(activation, epsilon, momentum,
                                                          chs, init_bias, init_scale)
end

function initialparameters(rng::AbstractRNG, l::BatchNorm{affine}) where {affine}
    return affine ? (scale=l.init_scale(rng, l.chs), bias=l.init_bias(rng, l.chs)) :
           NamedTuple()
end
function initialstates(rng::AbstractRNG,
                       l::BatchNorm{affine, track_stats}) where {affine, track_stats}
    return if track_stats
        (running_mean=zeros32(rng, l.chs), running_var=ones32(rng, l.chs),
         training=Val(true))
    else
        (running_mean=nothing, running_var=nothing, training=Val(true))
    end
end

parameterlength(l::BatchNorm{affine}) where {affine} = affine ? (l.chs * 2) : 0
function statelength(l::BatchNorm{affine, track_stats}) where {affine, track_stats}
    (track_stats ? 2 * l.chs : 0) + 1
end

function (BN::BatchNorm)(x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    @assert size(x, N - 1) == BN.chs
    @assert !istraining(st)||size(x, N) > 1 "During `training`, `BatchNorm` can't handle Batch Size == 1"

    x_normalized, xmean, xvar = normalization(x,
                                              st.running_mean,
                                              st.running_var,
                                              ps.scale,
                                              ps.bias,
                                              BN.activation,
                                              collect([1:(N - 2); N]),
                                              st.training,
                                              BN.momentum,
                                              BN.epsilon)

    st = merge(st, (running_mean=xmean, running_var=xvar))

    return x_normalized, st
end

function (BN::BatchNorm{affine, track_stats})(x::Union{CuArray{T, 2}, CuArray{T, 4},
                                                       CuArray{T, 5}}, ps,
                                              st::NamedTuple) where {
                                                                     T <:
                                                                     Union{Float32, Float64
                                                                           }, affine,
                                                                     track_stats}
    # NNlibCUDA silently updates running_mean and running_var so copying them
    if istraining(st)
        running_mean2 = track_stats ? copy(st.running_mean) : nothing
        running_var2 = track_stats ? copy(st.running_var) : nothing
    else
        if track_stats
            running_mean2 = copy(st.running_mean)
            running_var2 = copy(st.running_var)
        else
            N = ndims(x)
            reduce_dims = collect([1:(N - 2); N])
            running_mean2 = mean(x; dims=reduce_dims)
            running_var2 = var(x; mean=running_mean2, dims=reduce_dims,
                               corrected=false)
        end
    end
    res = applyactivation(BN.activation,
                          batchnorm(affine ? ps.scale : nothing,
                                    affine ? ps.bias : nothing,
                                    x,
                                    running_mean2,
                                    running_var2,
                                    BN.momentum;
                                    eps=BN.epsilon,
                                    training=istraining(st)))
    if track_stats
        st = merge(st, (running_mean=running_mean2, running_var=running_var2))
    end
    return res, st
end

function Base.show(io::IO, l::BatchNorm{affine, track_stats}) where {affine, track_stats}
    print(io, "BatchNorm($(l.chs)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    affine || print(io, ", affine=false")
    track_stats || print(io, ", track_stats=false")
    return print(io, ")")
end

"""
    GroupNorm(chs::Integer, groups::Integer, activation=identity; init_bias=zeros32, init_scale=ones32, affine=true, track_stats=false, epsilon=1f-5, momentum=0.1f0)

[Group Normalization](https://arxiv.org/abs/1803.08494) layer.

## Arguments

* `chs`: Size of the channel dimension in your data. Given an array with `N` dimensions, call the `N-1`th the channel dimension. For a batch of feature vectors this is just the data dimension, for `WHCN` images it's the usual channel dimension.
* `groups` is the number of groups along which the statistics are computed. The number of channels must be an integer multiple of the number of groups.
* `activation`: After normalisation, elementwise activation `activation` is applied.

## Keyword Arguments

* If `affine=true`, it also applies  a shift and a rescale to the input through to learnable per-channel bias bias and scale scale parameters.
    - `init_bias`: Controls how the `bias` is initiliazed
    - `init_scale`: Controls how the `scale` is initiliazed
* If `track_stats=true`, accumulates mean and variance statistics in training phase that will be used to renormalize the input in test phase.
* `epsilon`: a value added to the denominator for numerical stability
* `momentum`:  the value used for the `running_mean` and `running_var` computation

## Inputs

* `x`: Array where `size(x, N - 1) = chs` and `ndims(x) > 2`

## Returns

* `y`: Normalized Array
* Update model state

## Parameters

* `affine=true`
    * `bias`: Bias of shape `(chs,)` 
    * `scale`: Scale of shape `(chs,)`
* `affine=false` - Empty `NamedTuple()`

## States

* Statistics if `track_stats=true`
    - `running_mean`: Running mean of shape `(groups,)`
    - `running_var`: Running variance of shape `(groups,)`
* Statistics if `track_stats=false`
    - `running_mean`: nothing
    - `running_var`: nothing
* `training`: Used to check if training/inference mode

Use [`Lux.testmode`](@ref) during inference.

## Example

```julia
m = Chain(
    Dense(784 => 64),
    GroupNorm(64, 4, relu),
    Dense(64 => 10),
    GroupNorm(10, 5)
)
```

!!! warning
    GroupNorm doesn't have CUDNN support. The GPU fallback is not very efficient.

See also [`BatchNorm`](@ref)
"""
struct GroupNorm{affine, track_stats, F1, F2, F3, N} <:
       AbstractNormalizationLayer{affine, track_stats}
    activation::F1
    epsilon::N
    momentum::N
    chs::Int
    init_bias::F2
    init_scale::F3
    groups::Int
end

function GroupNorm(chs::Int,
                   groups::Int,
                   activation=identity;
                   init_bias=zeros32,
                   init_scale=ones32,
                   affine::Bool=true,
                   track_stats::Bool=true,
                   epsilon=1.0f-5,
                   momentum=0.1f0)
    @assert chs % groups==0 "The number of groups ($(groups)) must divide the number of channels ($chs)"
    activation = NNlib.fast_act(activation)
    return GroupNorm{affine, track_stats, typeof(activation), typeof(init_bias),
                     typeof(init_scale), typeof(epsilon)}(activation, epsilon, momentum,
                                                          chs, init_bias, init_scale,
                                                          groups)
end

function initialparameters(rng::AbstractRNG, l::GroupNorm{affine}) where {affine}
    return affine ? (scale=l.init_scale(rng, l.chs), bias=l.init_bias(rng, l.chs)) :
           NamedTuple()
end
function initialstates(rng::AbstractRNG,
                       l::GroupNorm{affine, track_stats}) where {affine, track_stats}
    return if track_stats
        (running_mean=zeros32(rng, l.groups), running_var=ones32(rng, l.groups),
         training=Val(true))
    else
        (running_mean=nothing, running_var=nothing, training=Val(true))
    end
end

parameterlength(l::GroupNorm{affine}) where {affine} = affine ? (l.chs * 2) : 0
function statelength(l::GroupNorm{affine, track_stats}) where {affine, track_stats}
    (track_stats ? 2 * l.groups : 0) + 1
end

function (GN::GroupNorm)(x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    sz = size(x)
    @assert N > 2
    @assert sz[N - 1] == GN.chs

    x_ = reshape(x, sz[1:(N - 2)]..., sz[N - 1] ÷ GN.groups, GN.groups, sz[N])

    x_normalized, xmean, xvar = normalization(x_,
                                              st.running_mean,
                                              st.running_var,
                                              ps.scale,
                                              ps.bias,
                                              GN.activation,
                                              collect(1:(N - 1)),
                                              st.training,
                                              GN.momentum,
                                              GN.epsilon)

    st = merge(st, (running_mean=xmean, running_var=xvar))

    return reshape(x_normalized, sz), st
end

function Base.show(io::IO, l::GroupNorm{affine, track_stats}) where {affine, track_stats}
    print(io, "GroupNorm($(l.chs), $(l.groups)")
    (l.activation == identity) || print(io, ", $(l.activation)")
    affine || print(io, ", affine=false")
    track_stats || print(io, ", track_stats=false")
    return print(io, ")")
end

"""
    WeightNorm(layer::AbstractExplicitLayer, which_params::NTuple{N,Symbol}, dims::Union{Tuple,Nothing}=nothing)

Applies [weight normalization](https://arxiv.org/abs/1602.07868) to a parameter in the given layer.

``w = g\\frac{v}{\\|v\\|}``

Weight normalization is a reparameterization that decouples the magnitude of a weight tensor from its direction. This updates the parameters in `which_params` (e.g. `weight`) using two parameters: one specifying the magnitude (e.g. `weight_g`) and one specifying the direction (e.g. `weight_v`).

## Arguments

* `layer` whose parameters are being reparameterized
* `which_params`: parameter names for the parameters being reparameterized
* By default, a norm over the entire array is computed. Pass `dims` to modify the dimension.

## Inputs

* `x`: Should be of valid type for input to `layer`

## Returns

* Output from `layer`
* Updated model state of `layer`

## Parameters

* `normalized`: Parameters of `layer` that are being normalized
* `unnormalized`: Parameters of `layer` that are not being normalized

## States

* Same as that of `layer`
"""
struct WeightNorm{which_params, L <: AbstractExplicitLayer, D} <: AbstractExplicitLayer
    layer::L
    dims::D
end

function WeightNorm(layer::AbstractExplicitLayer, which_params::NTuple{N, Symbol},
                    dims::Union{Tuple, Nothing}=nothing) where {N}
    return WeightNorm{Val{which_params}, typeof(layer), typeof(dims)}(layer, dims)
end

function initialparameters(rng::AbstractRNG,
                           wn::WeightNorm{Val{which_params}}) where {which_params}
    ps_layer = initialparameters(rng, wn.layer)
    ps_normalized = []
    ps_unnormalized = []
    i = 1
    for k in propertynames(ps_layer)
        v = ps_layer[k]
        if k ∈ which_params
            dim = wn.dims === nothing ? ndims(v) : wn.dims[i]
            push!(ps_normalized, Symbol(string(k) * "_g") => _norm_except(v, dim))
            push!(ps_normalized, Symbol(string(k) * "_v") => v)
            i += 1
        else
            push!(ps_unnormalized, k => v)
        end
    end
    ps_unnormalized = length(ps_unnormalized) == 0 ? NamedTuple() : (; ps_unnormalized...)
    return (normalized=(; ps_normalized...), unnormalized=ps_unnormalized)
end

initialstates(rng::AbstractRNG, wn::WeightNorm) = initialstates(rng, wn.layer)

function (wn::WeightNorm)(x, ps::VALID_PARAMETER_TYPES, s::NamedTuple)
    _ps = get_normalized_parameters(wn, wn.dims, ps.normalized)
    return wn.layer(x, merge(_ps, ps.unnormalized), s)
end

@inbounds @generated function get_normalized_parameters(::WeightNorm{Val{which_params}},
                                                        dims::T,
                                                        ps::Union{ComponentArray, NamedTuple
                                                                  }) where {T, which_params}
    parameter_names = string.(which_params)
    v_parameter_names = Symbol.(parameter_names .* "_v")
    g_parameter_names = Symbol.(parameter_names .* "_g")
    normalized_params_symbol = [gensym(p) for p in parameter_names]

    function get_norm_except_invoke(i)
        return if T <: Tuple
            :(_norm_except(ps.$(v_parameter_names[i]), dims[$i]))
        else
            :(_norm_except(ps.$(v_parameter_names[i])))
        end
    end

    calls = []
    for i in 1:length(parameter_names)
        push!(calls,
              :($(normalized_params_symbol[i]) = ps.$(v_parameter_names[i]) .*
                                                 (ps.$(g_parameter_names[i]) ./
                                                  $(get_norm_except_invoke(i)))))
    end
    push!(calls,
          :(return NamedTuple{$(which_params)}(tuple($(Tuple(normalized_params_symbol)...)))))

    return Expr(:block, calls...)
end

function Base.show(io::IO, w::WeightNorm{Val{which_params}}) where {which_params}
    return print(io, "WeightNorm{", which_params, "}(", w.layer, ")")
end
