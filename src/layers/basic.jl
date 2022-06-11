"""
    ReshapeLayer(dims)

Reshapes the passed array to have a size of `(dims..., :)`

## Arguments

* `dims`: The new dimensions of the array (excluding the last dimension).

## Inputs

* `x`: AbstractArray of any shape which can be reshaped in `(dims..., size(x, ndims(x)))`

## Returns

* AbstractArray of size `(dims..., size(x, ndims(x)))`
* Empty `NamedTuple()`
"""
struct ReshapeLayer{N} <: AbstractExplicitLayer
    dims::NTuple{N, Int}
end

@inline function (r::ReshapeLayer)(x::AbstractArray, ps, st::NamedTuple)
    return reshape(x, r.dims..., size(x, ndims(x))), st
end

function Base.show(io::IO, r::ReshapeLayer)
    print(io, "ReshapeLayer(output_dims = (", join(r.dims, ", "), ", :))")
end

"""
    FlattenLayer()

Flattens the passed array into a matrix.

## Inputs

* `x`: AbstractArray

## Returns

* AbstractMatrix of size `(:, size(x, ndims(x)))`
* Empty `NamedTuple()`
"""
struct FlattenLayer <: AbstractExplicitLayer end

@inline function (f::FlattenLayer)(x::AbstractArray{T, N}, ps, st::NamedTuple) where {T, N}
    return reshape(x, :, size(x, N)), st
end

"""
    SelectDim(dim, i)

Return a view of all the data of the input `x` where the index for dimension `dim` equals `i`. Equivalent to `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`.

## Arguments

* `dim`: Dimension for indexing
* `i`: Index for dimension `dim`

## Inputs

* `x`: AbstractArray that can be indexed with `view(x,:,:,...,i,:,:,...)`

## Returns

* `view(x,:,:,...,i,:,:,...)` where `i` is in position `d`
* Empty `NamedTuple()`
"""
struct SelectDim{dim, index} <: AbstractExplicitLayer end

SelectDim(dim, index) = SelectDim{Val(dim), Val(index)}()

@inline function (s::SelectDim{dim, index})(x, ps, st::NamedTuple) where {dim, index}
    selectdim(x, get_known(dim), get_known(index)), st
end

function Base.show(io::IO, s::SelectDim{dim, index}) where {dim, index}
    print(io, "SelectDim(dim = ", get_known(dim), ", index = ", get_known(index), ")")
end

"""
    NoOpLayer()

As the name suggests does nothing but allows pretty printing of layers. Whatever input is passed is returned.
"""
struct NoOpLayer <: AbstractExplicitLayer end

@inline (noop::NoOpLayer)(x, ps, st::NamedTuple) = x, st

"""
    WrappedFunction(f)

Wraps a stateless and parameter less function. Might be used when a function is added to `Chain`. For example, `Chain(x -> relu.(x))` would not work and the right thing to do would be `Chain((x, ps, st) -> (relu.(x), st))`. An easier thing to do would be `Chain(WrappedFunction(Base.Fix1(broadcast, relu)))`

## Arguments

* `f::Function`: A stateless and parameterless function

## Inputs

* `x`: s.t `hasmethod(f, (typeof(x),))` is `true`

## Returns

* Output of `f(x)`
* Empty `NamedTuple()`
"""
struct WrappedFunction{F} <: AbstractExplicitLayer
    func::F
end

(wf::WrappedFunction)(x, ps, st::NamedTuple) = wf.func(x), st

function Base.show(io::IO, w::WrappedFunction)
    return print(io, "WrappedFunction(", w.func, ")")
end

"""
    ActivationFunction(f)

Broadcast `f` on the input but fallback to CUDNN for Backward Pass. Internally calls [`Lux.applyactivation`](@ref)

## Arguments

* `f`: Activation function

## Inputs

* `x`: Any array type s.t. `f` can be broadcasted over it

## Returns

* Broadcasted Activation `f.(x)`
* Empty `NamedTuple()`
"""
struct ActivationFunction{F} <: AbstractExplicitLayer
    func::F
end

(af::ActivationFunction)(x, ps, st::NamedTuple) = applyactivation(af.func, x), st

function Base.show(io::IO, af::ActivationFunction)
    return print(io, "ActivationFunction(", af.func, ")")
end

"""
    SkipConnection(layer, connection)

Create a skip connection which consists of a layer or [`Chain`](@ref) of consecutive layers and a shortcut connection linking the block's input to the output through a user-supplied 2-argument callable. The first argument to the callable will be propagated through the given `layer` while the second is the unchanged, "skipped" input.

The simplest "ResNet"-type connection is just `SkipConnection(layer, +)`.

## Arguments

* `layer`: Layer or `Chain` of layers to be applied to the input
* `connection`: A 2-argument function that takes `layer(input)` and the input

## Inputs

* `x`: Will be passed directly to `layer`

## Returns

* Output of `connection(layer(input), input)`
* Updated state of `layer`

## Parameters

* Parameters of `layer`

## States

* States of `layer`

See [`Parallel`](@ref) for a more general implementation.
"""
struct SkipConnection{T <: AbstractExplicitLayer, F} <:
       AbstractExplicitContainerLayer{(:layers,)}
    layers::T
    connection::F
end

@inline function (skip::SkipConnection)(x, ps::VALID_PARAMETER_TYPES,
                                        st::NamedTuple)
    mx, st = skip.layers(x, ps.layers, st.layers)
    return skip.connection(mx, x), (layers=st,)
end

"""
    Parallel(connection, layers...)

Create a layer which passes an input to each path in `layers`, before reducing the output with `connection`.

## Arguments

* `layers`: A list of `N` Lux layers
* `connection`: An `N`-argument function that is called after passing the input through each layer. If `connection = nothing`, we return a tuple `Parallel(nothing, f, g)(x, y) = (f(x), g(y))`

## Inputs

* `x`: if `x` is not a tuple, then return is computed as `connection([l(x) for l in layers]...)`. Else one is passed to each layer, thus `Parallel(+, f, g)(x, y) = f(x) + g(y)`.

## Returns

* See the Inputs section for how the output is computed
* Updated state of the `layers`

## Parameters

* Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N`

## States

* States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N`

See also [`SkipConnection`](@ref) which is `Parallel` with one identity.
"""
struct Parallel{F, T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    connection::F
    layers::T
end

function Parallel(connection, layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return Parallel(connection, NamedTuple{names}(layers))
end

function (m::Parallel)(x, ps::VALID_PARAMETER_TYPES, st::NamedTuple)
    return applyparallel(m.layers, m.connection, x, ps, st)
end

@generated function applyparallel(layers::NamedTuple{names}, connection::C, x::T,
                                  ps::VALID_PARAMETER_TYPES,
                                  st::NamedTuple) where {names, C, T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = []
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i]($(getinput(i)),
                                                                ps.layers.$(names[i]),
                                                                st.layers.$(names[i])))
             for
             i in 1:N])
    push!(calls, :(st = (layers = NamedTuple{$names}((($(Tuple(st_symbols)...),))))))
    if C == Nothing
        push!(calls, :($(y_symbols[N + 1]) = tuple($(Tuple(y_symbols[1:N])...))))
    else
        push!(calls, :($(y_symbols[N + 1]) = connection($(Tuple(y_symbols[1:N])...))))
    end
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.keys(m::Parallel) = Base.keys(getfield(m, :layers))

"""
    BranchLayer(layers...)

Takes an input `x` and passes it through all the `layers` and returns a tuple of the outputs.

## Arguments

* `layers`: A list of `N` Lux layers

## Inputs

* `x`: Will be directly passed to each of the `layers`

## Returns

* Tuple: `(layer_1(x), layer_2(x), ..., layer_N(x))`
* Updated state of the `layers`

## Parameters

* Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N`

## States

* States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N`

## Comparison with [`Parallel`](@ref)

This is slightly different from `Parallel(nothing, layers...)`

* If the input is a tuple, `Parallel` will pass each element individually to each layer

* `BranchLayer` essentially assumes 1 input comes in and is branched out into `N` outputs

## Example

An easy way to replicate an input to an NTuple is to do

```julia
l = BranchLayer(
    NoOpLayer(),
    NoOpLayer(),
    NoOpLayer(),
)
```
"""
struct BranchLayer{T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
end

function BranchLayer(layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return BranchLayer(NamedTuple{names}(layers))
end

function (m::BranchLayer)(x, ps::VALID_PARAMETER_TYPES, st::NamedTuple)
    applybranching(m.layers, x, ps, st)
end

@generated function applybranching(layers::NamedTuple{names}, x,
                                   ps::VALID_PARAMETER_TYPES,
                                   st::NamedTuple) where {names}
    N = length(names)
    y_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = []
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i](x, ps.layers.$(names[i]),
                                                                st.layers.$(names[i])))
             for i in 1:N])
    push!(calls, :(st = (layers = NamedTuple{$names}((($(Tuple(st_symbols)...),))))))
    push!(calls, :(return tuple($(Tuple(y_symbols)...)), st))
    return Expr(:block, calls...)
end

Base.keys(m::BranchLayer) = Base.keys(getfield(m, :layers))

"""
    PairwiseFusion(connection, layers...)

```
x1 --> layer1 --> y1
                  |
                  |--> connection --> layer2 --> y2
                  |                              |
                  x2                             |--> connection --> layer3 --> y3
                                                 |                              |
                                                 x3                             |--> connection --> y4
                                                                                |
                                                                                x4
```

## Arguments

* `connection`: Takes 2 inputs and combines them
* `layers`: [`AbstractExplicitLayer`](@ref)s 

## Inputs

Layer behaves differently based on input type:
1. If the input `x` is a tuple of length `N + 1`, then the `layers` must be a tuple of length `N`. The computation is as follows

```julia
y = x[1]
for i in 1:N
    y = connection(x[i + 1], layers[i](y))
end
```

2. Any other kind of input

```julia
y = x
for i in 1:N
    y = connection(x, layers[i](y))
end
```

## Returns

* See Inputs section for how the return value is computed
* Updated model state for all the contained layers

## Parameters

* Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N`

## States

* States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N`
"""
struct PairwiseFusion{F, T <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
    connection::F
    layers::T
end

function PairwiseFusion(connection, layers...)
    names = ntuple(i -> Symbol("layer_$i"), length(layers))
    return PairwiseFusion(connection, NamedTuple{names}(layers))
end

function (m::PairwiseFusion)(x, ps::VALID_PARAMETER_TYPES, st::NamedTuple)
    return applypairwisefusion(m.layers, m.connection, x, ps, st)
end

@generated function applypairwisefusion(layers::NamedTuple{names}, connection::C, x::T,
                                        ps::VALID_PARAMETER_TYPES,
                                        st::NamedTuple) where {names, C, T}
    N = length(names)
    y_symbols = [gensym() for _ in 1:(N + 1)]
    st_symbols = [gensym() for _ in 1:N]
    getinput(i) = T <: Tuple ? :(x[$i]) : :x
    calls = [:($(y_symbols[N + 1]) = $(getinput(1)))]
    append!(calls,
            [:(($(y_symbols[i]), $(st_symbols[i])) = layers[$i]($(y_symbols[N + 1]),
                                                                ps.layers.$(names[i]),
                                                                st.layers.$(names[i]));
               $(y_symbols[N + 1]) = connection($(y_symbols[i]), $(getinput(i + 1))))
             for i in 1:N])
    push!(calls, :(st = (layers = NamedTuple{$names}((($(Tuple(st_symbols)...),))))))
    push!(calls, :(return $(y_symbols[N + 1]), st))
    return Expr(:block, calls...)
end

Base.keys(m::PairwiseFusion) = Base.keys(getfield(m, :layers))

"""
    Chain(layers...; disable_optimizations::Bool = false)

Collects multiple layers / functions to be called in sequence on a given input.

## Arguments

* `layers`: A list of `N` Lux layers

## Keyword Arguments

* `disable_optimizations`: Prevents any structural optimization

## Inputs

Input `x` is passed sequentially to each layer, and must conform to the input requirements of the internal layers.

## Returns

* Output after sequentially applying all the layers to `x`
* Updated model states

## Parameters

* Parameters of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N`

## States

* States of each `layer` wrapped in a NamedTuple with `fields = layer_1, layer_2, ..., layer_N`

## Optimizations

Performs a few optimizations to generate reasonable architectures. Can be disabled using keyword argument `disable_optimizations`.
* All sublayers are recursively optimized.
* If a function `f` is passed as a layer and it doesn't take 3 inputs, it is converted to a WrappedFunction(`f`) which takes only one input.
* If the layer is a Chain, it is flattened.
* [`NoOpLayer`](@ref)s are removed.
* If there is only 1 layer (left after optimizations), then it is returned without the `Chain` wrapper.
* If there are no layers (left after optimizations), a [`NoOpLayer`](@ref) is returned.

## Example

```julia
c = Chain(Dense(2, 3, relu), BatchNorm(3), Dense(3, 2))
```
"""
struct Chain{T} <: AbstractExplicitContainerLayer{(:layers,)}
    layers::T
    function Chain(xs...; disable_optimizations::Bool=false)
        xs = disable_optimizations ? xs : flatten_model(xs)
        length(xs) == 0 && return NoOpLayer()
        length(xs) == 1 && return first(xs)
        names = ntuple(i -> Symbol("layer_$i"), length(xs))
        layers = NamedTuple{names}(xs)
        return new{typeof(layers)}(layers)
    end
    function Chain(xs::AbstractVector; disable_optimizations::Bool=false)
        Chain(xs...; disable_optimizations)
    end
end

function flatten_model(layers::Union{AbstractVector, Tuple})
    new_layers = []
    for l in layers
        f = flatten_model(l)
        if f isa Tuple || f isa AbstractVector
            append!(new_layers, f)
        elseif f isa Function
            if !hasmethod(f, (Any, VALID_PARAMETER_TYPES, NamedTuple))
                if f === identity
                    continue
                else
                    push!(new_layers, WrappedFunction(f))
                end
            else
                push!(new_layers, f)
            end
        elseif f isa Chain
            append!(new_layers, f.layers)
        elseif f isa NoOpLayer
            continue
        else
            push!(new_layers, f)
        end
    end
    return layers isa AbstractVector ? new_layers : Tuple(new_layers)
end

flatten_model(x) = x

function (c::Chain)(x, ps::VALID_PARAMETER_TYPES, st::NamedTuple)
    applychain(c.layers, x, ps, st.layers)
end

@generated function applychain(layers::NamedTuple{fields}, x,
                               ps::VALID_PARAMETER_TYPES,
                               st::NamedTuple{fields}) where {fields}
    N = length(fields)
    x_symbols = [gensym() for _ in 1:N]
    st_symbols = [gensym() for _ in 1:N]
    calls = [
        :(($(x_symbols[1]), $(st_symbols[1])) = layers[1](x, ps.layers.layer_1,
                                                          st.layer_1)),
    ]
    append!(calls,
            [:(($(x_symbols[i]), $(st_symbols[i])) = layers[$i]($(x_symbols[i - 1]),
                                                                ps.layers.$(fields[i]),
                                                                st.$(fields[i])))
             for i in 2:N])
    push!(calls, :(st = (layers=NamedTuple{$fields}((($(Tuple(st_symbols)...),))),)))
    push!(calls, :(return $(x_symbols[N]), st))
    return Expr(:block, calls...)
end

Base.keys(m::Chain) = Base.keys(getfield(m, :layers))

"""
    Dense(in_dims => out_dims, activation=identity; init_weight=glorot_uniform, init_bias=zeros32, bias::Bool=true)

Create a traditional fully connected layer, whose forward pass is given by: `y = activation.(weight * x .+ bias)`

## Arguments

* `in_dims`: number of input dimensions
* `out_dims`: number of output dimensions
* `activation`: activation function

## Keyword Arguments

* `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in_dims)`)
* `init_bias`: initializer for the bias vector (ignored if `bias=false`)
* `bias`: whether to include a bias vector

## Input

* `x` must be a Matrix of size `in_dims × B` or a Vector of length `in_dims`

## Returns

* Matrix of size `out_dims × B` or a Vector of length `out_dims`
* Empty `NamedTuple()`

## Parameters

* `weight`: Weight Matrix of size `out_dims × in_dims`
* `bias`: Bias of size `out_dims × 1` (present if `bias=true`)
"""
struct Dense{bias, F1, F2, F3} <: AbstractExplicitLayer
    activation::F1
    in_dims::Int
    out_dims::Int
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, d::Dense{bias}) where {bias}
    print(io, "Dense($(d.in_dims) => $(d.out_dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    bias || print(io, ", bias=false")
    return print(io, ")")
end

function Dense(mapping::Pair{<:Int, <:Int}, activation=identity; init_weight=glorot_uniform,
               init_bias=zeros32, bias::Bool=true)
    return Dense(first(mapping), last(mapping), activation; init_weight=init_weight,
                 init_bias=init_bias, bias=bias)
end

function Dense(in_dims::Int, out_dims::Int, activation=identity; init_weight=glorot_uniform,
               init_bias=zeros32, bias::Bool=true)
    activation = NNlib.fast_act(activation)
    return Dense{bias, typeof(activation), typeof(init_weight), typeof(init_bias)}(activation,
                                                                                   in_dims,
                                                                                   out_dims,
                                                                                   init_weight,
                                                                                   init_bias)
end

function initialparameters(rng::AbstractRNG, d::Dense{bias}) where {bias}
    if bias
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),
                bias=d.init_bias(rng, d.out_dims, 1))
    else
        return (weight=d.init_weight(rng, d.out_dims, d.in_dims),)
    end
end

function parameterlength(d::Dense{bias}) where {bias}
    return bias ? d.out_dims * (d.in_dims + 1) : d.out_dims * d.in_dims
end
statelength(d::Dense) = 0

@inline function (d::Dense{false})(x::AbstractVecOrMat,
                                   ps::VALID_PARAMETER_TYPES,
                                   st::NamedTuple)
    return applyactivation(d.activation, ps.weight * x), st
end

@inline function (d::Dense{false, typeof(identity)})(x::AbstractVecOrMat,
                                                     ps::VALID_PARAMETER_TYPES,
                                                     st::NamedTuple)
    return ps.weight * x, st
end

@inline function (d::Dense{false})(x::AbstractArray,
                                   ps::VALID_PARAMETER_TYPES,
                                   st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return reshape(applyactivation(d.activation, ps.weight * x_reshaped), d.out_dims,
                   sz[2:end]...), st
end

@inline function (d::Dense{false, typeof(identity)})(x::AbstractArray,
                                                     ps::VALID_PARAMETER_TYPES,
                                                     st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return reshape(ps.weight * x_reshaped, d.out_dims, sz[2:end]...), st
end

@inline function (d::Dense{true})(x::AbstractVector, ps::VALID_PARAMETER_TYPES,
                                  st::NamedTuple)
    return applyactivation(d.activation, elementwise_add(ps.weight * x, vec(ps.bias))), st
end

@inline function (d::Dense{true, typeof(identity)})(x::AbstractVector,
                                                    ps::VALID_PARAMETER_TYPES,
                                                    st::NamedTuple)
    return elementwise_add(ps.weight * x, vec(ps.bias)), st
end

@inline function (d::Dense{true})(x::AbstractMatrix, ps::VALID_PARAMETER_TYPES,
                                  st::NamedTuple)
    return applyactivation(d.activation, elementwise_add(ps.weight * x, ps.bias)), st
end

@inline function (d::Dense{true, typeof(identity)})(x::AbstractMatrix,
                                                    ps::VALID_PARAMETER_TYPES,
                                                    st::NamedTuple)
    return elementwise_add(ps.weight * x, ps.bias), st
end

@inline function (d::Dense{true})(x::AbstractArray, ps::VALID_PARAMETER_TYPES,
                                  st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return (reshape(applyactivation(d.activation,
                                    elementwise_add(ps.weight * x_reshaped, ps.bias)),
                    d.out_dims, sz[2:end]...), st)
end

@inline function (d::Dense{true, typeof(identity)})(x::AbstractArray,
                                                    ps::VALID_PARAMETER_TYPES,
                                                    st::NamedTuple)
    sz = size(x)
    x_reshaped = reshape(x, sz[1], :)
    return (reshape(elementwise_add(ps.weight * x_reshaped, ps.bias), d.out_dims,
                    sz[2:end]...), st)
end

"""
    Scale(dims, activation=identity; init_weight=ones32, init_bias=zeros32, bias::Bool=true)

Create a Sparsely Connected Layer with a very specific structure (only Diagonal Elements are non-zero). The forward pass is given by: `y = activation.(weight .* x .+ bias)`

## Arguments

* `dims`: size of the learnable scale and bias parameters.
* `activation`: activation function

## Keyword Arguments

* `init_weight`: initializer for the weight matrix (`weight = init_weight(rng, out_dims, in_dims)`)
* `init_bias`: initializer for the bias vector (ignored if `bias=false`)
* `bias`: whether to include a bias vector

## Input

* `x` must be an Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])` for `k ≤ size(dims)`

## Returns

* Array of size `(dims..., B)` or `(dims...[0], ..., dims[k])` for `k ≤ size(dims)`
* Empty `NamedTuple()`

## Parameters

* `weight`: Weight Array of size `(dims...)`
* `bias`: Bias of size `(dims...)`

!!! compat "Lux 0.4.3"
    `Scale` with multiple dimensions requires at least Lux 0.4.3.
"""
struct Scale{bias, F1, D, F2, F3} <: AbstractExplicitLayer
    activation::F1
    dims::D
    init_weight::F2
    init_bias::F3
end

function Base.show(io::IO, d::Scale)
    print(io, "Scale($(d.dims)")
    (d.activation == identity) || print(io, ", $(d.activation)")
    return print(io, ")")
end

function Scale(dims::Tuple{Vararg{Integer}}, activation=identity;
               init_weight=glorot_uniform,
               init_bias=zeros32, bias::Bool=true)
    activation = NNlib.fast_act(activation)
    return Scale{bias, typeof(activation), typeof(dims), typeof(init_weight),
                 typeof(init_bias)}(activation, dims, init_weight, init_bias)
end

function Scale(s1::Integer, s23::Integer...; _act=identity, kw...)
    Scale(tuple(s1, s23...), _act; kw...)
end
Scale(size_act...; kw...) = Scale(size_act[1:(end - 1)]...; _act=size_act[end], kw...)

function initialparameters(rng::AbstractRNG, d::Scale{true})
    return (weight=d.init_weight(rng, d.dims...), bias=d.init_bias(rng, d.dims...))
end
function initialparameters(rng::AbstractRNG, d::Scale{false})
    (weight=d.init_weight(rng, d.dims...),)
end

parameterlength(d::Scale{bias}) where {bias} = (1 + bias) * prod(d.dims)
statelength(d::Scale) = 0

function (d::Scale{true})(x::AbstractArray, ps::VALID_PARAMETER_TYPES,
                          st::NamedTuple)
    return applyactivation(d.activation,
                           elementwise_add(elementwise_mul(ps.weight, x), ps.bias)), st
end

function (d::Scale{true, typeof(identity)})(x::AbstractArray,
                                            ps::VALID_PARAMETER_TYPES,
                                            st::NamedTuple)
    return elementwise_add(elementwise_mul(ps.weight, x), ps.bias), st
end

function (d::Scale{false})(x::AbstractArray, ps::VALID_PARAMETER_TYPES,
                           st::NamedTuple)
    return applyactivation(d.activation, elementwise_mul(ps.weight, x)), st
end

function (d::Scale{false, typeof(identity)})(x::AbstractArray,
                                             ps::VALID_PARAMETER_TYPES,
                                             st::NamedTuple)
    return elementwise_mul(ps.weight, x), st
end
