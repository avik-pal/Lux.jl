"""
    Conv(k::NTuple{N,Integer}, (in_chs => out_chs)::Pair{<:Integer,<:Integer}, activation=identity; init_weight=glorot_uniform, stride=1, pad=0, dilation=1, groups=1, bias=true)

Standard convolutional layer.

Image data should be stored in WHCN order (width, height, channels, batch). In other words, a `100 × 100` RGB image would be a `100 × 100 × 3 × 1` array, and a batch of 50 would be a `100 × 100 × 3 × 50` array. This has `N = 2` spatial dimensions, and needs a kernel size like `(5, 5)`, a 2-tuple of integers. To take convolutions along `N` feature dimensions, this layer expects as input an array with `ndims(x) == N + 2`, where `size(x, N + 1) == in_chs` is the number of input channels, and `size(x, ndims(x))` is the number of observations in a batch.

!!! note
    Frameworks like [`Pytorch`](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d) perform cross-correlation in their convolution layers

## Arguments

* `k`: Tuple of integers specifying the size of the convolutional kernel. Eg, for 2D convolutions `length(k) == 2`
* `in_chs`: Number of input channels
* `out_chs`: Number of input and output channels
* `activation`: Activation Function

## Keyword Arguments

* `init_weight`: Controls the initialization of the weight parameter
* `stride`: Should each be either single integer, or a tuple with `N` integers
* `dilation`: Should each be either single integer, or a tuple with `N` integers
* `pad`: Specifies the number of elements added to the borders of the data array. It can be
    - a single integer for equal padding all around,
    - a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
    - a tuple of `2*N` integers, for asymmetric padding, or
    - the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.
* `groups`: Expected to be an `Int`. It specifies the number of groups to divide a convolution into (set `groups = in_chs` for Depthwise Convolutions). `in_chs` and `out_chs` must be divisible by `groups`.
* `bias`: The initial bias vector is all zero by default. Trainable bias can be disabled entirely by setting this to `false`.

## Inputs

* `x`: Data satisfying `ndims(x) == N + 2 && size(x, N - 1) == in_chs`, i.e. `size(x) = (I_N, ..., I_1, C_in, N)`

## Returns

* Output of the convolution `y` of size `(O_N, ..., O_1, C_out, N)` where ``O_i = floor\\left(\\frac{I_i + pad[i] + pad[(i + N) \\% length(pad)] - dilation[i] \\times (k[i] - 1)}{stride[i]} + 1\\right)``
* Empty `NamedTuple()`

## Parameters

* `weight`: Convolution kernel
* `bias`: Bias (present if `bias=true`)
"""
struct Conv{N, bias, M, F1, F2} <: AbstractExplicitLayer
    activation::F1
    in_chs::Int
    out_chs::Int
    kernel_size::NTuple{N, Int}
    stride::NTuple{N, Int}
    pad::NTuple{M, Int}
    dilation::NTuple{N, Int}
    groups::Int
    init_weight::F2
end

function Conv(k::NTuple{N, Integer},
              ch::Pair{<:Integer, <:Integer},
              activation=identity;
              init_weight=glorot_uniform,
              stride=1,
              pad=0,
              dilation=1,
              groups=1,
              bias=true) where {N}
    stride = expand(Val(N), stride)
    dilation = expand(Val(N), dilation)
    pad = calc_padding(Conv, pad, k, dilation, stride)
    activation = NNlib.fast_act(activation)
    return Conv{N, bias, length(pad), typeof(activation), typeof(init_weight)}(activation,
                                                                               first(ch),
                                                                               last(ch), k,
                                                                               stride, pad,
                                                                               dilation,
                                                                               groups,
                                                                               init_weight)
end

function initialparameters(rng::AbstractRNG, c::Conv{N, bias}) where {N, bias}
    weight = convfilter(rng, c.kernel_size, c.in_chs => c.out_chs; init=c.init_weight,
                        groups=c.groups)
    if bias
        return (weight=weight,
                bias=zeros(eltype(weight), ntuple(_ -> 1, N)..., c.out_chs, 1))
    else
        return (weight=weight,)
    end
end

function parameterlength(c::Conv{N, bias}) where {N, bias}
    return prod(c.kernel_size) * c.in_chs * c.out_chs ÷ c.groups + (bias ? c.out_chs : 0)
end

@inline function (c::Conv{N, false})(x::AbstractArray,
                                     ps::VALID_PARAMETER_TYPES,
                                     st::NamedTuple) where {N}
    cdims = DenseConvDims(x, ps.weight; stride=c.stride, padding=c.pad, dilation=c.dilation,
                          groups=c.groups)
    return applyactivation(c.activation, conv_wrapper(x, ps.weight, cdims)), st
end

@inline function (c::Conv{N, true})(x::AbstractArray, ps::VALID_PARAMETER_TYPES,
                                    st::NamedTuple) where {N}
    cdims = DenseConvDims(x, ps.weight; stride=c.stride, padding=c.pad, dilation=c.dilation,
                          groups=c.groups)
    return applyactivation(c.activation,
                           elementwise_add(conv_wrapper(x, ps.weight, cdims), ps.bias)), st
end

function Base.show(io::IO, l::Conv)
    print(io, "Conv(", l.kernel_size)
    print(io, ", ", l.in_chs, " => ", l.out_chs)
    _print_conv_opt(io, l)
    return print(io, ")")
end

function _print_conv_opt(io::IO, l::Conv{N, bias}) where {N, bias}
    l.activation == identity || print(io, ", ", l.activation)
    all(==(0), l.pad) || print(io, ", pad=", _maybetuple_string(l.pad))
    all(==(1), l.stride) || print(io, ", stride=", _maybetuple_string(l.stride))
    all(==(1), l.dilation) || print(io, ", dilation=", _maybetuple_string(l.dilation))
    (l.groups == 1) || print(io, ", groups=", l.groups)
    (bias == false) && print(io, ", bias=false")
    return nothing
end

"""
    MaxPool(window::NTuple; pad=0, stride=window)

Max pooling layer, which replaces all pixels in a block of size `window` with the maximum value.

# Arguments

* `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling `length(window) == 2`

## Keyword Arguments

* `stride`: Should each be either single integer, or a tuple with `N` integers
* `pad`: Specifies the number of elements added to the borders of the data array. It can be
    - a single integer for equal padding all around,
    - a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
    - a tuple of `2*N` integers, for asymmetric padding, or
    - the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.

## Inputs

* `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`
  
## Returns

* Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where ``O_i = floor\\left(\\frac{I_i + pad[i] + pad[(i + N) \\% length(pad)] - dilation[i] \\times (k[i] - 1)}{stride[i]} + 1\\right)``
* Empty `NamedTuple()`

See also [`Conv`](@ref), [`MeanPool`](@ref), [`GlobalMaxPool`](@ref), [`AdaptiveMaxPool`](@ref)
"""
struct MaxPool{N, M} <: AbstractExplicitLayer
    k::NTuple{N, Int}
    pad::NTuple{M, Int}
    stride::NTuple{N, Int}
end

function MaxPool(k::NTuple{N, Integer}; pad=0, stride=k) where {N}
    stride = expand(Val(N), stride)
    pad = calc_padding(MaxPool, pad, k, 1, stride)
    return MaxPool{N, length(pad)}(k, pad, stride)
end

function (m::MaxPool{N, M})(x, ps, st::NamedTuple) where {N, M}
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return maxpool(x, pdims), st
end

function Base.show(io::IO, m::MaxPool)
    print(io, "MaxPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    return print(io, ")")
end

"""
    MeanPool(window::NTuple; pad=0, stride=window)

Mean pooling layer, which replaces all pixels in a block of size `window` with the mean value.

# Arguments

* `window`: Tuple of integers specifying the size of the window. Eg, for 2D pooling `length(window) == 2`

## Keyword Arguments

* `stride`: Should each be either single integer, or a tuple with `N` integers
* `pad`: Specifies the number of elements added to the borders of the data array. It can be
    - a single integer for equal padding all around,
    - a tuple of `N` integers, to apply the same padding at begin/end of each spatial dimension,
    - a tuple of `2*N` integers, for asymmetric padding, or
    - the singleton `SamePad()`, to calculate padding such that `size(output,d) == size(x,d) / stride` (possibly rounded) for each spatial dimension.

## Inputs

* `x`: Data satisfying `ndims(x) == N + 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`
  
## Returns

* Output of the pooling `y` of size `(O_N, ..., O_1, C, N)` where ``O_i = floor\\left(\\frac{I_i + pad[i] + pad[(i + N) \\% length(pad)] - dilation[i] \\times (k[i] - 1)}{stride[i]} + 1\\right)``
* Empty `NamedTuple()`

See also [`Conv`](@ref), [`MaxPool`](@ref), [`GlobalMeanPool`](@ref), [`AdaptiveMeanPool`](@ref)
"""
struct MeanPool{N, M} <: AbstractExplicitLayer
    k::NTuple{N, Int}
    pad::NTuple{M, Int}
    stride::NTuple{N, Int}
end

function MeanPool(k::NTuple{N, Integer}; pad=0, stride=k) where {N}
    stride = expand(Val(N), stride)
    pad = calc_padding(MeanPool, pad, k, 1, stride)
    return MeanPool{N, length(pad)}(k, pad, stride)
end

function (m::MeanPool{N, M})(x, ps, st::NamedTuple) where {N, M}
    pdims = PoolDims(x, m.k; padding=m.pad, stride=m.stride)
    return meanpool(x, pdims), st
end

function Base.show(io::IO, m::MeanPool)
    print(io, "MeanPool(", m.k)
    all(==(0), m.pad) || print(io, ", pad=", _maybetuple_string(m.pad))
    m.stride == m.k || print(io, ", stride=", _maybetuple_string(m.stride))
    return print(io, ")")
end

"""
    Upsample(mode = :nearest; [scale, size]) 
    Upsample(scale, mode = :nearest)  

Upsampling Layer.

## Layer Construction

### Option 1

* `mode`: Set to `:nearest`, `:linear`, `:bilinear` or `:trilinear`

One of two keywords must be given:

* If `scale` is a number, this applies to all but the last two dimensions (channel and batch) of the input.  It may also be a tuple, to control dimensions individually.
* Alternatively, keyword `size` accepts a tuple, to directly specify the leading dimensions of the output.

### Option 2

* If `scale` is a number, this applies to all but the last two dimensions (channel and batch) of the input.  It may also be a tuple, to control dimensions individually.
* `mode`: Set to `:nearest`, `:linear`, `:bilinear` or `:trilinear`

Currently supported upsampling `mode`s and corresponding NNlib's methods are:
  - `:nearest` -> `NNlib.upsample_nearest`
  - `:linear` -> `NNlib.upsample_linear`
  - `:bilinear` -> `NNlib.upsample_bilinear`
  - `:trilinear` -> `NNlib.upsample_trilinear`

## Inputs

* `x`: For the input dimensions look into the documentation for the corresponding `NNlib` function
    - As a rule of thumb, `:nearest` should work with arrays of arbitrary dimensions
    - `:linear` works with 3D Arrays, `:bilinear` works with 4D Arrays, and `:trilinear` works with 5D Arrays
  
## Returns

* Upsampled Input of size `size` or of size `(I_1 × scale[1], ..., I_N × scale[N], C, N)`
* Empty `NamedTuple()`
"""
struct Upsample{mode, S, T} <: AbstractExplicitLayer
    scale::S
    size::T
end

function Upsample(mode::Symbol=:nearest; scale=nothing, size=nothing)
    mode in [:nearest, :bilinear, :trilinear] ||
        throw(ArgumentError("mode=:$mode is not supported."))
    if !(isnothing(scale) ⊻ isnothing(size))
        throw(ArgumentError("Either scale or size should be specified (but not both)."))
    end
    return Upsample{mode, typeof(scale), typeof(size)}(scale, size)
end

Upsample(scale, mode::Symbol=:nearest) = Upsample(mode; scale)

function (m::Upsample{:nearest})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_nearest(x, m.scale), st
end
function (m::Upsample{:nearest, Int})(x::AbstractArray{T, N}, ps,
                                      st::NamedTuple) where {T, N}
    return NNlib.upsample_nearest(x, ntuple(i -> m.scale, N - 2)), st
end
function (m::Upsample{:nearest, Nothing})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_nearest(x; size=m.size), st
end

function (m::Upsample{:bilinear})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_bilinear(x, m.scale), st
end
function (m::Upsample{:bilinear, Nothing})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_bilinear(x; size=m.size), st
end

function (m::Upsample{:trilinear})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_trilinear(x, m.scale), st
end
function (m::Upsample{:trilinear, Nothing})(x::AbstractArray, ps, st::NamedTuple)
    return NNlib.upsample_trilinear(x; size=m.size), st
end

function Base.show(io::IO, u::Upsample{mode}) where {mode}
    print(io, "Upsample(")
    print(io, ":", mode)
    u.scale !== nothing && print(io, ", scale = $(u.scale)")
    u.size !== nothing && print(io, ", size = $(u.size)")
    return print(io, ")")
end

"""
    GlobalMaxPool()

Global Mean Pooling layer. Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output, by performing max pooling on the complete (w,h)-shaped feature maps.

## Inputs

* `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`
  
## Returns

* Output of the pooling `y` of size `(1, ..., 1, C, N)`
* Empty `NamedTuple()`

See also [`MaxPool`](@ref), [`AdaptiveMaxPool`](@ref), [`GlobalMeanPool`](@ref)
"""
struct GlobalMaxPool <: AbstractExplicitLayer end

function (g::GlobalMaxPool)(x, ps, st::NamedTuple)
    return maxpool(x, PoolDims(x, size(x)[1:(end - 2)])), st
end

"""
    GlobalMeanPool()

Global Mean Pooling layer. Transforms (w,h,c,b)-shaped input into (1,1,c,b)-shaped output, by performing mean pooling on the complete (w,h)-shaped feature maps.

## Inputs

* `x`: Data satisfying `ndims(x) > 2`, i.e. `size(x) = (I_N, ..., I_1, C, N)`
  
## Returns

* Output of the pooling `y` of size `(1, ..., 1, C, N)`
* Empty `NamedTuple()`

See also [`MeanPool`](@ref), [`AdaptiveMeanPool`](@ref), [`GlobalMaxPool`](@ref)
"""
struct GlobalMeanPool <: AbstractExplicitLayer end

function (g::GlobalMeanPool)(x, ps, st::NamedTuple)
    return meanpool(x, PoolDims(x, size(x)[1:(end - 2)])), st
end

"""
    AdaptiveMaxPool(out::NTuple)

Adaptive Max Pooling layer. Calculates the necessary window size such that its output has `size(y)[1:N] == out`.

## Arguments

* `out`: Size of the first `N` dimensions for the output

## Inputs

* `x`: Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

## Returns

* Output of size `(out..., C, N)`
* Empty `NamedTuple()`

See also [`MaxPool`](@ref), [`AdaptiveMeanPool`](@ref).
"""
struct AdaptiveMaxPool{S, O} <: AbstractExplicitLayer
    out::NTuple{O, Int}
    AdaptiveMaxPool(out::NTuple{O, Int}) where {O} = new{O + 2, O}(out)
end

function (a::AdaptiveMaxPool{S})(x::AbstractArray{T, S}, ps, st::NamedTuple) where {S, T}
    pdims = compute_adaptive_pooling_dims(x, a.out)
    return maxpool(x, pdims), st
end

function Base.show(io::IO, a::AdaptiveMaxPool)
    return print(io, "AdaptiveMaxPool(", a.out, ")")
end

"""
    AdaptiveMeanPool(out::NTuple)

Adaptive Mean Pooling layer. Calculates the necessary window size such that its output has `size(y)[1:N] == out`.

## Arguments

* `out`: Size of the first `N` dimensions for the output

## Inputs

* `x`: Expects as input an array with `ndims(x) == N+2`, i.e. channel and batch dimensions, after the `N` feature dimensions, where `N = length(out)`.

## Returns

* Output of size `(out..., C, N)`
* Empty `NamedTuple()`

See also [`MeanPool`](@ref), [`AdaptiveMaxPool`](@ref).
"""
struct AdaptiveMeanPool{S, O} <: AbstractExplicitLayer
    out::NTuple{O, Int}
    AdaptiveMeanPool(out::NTuple{O, Int}) where {O} = new{O + 2, O}(out)
end

function (a::AdaptiveMeanPool{S})(x::AbstractArray{T, S}, ps, st::NamedTuple) where {S, T}
    pdims = compute_adaptive_pooling_dims(x, a.out)
    return meanpool(x, pdims), st
end

function Base.show(io::IO, a::AdaptiveMeanPool)
    return print(io, "AdaptiveMeanPool(", a.out, ")")
end
