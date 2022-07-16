import NNlib, Statistics

"""
    EmptyFixedSizedArray{T, N, S} <: AbstractArray{T, N}

An EmptyFixedSizedArray to test out static size inference of a model. This has multiple
usecases:

  - Test out that the neural network works without actually doing expensive computations.
  - Statically infer sizes of intermediate arrays. Especially useful if we want to generate
    an XLA computation which requires static shape inference.

Semantics of the Array:

  - `getfield` always returns T(0).
  - `setfield` is a no-op.
"""
struct EmptyFixedSizedArray{T, N, S} <: AbstractArray{T, N} end

function EmptyFixedSizedArray(x::AbstractArray)
    return EmptyFixedSizedArray{eltype(x), ndims(x), size(x)}()
end

function Base.show(io::IO, x::EmptyFixedSizedArray{T, N, S}) where {T, N, S}
    print(io, "$(join(S, "x")) EmptyFixedSizedArray{$T, $N}")
    return nothing
end
Base.show(io::IO, ::MIME, x::EmptyFixedSizedArray) = show(io, x)
function Base.display(x::EmptyFixedSizedArray)
    show(stdout, x)
    println()
    return nothing
end

Base.size(::EmptyFixedSizedArray{T, N, S}) where {T, N, S} = S
Base.eltype(::EmptyFixedSizedArray{T}) where {T} = T
Base.getindex(::EmptyFixedSizedArray{T}, i...) where {T} = T(0)
Base.setindex!(::EmptyFixedSizedArray, i, v) = nothing

Base.similar(x::EmptyFixedSizedArray) = x
function Base.similar(::EmptyFixedSizedArray{T1, N, S}, ::Type{T}) where {T1, N, S, T}
    return EmptyFixedSizedArray{T, N, S}()
end
function Base.similar(::EmptyFixedSizedArray, ::Type{T},
                      dims::Union{Integer, AbstractUnitRange}...) where {T}
    dims = dims isa Integer ? (dims,) : dims
    return EmptyFixedSizedArray{T, length(dims), dims}()
end
function Base.similar(x::EmptyFixedSizedArray{T},
                      dims::Union{Integer, AbstractUnitRange}...) where {T}
    return similar(x, T, dims...)
end

function Base.reshape(x::EmptyFixedSizedArray, ::Val{shape}) where {shape}
    return reshape(x, shape...)
end

# NOTE(@avik-pal): Type Inference not possible
function Base.reshape(x::EmptyFixedSizedArray{T, N, S},
                      dims::Union{Colon, Int, UnitRange}...) where {T, N, S}
    dims_ = filter(x -> !isa(x, Colon), dims)
    colons = length(dims) - length(dims_)
    @assert colons<=1 AssertionError("Atmax 1 Colon() is allowed in `dims`.")
    if colons == 1
        cidx = findfirst(x -> isa(x, Colon), dims)
        dims = (dims[1:(cidx - 1)]..., div(prod(S), prod(dims_)), dims[(cidx + 1):end]...)
    end
    @assert prod(dims)==prod(S) AssertionError("Array of size $S cannot be reshaped " *
                                               "into size $dims.")
    return EmptyFixedSizedArray{T, length(dims), dims}()
end

# NOTE(@avik-pal): Type Inference not possible
function Base.view(x::EmptyFixedSizedArray{T},
                   dims::Union{Colon, Int, UnitRange}...) where {T}
    dims_ = to_indices(x, dims)
    return EmptyFixedSizedArray{T, length(dims_), dims_}()
end

function Base.:+(::EmptyFixedSizedArray{T1, N, S},
                 ::EmptyFixedSizedArray{T2, N, S}) where {T1, T2, N, S}
    T = promote_type(T1, T2)
    return EmptyFixedSizedArray{T, N, S}()
end

function Base.:-(::EmptyFixedSizedArray{T1, N, S},
                 ::EmptyFixedSizedArray{T2, N, S}) where {T1, T2, N, S}
    T = promote_type(T1, T2)
    return EmptyFixedSizedArray{T, N, S}()
end

function Base.:*(::EmptyFixedSizedArray{T1, 2, S1},
                 ::EmptyFixedSizedArray{T2, 2, S2}) where {T1, T2, S1, S2}
    @assert S1[2]==S2[1] AssertionError("Sizes $S1 and $S2 are not compatible for " *
                                        "matrix multiplication.")
    T = promote_type(T1, T2)
    return EmptyFixedSizedArray{T, 2, (S1[1], S2[2])}()
end

function Base.BroadcastStyle(::Type{<:EmptyFixedSizedArray})
    return Broadcast.ArrayStyle{EmptyFixedSizedArray}()
end

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{EmptyFixedSizedArray}},
                      ::Type{ElType}) where {ElType}
    return EmptyFixedSizedArray{ElType, length(axes(bc)), length.(axes(bc))}()
end

Base.copyto!(dest::EmptyFixedSizedArray, bc::Base.Broadcast.Broadcasted) = dest

function NNlib.conv!(out::EmptyFixedSizedArray, in1::EmptyFixedSizedArray,
                     in2::EmptyFixedSizedArray, cdims::NNlib.DenseConvDims; kwargs...)
    return out
end

function NNlib.maxpool!(out::EmptyFixedSizedArray, x::EmptyFixedSizedArray,
                        pdims::NNlib.PoolDims; kwargs...)
    return out
end

function NNlib.meanpool!(out::EmptyFixedSizedArray, x::EmptyFixedSizedArray,
                         pdims::NNlib.PoolDims; kwargs...)
    return out
end

@inline function _reshape_into_proper_shape(x::EmptyFixedSizedArray,
                                            y::EmptyFixedSizedArray)
    return reshape(x, _get_reshape_dims(size(y), length(x))...)
end

@generated function _compute_reduced_dimensions(::EmptyFixedSizedArray{T, N, shape},
                                                ::Val{dims}) where {T, N, dims, shape}
    @assert minimum(dims) > 0 && maximum(dims) <= N
    d = dims isa Int ? (dims,) : (dims isa Vector ? Tuple(dims) : dims)
    res = ntuple(i -> i in d ? 1 : shape[i], N)
    return :(return $res)
end

function _compute_reduced_dimensions(x::EmptyFixedSizedArray, dims)
    return _compute_reduced_dimensions(x, Val(dims))
end

function _generic_reduction(x::EmptyFixedSizedArray{T, N}, dims::Val) where {T, N}
    return EmptyFixedSizedArray{T, N, _compute_reduced_dimensions(x, dims)}()
end

Base._sum(x::EmptyFixedSizedArray{T}, ::Colon) where {T, N} = T(0)
function Base._sum(x::EmptyFixedSizedArray{T, N}, dims) where {T, N}
    return EmptyFixedSizedArray{T, N, _compute_reduced_dimensions(x, dims)}()
end
Base._sum(f::Function, x::EmptyFixedSizedArray{T}, dims::Colon) where {T} = T(0)
function Base._sum(f::Function, x::EmptyFixedSizedArray{T, N}, dims) where {T, N}
    return EmptyFixedSizedArray{T, N, _compute_reduced_dimensions(x, dims)}()
end

Statistics._mean(::Function, x::EmptyFixedSizedArray, dims) = Base._sum(x, dims)
Statistics._var(x::EmptyFixedSizedArray, corrected::Bool, mean, dims) = Base._sum(x, dims)
