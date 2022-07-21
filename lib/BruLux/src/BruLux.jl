module BruLux

using Boltz
using LinearAlgebra
using Lux
using MacroTools
using NNlib

"""
    BruLuxArray

Brutus cannot compile BLAS code currently. We use `BruLuxArray` to dispatch to code which can be compiled using Brutus
"""
struct BruLuxArray{T,N,D} <: AbstractArray{T,N}
    data::D
end

MacroTools.@forward BruLuxArray.data Base.size, Base.getindex, Base.setindex!

BruLuxArray(a::AbstractArray{T,N}) where {T,N} = BruLuxArray{T,N,typeof(a)}(a)

function Base.:*(A::BruLuxArray{T,2}, B::BruLuxArray{T,2}) where {T}
    Y = BruLuxArray(fill!(similar(A, (size(A, 1), size(B, 2))), zero(T)))
    mul!(Y, A, B)
    return Y
end

Base.similar(b::BruLuxArray) = BruLuxArray(similar(b.data, eltype(b), size(b)))
Base.similar(b::BruLuxArray, ::Type{T}) where {T} = BruLuxArray(similar(b.data, T, size(b)))
function Base.similar(b::BruLuxArray, ::Type{T}, dims::Union{Integer,AbstractUnitRange}...) where {T}
    return BruLuxArray(similar(b.data, T, dims))
end
function Base.similar(b::BruLuxArray, dims::Union{Integer,AbstractUnitRange}...)
    return BruLuxArray(similar(b.data, eltype(b), dims))
end

Base.reshape(b::BruLuxArray, dims::Union{Colon,Int,UnitRange}...) = BruLuxArray(reshape(b.data, dims))

Base.view(b::BruLuxArray, dims::Union{Colon,Int,UnitRange}...) = BruLuxArray(view(b.data, dims...))

function LinearAlgebra.mul!(Y::BruLuxArray{T,N}, A::BruLuxArray{T,N}, B::BruLuxArray{T,N}) where {T,N}
    for j in axes(B, 2), k in axes(B, 1), i in axes(A, 1)
        Y[i, j] += A[i, k] * B[k, j]
    end
    return Y
end

function NNlib.conv_im2col!(
    y::BruLuxArray{T,5},
    x::BruLuxArray{T,5},
    w::BruLuxArray{T,5},
    cdims::DenseConvDims;
    col::BruLuxArray{T,3}=similar(x, NNlib.im2col_dims(cdims)...),
    alpha::T=T(1),
    beta::T=T(0),
) where {T}
    NNlib.check_dims(size(x), size(w), size(y), cdims)
    M = prod(NNlib.output_size(cdims))
    N = NNlib.channels_out(cdims)
    K = prod(NNlib.kernel_size(cdims)) * NNlib.channels_in(cdims)

    Threads.@threads for batch_idx in 1:size(x, 5)
        # col_slice is a thread-local workspace
        col_slice = view(col, :, :, Threads.threadid())

        NNlib.im2col!(col_slice, view(x, :, :, :, :, batch_idx), cdims)

        view(reshape(y, M, N, :), :, :, batch_idx) .= col_slice * reshape(w, K, N)
    end
    return y
end

## Broadcasting
Base.BroadcastStyle(::Type{<:BruLuxArray}) = Broadcast.ArrayStyle{BruLuxArray}()

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{BruLuxArray}}, ::Type{ElType}) where ElType
    BruLuxArray(similar(Array{ElType}, axes(bc)))
end

export BruLuxArray

end
