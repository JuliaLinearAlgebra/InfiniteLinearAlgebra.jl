module InfiniteLinearAlgebra
using BlockArrays, BlockBandedMatrices, BandedMatrices, LazyArrays, FillArrays, InfiniteArrays, MatrixFactorizations, LinearAlgebra

import Base: +, -, *, /, \, ^, OneTo, getindex, promote_op, _unsafe_getindex, print_matrix_row, size, axes,
            AbstractMatrix, AbstractArray, Matrix, Array, Vector, AbstractVector,
            show, getproperty
import Base.Broadcast: BroadcastStyle

import InfiniteArrays: OneToInf, InfUnitRange, Infinity, InfStepRange
import FillArrays: AbstractFill
import BandedMatrices: BandedMatrix, _BandedMatrix, bandeddata, bandwidths
import LinearAlgebra: lmul!,  ldiv!, matprod, qr, AbstractTriangular, AbstractQ, adjoint, transpose
import LazyArrays: CachedArray, DenseColumnMajor, FillLayout, ApplyMatrix, check_mul_axes, ApplyStyle, LazyArrayApplyStyle, LazyArrayStyle,
                    CachedMatrix, CachedArray, resizedata!, MemoryLayout
import MatrixFactorizations: ql, ql!, QLPackedQ, getL, getR, reflector!, reflectorApply!, QL, QR, QRPackedQ

import BlockArrays: BlockSizes, cumulsizes, _find_block, AbstractBlockVecOrMat, sizes_from_blocks

import BandedMatrices: BandedMatrix, bandwidths, AbstractBandedLayout, _banded_qr!

import BlockBandedMatrices: _BlockSkylineMatrix, _BandedMatrix, AbstractBlockSizes, cumulsizes, _BlockSkylineMatrix, BlockSizes, blockstart, blockstride,
        BlockSkylineSizes, BlockSkylineMatrix, BlockBandedMatrix, _BlockBandedMatrix, BlockTridiagonal


# Fix ∞ BandedMatrix
ApplyStyle(::typeof(*), ::Type{<:BandedMatrix{<:Any,<:Any,<:OneToInf}}, _::Type{<:AbstractArray}...) =
    LazyArrayApplyStyle()

# BroadcastStyle(::Type{<:BandedMatrix{<:Any,<:Any,<:OneToInf}}) = LazyArrayStyle{2}()

^(A::BandedMatrix{T,<:Any,<:OneToInf}, p::Integer) where T =
    if p < 0 
        inv(A)^(-p)
    elseif p == 0
        Eye{T}(∞)
    elseif p == 1
        copy(A)
    else
        A*A^(p-1)
    end
    

if VERSION < v"1.2-"
    import Base: has_offset_axes
    require_one_based_indexing(A...) = !has_offset_axes(A...) || throw(ArgumentError("offset arrays are not supported but got an array with index other than 1"))
else
    import Base: require_one_based_indexing    
end             

export Vcat, Fill, ql, ql!, ∞, ContinuousSpectrumError, BlockTridiagonal

include("banded/hessenbergq.jl")

include("banded/infbanded.jl")
include("blockbanded/infblocktridiagonal.jl")
include("banded/infqltoeplitz.jl")
include("infql.jl")
include("infqr.jl")

end # module
