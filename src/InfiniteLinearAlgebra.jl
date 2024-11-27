module InfiniteLinearAlgebra
using InfiniteArrays: InfRanges
using BlockArrays, BlockBandedMatrices, BandedMatrices, LazyArrays, LazyBandedMatrices, SemiseparableMatrices,
        FillArrays, InfiniteArrays, MatrixFactorizations, ArrayLayouts, LinearAlgebra

import Base: *, +, -, /, \, ^, AbstractArray, AbstractMatrix, AbstractVector, Array,
             Matrix, OneTo, Slice, Vector, adjoint,
             axes, copy, copymutable, copyto!, getindex, getproperty, inv,
             length, map, oneto, promote_op, require_one_based_indexing, show,
             similar, size, transpose, adjoint, copymutable, transpose,
             adjoint, copymutable, transpose

import Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted

import ArrayLayouts: AbstractBandedLayout, AbstractQLayout, AdjQRPackedQLayout, CNoPivot, DenseColumnMajor, FillLayout,
                     MatLdivVec, MatLmulMat, MatLmulVec, MemoryLayout, QRPackedQLayout, RangeCumsum, TriangularLayout,
                     TridiagonalLayout, __qr, _bidiag_forwardsub!, _factorize, _qr, check_mul_axes, colsupport,
                     diagonaldata, ldiv!, lmul!, mul, mulreduce, reflector!, reflectorApply!,
                     rowsupport, sub_materialize, subdiagonaldata, sublayout, supdiagonaldata, transposelayout,
                     triangulardata, triangularlayout, zero!, materialize!

import BandedMatrices: BandedColumns, BandedMatrix, BandedMatrix, _BandedMatrix, AbstractBandedMatrix,
                       _BandedMatrix, _BandedMatrix, _banded_qr, _banded_qr!, _default_banded_broadcast, banded_chol!,
                       banded_similar, bandedcolumns, bandeddata, bandwidths

import BlockArrays: AbstractBlockLayout, BlockLayout, BlockSlice, BlockSlice1, BlockedOneTo,
                    blockcolsupport, sizes_from_blocks, OneToCumsum, AbstractBlockedUnitRange

import BlockBandedMatrices: AbstractBlockBandedLayout, BlockBandedMatrix, BlockSkylineMatrix,
                            BlockSkylineSizes, BlockTridiagonal, _BlockBandedMatrix, _BlockSkylineMatrix,
                            _blockbanded_qr!

import FillArrays: AbstractFill, AbstractFillMatrix, axes_print_matrix_row, getindex_value

import InfiniteArrays: AbstractInfUnitRange, InfAxes, InfRanges, InfStepRange, InfUnitRange, OneToInf, PosInfinity, InfIndexRanges

import Infinities: InfiniteCardinal, Infinity, RealInfinity

import LazyArrays: AbstractCachedMatrix, AbstractCachedVector, AbstractLazyLayout, ApplyArray, ApplyLayout, ApplyMatrix,
                   CachedArray, CachedLayout, CachedMatrix, CachedVector, LazyArrayStyle, LazyLayout,
                   LazyLayouts, LazyMatrix, LazyVector, AbstractPaddedLayout, PaddedColumns, _broadcast_sub_arguments,
                   applybroadcaststyle, applylayout, arguments, cacheddata, paddeddata, resizedata!, simplifiable,
                   simplify, islazy, islazy_layout, cache_getindex

import LazyBandedMatrices: AbstractLazyBandedBlockBandedLayout, AbstractLazyBandedLayout, ApplyBandedLayout, BlockVec,
                           BroadcastBandedLayout, KronTravBandedBlockBandedLayout, LazyBandedLayout,
                           _block_interlace_axes, _krontrav_axes, krontravargs

import LinearAlgebra: AbstractQ, AdjOrTrans, factorize, matprod, qr

import MatrixFactorizations: AdjQLPackedQLayout, LayoutQ, QL, QLPackedQ, QLPackedQLayout, QR, QRPackedQ,
                             copymutable_size, getL, getQ, getR, getU, ql, ql!, ql_layout, reversecholesky_layout, ul,
                             ul!, ul_layout

import SemiseparableMatrices: AbstractAlmostBandedLayout, _almostbanded_qr!
import InfiniteArrays: UpperOrLowerTriangular

# BroadcastStyle(::Type{<:BandedMatrix{<:Any,<:Any,<:OneToInf}}) = LazyArrayStyle{2}()



function choplength(c::AbstractVector, tol)
    @inbounds for k = length(c):-1:1
        if abs(c[k]) > tol
            return k
            break
        end
    end
    return 0
end

# resize! to nearest block
"""
compatible_resize!(c::AbstractVector, n)

resizes a vector `c` but in a way that block sizes are not changed when `c` has blocked axes.
It may allocate a new vector in some settings.
"""
compatible_resize!(_, c::AbstractVector, n) = resize!(c, n)
compatible_resize!(ax::BlockedOneTo, c::AbstractVector, n) = resize!(c, iszero(n) ? Block(0) : findblock(ax, n))
compatible_resize!(c, n) = compatible_resize!(axes(c,1), c, n)
chop!(c::AbstractVector{T}, tol::Real=zero(real(T))) where T = compatible_resize!(c, choplength(c, tol))

function chop(A::AbstractMatrix{T}, tol::Real=zero(real(T))) where T
    for k = size(A,1):-1:1
        if norm(view(A,k,:))>tol
            A=A[1:k,:]
            break
        end
    end
    for j = size(A,2):-1:1
        if norm(view(A,:,j))>tol
            A=A[:,1:j]
            break
        end
    end
    return A
end

pad(c::AbstractVector{T}, ax::Union{OneTo,OneToInf}) where T = [c; Zeros{T}(length(ax)-length(c))]
pad(c, ax...) = PaddedArray(c, ax)

pad(c::Transpose, ax, bx) = transpose(pad(parent(c), bx, ax))
pad(c::Adjoint, ax, bx) = adjoint(pad(parent(c), bx, ax))
pad(c::BlockVec, ax::BlockedOneTo{Int,<:InfStepRange}) = BlockVec(pad(c.args[1], size(c.args[1],1), ∞))

export ∞, ContinuousSpectrumError, BlockTridiagonal

include("banded/hessenbergq.jl")

include("banded/infbanded.jl")
include("blockbanded/blockbanded.jl")
include("banded/infqltoeplitz.jl")
include("banded/infreversecholeskytoeplitz.jl")
include("banded/infreversecholeskytridiagonal.jl")
include("infql.jl")
include("infqr.jl")
include("inful.jl")
include("infcholesky.jl")
include("banded/bidiagonalconjugation.jl")

end # module
