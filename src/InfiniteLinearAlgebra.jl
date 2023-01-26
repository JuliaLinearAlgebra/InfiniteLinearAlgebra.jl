module InfiniteLinearAlgebra
using InfiniteArrays: InfRanges
using BlockArrays, BlockBandedMatrices, BandedMatrices, LazyArrays, LazyBandedMatrices, SemiseparableMatrices,
        FillArrays, InfiniteArrays, MatrixFactorizations, ArrayLayouts, LinearAlgebra

import Base: +, -, *, /, \, ^, OneTo, getindex, promote_op, _unsafe_getindex, size, axes, length,
            AbstractMatrix, AbstractArray, Matrix, Array, Vector, AbstractVector, Slice,
            show, getproperty, copy, copyto!, map, require_one_based_indexing, similar, inv
import Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted

import ArrayLayouts: colsupport, rowsupport, triangularlayout, MatLdivVec, triangulardata, TriangularLayout, TridiagonalLayout,
                        sublayout, _qr, __qr, MatLmulVec, MatLmulMat, AbstractQLayout, materialize!, diagonaldata, subdiagonaldata, supdiagonaldata,
                        _bidiag_forwardsub!, mulreduce, RangeCumsum, _factorize, transposelayout, ldiv!, lmul!, mul, CNoPivot
import BandedMatrices: BandedMatrix, _BandedMatrix, AbstractBandedMatrix, bandeddata, bandwidths, BandedColumns, bandedcolumns,
                        _default_banded_broadcast, banded_similar
import FillArrays: AbstractFill, getindex_value, axes_print_matrix_row
import InfiniteArrays: OneToInf, InfUnitRange, Infinity, PosInfinity, InfiniteCardinal, InfStepRange, AbstractInfUnitRange, InfAxes, InfRanges
import LinearAlgebra: matprod, qr, AbstractTriangular, AbstractQ, adjoint, transpose, AdjOrTrans
import LazyArrays: applybroadcaststyle, CachedArray, CachedMatrix, CachedVector, DenseColumnMajor, FillLayout, ApplyMatrix, check_mul_axes, LazyArrayStyle,
                    resizedata!, MemoryLayout, most,
                    factorize, sub_materialize, LazyLayout, LazyArrayStyle,
                    applylayout, ApplyLayout, PaddedLayout, CachedLayout, cacheddata, zero!, MulAddStyle,
                    LazyArray, LazyMatrix, LazyVector, paddeddata, arguments
import MatrixFactorizations: ul, ul!, _ul, ql, ql!, _ql, QLPackedQ, getL, getR, getU, reflector!, reflectorApply!, QL, QR, QRPackedQ,
                            QRPackedQLayout, AdjQRPackedQLayout, QLPackedQLayout, AdjQLPackedQLayout, LayoutQ

import BlockArrays: AbstractBlockVecOrMat, sizes_from_blocks, _length, BlockedUnitRange, blockcolsupport, BlockLayout, AbstractBlockLayout, BlockSlice

import BandedMatrices: BandedMatrix, bandwidths, AbstractBandedLayout, _banded_qr!, _banded_qr, _BandedMatrix, banded_chol!

import LazyBandedMatrices: ApplyBandedLayout, BroadcastBandedLayout, _krontrav_axes, _block_interlace_axes, LazyBandedLayout, AbstractLazyBandedBlockBandedLayout,
                            AbstractLazyBandedLayout, OneToCumsum, BlockSlice1, KronTravBandedBlockBandedLayout, krontravargs, _broadcast_sub_arguments

import BlockBandedMatrices: _BlockSkylineMatrix, _BandedMatrix, _BlockSkylineMatrix, blockstart, blockstride,
        BlockSkylineSizes, BlockSkylineMatrix, BlockBandedMatrix, _BlockBandedMatrix, BlockTridiagonal,
        AbstractBlockBandedLayout, _blockbanded_qr!, BlockBandedLayout

import DSP: conv

import SemiseparableMatrices: AbstractAlmostBandedLayout, _almostbanded_qr!


if VERSION < v"1.6-"
    oneto(n) = Base.OneTo(n)
else
    import Base: oneto, unitrange
end

if VERSION ≥ v"1.7-"
    LinearAlgebra._cut_B(x::AbstractVector, r::InfUnitRange) = x
    LinearAlgebra._cut_B(X::AbstractMatrix, r::InfUnitRange) = X
end

# BroadcastStyle(::Type{<:BandedMatrix{<:Any,<:Any,<:OneToInf}}) = LazyArrayStyle{2}()

function ArrayLayouts._power_by_squaring(_, ::NTuple{2,InfiniteCardinal{0}}, A::AbstractMatrix{T}, p::Integer) where T
    if p < 0
        inv(A)^(-p)
    elseif p == 0
        Eye{T}(∞)
    elseif p == 1
        copy(A)
    else
        A*A^(p-1)
    end
end

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
compatible_resize!(_, c::AbstractVector, n) = resize!(c, n)
compatible_resize!(ax::BlockedUnitRange, c::AbstractVector, n) = resize!(c, iszero(n) ? Block(0) : findblock(ax, n))
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

export Vcat, Fill, ql, ql!, ∞, ContinuousSpectrumError, BlockTridiagonal

include("infconv.jl")

include("banded/hessenbergq.jl")

include("banded/infbanded.jl")
include("blockbanded/blockbanded.jl")
include("banded/infqltoeplitz.jl")
include("infql.jl")
include("infqr.jl")
include("inful.jl")
include("infcholesky.jl")


end # module
