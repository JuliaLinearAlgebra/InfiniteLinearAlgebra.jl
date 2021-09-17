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
                        _bidiag_forwardsub!, mulreduce, RangeCumsum, _factorize, transposelayout, ldiv!, lmul!, mul
import BandedMatrices: BandedMatrix, _BandedMatrix, AbstractBandedMatrix, bandeddata, bandwidths, BandedColumns, bandedcolumns,
                        _default_banded_broadcast, banded_similar
import FillArrays: AbstractFill, getindex_value, axes_print_matrix_row
import InfiniteArrays: OneToInf, InfUnitRange, Infinity, PosInfinity, InfiniteCardinal, InfStepRange, AbstractInfUnitRange, InfAxes, InfRanges
import LinearAlgebra: matprod, qr, AbstractTriangular, AbstractQ, adjoint, transpose, AdjOrTrans
import LazyArrays: applybroadcaststyle, CachedArray, CachedMatrix, CachedVector, DenseColumnMajor, FillLayout, ApplyMatrix, check_mul_axes, LazyArrayStyle,
                    resizedata!, MemoryLayout,
                    factorize, sub_materialize, LazyLayout, LazyArrayStyle, layout_getindex,
                    applylayout, ApplyLayout, PaddedLayout, zero!, MulAddStyle,
                    LazyArray, LazyMatrix, LazyVector, paddeddata
import MatrixFactorizations: ul, ul!, _ul, ql, ql!, _ql, QLPackedQ, getL, getR, getU, reflector!, reflectorApply!, QL, QR, QRPackedQ,
                            QRPackedQLayout, AdjQRPackedQLayout, QLPackedQLayout, AdjQLPackedQLayout, LayoutQ

import BlockArrays: AbstractBlockVecOrMat, sizes_from_blocks, _length, BlockedUnitRange, blockcolsupport, BlockLayout, AbstractBlockLayout, BlockSlice

import BandedMatrices: BandedMatrix, bandwidths, AbstractBandedLayout, _banded_qr!, _banded_qr, _BandedMatrix, banded_chol!

import LazyBandedMatrices: ApplyBandedLayout, BroadcastBandedLayout, _krontrav_axes, _block_interlace_axes, LazyBandedLayout,AbstractLazyBandedLayout

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

if VERSION ≥ v"1.7-"
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


function chop!(c::AbstractVector, tol::Real)
    @assert tol >= 0

    @inbounds for k=length(c):-1:1
        if abs(c[k]) > tol
            resize!(c,k)
            return c
        end
    end
    resize!(c,0)
    c
end

function chop(A::AbstractMatrix, tol)
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
