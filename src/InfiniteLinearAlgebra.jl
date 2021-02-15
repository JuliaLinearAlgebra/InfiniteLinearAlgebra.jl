module InfiniteLinearAlgebra
using BlockArrays, BlockBandedMatrices, BandedMatrices, LazyArrays, LazyBandedMatrices, SemiseparableMatrices,
        FillArrays, InfiniteArrays, MatrixFactorizations, ArrayLayouts, LinearAlgebra

import Base: +, -, *, /, \, ^, OneTo, getindex, promote_op, _unsafe_getindex, size, axes,
            AbstractMatrix, AbstractArray, Matrix, Array, Vector, AbstractVector, Slice,
            show, getproperty, copy, map, require_one_based_indexing, similar, inv
import Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted

import ArrayLayouts: colsupport, rowsupport, triangularlayout, MatLdivVec, triangulardata, TriangularLayout, TridiagonalLayout, 
                        sublayout, _qr, __qr, MatLmulVec, MatLmulMat, AbstractQLayout, materialize!, subdiag, supdiag,
                        _bidiag_forwardsub!, mulreduce, RangeCumsum
import BandedMatrices: BandedMatrix, _BandedMatrix, AbstractBandedMatrix, bandeddata, bandwidths, BandedColumns, bandedcolumns,
                        _default_banded_broadcast, banded_similar
import FillArrays: AbstractFill, getindex_value, axes_print_matrix_row
import InfiniteArrays: OneToInf, InfUnitRange, Infinity, PosInfinity, InfiniteCardinal, InfStepRange, AbstractInfUnitRange, InfAxes, InfRanges
import LinearAlgebra: lmul!,  ldiv!, matprod, qr, AbstractTriangular, AbstractQ, adjoint, transpose
import LazyArrays: applybroadcaststyle, CachedArray, CachedMatrix, CachedVector, DenseColumnMajor, FillLayout, ApplyMatrix, check_mul_axes, ApplyStyle, LazyArrayApplyStyle, LazyArrayStyle,
                    resizedata!, MemoryLayout,
                    factorize, sub_materialize, LazyLayout, LazyArrayStyle, layout_getindex,
                    applylayout, ApplyLayout, PaddedLayout, zero!, MulAddStyle,
                    LazyArray, LazyMatrix, LazyVector, paddeddata
import MatrixFactorizations: ul, ul!, _ul, ql, ql!, _ql, QLPackedQ, getL, getR, getU, reflector!, reflectorApply!, QL, QR, QRPackedQ,
                            QRPackedQLayout, AdjQRPackedQLayout, QLPackedQLayout, AdjQLPackedQLayout, LayoutQ

import BlockArrays: AbstractBlockVecOrMat, sizes_from_blocks, _length, BlockedUnitRange, blockcolsupport, BlockLayout, AbstractBlockLayout, BlockSlice

import BandedMatrices: BandedMatrix, bandwidths, AbstractBandedLayout, _banded_qr!, _banded_qr, _BandedMatrix

import LazyBandedMatrices: ApplyBandedLayout, BroadcastBandedLayout, _krontrav_axes, _block_interlace_axes

import BlockBandedMatrices: _BlockSkylineMatrix, _BandedMatrix, _BlockSkylineMatrix, blockstart, blockstride,
        BlockSkylineSizes, BlockSkylineMatrix, BlockBandedMatrix, _BlockBandedMatrix, BlockTridiagonal,
        AbstractBlockBandedLayout, _blockbanded_qr!, BlockBandedLayout

import SemiseparableMatrices: AbstractAlmostBandedLayout, _almostbanded_qr!


if VERSION < v"1.6-"
    oneto(n) = Base.OneTo(n)
else
    import Base: oneto, unitrange
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

export Vcat, Fill, ql, ql!, ∞, ContinuousSpectrumError, BlockTridiagonal

include("banded/hessenbergq.jl")

include("banded/infbanded.jl")
include("blockbanded/blockbanded.jl")
include("banded/infqltoeplitz.jl")
include("infql.jl")
include("infqr.jl")
include("inful.jl")


end # module
