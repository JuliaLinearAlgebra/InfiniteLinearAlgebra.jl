module InfiniteLinearAlgebra
using BlockArrays, BlockBandedMatrices, BandedMatrices, LazyArrays, LazyBandedMatrices, SemiseparableMatrices,
        FillArrays, InfiniteArrays, MatrixFactorizations, ArrayLayouts, LinearAlgebra

import Base: +, -, *, /, \, ^, OneTo, getindex, promote_op, _unsafe_getindex, print_matrix_row, size, axes,
            AbstractMatrix, AbstractArray, Matrix, Array, Vector, AbstractVector, Slice,
            show, getproperty, copy, map, require_one_based_indexing
import Base.Broadcast: BroadcastStyle, Broadcasted, broadcasted

import ArrayLayouts: colsupport, rowsupport, triangularlayout, MatLdivVec, triangulardata, TriangularLayout, sublayout, _qr, MatLmulVec, AbstractQLayout
import BandedMatrices: BandedMatrix, _BandedMatrix, AbstractBandedMatrix, bandeddata, bandwidths, BandedColumns, bandedcolumns,
                        _default_banded_broadcast
import FillArrays: AbstractFill, getindex_value      
import InfiniteArrays: OneToInf, InfUnitRange, Infinity, InfStepRange, AbstractInfUnitRange, InfAxes                  
import LinearAlgebra: lmul!,  ldiv!, matprod, qr, AbstractTriangular, AbstractQ, adjoint, transpose
import LazyArrays: applybroadcaststyle, CachedArray, CachedMatrix, CachedVector, DenseColumnMajor, FillLayout, ApplyMatrix, check_mul_axes, ApplyStyle, LazyArrayApplyStyle, LazyArrayStyle,
                    resizedata!, MemoryLayout, mulapplystyle, LmulStyle, RmulStyle,
                    factorize, sub_materialize, LazyLayout, LazyArrayStyle, layout_getindex,
                    @lazymul, applylayout, ApplyLayout, PaddedLayout, materialize!, zero!, MulAddStyle,
                    LazyArray, LazyMatrix, LazyVector, paddeddata
import MatrixFactorizations: ql, ql!, _ql, QLPackedQ, getL, getR, reflector!, reflectorApply!, QL, QR, QRPackedQ,
                            QRPackedQLayout, AdjQRPackedQLayout, QLPackedQLayout, AdjQLPackedQLayout

import BlockArrays: AbstractBlockVecOrMat, sizes_from_blocks, _length, BlockedUnitRange, blockcolsupport

import BandedMatrices: BandedMatrix, bandwidths, AbstractBandedLayout, _banded_qr!, _banded_qr, _BandedMatrix

import LazyBandedMatrices: MulBandedLayout, BroadcastBandedLayout, _krontrav_axes

import BlockBandedMatrices: _BlockSkylineMatrix, _BandedMatrix, _BlockSkylineMatrix, blockstart, blockstride,
        BlockSkylineSizes, BlockSkylineMatrix, BlockBandedMatrix, _BlockBandedMatrix, BlockTridiagonal,
        AbstractBlockBandedLayout, _blockbanded_qr!, BlockBandedLayout

import SemiseparableMatrices: AbstractAlmostBandedLayout, _almostbanded_qr!

LazyArrays.@lazymul BandedMatrix{<:Any,<:Any,<:OneToInf}
*(A::BandedMatrix{<:Any,<:Any,<:OneToInf}, b::CachedVector) = apply(*,A,b)


# BroadcastStyle(::Type{<:BandedMatrix{<:Any,<:Any,<:OneToInf}}) = LazyArrayStyle{2}()

function ^(A::BandedMatrix{T,<:Any,<:OneToInf}, p::Integer) where T
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

#######
# block broadcasted
######

const CumsumOneToInf2 = BroadcastArray{Int64,1,typeof(div),Tuple{BroadcastArray{Int64,1,typeof(*),Tuple{InfiniteArrays.OneToInf{Int64},InfiniteArrays.InfUnitRange{Int64}}},Int64}}
BlockArrays.sortedunion(a::CumsumOneToInf2, ::CumsumOneToInf2) = a


map(::typeof(length), A::BroadcastArray{OneTo{Int},1,Type{OneTo}}) = A.args[1]
map(::typeof(length), A::BroadcastArray{<:Fill,1,Type{Fill}}) = A.args[2]    
map(::typeof(length), A::BroadcastArray{<:Zeros,1,Type{Zeros}}) = A.args[1]    
map(::typeof(length), A::BroadcastArray{<:Vcat,1,Type{Vcat}}) = broadcast(+,map.(length,A.args)...)
broadcasted(::LazyArrayStyle{1}, ::typeof(length), A::BroadcastArray{OneTo{Int},1,Type{OneTo}}) =
    A.args[1]
broadcasted(::LazyArrayStyle{1}, ::typeof(length), A::BroadcastArray{<:Fill,1,Type{Fill}}) =
    A.args[2]   

BlockArrays._length(::BlockedUnitRange, ::OneToInf) = ∞
BlockArrays._last(::BlockedUnitRange, ::OneToInf) = ∞

###
# KronTrav
###

_krontrav_axes(A::NTuple{N,OneToInf{Int}}, B::NTuple{N,OneToInf{Int}}) where N = 
     @. blockedrange(OneTo(length(A)))

    
end # module
