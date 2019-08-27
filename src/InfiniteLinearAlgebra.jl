module InfiniteLinearAlgebra
using BlockArrays, BlockBandedMatrices, BandedMatrices, LazyArrays, FillArrays, InfiniteArrays, MatrixFactorizations, LinearAlgebra

import Base: +, -, *, /, \, OneTo, getindex, promote_op, _unsafe_getindex, print_matrix_row, size,
            AbstractMatrix, AbstractArray, Matrix, Array, Vector, AbstractVector,
            show, getproperty
import InfiniteArrays: OneToInf, InfUnitRange, Infinity, InfStepRange
import FillArrays: AbstractFill
import BandedMatrices: BandedMatrix, _BandedMatrix, bandeddata
import LinearAlgebra: lmul!,  ldiv!, matprod, qr, QRPackedQ, AbstractTriangular, AbstractQ, adjoint, transpose,
                        QR
import LazyArrays: CachedArray, DenseColumnMajor, FillLayout, ApplyMatrix
import MatrixFactorizations: ql, ql!, QLPackedQ, getL, reflector!, reflectorApply!,
                            QL

import BlockArrays: BlockSizes, cumulsizes, _find_block, AbstractBlockVecOrMat, sizes_from_blocks

import BandedMatrices: BandedMatrix, bandwidths

import BlockBandedMatrices: _BlockSkylineMatrix, _BandedMatrix, AbstractBlockSizes, cumulsizes, _BlockSkylineMatrix, BlockSizes, blockstart, blockstride,
        BlockSkylineSizes, BlockSkylineMatrix, BlockBandedMatrix, _BlockBandedMatrix, BlockTridiagonal


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
