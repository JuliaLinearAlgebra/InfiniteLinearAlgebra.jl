using InfiniteLinearAlgebra, LinearAlgebra, BandedMatrices, InfiniteArrays, MatrixFactorizations, LazyArrays, FillArrays
import BandedMatrices: _BandedMatrix, colsupport
import MatrixFactorizations: QR
import LazyArrays: CachedMatrix

struct AdaptiveQRFactors{T,DM<:CachedMatrix{T}} <: AbstractMatrix{T}
    factors::DM
    τ::Vector{T}
end

struct AdaptiveQRTau{T,DM<:CachedMatrix{T}} <: AbstractMatrix{T}
    factors::DM
    τ::Vector{T}
end


A = _BandedMatrix(Vcat(Ones(1,∞), (1:∞)', Ones(1,∞)), ∞, 1, 1)
C = cache(A)


V = view(C.data,1:10,1:11)
V isa BandedMatrices.BandedSubBandedMatrix
qr!(V)
qr!(C.data)

F = QR(


vierwC.data

@test C[100_000,100_000] === 100_000.0

@time C[100_000,100_000]
M = 
@time F = qr!(C.data)

@which C[1:100,1:100]

@which bandwidths(C)

@which cache(A)
C.data

2