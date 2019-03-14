
module InfiniteBandedMatrices
using BlockArrays, BlockBandedMatrices, BandedMatrices, LazyArrays, FillArrays, InfiniteArrays, MatrixFactorizations, LinearAlgebra

import Base: +, -, *, /, \, OneTo, getindex, promote_op
import InfiniteArrays: OneToInf
import FillArrays: AbstractFill
import BandedMatrices: BandedMatrix, _BandedMatrix, bandeddata
import LinearAlgebra: reflector!, reflectorApply!, lmul!, has_offset_axes, matprod
import LazyArrays: CachedArray
import MatrixFactorizations: ql, ql!, QLPackedQ, getL

export Vcat, Fill, ql, ql!, ∞

const SymTriPertToeplitz{T} = SymTridiagonal{T,Vcat{T,1,Tuple{Vector{T},Fill{T,1,Tuple{OneToInf{Int}}}}}}
const TriPertToeplitz{T} = Tridiagonal{T,Vcat{T,1,Tuple{Vector{T},Fill{T,1,Tuple{OneToInf{Int}}}}}}
const InfBandedMatrix{T,V<:AbstractMatrix{T}} = BandedMatrix{T,V,OneToInf{Int}}

for op in (:-, :+)
    @eval begin
        $op(A::SymTriPertToeplitz{T}, λ::UniformScaling) where T = SymTridiagonal(broadcast($op, A.dv, λ.λ), A.ev)
        $op(λ::UniformScaling, A::SymTriPertToeplitz{T}) where T = SymTridiagonal(broadcast($op, λ.λ, A.dv), A.ev)
        $op(A::TriPertToeplitz{T}, λ::UniformScaling) where T = Tridiagonal(A.dl, broadcast($op, A.d, λ.λ), A.du)
        $op(λ::UniformScaling, A::TriPertToeplitz{T}) where T = Tridiagonal(A.dl, broadcast($op, λ.λ, A.d), A.du)
    end
end

*(a::AbstractVector, b::AbstractFill{<:Any,2,Tuple{OneTo{Int},OneToInf{Int}}}) = MulArray(a,b)


function BandedMatrix(A::SymTriPertToeplitz{T}, (l,u)::Tuple{Int,Int}) where T
    a,a∞ = A.dv.arrays
    b,b∞ = A.ev.arrays
    n = max(length(a), length(b)+1) + 1
    data = zeros(T, l+u+1, n)
    data[u,2:length(b)+1] .= b
    data[u,length(b)+2:end] .= b∞.value
    data[u+1,1:length(a)] .= a
    data[u+1,length(a)+1:end] .= a∞.value
    data[u+2,1:length(b)] .= b
    data[u+2,length(b)+1:end] .= b∞.value
    _BandedMatrix(Hcat(data, [Zeros{T}(u-1); b∞.value; a∞.value; b∞.value; Zeros{T}(l-1)] * Ones(1,∞)), ∞, l, u)
end

function BandedMatrix(A::TriPertToeplitz{T}, (l,u)::Tuple{Int,Int}) where T
    a,a∞ = A.d.arrays
    b,b∞ = A.du.arrays
    c,c∞ = A.dl.arrays
    n = max(length(a), length(b)+1, length(c)-1) + 1
    data = zeros(T, l+u+1, n)
    data[u,2:length(b)+1] .= b
    data[u,length(b)+2:end] .= b∞.value
    data[u+1,1:length(a)] .= a
    data[u+1,length(a)+1:end] .= a∞.value
    data[u+2,1:length(c)] .= c
    data[u+2,length(c)+1:end] .= c∞.value
    _BandedMatrix(Hcat(data, [Zeros{T}(u-1); b∞.value; a∞.value; c∞.value; Zeros{T}(l-1)] * Ones(1,∞)), ∞, l, u)
end


# function tailiterate(X)
#     Q,L = ql(X)
#     I = Eye(2)
#     Z = Zeros(2,2)
#     [I Z; Z Z]*X + [Z Z; I Z]*L*
#                 [Z I Z; Z Z I; Z Z Z]
# end

function tailiterate!(X::AbstractMatrix{T}) where T
    c,a,b = X[1,:]
    h = zero(T)
    for _=1:10_000
        QL = ql!(X)     
        h == X[1,3] && return X, QL.τ[end]
        h = X[1,3]
        X[2,:] .= (zero(T), X[1,1], X[1,2]);
        X[1,:] .= (c,a,b);
    end
    error("Did not converge")
end
tailiterate(c,a,b) = tailiterate!([c a b; 0 c a])

ql(A::SymTriPertToeplitz{T}) where T = ql!(BandedMatrix(A, (2,1)))
ql(A::TriPertToeplitz{T}) where T = ql!(BandedMatrix(A, (2,1)))

toeptail(B::BandedMatrix) = B.data.arrays[end].applied.args[1]

function ql!(B::InfBandedMatrix{T}) where T
    @assert bandwidths(B) == (2,1)
    b,a,c,_ = toeptail(B)
    X, τ = tailiterate(c,a,b)
    data = bandeddata(B).arrays[1]
    B̃ = _BandedMatrix(data, size(data,2), 2,1)
    B̃[end,end-1:end] .= (X[1,1], X[1,2])
    F = ql!(B̃)
    B̃.data[3:end,end] .= (X[2,2], X[2,1]) # fill in L
    B̃.data[4,end-1] = X[2,1] # fill in L
    H = Hcat(B̃.data, [X[1,3], X[2,3], X[2,2], X[2,1]] * Ones{T}(1,∞))
    QL(_BandedMatrix(H, ∞, 2, 1), Vcat(F.τ,Fill(τ,∞)))
end

getindex(Q::QLPackedQ{T,<:InfBandedMatrix{T}}, i::Integer, j::Integer) where T =
    (Q'*Vcat(Zeros{T}(i-1), one(T), Zeros{T}(∞)))[j]

function getL(Q::QL{T,<:InfBandedMatrix{T}}) where T
    LowerTriangular(Q.factors)
end

# number of structural non-zeros
nzzeros(B::Vcat, k) = sum(size.(B.arrays[1:end-1],k))
nzzeros(B::CachedArray, k) = max(size(B.data,k), nzzeros(B.array,k))


function lmul!(A::QLPackedQ{<:Any,<:InfBandedMatrix}, B::AbstractVecOrMat)
    @assert !has_offset_axes(B)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    D = Afactors.data
    begin
        for k = 1:∞
            ν = k
            allzero = k > nzzeros(B,1) ? true : false
            for j = 1:nB
                vBj = B[k,j]
                for i = max(1,ν-u):k-1
                    if !iszero(B[i,j])
                        allzero = false
                        vBj += conj(D[i-ν+u+1,ν])*B[i,j]
                    end
                end
                vBj = A.τ[k]*vBj
                B[k,j] -= vBj
                for i = max(1,ν-u):k-1
                    B[i,j] -= D[i-ν+u+1,ν]*vBj
                end
            end
            allzero && break
        end
    end
    B
end

function lmul!(adjA::Adjoint{<:Any,<:QLPackedQ{<:Any,<:InfBandedMatrix}}, B::AbstractVecOrMat)
    @assert !has_offset_axes(B)
    A = adjA.parent
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    D = Afactors.data
    @inbounds begin
        for k = nzzeros(B,1)+u:-1:1
            ν = k
            for j = 1:nB
                vBj = B[k,j]
                for i = max(1,ν-u):k-1
                    vBj += conj(D[i-ν+u+1,ν])*B[i,j]
                end
                vBj = conj(A.τ[k])*vBj
                B[k,j] -= vBj
                for i = max(1,ν-u):k-1
                    B[i,j] -= D[i-ν+u+1,ν]*vBj
                end
            end
        end
    end
    B
end

function (*)(A::QLPackedQ{T,<:InfBandedMatrix}, x::AbstractVector{S}) where {T,S}
    TS = promote_op(matprod, T, S)
    lmul!(A, cache(convert(AbstractVector{TS},x)))
end

function (*)(A::Adjoint{T,<:QLPackedQ{T,<:InfBandedMatrix}}, x::AbstractVector{S}) where {T,S}
    TS = promote_op(matprod, T, S)
    lmul!(A, cache(convert(AbstractVector{TS},x)))
end

end # module
