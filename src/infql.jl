# function tailiterate(X)
#     Q,L = ql(X)
#     I = Eye(2)
#     Z = Zeros(2,2)
#     [I Z; Z Z]*X + [Z Z; I Z]*L*
#                 [Z I Z; Z Z I; Z Z Z]
# end

# doesn't normalize last column
function _qlfactUnblocked!(A::AbstractMatrix{T}) where {T}
    require_one_based_indexing(A)
    m, n = size(A)
    τ = zeros(T, min(m,n))
    for k = min(m,n):-1:2
        ν = k+n-min(m,n)
        x = view(A, k:-1:1, ν)
        τk = reflector!(x)
        τ[k] = τk
        reflectorApply!(x, τk, view(A, k:-1:1, 1:ν-1))
    end
    QL(A, τ)
end


function tailiterate!(X::AbstractMatrix{T}) where T
    c,a,b = X[1,:]
    h = zero(T)
    for _=1:10_000_000
        QL = ql!(X)     
        if h == X[1,3] 
            return QL
        end
        h = X[1,3]
        X[2,:] .= (zero(T), X[1,1], X[1,2]);
        X[1,:] .= (c,a,b);
    end
    error("Did not converge")
end
tailiterate(c,a,b) = tailiterate!([c a b; 0 c a])

struct ContinuousSpectrumError <: Exception end

function qltail(Z::Number, A::Number, B::Number)
    T = promote_type(eltype(Z),eltype(A),eltype(B))
    ñ1 = (A + sqrt(A^2-4B*Z))/2
    ñ2 = (A - sqrt(A^2-4B*Z))/2
    ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
    (n,σ) = (abs(ñ),conj(sign(ñ)))
    if n^2 < abs2(B)
        throw(ContinuousSpectrumError())
    end

    e = sqrt(n^2 - abs2(B))
    d = σ*e*Z/n

    X = [Z A B;
         0 d e]
    QL = _qlfactUnblocked!(X)

    # two iterations to correct for sign
    X[2,:] .= (zero(T), X[1,1], X[1,2]);
    X[1,:] .= (Z,A,B);
    QL = _qlfactUnblocked!(X)

    X, QL.τ[end]         
end

ql(A::SymTriPertToeplitz{T}) where T = ql!(BandedMatrix(A, (2,1)))
ql(A::TriPertToeplitz{T}) where T = ql!(BandedMatrix(A, (2,1)))
ql(A::InfBandedMatrix{T}) where T = ql!(BandedMatrix(A, (2,1)))

toeptail(B::BandedMatrix) = B.data.arrays[end].applied.args[1]

function ql!(B::InfBandedMatrix{T}) where T
    @assert bandwidths(B) == (2,1)
    b,a,c,_ = toeptail(B)
    X, τ = qltail(c,a,b)
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
    (Q'*Vcat(Zeros{T}(i-1), one(T), Zeros{T}(∞)))[j]'

function getL(Q::QL{T,<:InfBandedMatrix{T}}) where T
    LowerTriangular(Q.factors)
end

# number of structural non-zeros
nzzeros(B::Vcat, k) = sum(size.(B.arrays[1:end-1],k))
nzzeros(B::CachedArray, k) = max(size(B.data,k), nzzeros(B.array,k))


function lmul!(A::QLPackedQ{<:Any,<:InfBandedMatrix}, B::AbstractVecOrMat)
    require_one_based_indexing(B)
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
    require_one_based_indexing(B)
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