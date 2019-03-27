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
ql(A::SymTridiagonal{T}) where T = ql!(BandedMatrix(A, (2,1)))
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

getL(Q::QL, ::Tuple{OneToInf{Int},OneToInf{Int}}) where T = LowerTriangular(Q.factors)

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



function blocktailiterate(c,a,b, d=c, e=a)
    z = zero(c)
    for _=1:1_000_000
        X = [c a b; z d e]
        F = ql!(X)
        d̃,ẽ = F.L[1:2,1:2], F.L[1:2,3:4]
        
        d̃,ẽ = QLPackedQ(F.factors[1:2,3:4],F.τ[1:2])*d̃,QLPackedQ(F.factors[1:2,3:4],F.τ[1:2])*ẽ  # undo last rotation
        if ≈(d̃, d; atol=1E-10) && ≈(ẽ, e; atol=1E-10)
            X[1:2,1:2] = d̃; X[1:2,3:4] = ẽ
            return PseudoBlockArray(X,fill(2,2), fill(2,3)), F.τ[3:end]
        end
        d,e = d̃,ẽ
    end
    error("Did not converge")
end


###
# BlockTridiagonal
####

function _ql(A::BlockTriPertToeplitz, d, e)
    N = max(length(A.blocks.du.arrays[1])+1,length(A.blocks.d.arrays[1]),length(A.blocks.dl.arrays[1]))
    c,a,b = A[Block(N+1,N)],A[Block(N,N)],A[Block(N-1,N)]
    P,τ = blocktailiterate(c,a,b,d,e)
    B = BlockBandedMatrix(A,(2,1))
    

    BB = _BlockBandedMatrix(B.data.arrays[1], (fill(2,N+2), fill(2,N)), (2,1))
    BB[Block(N),Block.(N-1:N)] .= P[Block(1), Block.(1:2)]
    F = ql!(view(BB, Block.(1:N), Block.(1:N)))
    BB[Block(N+1),Block.(N-1:N)] .= P[Block(2), Block.(1:2)]
    BB[Block(N+2),Block(N)] .= P[Block(2), Block.(1)]


    QL(_BlockSkylineMatrix(Vcat(BB.data, mortar(Fill(vec(Vcat(P[Block(1,3)], P[Block(2,3)], P[Block(2,2)], P[Block(2,1)])),∞))),B.block_sizes),
            Vcat(F.τ,mortar(Fill(τ,∞)))), P[Block(1,1)], P[Block(1,2)]
end

ql(A::BlockTriPertToeplitz) = _ql(A, A[Block(2,3)], A[Block(3,3)])[1]

ql(A::Adjoint{T,BlockTriPertToeplitz{T}}) where T = ql(BlockTridiagonal(A))

const InfBlockBandedMatrix{T} = BlockSkylineMatrix{T,<:Vcat{T,1,<:Tuple{Vector{T},<:BlockArray{T,1,<:Fill{<:Any,1,Tuple{OneToInf{Int64}}}}}}}

function lmul!(adjA::Adjoint{<:Any,<:QLPackedQ{<:Any,<:InfBlockBandedMatrix}}, B::AbstractVector)
    require_one_based_indexing(B)
    A = adjA.parent
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = blockbandwidths(Afactors)
    # todo: generalize
    l = 2l+1
    u = 2u+1
    @inbounds begin
        for k = nzzeros(B,1)+u:-1:1
            ν = k
            for j = 1:nB
                vBj = B[k,j]
                for i = max(1,ν-u):k-1
                    vBj += conj(Afactors[i,ν])*B[i,j]
                end
                vBj = conj(A.τ[k])*vBj
                B[k,j] -= vBj
                for i = max(1,ν-u):k-1
                    B[i,j] -= Afactors[i,ν]*vBj
                end
            end
        end
    end
    B
end

getindex(Q::QLPackedQ{T,<:InfBlockBandedMatrix{T}}, i::Integer, j::Integer) where T =
    (Q'*Vcat(Zeros{T}(i-1), one(T), Zeros{T}(∞)))[j]'

function (*)(A::QLPackedQ{T,<:InfBlockBandedMatrix}, x::AbstractVector{S}) where {T,S}
    TS = promote_op(matprod, T, S)
    lmul!(A, cache(convert(AbstractVector{TS},x)))
end

function (*)(A::Adjoint{T,<:QLPackedQ{T,<:InfBlockBandedMatrix}}, x::AbstractVector{S}) where {T,S}
    TS = promote_op(matprod, T, S)
    lmul!(A, cache(convert(AbstractVector{TS},x)))
end