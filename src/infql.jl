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
    (n,σ) = (abs(ñ),sign(ñ))
    if n^2 < abs2(B)
        throw(ContinuousSpectrumError())
    end

    e = sqrt(n^2 - abs2(B))
    d = σ*e*Z/n

    ql!([Z A B; 0 d e])
end


ql_hessenberg(A::InfBandedMatrix{T}; kwds...) where T = ql_hessenberg!(BandedMatrix(A, (bandwidth(A,1)+bandwidth(A,2),bandwidth(A,2))); kwds...)

toeptail(B::BandedMatrix{T}) where T =
    _BandedMatrix(B.data.args[end].args[1][1:end-B.u]*Ones{T}(1,∞), size(B,1), B.l-B.u, B.u)

# asymptotics of A[:,j:end] as j -> ∞
rightasymptotics(d::Hcat) = last(d.args)
rightasymptotics(d::Vcat) = Vcat(rightasymptotics.(d.args)...)
rightasymptotics(d) = d

function ql_hessenberg!(B::InfBandedMatrix{TT}; kwds...) where TT
    l,u = bandwidths(B)
    @assert u == 1
    T = toeptail(B)
    # Tail QL
    F∞ = ql_hessenberg(T; kwds...)
    Q∞, L∞ = F∞

    # populate finite data and do ql!
    data = bandeddata(B).args[1]
    B̃ = _BandedMatrix(data, size(data,2), l,u)
    B̃[end,end] = L∞[1,1]
    B̃[end:end,end-l+1:end-1] = adjoint(Q∞)[1:1,1:l-1]*T[l:2(l-1),1:l-1]
    F = ql!(B̃)

    # fill in data with L∞
    B̃ = _BandedMatrix(B̃.data, size(data,2)+l, l,u)
    B̃[size(data,2)+1:end,end-l+1:end] = adjoint(Q∞)[2:l+1,1:l+1] * T[l:2l,1:l]


    # combine finite and infinite data
    H = Hcat(B̃.data, rightasymptotics(F∞.factors.data))
    QLHessenberg(_BandedMatrix(H, ℵ₀, l, 1), Vcat( LowerHessenbergQ(F.Q).q, F∞.q))
end

getindex(Q::QLPackedQ{T,<:InfBandedMatrix{T}}, i::Int, j::Int) where T =
    (Q'*[Zeros{T}(i-1); one(T); Zeros{T}(∞)])[j]'

getL(Q::QL, ::NTuple{2,InfiniteCardinal{0}}) = LowerTriangular(Q.factors)
getL(Q::QLHessenberg, ::NTuple{2,InfiniteCardinal{0}}) = LowerTriangular(Q.factors)

# number of structural non-zeros in axis k
nzzeros(A::AbstractArray, k) = size(A,k)
nzzeros(::Zeros, k) = 0
nzzeros(B::Vcat, k) = sum(size.(B.args[1:end-1],k))
nzzeros(B::CachedArray, k) = max(B.datasize[k], nzzeros(B.array,k))
function nzzeros(B::AbstractMatrix, k)
    l,u = bandwidths(B)
    k == 1 ? size(B,2) + l : size(B,1) + u
end

function materialize!(M::Lmul{<:QLPackedQLayout{<:BandedColumns},<:PaddedLayout})
    A,B = M.A,M.B
    require_one_based_indexing(B)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    D = Afactors.data
    for k = 1:∞
        ν = k
        allzero = k > nzzeros(B,1) ? true : false
        for j = 1:nB
            vBj = B[k,j]
            for i = max(1,ν-u):k-1
                Bij = B[i,j]
                if !iszero(Bij)
                    allzero = false
                    vBj += conj(D[i-ν+u+1,ν])*Bij
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
    B
end

function materialize!(M::Lmul{<:AdjQLPackedQLayout{<:BandedColumns},<:PaddedLayout})
    adjA,B = M.A,M.B
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

function _lmul_cache(A::AbstractMatrix{T}, x::AbstractVector{S}) where {T,S}
    TS = promote_op(matprod, T, S)
    lmul!(A, cache(convert(AbstractVector{TS},x)))
end

(*)(A::QLPackedQ{T,<:InfBandedMatrix}, x::AbstractVector) where {T} = _lmul_cache(A, x)
(*)(A::Adjoint{T,<:QLPackedQ{T,<:InfBandedMatrix}}, x::AbstractVector) where {T} = _lmul_cache(A, x)
(*)(A::QLPackedQ{T,<:InfBandedMatrix}, x::LazyVector) where {T} = _lmul_cache(A, x)
(*)(A::Adjoint{T,<:QLPackedQ{T,<:InfBandedMatrix}}, x::LazyVector) where {T} = _lmul_cache(A, x)


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

function _blocktripert_ql(A, d, e)
    N = max(length(A.blocks.du.args[1])+1,length(A.blocks.d.args[1]),length(A.blocks.dl.args[1]))
    c,a,b = A[Block(N+1,N)],A[Block(N,N)],A[Block(N-1,N)]
    P,τ = blocktailiterate(c,a,b,d,e)
    B = BlockBandedMatrix(A,(2,1))


    BB = _BlockBandedMatrix(B.data.args[1], (fill(2,N+2), fill(2,N)), (2,1))
    BB[Block(N),Block.(N-1:N)] .= P[Block(1), Block.(1:2)]
    F = ql!(view(BB, Block.(1:N), Block.(1:N)))
    BB[Block(N+1),Block.(N-1:N)] .= P[Block(2), Block.(1:2)]
    BB[Block(N+2),Block(N)] .= P[Block(2), Block.(1)]


    QL(_BlockSkylineMatrix(Vcat(BB.data, mortar(Fill(vec(Vcat(P[Block(1,3)], P[Block(2,3)], P[Block(2,2)], P[Block(2,1)])),∞))),B.block_sizes),
            Vcat(F.τ,mortar(Fill(τ,∞)))), P[Block(1,1)], P[Block(1,2)]
end

ql(A::BlockTriPertToeplitz) = _blocktripert_ql(A, A[Block(2,3)], A[Block(3,3)])[1]

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

ldiv!(F::QLProduct, b::AbstractVector) = ldiv!(F.L, lmul!(F.Q',b))
ldiv!(F::QLProduct, b::LayoutVector) = ldiv!(F.L, lmul!(F.Q',b))

function materialize!(M::MatLdivVec{<:TriangularLayout{'L','N',BandedColumns{PertConstRows}},<:PaddedLayout})
    A,b = M.A,M.B
    require_one_based_indexing(A, b)
    n = size(A, 2)
    if !(n == length(b))
        throw(DimensionMismatch("second dimension of left hand side A, $n, and length of right hand side b, $(length(b)), must be equal"))
    end
    data = triangulardata(A)
    nz = nzzeros(b,1)
    @inbounds for j in 1:n
        iszero(data[j,j]) && throw(SingularException(j))
        bj = b[j] = data[j,j] \ b[j]
        allzero = j > nz && iszero(bj)
        for i in (j+1:n) ∩ colsupport(data,j)
            b[i] -= data[i,j] * bj
            allzero = allzero && iszero(b[i])
        end
        allzero && break
    end
    b
end

_ql(layout, ::NTuple{2,OneToInf{Int}}, A, args...; kwds...) = error("Not implemented")

_data_tail(::PaddedLayout, a) = paddeddata(a), zero(eltype(a))
_data_tail(::AbstractFillLayout, a) = Vector{eltype(a)}(), getindex_value(a)
_data_tail(::CachedLayout, a) = cacheddata(a), getindex_value(a.array)
function _data_tail(::ApplyLayout{typeof(vcat)}, a)
    args = arguments(vcat, a)
    dat,tl = _data_tail(last(args))
    vcat(most(args)..., dat), tl
end
_data_tail(a) = _data_tail(MemoryLayout(a), a)

function _ql(::SymTridiagonalLayout, ::NTuple{2,OneToInf{Int}}, A, args...; kwds...)
    T = eltype(A)
    d,d∞ = _data_tail(A.dv)
    ev,ev∞ = _data_tail(A.ev)

    m = max(length(d), length(ev)+1)
    dat = zeros(T, 3, m)
    dat[1,2:1+length(ev)] .= ev
    dat[1,2+length(ev):end] .= ev∞
    dat[2,1:length(d)] .= d
    dat[2,1+length(d):end] .= d∞
    dat[3,1:length(ev)] .= ev
    dat[3,1+length(ev):end] .= ev∞

    ql(_BandedMatrix(Hcat(dat, [ev∞,d∞,ev∞] * Ones{T}(1,∞)), ℵ₀, 1, 1), args...; kwds...)
end



# TODO: This should be redesigned as ql(BandedMatrix(A))
# But we need to support dispatch on axes
function _ql(::TridiagonalLayout, ::NTuple{2,OneToInf{Int}}, A, args...; kwds...)
    T = eltype(A)
    d,d∞ = _data_tail(A.d)
    dl,dl∞ = _data_tail(A.dl)
    du,du∞ = _data_tail(A.du)

    m = max(length(d), length(du)+1, length(dl))
    dat = zeros(T, 3, m)
    dat[1,2:1+length(du)] .= du
    dat[1,2+length(du):end] .= du∞
    dat[2,1:length(d)] .= d
    dat[2,1+length(d):end] .= d∞
    dat[3,1:length(dl)] .= dl
    dat[3,1+length(dl):end] .= dl∞

    ql(_BandedMatrix(Hcat(dat, [du∞,d∞,dl∞] * Ones{T}(1,∞)), ℵ₀, 1, 1), args...; kwds...)
end


###
# L*Q special case
###

copy(M::Mul{TriangularLayout{'L', 'N', PertToeplitzLayout}, HessenbergQLayout{'L'}}) =
    ApplyArray(*, M.A, M.B)

copy(M::Mul{HessenbergQLayout{'L'}, TriangularLayout{'L', 'N', PertToeplitzLayout}}) =
    ApplyArray(*, M.A, M.B)


function LazyBandedMatrices._SymTridiagonal(::Tuple{TriangularLayout{'L', 'N', PertToeplitzLayout}, HessenbergQLayout{'L'}}, A)
    T = eltype(A)
    L,Q = arguments(*, A)
    Ldat,L∞ = arguments(hcat, L.data.data)
    Qdat, Q∞ = arguments(vcat, Q.q)

    m = max(size(Ldat,2)+2, length(Qdat)+1)
    dv = [A[k,k] for k=1:m]
    ev = [A[k,k+1] for k=1:m-1]
    SymTridiagonal([dv; Fill(dv[end],∞)], [ev; Fill(ev[end],∞)])
end


###
# Experimental adaptive finite section QL
###
mutable struct QLFiniteSectionQFactor{T} <: AbstractCachedMatrix{T}
    data::AbstractMatrix{T}
    M::AbstractMatrix{T}
    datasize::Int
    tol::Real
    QLFiniteSectionQFactor{T}(array::AbstractMatrix{T},M::AbstractMatrix{T},N::Int,tol) where T = new{T}(array, M, N, tol)
end

mutable struct QLFiniteSectionLFactor{T} <: AbstractCachedMatrix{T}
    data::AbstractMatrix{T}
    M::AbstractMatrix{T}
    datasize::Int
    tol::Real
    QLFiniteSectionLFactor{T}(array::AbstractMatrix{T},M::AbstractMatrix{T},N::Int,tol) where T = new{T}(array, M, N, tol)
end

size(::QLFiniteSectionQFactor) = (ℵ₀, ℵ₀)
size(::QLFiniteSectionLFactor) = (ℵ₀, ℵ₀)

mutable struct AdaptiveQLFiniteSection{T}
    Q::QLFiniteSectionQFactor{T}
    L::QLFiniteSectionLFactor{T}
    tol
end

# Computes the initial data for the finite section based QL decomposition
function AdaptiveQLFiniteSection(A::AbstractMatrix{T}, tol = eps(T), maxN = 10000) where T
    @assert size(A) == (ℵ₀, ℵ₀) # only makes sense for infinite matrices
    N = 50 # We initialize with a 50 × 50 block that is adaptively expanded
    Qerr = one(T)
    Lerr = one(T)
    Qs, Ls = ql(A[1:N,1:N])
    while norm(Qerr,2)>tol || norm(Lerr,2)>tol
        # compute QL for small (N×N) finite section and large (2N×2N) finite section
        Ql, Ll = ql(A[1:2N,1:2N])
        # stop if desired level of convergence achieved
        Qerr = Ql[1:50,1:50]-Qs[1:50,1:50]
        Lerr = Ll[1:50,1:50]-Ls[1:50,1:50]
        if N ≥ maxN
            error("Reached max. iterations in finite section QL without convergence to desired tolerance.")
        end
        Qs, Ls = Ql, Ll
        N = 2*N
    end
    return AdaptiveQLFiniteSection{T}(QLFiniteSectionQFactor{T}(Qs[1:50,1:50], A, 50, tol),QLFiniteSectionLFactor{T}(Ls[1:50,1:50], A, 50, tol),tol)
end

# Resize and filling functions for cached implementation
function resizedata!(K::QLFiniteSectionLFactor, nm::Integer)
    νμ = K.datasize
    if nm > νμ
        olddata = copy(K.data)
        K.data = similar(K.data, nm, nm)
        K.data[axes(olddata)...] = olddata
        inds = νμ:nm
        cache_filldata!(K, inds)
        K.datasize = size(K.data,1)
    end
    K
end

function cache_filldata!(L::QLFiniteSectionLFactor{T}, inds::UnitRange{Int}) where T
    j = maximum(inds)
    maxN = 1000*j
    Qerr = one(T)
    Lerr = one(T)
    N = j
    Qs, Ls = ql(L.M[1:N,1:N])
    while norm(Qerr,2)>L.tol || norm(Lerr,2)>L.tol
        # compute QL for small (N×N) finite section and large (2N×2N) finite section
        Ql, Ll = ql(L.M[1:2N,1:2N])
        # stop if desired level of convergence achieved
        Qerr = Ql[1:j,1:j]-Qs[1:j,1:j]
        Lerr = Ll[1:j,1:j]-Ls[1:j,1:j]
        if N == maxN
            error("Reached max. iterations in finite section QL without convergence to desired tolerance.")
        end
        Qs, Ls = Ql, Ll
        N = 2*N
    end
    L.data = Ls[1:j,1:j]
end
function resizedata!(K::QLFiniteSectionQFactor, nm::Integer)
    νμ = K.datasize
    if nm > νμ
        olddata = copy(K.data)
        K.data = similar(K.data, nm, nm)
        K.data[axes(olddata)...] = olddata
        inds = νμ:nm
        cache_filldata!(K, inds)
        K.datasize = size(K.data,1)
    end
    K
end
function cache_filldata!(Q::QLFiniteSectionQFactor{T}, inds::UnitRange{Int}) where T
    j = maximum(inds)
    maxN = 1000*j
    Qerr = one(T)
    Lerr = one(T)
    N = j
    Qs, Ls = ql(Q.M[1:N,1:N])
    while norm(Qerr,2)>Q.tol || norm(Lerr,2)>Q.tol
        # compute QL for small (N×N) finite section and large (2N×2N) finite section
        Ql, Ll = ql(Q.M[1:2N,1:2N])
        # stop if desired level of convergence achieved
        Qerr = Ql[1:j,1:j]-Qs[1:j,1:j]
        Lerr = Ll[1:j,1:j]-Ls[1:j,1:j]
        if N == maxN
            error("Reached max. iterations in finite section QL without convergence to desired tolerance.")
        end
        Qs, Ls = Ql, Ll
        N = 2*N
    end
    Q.data = Qs[1:j,1:j]
end

function getindex(K::QLFiniteSectionQFactor, k::Integer, j::Integer)
    resizedata!(K, max(k,j))
    K.data[k, j]
end
function getindex(K::QLFiniteSectionLFactor, k::Integer, j::Integer)
    resizedata!(K, max(k,j))
    K.data[k, j]
end
function getindex(K::QLFiniteSectionQFactor, kr::Integer, jr::UnitRange{Int})
    resizedata!(K, maximum(jr))
    K.data[kr, jr]
end
function getindex(K::QLFiniteSectionLFactor, kr::Integer, jr::UnitRange{Int})
    resizedata!(K, maximum(jr))
    K.data[kr, jr]
end
function getindex(K::QLFiniteSectionQFactor, kr::Integer, jr::UnitRange{Int})
    resizedata!(K, maximum(jr))
    K.data[kr, jr]
end
function getindex(K::QLFiniteSectionLFactor, jr::UnitRange{Int}, kr::Integer)
    resizedata!(K, maximum(jr))
    K.data[kr, jr]
end
function getindex(K::QLFiniteSectionQFactor, jr::UnitRange{Int}, kr::Integer)
    resizedata!(K, maximum(jr))
    K.data[kr, jr]
end
function getindex(K::QLFiniteSectionQFactor, I::Vararg{Int,2})
    resizedata!(K,maximum(I))
    getindex(K.data,I[1],I[2])
end
function getindex(K::QLFiniteSectionLFactor, I::Vararg{Int,2})
    resizedata!(K,maximum(I))
    getindex(K.data,I[1],I[2])
end
function getindex(K::QLFiniteSectionQFactor, kr::UnitRange{Int}, jr::UnitRange{Int})
    resizedata!(K, max(maximum(jr),maximum(kr)))
    K.data[kr, jr]
end
function getindex(K::QLFiniteSectionLFactor, kr::UnitRange{Int}, jr::UnitRange{Int})
    resizedata!(K, max(maximum(jr),maximum(kr)))
    K.data[kr, jr]
end