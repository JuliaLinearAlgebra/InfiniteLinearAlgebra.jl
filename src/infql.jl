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

getL(Q::QL, ::NTuple{2,InfiniteCardinal{0}}) = LowerTriangular(Q.factors)
getL(Q::QLHessenberg, ::NTuple{2,InfiniteCardinal{0}}) = LowerTriangular(Q.factors)


function materialize!(M::Lmul{<:QLPackedQLayout{<:BandedColumns},<:AbstractPaddedLayout})
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
        allzero = k > last(colsupport(B)) ? true : false
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

function materialize!(M::Lmul{<:AdjQLPackedQLayout{<:BandedColumns},<:AbstractPaddedLayout})
    adjA,B = M.A,M.B
    require_one_based_indexing(B)
    A = parent(adjA)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    D = Afactors.data
    @inbounds begin
        for k = last(colsupport(B))+u:-1:1
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



###
# BlockTridiagonal
####

"""
    blocktailql(c, a, b)

computes the stationary tail of the QL factorisation of the block-tridiagonal Toeplitz operator
with sub-diagonal, diagonal and super-diagonal blocks `c`, `a` and `b`. That is, it returns
a unitary `G`, blocks `d` and `e`, and blocks `L₂`, `L₁` and lower-triangular `L₀` such that

    G' * [c a b; 0 d e] == [d e 0; L₂ L₁ L₀]

Applying `G'` to consecutive block rows, starting infinitely far down, thus reduces the operator to
lower-triangular form, where `[d e]` is the block row carried up to the next application.
"""
function blocktailql(c, a, b)
    T = float(promote_type(eltype(c), eltype(a), eltype(b)))
    n = LinearAlgebra.checksquare(a)
    Z = zeros(T, n, n)
    # Since [d e 0] lies in the row space of [c a b; 0 d e], eliminating the last block column
    # shows that W = e \ d satisfies b*W^2 - a*W + c == 0. The minimal solvent gives the decaying tail.
    W = _realifreal(T, matrixroot(b, -a, c))
    K = a - b*W # so that K*W == c
    # The first block column of G is then [e*inv(K) -e*inv(K)*b*inv(e)]', which is orthonormal iff
    # E = e'e satisfies E == K'*inv(I + b*inv(E)*b')*K. Writing F = E + b'b this becomes
    # F == D - B'*inv(F)*B, i.e., F \ B is a solvent of B'*X^2 - D*X + B == 0.
    D = K'K + b'b
    B = b'K
    λ, V = eigen(Hermitian(D - B'*_realifreal(T, matrixroot(B', -D, B)) - b'b))
    e = Diagonal(sqrt.(max.(λ, 0))) * V' # any e with e'e == E will do
    d = e*W
    Q = ql([b; e]).Q * Matrix{T}(I, 2n, 2n)
    X = Q' * [c a b; Z d e] # == [Ω'*[d e] 0; L₂ L₁ L₀] for some unitary Ω
    # Ω solves the Procrustes problem of rotating X[1:n,1:2n] onto [d e]
    S = svd([d e] * X[1:n,1:2n]')
    G = Q * [S.V*S.U' Z; Z I]
    G, d, e, X[n+1:2n,1:n], X[n+1:2n,n+1:2n], tril!(X[n+1:2n,2n+1:3n])
end

"""
    BlockTailQ(Q₀, G)

represents the infinite unitary operator `⋯ * G₃ * G₂ * G₁ * Q₀` where `Q₀` acts on the first
`m = size(Q₀,1)` rows and `Gₖ` is `G` acting on rows `m+(k-2)n+1:m+k*n`, with `size(G) == (2n,2n)`.
"""
struct BlockTailQ{T,QQ<:Union{AbstractMatrix{T},AbstractQ{T}},GG<:AbstractMatrix{T}} <: AbstractQ{T}
    Q₀::QQ
    G::GG
end

size(::BlockTailQ) = (ℵ₀, ℵ₀)
function axes(Q::BlockTailQ)
    ax = blockedrange(Fill(size(Q.G,1) ÷ 2, ∞))
    (ax, ax)
end

function lmul!(Q::BlockTailQ, x::AbstractVector)
    G, m = Q.G, size(Q.Q₀,1)
    n = size(G,1) ÷ 2
    T = eltype(x)
    nz = last(colsupport(x))
    resizedata!(x, m)
    lmul!(Q.Q₀, view(x.data, 1:m))
    t = Vector{T}(undef, 2n)
    for k = 1:∞
        kr = m+(k-2)n+1:m+k*n
        resizedata!(x, last(kr))
        v = view(x.data, kr)
        mul!(t, G, v)
        v .= t
        # beyond the support of x only the bottom block is carried to the next Gₖ
        last(kr) > nz && norm(view(t, n+1:2n)) ≤ 10floatmin(real(T)) && break
    end
    x
end

function lmul!(adjQ::AdjointQ{<:Any,<:BlockTailQ}, x::AbstractVector)
    Q = parent(adjQ)
    G, m = Q.G, size(Q.Q₀,1)
    n = size(G,1) ÷ 2
    T = eltype(x)
    # Gₖ' acts as the identity once its rows are beyond the support of x
    K = fld(last(colsupport(x)) - m - 1, n) + 2
    resizedata!(x, max(m, m+K*n))
    t = Vector{T}(undef, 2n)
    for k = K:-1:1
        v = view(x.data, m+(k-2)n+1:m+k*n)
        mul!(t, G', v)
        v .= t
    end
    lmul!(Q.Q₀', view(x.data, 1:m))
    x
end

for Typ in (:BlockTailQ, :(AdjointQ{<:Any,<:BlockTailQ}))
    @eval begin
        function (*)(Q::$Typ, x::AbstractVector{S}) where S
            TS = promote_op(matprod, eltype(Q), S)
            lmul!(Q, cache(convert(AbstractVector{TS}, x)))
        end
        function (*)(Q::$Typ, x::LayoutVector{S}) where S
            TS = promote_op(matprod, eltype(Q), S)
            lmul!(Q, cache(convert(AbstractVector{TS}, x)))
        end
    end
end

getindex(Q::BlockTailQ{T}, i::Int, j::Int) where T = (Q'*Vcat(Zeros{T}(i-1), one(T), Zeros{T}(∞)))[j]'
function getindex(Q::BlockTailQ{T}, I::AbstractVector{Int}, J::AbstractVector{Int}) where T
    ret = Matrix{T}(undef, length(I), length(J))
    for (k,i) in enumerate(I)
        ret[k,:] .= conj.((Q'*Vcat(Zeros{T}(i-1), one(T), Zeros{T}(∞)))[J])
    end
    ret
end

@inline getQ(F::QLProduct{<:Any,<:Tuple{BlockTailQ}}) = only(F.Qs)

function ql(A::BlockTriPertToeplitz{T}) where T
    dl, d, du = A.blocks.dl, A.blocks.d, A.blocks.du
    c, a, b = getindex_value(last(dl.args)), getindex_value(last(d.args)), getindex_value(last(du.args))
    n = size(a,1)
    S = float(T)
    G, dₜ, eₜ, L₂, L₁, L₀ = blocktailql(c, a, b)
    # block rows N+1, N+2, … of A are Toeplitz, and are reduced by Gₖ, which carry into block row N
    N = max(length(first(dl.args))+1, length(first(d.args)), length(first(du.args))+1, 2)
    B = BlockBandedMatrix(A, (2,1))
    BB = BlockBandedMatrix{S}(undef, fill(n,N+2), fill(n,N), (2,1))
    fill!(BB.data, zero(S))
    for J = 1:N, K = max(1,J-1):J+1
        BB[Block(K,J)] = A[Block(K,J)]
    end
    Y = G' * [A[Block(N,N-1)] A[Block(N,N)] A[Block(N,N+1)]; zeros(S,n,n) dₜ eₜ]
    BB[Block(N,N-1)] = Y[1:n,1:n]
    BB[Block(N,N)] = Y[1:n,n+1:2n]
    F = ql!(view(BB, Block.(1:N), Block.(1:N)))
    Q₀ = QLPackedQ(Matrix(F.factors), F.τ)
    BB[Block(N+1,N-1)] = Y[n+1:2n,1:n]
    BB[Block(N+1,N)] = Y[n+1:2n,n+1:2n]
    BB[Block(N+2,N)] = L₂
    tail = vec(Vcat(zeros(S,n,n), L₀, L₁, L₂))
    QLProduct((BlockTailQ(Q₀, G),), LowerTriangular(_BlockSkylineMatrix(Vcat(BB.data, mortar(Fill(tail,∞))), B.block_sizes)))
end

ql(A::Adjoint{T,BlockTriPertToeplitz{T}}) where T = ql(BlockTridiagonal(A))

const InfBlockBandedMatrix{T} = BlockSkylineMatrix{T,<:Vcat{T,1,<:Tuple{Vector{T},<:BlockArray{T,1,<:Fill{<:Any,1,Tuple{OneToInf{Int64}}}}}}}

ldiv!(F::QLProduct, b::AbstractVector) = ldiv!(F.L, lmul!(F.Q',b))
ldiv!(F::QLProduct, b::LayoutVector) = ldiv!(F.L, lmul!(F.Q',b))

function materialize!(M::MatLdivVec{<:TriangularLayout{'L','N',BandedColumns{PertConstRows}},<:AbstractPaddedLayout})
    A,b = M.A,M.B
    require_one_based_indexing(A, b)
    n = size(A, 2)
    if !(n == length(b))
        throw(DimensionMismatch("second dimension of left hand side A, $n, and length of right hand side b, $(length(b)), must be equal"))
    end
    data = triangulardata(A)
    nz = last(colsupport(b))
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

ql_layout(layout, ::NTuple{2,OneToInf{Int}}, A, args...; kwds...) = error("Not implemented")

_data_tail(::PaddedColumns, a) = paddeddata(a), zero(eltype(a))
_data_tail(::AbstractFillLayout, a) = Vector{eltype(a)}(), getindex_value(a)
_data_tail(::CachedLayout, a) = cacheddata(a), getindex_value(a.array)
function _data_tail(::ApplyLayout{typeof(vcat)}, a)
    args = arguments(vcat, a)
    dat,tl = _data_tail(last(args))
    vcat(Base.front(args)..., dat), tl
end
_data_tail(a) = _data_tail(MemoryLayout(a), a)

function ql_layout(::Union{PertTridiagonalToeplitzLayout,SymTridiagonalLayout}, ::NTuple{2,OneToInf{Int}}, A, args...; kwds...)
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
function ql_layout(::TridiagonalLayout, ::NTuple{2,OneToInf{Int}}, A, args...; kwds...)
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
mutable struct AdaptiveQLTau{T} <: AbstractCachedVector{T}
    data::Vector{T}
    M::AbstractMatrix{T}
    datasize::Integer
    tol::Real
    AdaptiveQLTau{T}(D, M, N::Integer, tol) where T = new{T}(D, M, N, tol)
end
mutable struct AdaptiveQLFactors{T} <: AbstractCachedMatrix{T}
    data::BandedMatrix{T}
    M::AbstractMatrix{T}
    datasize::Tuple{Int, Int}
    tol::Real
    AdaptiveQLFactors{T}(D, M, N::Tuple{Int, Int}, tol) where T = new{T}(D, M, N, tol)
end

size(::AdaptiveQLFactors) = (ℵ₀, ℵ₀)
size(::AdaptiveQLTau) = (ℵ₀, )

# adaptive QL accepts optional tolerance
function ql(A::InfBandedMatrix{T}, tol = eps(float(real(T)))) where T
    factors, τ = initialadaptiveQLblock(A, tol)
    QL(AdaptiveQLFactors{T}(factors, A, size(factors), tol),AdaptiveQLTau{T}(τ, A, length(τ), tol))
end

# Computes the initial data for the finite section based QL decomposition
function initialadaptiveQLblock(A::AbstractMatrix{T}, tol) where T
    maxN = 10000   # Prevent runaway loop
    j = 50         # We initialize with a 50 × 50 block that is adaptively expanded
    Lerr = one(real(T))
    N = j
    checkinds = max(1,j-bandwidth(A,1)-bandwidth(A,2))
    @inbounds Ls = ql(A[checkinds:N,checkinds:N]).L[2:j-checkinds+1,2:j-checkinds+1]
    @inbounds while Lerr > tol
        # compute QL for small finite section and large finite section
        Ll = ql(A[checkinds:2N,checkinds:2N]).L[2:j-checkinds+1,2:j-checkinds+1]
        # compare bottom right sections and stop if desired level of convergence achieved
        Lerr = norm(Ll-Ls,2)
        if N >= maxN
            error("Reached max. iterations in adaptive QL without convergence to desired tolerance.")
        end
        Ls = Ll
        N = 2*N
    end
    F = ql(A[1:(N÷2),1:(N÷2)])
    return (F.factors[1:50,1:50], F.τ[1:50])
end

# Resize and filling functions for cached implementation
function resizedata!(K::AdaptiveQLFactors, nm...)
    nm = maximum(nm)
    νμ = K.datasize[1]
    if nm > νμ
        olddata = copy(K.data)
        K.data = similar(K.data, nm, nm)
        K.data[axes(olddata)...] = olddata
        inds = νμ:nm
        cache_filldata!(K, inds)
        K.datasize = size(K.data)
    end
    K
end

function resizedata!(K::AdaptiveQLTau, nm...)
    nm = maximum(nm)
    νμ = K.datasize
    if nm > νμ
        resize!(K.data,nm)
        cache_filldata!(K, νμ:nm)
        K.datasize = size(K.data,1)
    end
    K
end

function cache_filldata!(A::AdaptiveQLFactors{T}, inds::UnitRange{Int}) where T
    j = maximum(inds)
    maxN = 1000*j # Prevent runaway loop
    Lerr = one(real(T))
    N = j
    checkinds = max(1,j-bandwidth(A.M,1)-bandwidth(A.M,2))
    @inbounds Ls = ql(A.M[checkinds:N,checkinds:N]).L[2:j-checkinds+1,2:j-checkinds+1]
    @inbounds while Lerr > A.tol
        # compute QL for small finite section and large finite section
        Ll = ql(A.M[checkinds:2N,checkinds:2N]).L[2:j-checkinds+1,2:j-checkinds+1]
        # compare bottom right sections and stop if desired level of convergence achieved
        Lerr = norm(Ll-Ls,2)
        if N >= maxN
            error("Reached max. iterations in adaptive QL without convergence to desired tolerance.")
        end
        Ls = Ll
        N = 2*N
    end
    A.data = ql(A.M[1:(N÷2),1:(N÷2)]).factors[1:j,1:j]
end

function cache_filldata!(A::AdaptiveQLTau{T}, inds::UnitRange{Int}) where T
    j = maximum(inds)
    maxN = 1000*j
    Lerr = one(real(T))
    N = j
    checkinds = max(1,j-bandwidth(A.M,1)-bandwidth(A.M,2))
    @inbounds Ls = ql(A.M[checkinds:N,checkinds:N]).L[2:j-checkinds+1,2:j-checkinds+1]
    @inbounds while Lerr > A.tol
        # compute QL for small finite section and large finite section
        Ll = ql(A.M[checkinds:2N,checkinds:2N]).L[2:j-checkinds+1,2:j-checkinds+1]
        # compare bottom right sections and stop if desired level of convergence achieved
        Lerr = norm(Ll-Ls,2)
        if N >= maxN
            error("Reached max. iterations in adaptive QL without convergence to desired tolerance.")
        end
        Ls = Ll
        N = 2*N
    end
    A.data = ql(A.M[1:(N÷2),1:(N÷2)]).τ[1:j]
end

# TODO: adaptively build L*b using caching and forward-substitution
*(L::LowerTriangular{T, AdaptiveQLFactors{T}}, b::LayoutVector) where T = ApplyArray(*, L, b)

MemoryLayout(::AdaptiveQLFactors) = LazyBandedLayout()
bandwidths(F::AdaptiveQLFactors) = bandwidths(F.data)

# Q = \\prod_{i=1}^{\\min(m,n)} (I - \\tau_i v_i v_i^T)
getindex(Q::QLPackedQ{T,<:AdaptiveQLFactors{T}}, i::Int, j::Int) where T =
(Q'*[Zeros{T}(i-1); one(T); Zeros{T}(∞)])[j]'
getindex(Q::QLPackedQ{<:Any,<:AdaptiveQLFactors}, I::AbstractVector{Int}, J::AbstractVector{Int}) =
    [Q[i,j] for i in I, j in J]
getindex(Q::QLPackedQ{<:Any,<:AdaptiveQLFactors}, I::Int, J::UnitRange{Int}) =
    [Q[i,j] for i in I, j in J]
getindex(Q::QLPackedQ{<:Any,<:AdaptiveQLFactors}, I::UnitRange{Int}, J::Int) =
    [Q[i,j] for i in I, j in J]

materialize!(M::Lmul{<:QLPackedQLayout{<:AbstractLazyLayout},<:AbstractPaddedLayout}) = ApplyArray(*,M.A,M.B)

function materialize!(M::Lmul{<:AdjQLPackedQLayout{<:AbstractLazyLayout},<:AbstractPaddedLayout})
    adjA,B = M.A,M.B
    A = parent(adjA)
    mA, nA = size(A.factors)
    mB, nB = size(B,1), size(B,2)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors
    l,u = bandwidths(Afactors)
    l = 2l+1
    u = 2u+1
    @inbounds begin
        for k = last(colsupport(B))+u:-1:1
            for j = 1:nB
                vBj = B[k,j]
                for i = max(1,k-u):k-1
                    vBj += conj(Afactors[i,k])*B[i,j]
                end
                vBj = conj(A.τ[k])*vBj
                B[k,j] -= vBj
                for i = max(1,k-u):k-1
                    B[i,j] -= Afactors[i,k]*vBj
                end
            end
        end
    end
    B
end
