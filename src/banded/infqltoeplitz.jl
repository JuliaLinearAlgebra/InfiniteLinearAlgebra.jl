
# this gives the d and e so that
# [Z A B;
#  0 d e]
#
# is the fixed point
function tail_de(a::AbstractVector{T}; branch=findmax) where {T<:Real}
    m = length(a)
    C = [view(a, m-1:-1:1) Vcat(-a[end] * Eye(m - 2), Zeros{T}(1, m - 2))]
    λ, V = eigen(C)
    n2, j = branch(abs2.(λ))
    isreal(λ[j]) || throw(DomainError(a, "Real-valued QL factorization does not exist. Try ql(complex(A)) to see if a complex-valued QL factorization exists."))
    n2 ≥ a[end]^2 || throw(DomainError(a, "QL factorization does not exist. This could indicate that the operator is not Fredholm or that the dimension of the kernel exceeds that of the co-kernel. Try again with the adjoint."))
    c = sqrt((n2 - a[end]^2) / real(V[1, j])^2)
    c * real(V[end:-1:1, j])
end

function tail_de(a::AbstractVector{T}; branch=findmax) where {T}
    m = length(a)
    C = [view(a, m-1:-1:1) Vcat(-a[end] * Eye(m - 2), Zeros{T}(1, m - 2))]
    λ, V = eigen(C)::Eigen{float(T),float(T),Matrix{float(T)},Vector{float(T)}}
    n2, j = branch(abs2.(λ))
    n2 ≥ abs2(a[end]) || throw(DomainError(a, "QL factorization does not exist. This could indicate that the operator is not Fredholm or that the dimension of the kernel exceeds that of the co-kernel. Try again with the adjoint."))
    c_abs = sqrt((n2 - abs2(a[end])) / abs2(V[1, j]))
    c_sgn = -sign(λ[j]) / sign(V[1, j] * a[end-1] - V[2, j] * a[end])
    c_sgn * c_abs * V[end:-1:1, j]
end


# this calculates the QL decomposition of X and corrects sign
function ql_X!(X)
    s = sign(real(X[2, end]))
    F = ql!(X)
    if s ≠ sign(real(X[1, end-1])) # we need to normalise the sign if ql! flips it
        F.τ[1] = 2 - F.τ[1] # 1-F.τ[1] is the sign so this flips it
        X[1, 1:end-1] *= -1
    end
    F
end




function ql(Op::TriToeplitz{T}; kwds...) where {T<:Real}
    Z, A, B = Op.dl.value, Op.d.value, Op.du.value
    d, e = tail_de([Z, A, B]; kwds...) # fixed point of QL but with two Qs, one that changes sign
    X = [Z A B; zero(T) d e]
    F = ql_X!(X)
    t, ω = F.τ[2], X[1, end]
    QL(_BandedMatrix(Hcat([zero(T), e, X[2, 2], X[2, 1]], [ω, X[2, 3], X[2, 2], X[2, 1]] * Ones{T}(1, ∞)), ℵ₀, 2, 1), Vcat(F.τ[1], Fill(t, ∞)))
end

ql(Op::TriToeplitz{T}) where {T} = ql(InfToeplitz(Op))

# ql for Lower hessenberg InfToeplitz
function ql_hessenberg(A::InfToeplitz{T}; kwds...) where {T}
    l, u = bandwidths(A)
    @assert u == 1
    a = reverse(A.data.args[1])
    de = tail_de(a; kwds...)
    X = [transpose(a); zero(T) transpose(de)]::Matrix{float(T)}
    F = ql_X!(X) # calculate data for fixed point
    factors = _BandedMatrix(Hcat([zero(T); X[1, end-1]; X[2, end-1:-1:1]], [0; X[2, end:-1:1]] * Ones{float(T)}(1, ∞)), ℵ₀, l + u, 1)
    QLHessenberg(factors, Fill(F.Q, ∞))
end


# remove one band of A
function ql_pruneband(A; kwds...)
    l, u = bandwidths(A)
    A_hess = A[:, u:end]
    Q, L = ql_hessenberg(A_hess; kwds...)
    p = size(_pertdata(bandeddata(parent(L))), 2) + u + 1 # pert size
    dat = (UpperHessenbergQ((Q').q[1:(p+l)])) * A[1:p+l+1, 1:p]
    pert = Array{eltype(dat)}(undef, l + u + 1, size(dat, 2) - 1)
    for j = 1:u
        pert[u-j+1:end, j] .= view(dat, 1:l+j+1, j)
    end
    for j = u+1:size(pert, 2)
        pert[:, j] .= view(dat, j-u+1:j+l+1, j)
    end
    H = _BandedMatrix(Hcat(pert, dat[end-l-u:end, end] * Ones{eltype(dat)}(1, ∞)), ℵ₀, l + 1, u - 1)
    Q, H
end

# represent Q as a product of orthogonal operations
struct ProductQ{T,QQ<:Tuple} <: LayoutQ{T}
    Qs::QQ
end

ArrayLayouts.@layoutmatrix ProductQ
ArrayLayouts.@_layoutlmul ProductQ

ProductQ(Qs::AbstractMatrix...) = ProductQ{mapreduce(eltype, promote_type, Qs),typeof(Qs)}(Qs)

adjoint(Q::ProductQ) = ProductQ(reverse(map(adjoint, Q.Qs))...)

size(Q::ProductQ, dim::Integer) = size(dim == 1 ? Q.Qs[1] : last(Q.Qs), dim == 2 ? 1 : dim)
axes(Q::ProductQ, dim::Integer) = axes(dim == 1 ? Q.Qs[1] : last(Q.Qs), dim == 2 ? 1 : dim)

function lmul!(Q::ProductQ, v::AbstractVecOrMat)
    for j = length(Q.Qs):-1:1
        lmul!(Q.Qs[j], v)
    end
    v
end

# Avoid ambiguities
getindex(Q::ProductQ, i::Int, j::Int) = Q[:, j][i]

function getindex(Q::ProductQ, ::Colon, j::Int)
    y = zeros(eltype(Q), size(Q, 2))
    y[j] = 1
    lmul!(Q, y)
end
getindex(Q::ProductQ{<:Any,<:Tuple{Vararg{LowerHessenbergQ}}}, i::Int, j::Int) = (Q')[j, i]'

function _productq_mul(A::ProductQ{T}, x::AbstractVector{S}) where {T,S}
    TS = promote_op(matprod, T, S)
    lmul!(A, Base.copymutable(convert(AbstractVector{TS}, x)))
end

mul(A::ProductQ, x::AbstractVector) = _productq_mul(A, x)



mul(Q::ProductQ, X::AbstractMatrix) = ApplyArray(*, Q.Qs...) * X
mul(X::AbstractMatrix, Q::ProductQ) = X * ApplyArray(*, Q.Qs...)



# LQ where Q is a product of orthogonal operations
struct QLProduct{T,QQ<:Tuple,LL} <: Factorization{T}
    Qs::QQ
    L::LL
end



QLProduct(Qs::Tuple, L::AbstractMatrix{T}) where {T} = QLProduct{T,typeof(Qs),typeof(L)}(Qs, L)
QLProduct(F::QLHessenberg) = QLProduct(tuple(F.Q), F.L)

# iteration for destructuring into components
Base.iterate(S::QLProduct) = (S.Q, Val(:L))
Base.iterate(S::QLProduct, ::Val{:L}) = (S.L, Val(:done))
Base.iterate(S::QLProduct, ::Val{:done}) = nothing

QLProduct{T}(A::QLProduct) where {T} = QLProduct(convert.(AbstractMatrix{T}, A.Qs), convert(AbstractMatrix{T}, A.L))
Factorization{T}(A::QLProduct{T}) where {T} = A
Factorization{T}(A::QLProduct) where {T} = QLProduct{T}(A)
AbstractMatrix(F::QLProduct) = F.Q * F.L
AbstractArray(F::QLProduct) = AbstractMatrix(F)
Matrix(F::QLProduct) = Array(AbstractArray(F))
Array(F::QLProduct) = Matrix(F)

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::QLProduct)
    summary(io, F)
    println(io)
    println(io, "Q factor:")
    show(io, mime, F.Q)
    println(io, "\nL factor:")
    show(io, mime, F.L)
end

@inline getL(F::QLProduct) = getfield(F, :L)
@inline getQ(F::QLProduct) = ProductQ(F.Qs...)

function getproperty(F::QLProduct, d::Symbol)
    if d == :L
        return getL(F)
    elseif d == :Q
        return getQ(F)
    else
        getfield(F, d)
    end
end

Base.propertynames(F::QLProduct, private::Bool=false) =
    (:L, :Q, (private ? fieldnames(typeof(F)) : ())...)

function _inf_ql(A::AbstractMatrix{T}; kwds...) where {T}
    _, u = bandwidths(A)
    u ≤ 0 && return QLProduct(tuple(Eye{float(T)}(∞)), A)
    u == 1 && return QLProduct(ql_hessenberg(A; kwds...))
    Q1, H1 = ql_pruneband(A; kwds...)
    F̃ = ql(H1; kwds...)
    QLProduct(tuple(Q1, F̃.Qs...), F̃.L)
end

ql(A::InfToeplitz; kwds...) = _inf_ql(A; kwds...)
ql(A::PertToeplitz; kwds...) = _inf_ql(A; kwds...)

ql(A::Adjoint{<:Any,<:InfToeplitz}) = ql(BandedMatrix(A))
ql(A::Adjoint{<:Any,<:PertToeplitz}) = ql(BandedMatrix(A))