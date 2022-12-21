isorthogonal(::AbstractQ) = true
isorthogonal(q) = q'q ≈ I

convert_eltype(Q::AbstractMatrix, ::Type{T}) where {T} = convert(AbstractMatrix{T}, Q)
if !(AbstractQ <: AbstractMatrix)
    convert_eltype(Q::AbstractQ, ::Type{T}) where {T} = convert(AbstractQ{T}, Q)
end

"""
    QLHessenberg(factors, q)

represents a Hessenberg QL factorization where factors contains L in its
lower triangular components and q is a vector of 2x2 orthogonal transformations
whose product gives Q.
"""
struct QLHessenberg{T,S<:AbstractMatrix{T},QT<:AbstractVector{<:Union{AbstractMatrix{T},AbstractQ{T}}}} <: Factorization{T}
    factors::S
    q::QT

    function QLHessenberg{T,S,QT}(factors, q) where {T,S<:AbstractMatrix{T},QT<:AbstractVector{<:Union{AbstractMatrix{T},AbstractQ{T}}}}
        require_one_based_indexing(factors)
        new{T,S,QT}(factors, q)
    end
end

QLHessenberg(factors::AbstractMatrix{T}, q::AbstractVector{<:Union{AbstractMatrix{T},AbstractQ{T}}}) where {T} =
    QLHessenberg{T,typeof(factors),typeof(q)}(factors, q)
QLHessenberg{T}(factors::AbstractMatrix, q::AbstractVector) where {T} =
    QLHessenberg(convert(AbstractMatrix{T}, factors), convert_eltype.(q, T))

# iteration for destructuring into components
Base.iterate(S::QLHessenberg) = (S.Q, Val(:L))
Base.iterate(S::QLHessenberg, ::Val{:L}) = (S.L, Val(:done))
Base.iterate(S::QLHessenberg, ::Val{:done}) = nothing

QLHessenberg{T}(A::QLHessenberg) where {T} = QLHessenberg(convert(AbstractMatrix{T}, A.factors), convert_eltype.(A.q, T))
Factorization{T}(A::QLHessenberg{T}) where {T} = A
Factorization{T}(A::QLHessenberg) where {T} = QLHessenberg{T}(A)
AbstractMatrix(F::QLHessenberg) = F.Q * F.L
AbstractArray(F::QLHessenberg) = AbstractMatrix(F)
Matrix(F::QLHessenberg) = Array(AbstractArray(F))
Array(F::QLHessenberg) = Matrix(F)

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::QLHessenberg)
    summary(io, F); println(io)
    println(io, "Q factor:")
    show(io, mime, F.Q)
    println(io, "\nL factor:")
    show(io, mime, F.L)
end

@inline function getL(F::QLHessenberg, _)
    m, n = size(F)
    tril!(getfield(F, :factors)[end-min(m,n)+1:end, 1:n], max(n-m,0))
end
@inline getQ(F::QLHessenberg, _) = LowerHessenbergQ(F.q)

getL(F::QLHessenberg) = getL(F, size(F.factors))
getQ(F::QLHessenberg) = getQ(F, size(F.factors))

function getproperty(F::QLHessenberg, d::Symbol)
    if d == :L
        return getL(F)
    elseif d == :Q
        return getQ(F)
    else
        getfield(F, d)
    end
end

Base.propertynames(F::QLHessenberg, private::Bool=false) =
    (:L, :Q, (private ? fieldnames(typeof(F)) : ())...)

abstract type AbstractHessenbergQ{T} <: LayoutQ{T} end

for Typ in (:UpperHessenbergQ, :LowerHessenbergQ)
    @eval begin
        struct $Typ{T, QT<:AbstractVector{<:Union{AbstractMatrix{T},AbstractQ{T}}}} <: AbstractHessenbergQ{T}
            q::QT
            function $Typ{T,QT}(q::QT) where {T,QT<:AbstractVector{<:Union{AbstractMatrix{T},AbstractQ{T}}}}
                all(isorthogonal.(q)) || throw(ArgumentError("input must be orthogonal"))
                all(size.(q) .== Ref((2,2))) ||  throw(ArgumentError("input must be 2x2"))
                new{T,QT}(q)
            end
        end

        $Typ(q::AbstractVector{<:Union{AbstractMatrix{T},AbstractQ{T}}}) where {T} =
            $Typ{T,typeof(q)}(q)
    end
end

size(Q::AbstractHessenbergQ, k::Integer) = length(Q.q)+1
axes(Q::AbstractHessenbergQ, k::Integer) = oneto(length(Q.q)+1)
size(F::QLHessenberg, dim::Integer) = size(getfield(F, :factors), dim)
size(F::QLHessenberg) = size(getfield(F, :factors))

bandwidths(Q::UpperHessenbergQ) = (1,size(Q,2)-1)
bandwidths(Q::LowerHessenbergQ) = (size(Q,1)-1,1)

adjoint(Q::UpperHessenbergQ) = LowerHessenbergQ(adjoint.(Q.q))
adjoint(Q::LowerHessenbergQ) = UpperHessenbergQ(adjoint.(Q.q))

check_mul_axes(A::AbstractHessenbergQ, B, C...) =
    axes(A,2) == axes(B,1) || throw(DimensionMismatch("Second axis of A, $(axes(A,2)), and first axis of B, $(axes(B,1)) must match"))

struct HessenbergQLayout{UPLO} <: AbstractQLayout end

MemoryLayout(::Type{<:UpperHessenbergQ}) = HessenbergQLayout{'U'}()
MemoryLayout(::Type{<:LowerHessenbergQ}) = HessenbergQLayout{'L'}()

function materialize!(L::MatLmulVec{<:HessenbergQLayout{'L'}})
    Q, x = L.A,L.B
    T = eltype(Q)
    t = Array{T}(undef, 2)
    nz = nzzeros(x,1)
    for n = 1:length(Q.q)
        v = view(x, n:n+1)
        mul!(t, Q.q[n], v)
        v .= t
        n > nz && norm(t) ≤ 10floatmin(real(T)) && return x
    end
    x
end

function materialize!(L::MatLmulVec{<:HessenbergQLayout{'U'}})
    Q, x = L.A,L.B
    T = eltype(Q)
    t = Array{T}(undef, 2)
    for n = min(length(Q.q),nzzeros(x,1)):-1:1
        v = view(x, n:n+1)
        mul!(t, Q.q[n], v)
        v .= t
    end
    x
end

function materialize!(L::MatLmulMat{<:HessenbergQLayout})
    Q,X = L.A,L.B
    for j in axes(X,2)
        ArrayLayouts.lmul!(Q, view(X,:,j))
    end
    X
end


###
# getindex
####

getindex(Q::UpperHessenbergQ, I::AbstractVector{Int}, J::AbstractVector{Int}) =
    hcat((Q[:,j][I] for j in J)...)

getindex(Q::LowerHessenbergQ, i::Int, j::Int) = (Q')[j,i]'
getindex(Q::LowerHessenbergQ, I::AbstractVector{Int}, J::AbstractVector{Int}) =
    [Q[i,j] for i in I, j in J]


###
# QLPackedQ, QRPackedQ <-> Lower/UpperHessenbergQ
###

UpperHessenbergQ(Q::LinearAlgebra.QRPackedQ) = UpperHessenbergQ(QRPackedQ(Q))

function UpperHessenbergQ(Q::QRPackedQ{T}) where T
    @assert bandwidth(Q.factors,1) == 1
    q = Vector{Matrix{T}}()
    for j = 1:length(Q.τ)-2
        push!(q, QRPackedQ(Q.factors[j:j+1,j:j+1], [Q.τ[j],zero(T)]))
    end
    push!(q, QRPackedQ(Q.factors[end-1:end,end-1:end], Q.τ[end-1:end]))
    UpperHessenbergQ(q)
end

function LowerHessenbergQ(Q::QLPackedQ{T}) where T
    @assert bandwidth(Q.factors,2) == 1
    q = Vector{Matrix{T}}()
    push!(q, QLPackedQ(Q.factors[1:2,1:2], Q.τ[1:2]))
    for j = 2:length(Q.τ)-1
        push!(q, QLPackedQ(Q.factors[j:j+1,j:j+1], [zero(T),Q.τ[j+1]]))
    end
    LowerHessenbergQ(q)
end
