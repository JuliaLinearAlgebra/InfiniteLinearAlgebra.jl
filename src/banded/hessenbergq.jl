isorthogonal(::AbstractQ) = true
isorthogonal(q) = q'q ≈ I


struct QLHessenberg{T,S<:AbstractMatrix{T},QT<:AbstractVector{<:AbstractMatrix{T}}} <: Factorization{T}
    factors::S
    q::QT

    function QLHessenberg{T,S,QT}(factors, q) where {T,S<:AbstractMatrix{T},QT<:AbstractVector{<:AbstractMatrix{T}}}
        require_one_based_indexing(factors)
        new{T,S,QT}(factors, q)
    end
end

QLHessenberg(factors::AbstractMatrix{T}, q::AbstractVector{<:AbstractMatrix{T}}) where {T} = QLHessenberg{T,typeof(factors),typeof(q)}(factors, q)
QLHessenberg{T}(factors::AbstractMatrix, q::AbstractVector) where {T} =
    QLHessenberg(convert(AbstractMatrix{T}, factors), convert.(AbstractMatrix{T}, q))

# iteration for destructuring into components
Base.iterate(S::QLHessenberg) = (S.Q, Val(:L))
Base.iterate(S::QLHessenberg, ::Val{:L}) = (S.L, Val(:done))
Base.iterate(S::QLHessenberg, ::Val{:done}) = nothing

QLHessenberg{T}(A::QLHessenberg) where {T} = QLHessenberg(convert(AbstractMatrix{T}, A.factors), convert.(AbstractMatrix{T}, A.q))
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

getL(F::QLHessenberg) = getL(F, axes(F.factors))
getQ(F::QLHessenberg) = getQ(F, axes(F.factors))

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

abstract type AbstractHessenbergQ{T} <: AbstractQ{T} end

for Typ in (:UpperHessenbergQ, :LowerHessenbergQ)
    @eval begin
        struct $Typ{T, QT<:AbstractVector{<:AbstractMatrix{T}}} <: AbstractHessenbergQ{T}
            q::QT
            function $Typ{T,QT}(q::QT) where {T,QT<:AbstractVector{<:AbstractMatrix{T}}}
                all(isorthogonal.(q)) || throw(ArgumentError("input must be orthogonal"))
                all(size.(q) .== Ref((2,2))) ||  throw(ArgumentError("input must be 2x2"))
                new{T,QT}(q)
            end
        end

        $Typ(q::AbstractVector{<:AbstractMatrix{T}}) where T = 
            $Typ{T,typeof(q)}(q)
    end
end

size(Q::AbstractHessenbergQ, k::Integer) = length(Q.q)+1
size(F::QLHessenberg, dim::Integer) = size(getfield(F, :factors), dim)
size(F::QLHessenberg) = size(getfield(F, :factors))

bandwidths(Q::UpperHessenbergQ) = (1,size(Q,2)-1)
bandwidths(Q::LowerHessenbergQ) = (size(Q,1)-1,1)

adjoint(Q::UpperHessenbergQ) = LowerHessenbergQ(adjoint.(Q.q))
adjoint(Q::LowerHessenbergQ) = UpperHessenbergQ(adjoint.(Q.q))

check_mul_axes(A::AbstractHessenbergQ, B, C...) =
    axes(A,2) == axes(B,1) || throw(DimensionMismatch("Second axis of A, $(axes(A,2)), and first axis of B, $(axes(B,1)) must match"))


function lmul!(Q::LowerHessenbergQ{T}, x::AbstractVector) where T
    t = Array{T}(undef, 2)
    for n = 1:length(Q.q)
        v = view(x, n:n+1)
        mul!(t, Q.q[n], v)
        v .= t
        all(iszero,t) && return x
    end
    x
end

function lmul!(Q::UpperHessenbergQ{T}, x::AbstractVector) where T
    t = Array{T}(undef, 2)
    for n = min(length(Q.q),nzzeros(x,1)):-1:1
        v = view(x, n:n+1)
        mul!(t, Q.q[n], v)
        v .= t
    end
    x
end


###
# getindex
####

function getindex(Q::UpperHessenbergQ, i::Integer, j::Integer)
    y = zeros(eltype(Q), size(Q, 2))
    y[j] = 1
    lmul!(Q, y)[i]
end

getindex(Q::LowerHessenbergQ, i::Integer, j::Integer) = (Q')[j,i]'


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
