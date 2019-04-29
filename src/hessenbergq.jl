isorthogonal(::AbstractQ) = true
isorthogonal(q) = q'q ≈ I

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

bandwidths(Q::UpperHessenbergQ) = (1,size(Q,2)-1)
bandwidths(Q::LowerHessenbergQ) = (size(Q,1)-1,1)

adjoint(Q::UpperHessenbergQ) = LowerHessenbergQ(adjoint.(Q.q))
adjoint(Q::LowerHessenbergQ) = UpperHessenbergQ(adjoint.(Q.q))

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
# Infinite
####

function getindex(Q::UpperHessenbergQ, i::Integer, j::Integer)
    y = zeros(eltype(Q), size(Q, 2))
    y[j] = 1
    lmul!(Q, y)[i]
end

getindex(Q::LowerHessenbergQ, i::Integer, j::Integer) = (Q')[j,i]'