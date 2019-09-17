
mutable struct AdaptiveQRData{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}}
    data::CachedMatrix{T,DM,M}
    τ::Vector{T}
    ncols::Int
end

function AdaptiveQRData(::AbstractBandedLayout, A::AbstractMatrix{T}) where T 
    l,u = bandwidths(A)
    data = BandedMatrix{T}(undef,(2l+u+1,0),(l,l+u)) # pad super
    AdaptiveQRData(CachedArray(data,A), Vector{T}(), 0)
end
AdaptiveQRData(A::AbstractMatrix{T}) where T = AdaptiveQRData(MemoryLayout(typeof(A)), A)

function partialqr!(F::AdaptiveQRData{<:Any,<:BandedMatrix}, n::Int)
    if n > F.ncols 
        l,u = bandwidths(F.data.array)
        resizedata!(F.data,n+l,n+u+l)
        resize!(F.τ,n)
        ñ = F.ncols
        factors = view(F.data.data,ñ+1:n+l,ñ+1:n)
        τ = view(F.τ,ñ+1:n)
        _banded_qr!(factors, τ)
        # multiply remaining columns
        n̄ = max(ñ+1,n-l-u+1) # first column interacting with extra data
        Q = QRPackedQ(view(F.data.data,n̄:n+l,n̄:n), view(F.τ,n̄:n))
        lmul!(Q',view(F.data.data,n̄:n+l,n+1:n+u+l))
        F.ncols = n
    end
    F
end

struct AdaptiveQRFactors{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::AdaptiveQRData{T,DM,M}
end

size(F::AdaptiveQRFactors) = size(F.data.data)
bandwidths(F::AdaptiveQRFactors) = bandwidths(F.data.data)
function getindex(F::AdaptiveQRFactors, k::Int, j::Int)
    partialqr!(F.data, j)
    F.data.data[k,j]
end

struct AdaptiveQRTau{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} <: AbstractVector{T}
    data::AdaptiveQRData{T,DM,M}
end

size(F::AdaptiveQRTau) = (size(F.data.data,1),)

function getindex(F::AdaptiveQRTau, j::Int)
    partialqr!(F.data, j)
    F.data.τ[j]
end

getR(Q::QR, ::Tuple{OneToInf{Int},OneToInf{Int}}) where T = UpperTriangular(Q.factors)


function qr(A::BandedMatrix{<:Any,<:Any,<:OneToInf}) 
    data = AdaptiveQRData(A)
    QR(AdaptiveQRFactors(data), AdaptiveQRTau(data))
end