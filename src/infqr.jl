
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
        resizedata!(F.data,n+l,n+u+l);
        resize!(F.τ,n);
        ñ = F.ncols
        factors = view(F.data.data,ñ+1:n+l,ñ+1:n);
        τ = view(F.τ,ñ+1:n);
        _banded_qr!(factors, τ);
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

struct AdaptiveLayout{M} <: MemoryLayout end
MemoryLayout(::Type{AdaptiveQRFactors{T,DM,M}}) where {T,DM,M} = AdaptiveLayout{typeof(MemoryLayout(DM))}()
triangularlayout(::Type{Tri}, ::ML) where {Tri, ML<:AdaptiveLayout} = Tri{ML}()

size(F::AdaptiveQRFactors) = size(F.data.data)
bandwidths(F::AdaptiveQRFactors) = bandwidths(F.data.data)
function colsupport(F::AdaptiveQRFactors, j)
    partialqr!(F.data, j)
    colsupport(F.data.data.data, j)
end

function rowsupport(F::AdaptiveQRFactors, j)
    partialqr!(F.data, j+bandwidth(F,2))
    rowsupport(F.data.data.data, j)
end

function getindex(F::AdaptiveQRFactors, k::Int, j::Int)
    partialqr!(F.data, j)
    F.data.data[k,j]
end

colsupport(F::QRPackedQ{<:Any,<:AdaptiveQRFactors}, j) = colsupport(F.factors, j)
rowsupport(F::QRPackedQ{<:Any,<:AdaptiveQRFactors}, j) = rowsupport(F.factors, j)


Base.replace_in_print_matrix(A::AdaptiveQRFactors, i::Integer, j::Integer, s::AbstractString) =
    i in colsupport(A,j) ? s : Base.replace_with_centered_mark(s)
Base.replace_in_print_matrix(A::UpperTriangular{<:Any,<:AdaptiveQRFactors}, i::Integer, j::Integer, s::AbstractString) =
    i in colsupport(A,j) ? s : Base.replace_with_centered_mark(s)

struct AdaptiveQRTau{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} <: AbstractVector{T}
    data::AdaptiveQRData{T,DM,M}
end

size(F::AdaptiveQRTau) = (size(F.data.data,1),)

function getindex(F::AdaptiveQRTau, j::Int)
    partialqr!(F.data, j)
    F.data.τ[j]
end

getR(Q::QR, ::Tuple{OneToInf{Int},OneToInf{Int}}) where T = UpperTriangular(Q.factors)


function _banded_qr(::NTuple{2,OneToInf{Int}}, A)
    data = AdaptiveQRData(A)
    QR(AdaptiveQRFactors(data), AdaptiveQRTau(data))
end

#########
# lmul!
#########


function lmul!(A::QRPackedQ{<:Any,<:AdaptiveQRFactors}, B::CachedVector{T,Vector{T},<:Zeros{T,1}}) where T
    require_one_based_indexing(B)
    mA, nA = size(A.factors)
    sB = length(B.data)
    mB = length(B)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Afactors = A.factors

    resizedata!(B, maximum(colsupport(A,sB)))
    b = B.data    
    
    sB_2 = length(b)
    partialqr!(A.factors.data, sB)

    @inbounds begin
        for k = sB:-1:1
            cs = colsupport(Afactors,k)
            vBj = b[k]
            for i = (k+1:sB_2) ∩ cs
                vBj += conj(Afactors[i,k])*b[i]
            end
            vBj = A.τ[k]*vBj
            b[k] -= vBj
            for i = (k+1:sB_2) ∩ cs
                b[i] -= Afactors[i,k]*vBj
            end
        end
    end
    B
end


function lmul!(adjA::Adjoint{<:Any,<:QRPackedQ{<:Any,<:AdaptiveQRFactors}}, B::CachedVector{T,Vector{T},<:Zeros{T,1}}) where T
    COLGROWTH = 100 # rate to grow columns
    tol = floatmin(T)

    require_one_based_indexing(B)
    A = adjA.parent
    mA, nA = size(A.factors)
    mB = length(B)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    sB = length(B.data)
    jr = 1:min(COLGROWTH,nA)

    @inbounds begin
        while first(jr) < nA
            cs = colsupport(A.factors, last(jr))
            cs_max = maximum(cs)
            kr = first(jr):cs_max
            resizedata!(B, min(cs_max,mB))
            if (first(jr) > sB && maximum(abs,view(B.data,colsupport(A.factors,first(jr)))) ≤ tol)
                break
            end
            partialqr!(A.factors.data, min(cs_max,nA))
            Q_N = QRPackedQ(view(A.factors.data.data.data,kr,jr), view(A.τ.data.τ,jr))
            lmul!(Q_N', view(B.data, kr))
            jr = last(jr)+1:min(last(jr)+COLGROWTH,nA)
        end
    end
    B
end

function (*)(A::QRPackedQ{T,<:AdaptiveQRFactors}, x::AbstractVector{S}) where {T,S}
    TS = promote_op(matprod, T, S)
    lmul!(A, Base.copymutable(convert(AbstractVector{TS},x)))
end

function (*)(A::Adjoint{T,<:QRPackedQ{T,<:AdaptiveQRFactors}}, x::AbstractVector{S}) where {T,S}
    TS = promote_op(matprod, T, S)
    lmul!(A, Base.copymutable(convert(AbstractVector{TS},x)))
end

ldiv!(R::UpperTriangular{<:Any,<:AdaptiveQRFactors}, B::AbstractVector) = materialize!(Ldiv(R, B))



ldiv!(dest::AbstractVector, F::QR{<:Any,<:AdaptiveQRFactors}, b::AbstractVector) = 
    ldiv!(F, copyto!(dest, b))
ldiv!(F::QR{<:Any,<:AdaptiveQRFactors}, b::AbstractVector) = ldiv!(F.R, lmul!(F.Q',b))
\(F::QR{<:Any,<:AdaptiveQRFactors}, B::AbstractVector) = ldiv!(F.R, F.Q'B)


factorize(A::BandedMatrix{<:Any,<:Any,<:OneToInf}) = qr(A)