
mutable struct AdaptiveCholeskyFactors{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} <: LayoutMatrix{T}
    data::CachedMatrix{T,DM,M}
    ncols::Int
end

size(U::AdaptiveCholeskyFactors) = size(U.data.array)
bandwidths(A::AdaptiveCholeskyFactors) = (0,bandwidth(A.data,2))
colsupport(A::AdaptiveCholeskyFactors,j) = colsupport(A.data,j)
AdaptiveCholeskyFactors(A::Symmetric) = AdaptiveCholeskyFactors(cache(parent(A)),0)


function partialcholesky!(F::AdaptiveCholeskyFactors{T,<:BandedMatrix}, n::Int) where T
    if n > F.ncols 
        _,u = bandwidths(F.data.array)
        resizedata!(F.data,n+u,n+u);
        ncols = F.ncols
        kr = ncols+1:n
        factors = view(F.data.data,kr,kr)
        banded_chol!(factors, UpperTriangular)
        # multiply remaining columns
        U1 = UpperTriangular(view(F.data.data,n-u+1:n,n-u+1:n))
        B = view(F.data.data,n-u+1:n,n+1:n+u)
        ldiv!(U1',B)
        muladd!(-one(T),B',B,one(T),view(F.data.data,n+1:n+u,n+1:n+u))
        F.ncols = n
    end
    F
end

function getindex(F::AdaptiveCholeskyFactors, k::Int, j::Int)
    partialcholesky!(F, max(k,j))
    F.data.data[k,j]
end


adaptivecholesky(A) = Cholesky(AdaptiveCholeskyFactors(A), :U, 0)


ArrayLayouts._cholesky(::SymmetricLayout{<:AbstractBandedLayout}, ::NTuple{2,OneToInf{Int}}, A) = adaptivecholesky(A)