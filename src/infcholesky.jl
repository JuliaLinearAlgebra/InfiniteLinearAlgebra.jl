


using BandedMatrices, ArrayLayouts, InfiniteLinearAlgebra, LazyArrays
import BandedMatrices: banded_chol!
import LazyArrays: resizedata!, CachedMatrix

A = Symmetric(brand(10,10,1,1) + 10I)
U = cholesky(A).U
banded_chol!(view(parent(A),1:3,1:3),UpperTriangular)

P = parent(A)
B = view(P,1:3,4:10)
ldiv!(UpperTriangular(view(P,1:3,1:3))', B)


B = BandedMatrix(view(P,1:3,4:10))
C = view(P,4:10,4:10)
muladd!(-1.0,B',B,1.0,C)
B'B


T = eltype(C)
_,u = bandwidths(C)
ncols = 0
n = 100


mutable struct AdaptiveCholeskyFactors{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} <: LayoutMatrix{T}
    data::CachedMatrix{T,DM,M}
    ncols::Int
end


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

A = Symmetric(BandedMatrix(0 => 1:∞, 1=> Ones(∞)))
F = AdaptiveCholeskyFactors(cache(parent(A)), 0);
partialcholesky!(F,5);
partialcholesky!(F,10);

@test F.data.data[1:10,1:10] ≈ cholesky(Symmetric(A[1:10,1:10])).U

kr = ncols+1:n








A[1:100,1:100]  |> Symmetric |> eigvals


cholesky(Symmetric(view(P,4:10,4:10) - B'B)).U - U[4:end,4:end]



A
U
U

A

A = Symmetric(brand(10,10,1,1) + 10I, :L)
L = cholesky(A).L

L = cholesky(A).L
L*L' - A
U'U - A
U*U'

BandedMatrices.banded_chol!(view(parent(A),1:5,1:5), UpperTriangular)

cholesky(A)