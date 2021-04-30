using InfiniteLinearAlgebra, LinearAlgebra, BandedMatrices, Test


A = Symmetric(BandedMatrix(0 => 1:∞, 1=> Ones(∞)))
cholesky(A)
qr(A)

F = AdaptiveCholeskyFactors(cache(parent(A)), 0)
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