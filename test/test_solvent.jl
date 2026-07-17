using InfiniteLinearAlgebra, Test, LinearAlgebra
using InfiniteLinearAlgebra: matrixroot

@testset "matrix quadratic" begin
    A₂,A₁,A₀ = randn(2,2),randn(2,2),randn(2,2)
    W = matrixroot(A₂,A₁,A₀)
    @test A₂*W^2 + A₁*W ≈ -A₀

    A,B = Symmetric(randn(2,2)), randn(2,2)

    D = matrixroot(B, -A, B')';
    X = D\B;
    @test A - B * inv(X) * B' ≈ X

    D = matrixroot(-B, -A, B')';
    X = D\B;
    @test A +  B * inv(X) * B' ≈ X
end