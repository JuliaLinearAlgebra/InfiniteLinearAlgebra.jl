using InfiniteLinearAlgebra, LinearAlgebra, BandedMatrices, ArrayLayouts, Test

@testset "infinite-cholesky" begin
    S = Symmetric(BandedMatrix(0 => 1:∞, 1=> Ones(∞)))
    b = [1; zeros(∞)]
    @test cholesky(S) \ b ≈ qr(S) \ b ≈ S \ b

    # go past adaptive 
    b = [randn(10_000); zeros(∞)]
    @test cholesky(S) \ b ≈ qr(S) \ b ≈ S \ b
end
