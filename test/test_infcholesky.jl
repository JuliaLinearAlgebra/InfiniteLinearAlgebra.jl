using InfiniteLinearAlgebra, LinearAlgebra, BandedMatrices, ArrayLayouts, Test

@testset "infinite-cholesky" begin
    S = Symmetric(BandedMatrix(0 => 1:∞, 1=> Ones(∞)))
    b = [1; zeros(∞)]
    @test cholesky(S) \ b ≈ qr(S) \ b ≈ S \ b

    # go past adaptive 
    b = [randn(10_000); zeros(∞)]
    @test cholesky(S) \ b ≈ qr(S) \ b ≈ S \ b

    @testset "singularly perturbed" begin
        ε = 0.0001
        S = Symmetric(BandedMatrix(0 => 2 .+ ε*(1:∞), 1=> -Ones(∞)))
        b = [1; zeros(∞)]
        @test cholesky(S) \ b ≈  S \ b
    end

    @testset "long bandwidths" begin
        S = Symmetric(BandedMatrix(0 => 1:∞, 50=> Ones(∞)))
        b = [1; zeros(∞)]
        @test cholesky(S) \ b ≈ qr(S) \ b ≈ S \ b
    end

    @testset "powers" begin
        b = [1; zeros(∞)]
        @test cholesky(S^2) \ b ≈ qr(S^2) \ b ≈ S^2 \ b
    end
end
