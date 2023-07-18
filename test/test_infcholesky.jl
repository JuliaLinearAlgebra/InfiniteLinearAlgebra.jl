using InfiniteLinearAlgebra, LinearAlgebra, BandedMatrices, ArrayLayouts, LazyBandedMatrices, Test
import InfiniteLinearAlgebra: SymmetricBandedLayouts

@testset "infinite-cholesky" begin
    S = Symmetric(BandedMatrix(0 => 1:∞, 1=> Ones(∞)))
    b = [1; zeros(∞)]
    @test cholesky(S) \ b ≈ qr(S) \ b ≈ S \ b

    # go past adaptive
    b = [randn(10_000); zeros(∞)]
    @test cholesky(S) \ b ≈ qr(S) \ b ≈ S \ b

    @testset "singularly perturbed" begin
        # using Symmetric(BandedMatrix(...))
        ε = 0.0001
        S = Symmetric(BandedMatrix(0 => 2 .+ ε*(1:∞), 1=> -Ones(∞)))
        b = [1; zeros(∞)]
        @test cholesky(S) \ b ≈  S \ b
        # using SymTridiagonal(...)
        ε = 0.0001
        S = LazyBandedMatrices.SymTridiagonal(2 .+ ε*(1:∞), -Ones(∞))
        @test MemoryLayout(S) isa SymmetricBandedLayouts
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

        @test cholesky(S^2).U[1:100,1:100] ≈ cholesky(Symmetric((S^2)[1:100,1:100])).U
    end

    @testset "row/colsupport" begin
        S = Symmetric(BandedMatrix(0 => 1:∞, 2 => Ones(∞)))
        F = cholesky(S)
        @test colsupport(F.factors,5) == rowsupport(F.factors,3) == 3:5
        @test rowsupport(F.factors) == colsupport(F.factors) == axes(F.factors,1)

        @test (F.U * F.U')[1:10,1:10]  ≈ F.U[1:10,1:12] * F.U[1:10,1:12]'
        @test (F.U' * F.U)[1:10,1:10]  ≈ S[1:10,1:10]
    end
end
