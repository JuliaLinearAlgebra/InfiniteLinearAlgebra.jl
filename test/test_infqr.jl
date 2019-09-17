using InfiniteLinearAlgebra, LinearAlgebra, BandedMatrices, InfiniteArrays, MatrixFactorizations, LazyArrays, FillArrays
import BandedMatrices: _BandedMatrix, _banded_qr!, colsupport, BandedColumns
import InfiniteLinearAlgebra: partialqr!, AdaptiveQRData


@testset "Adaptive QR" begin
    @testset "test partialqr!" begin
        A = _BandedMatrix(Vcat(Ones(1,∞), (1:∞)', Ones(1,∞)), ∞, 1, 1)
        l,u = bandwidths(A)
        F = AdaptiveQRData(A);
        @test bandwidths(F.data) == (1,2)
        n = 3
        partialqr!(F,n);
        F̃ = qrunblocked(Matrix(A[1:n+l,1:n]))
        @test F̃.factors ≈ F.data[1:n+l,1:n]
        @test F̃.τ  ≈ F.τ
        @test triu!(F.data[1:n+1,1:n+2]) ≈ F̃.Q'*A[1:n+1,1:n+2] # test extra columns have been modified
        n  = 6
        partialqr!(F,n);
        F̃ = qrunblocked(Matrix(A[1:n+l,1:n]))
        @test F̃.factors ≈ F.data[1:n+l,1:n]
        @test F̃.τ  ≈ F.τ
        @test triu!(F.data[1:n+1,1:n+2]) ≈ F̃.Q'*A[1:n+1,1:n+2] # test extra columns have been modified
    end

    @testset "AdaptiveQRFactors" begin
        A = _BandedMatrix(Vcat(Ones(1,∞), (1:∞)', Ones(1,∞)), ∞, 1, 1) 
        F = qr(A);
        @test F.factors[1,1] ≈ -sqrt(2)
        @test F.factors[100,100] ≈ qrunblocked(A[1:101,1:100]).factors[100,100]
        @test F.τ[1] ≈ 1+sqrt(2)/2
        @test F.τ[100] ≈ qrunblocked(A[1:101,1:100]).τ[100]
        Q,R = F;
        
    end
end

    