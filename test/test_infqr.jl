using InfiniteLinearAlgebra, LinearAlgebra, BandedMatrices, InfiniteArrays, MatrixFactorizations, LazyArrays, FillArrays
import LazyArrays: colsupport, rowsupport, MemoryLayout, DenseColumnMajor, TriangularLayout
import BandedMatrices: _BandedMatrix, _banded_qr!, BandedColumns
import InfiniteLinearAlgebra: partialqr!, AdaptiveQRData, AdaptiveLayout


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
        @test bandwidths(R) == (0,2)
        @test size(R) == (∞,∞)
        @test R[1,1] == -sqrt(2)
    end

    @testset "col/rowsupport" begin
        A = _BandedMatrix(Vcat(Ones(1,∞), (1:∞)', Ones(1,∞)), ∞, 1, 1) 
        F = qr(A);
        @test MemoryLayout(typeof(F.factors)) isa AdaptiveLayout{BandedColumns{DenseColumnMajor}}
        @test bandwidths(F.factors) == (1,2)
        @test colsupport(F.factors,1) ==  1:2
        @test colsupport(F.factors,5) ==  3:6
        @test rowsupport(F.factors,1) ==  1:3
        @test rowsupport(F.factors,5) ==  4:7
        Q,R = F;
        @test MemoryLayout(typeof(R)) isa TriangularLayout
        @test colsupport(R,1) == 1:1
        @test colsupport(R,5) == 3:5
        @test rowsupport(R,1) == 1:3
        @test rowsupport(R,5) == 5:7
        @test colsupport(Q,1) == 1:2
        @test colsupport(Q,5) == 3:6
        @test rowsupport(Q,1) == 1:3
        @test rowsupport(Q,5) == 4:7
    end

    @testset "Qmul" begin
        A = _BandedMatrix(Vcat(Ones(1,∞), (1:∞)', Ones(1,∞)), ∞, 1, 1) 
        Q,R = qr(A);
        b = Vcat([1.,2,3],Zeros(∞))
        @test lmul!(Q, Base.copymutable(b)).data ≈ qr(A[1:4,1:3]).Q*[1,2,3]
        @test Q[1,1] ≈ -1/sqrt(2)
        @test Q[200_000,200_000] ≈ -1.0
        @test Q[1:101,1:100] ≈ qr(A[1:101,1:100]).Q[:,1:100]

        r = lmul!(Q', Base.copymutable(b))
        nr = length(r.data)
        @test qr(A[1:nr+1,1:nr]).Q'b[1:nr+1] ≈ r[1:nr+1]

        materialize!(Ldiv(R, r))
    end
end

    