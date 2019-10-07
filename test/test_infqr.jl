using InfiniteLinearAlgebra, LinearAlgebra, BandedMatrices, InfiniteArrays, MatrixFactorizations, LazyArrays, FillArrays, SpecialFunctions
import LazyArrays: colsupport, rowsupport, MemoryLayout, DenseColumnMajor, TriangularLayout, resizedata!
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

        @test lmul!(Q, Base.copymutable(b)) == Q*b

        r = lmul!(Q', Base.copymutable(b))
        nr = length(r.data)
        @test qr(A[1:nr+1,1:nr]).Q'b[1:nr+1] ≈ r[1:nr+1]

        @test Q'*b == r

        @test factorize(A) isa typeof(qr(A))
        @test qr(A)\b == A\b
        @test (A*(A\b))[1:100] ≈ [1:3; Zeros(97)]
    end

    @testset "Bessel J" begin
        z = 1000; # the bigger z the longer before we see convergence
        A = BandedMatrix(0 => -2*(0:∞)/z, 1 => Ones(∞), -1 => Ones(∞))
        J = A \ Vcat([besselj(1,z)], Zeros(∞))
        @test J[1:2000] ≈ [besselj(k,z) for k=0:1999]

        z = 10_000; # the bigger z the longer before we see convergence
        A = BandedMatrix(0 => -2*(0:∞)/z, 1 => Ones(∞), -1 => Ones(∞))
        J = A \ Vcat([besselj(1,z)], Zeros(∞))
        @test J[1:20_000] ≈ [besselj(k,z) for k=0:20_000-1]
    end

    @testset "5-band" begin
        A = BandedMatrix(-2 => Ones(∞), -1 => Vcat(1, Zeros(∞)), 0 => Vcat([1,2,3],Zeros(∞)).+3, 1 => Vcat(1, Zeros(∞)), 2 => Ones(∞))
        b = Vcat([3,4,5],Zeros(∞))
        x = qr(A) \ b
        @test x[1:2000] ≈ (A[1:2000,1:2000]\b[1:2000])
    end

    @testset "broadcast" begin
        A = BandedMatrix(0 => -2*(0:∞), 1 => Ones(∞), -1 => Ones(∞))
        B = BandedMatrix(-2 => Ones(∞), -1 => Vcat(1, Zeros(∞)), 0 => Vcat([1,2,3],Zeros(∞)).+3, 1 => Vcat(1, Zeros(∞)), 2 => Ones(∞))

        AB = BroadcastArray(+,A,B)
        C = cache(AB);
        resizedata!(C,103,100); resizedata!(C,203,200);
        @test C[103,104] ≈ 1.0
        F = qr(AB);
        partialqr!(F.factors.data, 100);
        partialqr!(F.factors.data, 200);
        @test norm(F.factors.data.data.data) ≤ 4000
        b = Vcat([3,4,5],Zeros(∞))
        @time x = qr(AB) \ b;
        @test x[1:300] ≈ AB[1:300,1:300] \ b[1:300]
    end
end

