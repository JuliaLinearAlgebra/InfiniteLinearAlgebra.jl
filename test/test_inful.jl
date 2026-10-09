using InfiniteLinearAlgebra, InfiniteArrays, BlockArrays, ArrayLayouts, FillArrays, LinearAlgebra, Test
import InfiniteLinearAlgebra: BlockTridiagonalToeplitzLayout, ul, adaptiveqr

@testset "∞-UL" begin
    @testset "Toeplitz" begin
        Δ = SymTridiagonal(Fill(-2,∞), Fill(1,∞))
        h = 0.01
        A = I - h*Δ
        U,L = ul(A)
        N = 10
        @test U[1:N,1:N+1]*L[1:N+1,1:N] ≈ A[1:N,1:N]

        u =  adaptiveqr(A) \ [1; zeros(∞)]
        v = L \ (U \ [1;zeros(∞)])
        @test u ≈ v
    end

    @testset "Symmetric" begin
        B = randn(2,2)
        A = randn(2,2) - 10I #; A = A + A'
        C = Matrix(B')

        J = mortar(Tridiagonal(Fill(C,∞), Fill(A,∞), Fill(B,∞)))

        @test MemoryLayout(J) isa BlockTridiagonalToeplitzLayout
        U,L = ul(J, Val(false))
        N = 10
        @test U[Block.(1:N),Block.(1:N+1)] * L[Block.(1:N+1),Block.(1:N)] ≈ J[Block.(1:N),Block.(1:N)]

        @test (J \ [1; zeros(∞)])[Block(1)] ≈ inv(L[Block(1,1)])[:,1]
    end

    @testset "Non-symmetric" begin
        A = [-10.0 1; 0 -12]
        B = [1.0 0; 1 2]
        C = [2.0 1; 0 1]
        J = mortar(Tridiagonal(Fill(C,∞), Fill(A,∞), Fill(B,∞)))
        U,L = ul(J, Val(false))
        N = 10
        @test istril(L[1:2N,1:2N])
        @test istriu(U[1:2N,1:2N])
        @test diag(U[1:2N,1:2N]) == ones(2N)
        @test U[Block.(1:N),Block.(1:N+1)] * L[Block.(1:N+1),Block.(1:N)] ≈ J[Block.(1:N),Block.(1:N)]
    end

    @testset "Perturbed Toeplitz" begin
        J = Tridiagonal([[1.5]; Fill(2.0,∞)],
                        [[-8.0, -9, -11]; Fill(-10.0,∞)],
                        [[0.4, 0.7]; Fill(0.5,∞)])
        N = 10
        for A in (J,
                  Tridiagonal([Float64[]; Fill(2.0,∞)],
                              [[-8.0]; Fill(-10.0,∞)],
                              [Float64[]; Fill(0.5,∞)]),
                  Tridiagonal([Float64[]; Fill(2.0,∞)],
                              [Float64[]; Fill(-10.0,∞)],
                              [Float64[]; Fill(0.5,∞)]),
                  SymTridiagonal([[-8.0, -9]; Fill(-10.0,∞)],
                                 [[1.5]; Fill(1.0,∞)]))
            U,L = ul(A, Val(false))
            Un,Ln = ul(Matrix(A[1:50,1:50]), Val(false))
            @test U[1:N,1:N+1]*L[1:N+1,1:N] ≈ A[1:N,1:N]
            @test U[1:N,1:N] ≈ Un[1:N,1:N]
            @test L[1:N,1:N] ≈ Ln[1:N,1:N]
        end
        U,L = ul(J)
        @test U[1:N,1:N+1]*L[1:N+1,1:N] ≈ J[1:N,1:N]
    end

    @testset "Perturbed block Toeplitz" begin
        A = [-10.0 1; 0 -12]
        B = [1.0 0; 1 2]
        C = [2.0 1; 0 1]
        J = mortar(Tridiagonal([[C/2]; Fill(C,∞)],
                               [[A+I, A-2I]; Fill(A,∞)],
                               [[2B, B/2, B+I]; Fill(B,∞)]))
        U,L = ul(J, Val(false))
        Un,Ln = ul(Matrix(J[1:100,1:100]), Val(false))
        N = 10
        @test istril(L[1:2N,1:2N])
        @test istriu(U[1:2N,1:2N])
        @test diag(U[1:2N,1:2N]) == ones(2N)
        @test U[Block.(1:N),Block.(1:N+1)] * L[Block.(1:N+1),Block.(1:N)] ≈ J[Block.(1:N),Block.(1:N)]
        @test U[1:2N,1:2N] ≈ Un[1:2N,1:2N]
        @test L[1:2N,1:2N] ≈ Ln[1:2N,1:2N]
        @test_throws ErrorException ul(J, Val(true))

        # A larger leading block represents a dense top-left perturbation.
        A0 = [-9.0 1 2; 3 -10 1; 2 1 -11]
        B0 = [1.0 2; 0 1; 2 1]
        C0 = [2.0 0 1; 1 1 0]
        J = mortar(Tridiagonal([[C0]; Fill(C,∞)],
                               [[A0]; Fill(A,∞)],
                               [[B0]; Fill(B,∞)]))
        U,L = ul(J, Val(false))
        @test istril(L[1:2N+1,1:2N+1])
        @test istriu(U[1:2N+1,1:2N+1])
        @test U[Block.(1:N),Block.(1:N+1)] * L[Block.(1:N+1),Block.(1:N)] ≈ J[Block.(1:N),Block.(1:N)]
    end

    @testset "Periodic Jacobi" begin
        e = 0.1
        B = [e 0; 1 e]
        A = [1 1; 1 -1.0]
        C = Matrix(B')
        J = mortar(Tridiagonal(Fill(C,∞), Fill(A,∞), Fill(B,∞))) - 10I
        U,L = ul(J, Val(false))
        N = 10;
        @test U[Block.(1:N),Block.(1:N+1)] * L[Block.(1:N+1),Block.(1:N)] ≈ J[Block.(1:N),Block.(1:N)]
    end
end
