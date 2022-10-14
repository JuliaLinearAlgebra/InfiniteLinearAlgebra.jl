using InfiniteLinearAlgebra, BlockArrays, ArrayLayouts, Test
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
