using InfiniteLinearAlgebra, BlockArrays, ArrayLayouts, Test
import InfiniteLinearAlgebra: BlockTriToeplitzLayout, ul

@testset "Block Toeplitz UL" begin
    B = randn(2,2)
    A = randn(2,2) - 10I #; A = A + A'
    C = Matrix(B')

    J = mortar(Tridiagonal(Fill(C,∞), Fill(A,∞), Fill(B,∞)))

    @test MemoryLayout(J) isa BlockTriToeplitzLayout
    U,L = ul(J)
    N = 10; @test U[Block.(1:N),Block.(1:N+1)] * L[Block.(1:N+1),Block.(1:N)] ≈ J[Block.(1:N),Block.(1:N)]
end