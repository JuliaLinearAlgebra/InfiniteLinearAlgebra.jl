using InfiniteLinearAlgebra, MatrixFactorizations, ArrayLayouts, LinearAlgebra, Test



@testset "infreversecholesky" begin
    @testset "Tri Toeplitz" begin
        A = SymTridiagonal(Fill(3, ∞), Fill(1, ∞))
        U, = reversecholesky(A)
        @test (U*U')[1:10,1:10] ≈ A[1:10,1:10]
    end

    @testset "Pert Tri Toeplitz" begin
        A = SymTridiagonal([[4,5, 6]; Fill(3, ∞)], [[2,3]; Fill(1, ∞)])
        @test reversecholesky(A).U[1:100,1:100] ≈ reversecholesky(A[1:1000,1:1000]).U[1:100,1:100]
    end
end