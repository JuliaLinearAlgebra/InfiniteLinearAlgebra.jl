using InfiniteLinearAlgebra, LazyBandedMatrices, FillArrays, MatrixFactorizations, ArrayLayouts, LinearAlgebra, Test, LazyArrays

@testset "infreversecholeskytoeplitz" begin
    @testset "Tri Toeplitz" begin
        A = SymTridiagonal(Fill(3, ∞), Fill(1, ∞))
        U, = reversecholesky(A)
        @test (U*U')[1:10, 1:10] ≈ A[1:10, 1:10]
    end

    @testset "Pert Tri Toeplitz" begin
        A = SymTridiagonal([[4, 5, 6]; Fill(3, ∞)], [[2, 3]; Fill(1, ∞)])
        @test reversecholesky(A).U[1:100, 1:100] ≈ reversecholesky(A[1:1000, 1:1000]).U[1:100, 1:100]
    end
end

@testset "infreversecholeskytridiagonal" begin
    local LL, L
    @testset "Test on Toeplitz example first" begin
        A = SymTridiagonal(Fill(3, ∞), Fill(1, ∞))
        L = reversecholesky(A)
        LL = InfiniteLinearAlgebra.reversecholesky_layout(SymTridiagonalLayout{LazyArrays.LazyLayout,LazyArrays.LazyLayout}(), axes(A), A, NoPivot())
        @test L.L[1:1000, 1:1000] ≈ LL.L[1:1000, 1:1000]
        @test (LL.L'*LL.L)[1:1000, 1:1000] == (LL.U*LL.L)[1:1000, 1:1000] ≈ A[1:1000, 1:1000]
    end

    @testset "Basic tests" begin
        L = LL
        @test MemoryLayout(L.L) isa BidiagonalLayout
        @test L.U === L.L'
        @test L.uplo == 'L'
        @test L.info == 0
        @test size(L) == (∞, ∞)
        @test axes(L) == (1:∞, 1:∞)
        @test eltype(L) == Float64
        Lc = copy(L)
        @test !(Lc === L)
        @test !(Lc.U === L.U)
        @test Lc.L[1:1000, 1:1000] == L.L[1:1000, 1:1000]
        UUc = copy(L.L')
        @test !(UUc === L.U)
        @test UUc[1:1000, 1:1000] == L.U[1:1000, 1:1000]
    end

    @testset "Errors" begin
        err = InfiniteLinearAlgebra.InfiniteBoundsAccessError(4, 6)
        @test_throws InfiniteLinearAlgebra.InfiniteBoundsAccessError throw(err)
        @test_throws InfiniteLinearAlgebra.InfiniteBoundsAccessError L.L[1, InfiniteLinearAlgebra.MAX_TRIDIAG_CHOL_N+1]
        @test_throws InfiniteLinearAlgebra.InfiniteBoundsAccessError L.L[InfiniteLinearAlgebra.MAX_TRIDIAG_CHOL_N+1, 1]
        @test_throws InfiniteLinearAlgebra.InfiniteBoundsAccessError L.L[InfiniteLinearAlgebra.MAX_TRIDIAG_CHOL_N+1, InfiniteLinearAlgebra.MAX_TRIDIAG_CHOL_N+1]
    end

    @testset "Another example" begin
        A = LazyBandedMatrices.SymTridiagonal(Ones(∞), 1 ./ (2:∞))
        L = reversecholesky(A)
        @test (L.U*L.L)[1:1000, 1:1000] ≈ A[1:1000, 1:1000]
        A = 5I + LazyBandedMatrices.SymTridiagonal(1 ./ (2:∞), Ones(∞))
        L = reversecholesky(A)
        @test (L.U*L.L)[1:1000, 1:1000] ≈ A[1:1000, 1:1000]
    end
end