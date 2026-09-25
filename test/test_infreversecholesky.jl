using InfiniteLinearAlgebra, LazyBandedMatrices, FillArrays, MatrixFactorizations, ArrayLayouts, LinearAlgebra, Test, LazyArrays, BlockArrays

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

    @testset "Block Tri Toeplitz" begin
        N = 6
        @testset "2x2 blocks" begin
            B = [1.0 2; 3 4]/10
            D = [5.0 1; 1 6]
            J = mortar(Tridiagonal(Fill(Matrix(B'), ∞), Fill(D, ∞), Fill(B, ∞)))
            F = reversecholesky(J)
            U, L = F
            @test U === F.U
            @test L[Block.(1:N), Block.(1:N)] == U[Block.(1:N), Block.(1:N)]'
            V = Matrix(U[Block.(1:N), Block.(1:N+1)])
            @test V*V' ≈ J[Block.(1:N), Block.(1:N)]
        end

        @testset "3x3 blocks" begin
            B = [1.0 2 3; 4 5 6; 7 8 10]/20
            D = Matrix(Symmetric([1.0 2 3; 2 5 6; 3 6 10]) + 8I)
            J = mortar(Tridiagonal(Fill(Matrix(B'), ∞), Fill(D, ∞), Fill(B, ∞)))
            U, = reversecholesky(J)
            V = Matrix(U[Block.(1:N), Block.(1:N+1)])
            @test V*V' ≈ J[Block.(1:N), Block.(1:N)]
        end

        @testset "Hermitian blocks" begin
            B = [1.0+im 2; 3 4-2im]/10
            D = [5.0 1+im; 1-im 6+0im]
            J = mortar(Tridiagonal(Fill(Matrix(B'), ∞), Fill(D, ∞), Fill(B, ∞)))
            U, = reversecholesky(J)
            V = Matrix(U[Block.(1:N), Block.(1:N+1)])
            @test V*V' ≈ J[Block.(1:N), Block.(1:N)]
        end

        @testset "solves" begin
            B = [1.0 2; 3 4]/10
            D = [5.0 1; 1 6]
            J = mortar(Tridiagonal(Fill(Matrix(B'), ∞), Fill(D, ∞), Fill(B, ∞)))
            F = reversecholesky(J)

            b = [1.0; zeros(∞)]
            x = F \ b
            @test x isa BlockedVector
            @test x[1:40] ≈ (J \ b)[1:40]
            @test (J*x)[1:40] ≈ b[1:40]
            @test x[Block(2)] == x[3:4]

            # right-hand sides that do not fill a whole number of blocks
            for nz = 1:7
                b = [Float64.(1:nz); zeros(∞)]
                @test (J*(F \ b))[1:40] ≈ b[1:40]
            end

            # the tail is truncated once it drops below `tolerance`
            @test last(colsupport(\(F, [1.0; zeros(∞)]; tolerance=1e-15), 1)) <
                  last(colsupport(F \ [1.0; zeros(∞)], 1))

            # a zero right-hand side needs no tail at all
            z = F \ [0.0; zeros(∞)]
            @test iszero(z[1:10])

            @testset "3x3 blocks" begin
                B = [1.0 2 3; 4 5 6; 7 8 10]/20
                D = Matrix(Symmetric([1.0 2 3; 2 5 6; 3 6 10]) + 8I)
                J = mortar(Tridiagonal(Fill(Matrix(B'), ∞), Fill(D, ∞), Fill(B, ∞)))
                b = [1.0; zeros(∞)]
                @test (J*(reversecholesky(J) \ b))[1:40] ≈ b[1:40]
            end

            @testset "Hermitian blocks" begin
                B = [1.0+im 2; 3 4-2im]/10
                D = [5.0 1+im; 1-im 6+0im]
                J = mortar(Tridiagonal(Fill(Matrix(B'), ∞), Fill(D, ∞), Fill(B, ∞)))
                b = [1.0+0im; zeros(∞)]
                @test (J*(reversecholesky(J) \ b))[1:40] ≈ b[1:40]
            end
        end

        @testset "1x1 blocks match scalar" begin
            a, b = 3.0, 1.0
            J = mortar(Tridiagonal(Fill(fill(b,1,1), ∞), Fill(fill(a,1,1), ∞), Fill(fill(b,1,1), ∞)))
            U, = reversecholesky(J)
            Us, = reversecholesky(SymTridiagonal(Fill(a, ∞), Fill(b, ∞)))
            @test U[1:10, 1:10] ≈ Us[1:10, 1:10]
        end
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