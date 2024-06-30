@testset "BidiagonalConjugationData" begin
    for _ in 1:10
        ir = InfiniteArrays.InfRandVector
        r = () -> Random.seed!(rand(1:2^32)) # avoid https://github.com/JuliaArrays/InfiniteArrays.jl/issues/182
        V = BandedMatrix(-1 => ir(r()), 0 => ir(r()), 1 => ir(r()))
        A = BandedMatrix(0 => ir(r()), 1 => ir(r()))
        X = BandedMatrix(0 => ir(r()), 1 => ir(r()), 2 => ir(r()))
        U = X * V * inv(A)
        B = InfiniteLinearAlgebra.BidiagonalConjugationData(U, X, V)

        @test MemoryLayout(B) == BidiagonalLayout{LazyArrays.LazyLayout,LazyArrays.LazyLayout}()
        @test bandwidths(B) == (0, 1)
        @test size(B) == (ℵ₀, ℵ₀)
        @test axes(B) == (OneToInf(), OneToInf())
        @test eltype(B) == Float64
        @test copy(B)[1:10, 1:10] == B[1:10, 1:10]
        @test !(copy(B) === B)
        @test copy(B')[1:10, 1:10] == B[1:10, 1:10]'
        @test !(copy(B') === B')
        @test LazyBandedMatrices.bidiagonaluplo(B) == 'U'
        @test LazyBandedMatrices.Bidiagonal(B)[1:100, 1:100] == LazyBandedMatrices.Bidiagonal(B[band(0)], B[band(1)], 'U')[1:100, 1:100]
        @test B[1:100, 1:100] ≈ A[1:100, 1:100]
        @test B[band(0)][1:1000] ≈ A[band(0)][1:1000]
        @test B[band(1)][1:1000] ≈ A[band(1)][1:1000]
        @test (B+B)[1:100, 1:100] ≈ 2(A[1:100, 1:100])
        @test (B*B)[1:100, 1:100] ≈ (A*A)[1:100, 1:100]
        @test inv(B)[1:100, 1:100] ≈ inv(A)[1:100, 1:100]
        @test (B*I)[1:100, 1:100] ≈ B[1:100, 1:100]
        @test (B*Diagonal(1:∞))[1:100, 1:100] ≈ B[1:100, 1:100] * Diagonal(1:100)
        @test (U*B)[1:100, 1:100] ≈ (X*V)[1:100, 1:100] rtol=1e-2 atol=1e-4
        @test (B'B)[1:100, 1:100] ≈ B'[1:100, 1:100] * B[1:100, 1:100]
    end
end