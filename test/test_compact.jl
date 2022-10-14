using InfiniteLinearAlgebra, FillArrays, LazyArrays, ArrayLayouts, Test

@testset "Compact" begin
    A = ApplyMatrix(hvcat, 2, randn(5,5), Zeros(5,∞), Zeros(∞,5), Zeros(∞,∞))
    b = [randn(10); zeros(∞)];
    @test ((I + A) \ b)[1:10] ≈ (I+A)[1:10,1:10] \ b[1:10]

    C = zeros(∞,∞);
    C[1:5,1:5] .= randn.()
    @test_skip ((I + C) \ b)[1:10] ≈ (I+C)[1:10,1:10] \ b[1:10]
end
