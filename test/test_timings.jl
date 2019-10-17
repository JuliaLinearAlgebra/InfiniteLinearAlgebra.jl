using InfiniteLinearAlgebra, BandedMatrices, LazyArrays, FillArrays, BenchmarkTools, Test
import LazyArrays: MemoryLayout, arguments

@testset "Bessel timings" begin
    A = BandedMatrix(0 => -2*(0:∞)/10.0, 1 => Ones(∞), -1 => Ones(∞))
    n = 1_000_000
    V = view(A.data.args[2],:,1:n)
    b = similar(V)
    @test @belapsed(copyto!(b,V)) ≤ 0.01
    @test @belapsed(A.data[:,1:n]) ≤ 0.02
    @test @belapsed(A[1:n,1:n]) ≤ 0.02
end

