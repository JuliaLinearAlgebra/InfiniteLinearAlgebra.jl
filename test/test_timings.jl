using InfiniteLinearAlgebra, BandedMatrices, LazyArrays, FillArrays, BenchmarkTools, Test
import LazyArrays: MemoryLayout, arguments, resizedata!, partialqr!

@testset "Bessel timings" begin
    A = BandedMatrix(0 => -2*(0:∞)/1_000_000.0, 1 => Ones(∞), -1 => Ones(∞))
    n = 1_000_000
    V = view(A.data.args[2],:,1:n)
    b = similar(V)
    @test @belapsed(copyto!(b,V)) ≤ 0.01
    @test @belapsed(A.data[:,1:n]) ≤ 0.02
    @test @belapsed(A[1:n,1:n]) ≤ 0.02

    C = cache(A); @time resizedata!(C, n+1,n);
    s = 1000; C = cache(A); @time for m = s:s:1_000_000 resizedata!(C, m,m); end;
    s = 1000; F = qr(A); @time for m = s:s:1_000_000 partialqr!(F.factors.data,m); end;
    @time A \ Vcat([1.0],Zeros(∞));
end

