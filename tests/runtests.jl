using Revise, InfiniteBandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test

J = SymTridiagonal(Vcat([randn(5);0;0], Fill(0.0,∞)), Vcat([randn(4);0.5;0.5], Fill(0.5,∞)))
A = J + 3I
F = ql(A);
Q = F.Q;

x = Vcat([1.0], Zeros(∞))
Q*x
copy.(x.arrays)

copy(Zeros(∞))




copy(Vcat([1.0], Zeros(∞)))

Q*Vcat([1.0], Zeros(∞))

similar(Vcat([1.0], Zeros(∞)))

@which copy(1:10)