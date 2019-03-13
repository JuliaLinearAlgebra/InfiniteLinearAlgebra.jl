using Revise, InfiniteBandedMatrices, FillArrays, LazyArrays, Test

J = SymTridiagonal(Vcat([randn(5);0;0], Fill(0.0,∞)), Vcat([randn(4);0.5;0.5], Fill(0.5,∞)))
A = J + 3I
F = ql(A);
Q = F.Q;

x = cache(Vcat([1.0], Zeros(∞)))
lmul!(Q, x)
