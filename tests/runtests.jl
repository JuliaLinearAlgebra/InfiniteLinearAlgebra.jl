using Revise, InfiniteBandedMatrices, BandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test

J = SymTridiagonal(Vcat([randn(5);0;0], Fill(0.0,∞)), Vcat([randn(4);0.5;0.5], Fill(0.5,∞)))
A = J + 3I
F = ql(A);
Q,L = F;
Q̃, L̃ = ql(BandedMatrix(A)[1:100,1:100]);
@test L̃[1:20,1:20] ≈ L[1:20,1:20]
@test Q̃.τ[1:20] ≈ Q.τ[1:20]
@test Q̃[1:20,1:20] ≈ Q[1:20,1:20]

x = Vcat(randn(10), Zeros(∞))
@test (Q*(Q'x))[1:100] ≈ x[1:100]
@test (Q'*(Q*x))[1:100] ≈ x[1:100]

A = J + 1.00001I
Q,L = ql(A);
Q̃, L̃ = ql(BandedMatrix(A)[1:10000,1:10000]);
@test L̃[1:20,1:20] ≈ L[1:20,1:20]
@test Q̃.τ[1:20] ≈ Q.τ[1:20]
@test Q̃[1:20,1:20] ≈ Q[1:20,1:20]

