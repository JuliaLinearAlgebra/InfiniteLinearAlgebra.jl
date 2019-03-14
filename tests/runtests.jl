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


A = J + 3I
F = ql(A);
Q,L = F;

@test Q.factors === parent(L)


J = Tridiagonal(Vcat([randn(4);0.5;0.5], Fill(0.3,∞)), 
                Vcat([randn(5);0;0], Fill(0.0,∞)), 
                Vcat([randn(4);0.5;0.5], Fill(0.5,∞)))

@test BandedMatrix(J)[1:100,1:100] == J[1:100,1:100]
A = J - 2I
Q,L = ql(A)
Q̃, L̃ = ql(BandedMatrix(A)[1:1000,1:1000]);
@test L̃[1:20,1:20] ≈ L[1:20,1:20]
@test Q̃.τ[1:20] ≈ Q.τ[1:20]
@test Q̃[1:20,1:20] ≈ Q[1:20,1:20]

L*Q

Q, L = randn(

A = brand(10,10,1,1);
Q,L = ql(A);
L*Q


n = 500; kr = range(-2; length=n, stop=2); h = step(kr)

Q, R = qr(A);
    norm((Matrix(R)*Matrix(Q))[band(2)])



### Semiseparable
A = Tridiagonal((1/h^2)*ones(length(kr)-1).+0im, Vector(kr).^3 .*im .- (2/h^2) .+ 0.0im,(1/h^2)*ones(length(kr)-1).+0im)
λ = eigvals(Matrix(A))
    scatter(real.(λ), imag.(λ))

Q,R = qr(Matrix(A))    
R*Q
using Plots


Q, R = qr(Matrix(Tridiagonal(ones(length(kr)-1).+0im, Vector(kr).*im .+ 0.0im,ones(length(kr)-1).+0im)));
    (R*Q)[band(2)]

Q, L = ql(Tridiagonal(ones(length(kr)-1).+0im, Vector(kr) .+ 0.0im,ones(length(kr)-1).+0im))
L[1:100,1:100]*Q[1:100,1]

MulArray(L,Q)[1,1]

Q'

L'