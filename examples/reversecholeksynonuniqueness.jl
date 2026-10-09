Δ = SymTridiagonal(Fill(0,∞), Fill(1,∞))
A = Δ + 4I
U = reversecholesky(A).U

# matches finite section
@test U[1:100,1:100] ≈ reversecholesky(A[1:200,1:200]).U[1:100,1:100]


# But we can find another one. 
a,b = U[1,1:2]

@test a^2 + b^2 ≈ 4
@test a*b ≈ 1

@test a^4 + 1 ≈ 4a^2

@test a^2 ≈ (4 + sqrt(16 - 4))/2

# choose other solution to quadratic equation:
ã = sqrt((4 - sqrt(16 - 4))/2)
b̃ = 1/ã

Ũ = Bidiagonal(Fill(ã,∞), Fill(b̃,∞), :U)
Ũ*Ũ'