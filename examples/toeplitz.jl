using Revise, InfiniteBandedMatrices, LazyArrays, FillArrays, Plots

# create a complex valued Toeplitz matrix
T = Tridiagonal(Vcat(Float64[], Fill(1/2,∞)), 
                Vcat(Float64[], Fill(0.0,∞)), 
                Vcat(Float64[], Fill(2.0,∞)))

# This operator has spectrum inside the image of 1/(2z) +s 2z on the unit circle, which is a Bernstein ellipse through 1.
# But in finite-dimensions it is a conjugation of a symmetric Toeplitz matrix with symbol 1/z+z so the eigenvalues are on [-2,2]
n = 100; T_n = T[1:n,1:n]
λ = eigvals(Matrix(T_n))
scatter(real.(λ), imag.(λ)) # points near [-2,2]


# The QL decomposition gives a very good indicator of being inside the spectrum: if `L[1,1] == 0` we have a kernel. 
# Away from the borderline this works fine with finite-dimensional linear algebra
λ = 0.5im; n = 100; ql(BandedMatrix(T-λ*I)[1:n,1:n]).L[1,1] # == 0.0 exactly


# but as we approach the edge of the spectrum, we need increasingly large n to determine `L[1,1]`:
λ = 1.49im;  n = 1000; ql(BandedMatrix(T-λ*I)[1:n,1:n]).L[1,1] # ≈ 0.0032 
λ = 1.499im; n = 10_000; ql(BandedMatrix(T-λ*I)[1:n,1:n]).L[1,1] # ≈ 0.001
λ = 1.4999im; n = 100_000; ql(BandedMatrix(T-λ*I)[1:n,1:n]).L[1,1] # ≈ 0.0003



# Thus the simple test of "are we in the spectrum" is computationally challenging. 
# The infinite-dimensional QL decomposition avoids this issue because the cost is independent of λ
# 
λ = 1.49im; ql(T-λ*I).L[1,1] # ≈ 1E-13
λ = 1.499im; ql(T-λ*I).L[1,1] # ≈ 6E-13
λ = 1.4999im; ql(T-λ*I).L[1,1] # ≈ 1E-11

λ = 1.50001im; ql(T-λ*I).L[1,1] # ≈ 1E-11

# There are some round-off error as we approach 1.5, which we can overcome using Big Floats:

λ = BigFloat(1.49)im; ql(T-λ*I).L[1,1] # ≈ 1E-13

(T-3I)
X, τ = InfiniteBandedMatrices.qltail(0.5,-3,2)

τ

using BandedMatrices
F = ql(BandedMatrix(T-3I)[1:100,1:100])

import MatrixFactorizations: reflectorApply!
import InfiniteBandedMatrices: blocktailiterate
reflectorApply!(F.factors[2:-1:1,2], F.τ[2], [0.0,0.5])

X

F.factors

F.τ

c,a,b = [0 0.5; 0 0],[0 2.0; 0.5 0],[0 0.0; 2.0 0]; 
A = BlockTridiagonal(Vcat([c], Fill(c,∞)), 
                Vcat([a], Fill(a,∞)), 
                Vcat([b], Fill(b,∞)))

imag.(ql((A-5im*I)[1:100,1:100]).L)
blocktailiterate(c,a-5im*I,b)[1]

ql((A-5im*I)[1:100,1:100]).factors[1:10,1:10]



c,a,b = 
A

X

ql(T-λ*I)


R = Matrix(Eye(2,2))
E = [0.0 1.0; -1.0 0.0]
λ = 5.0
A = BlockTridiagonal(Vcat(Matrix{Float64}[2R], Fill(2R,∞)), 
                Vcat(Matrix{Float64}[λ*E], Fill(λ*E,∞)), 
                Vcat(Matrix{Float64}[0.5R], Fill(0.5R,∞)))

A

using BlockArrays

c,a,b = (2R, λ*E, 0.5*R)
X, τ = blocktailiterate(c,a,b,randn(2,2),randn(2,2))

X

ql(A[1:100,1:100]).L
Q,L = ql(A);
Q[1:20,1:20]*L[1:20,1:20]

L
Q.τ


Q.τ




A[1:10,1:10]


R = [c; zeros(2,2)]

reflectorApply!(F.factors[2:-1:1,2], F.τ[2], [0.0,0.5])

z = randn(2,2)
X = [c a b; z c a]
P = PseudoBlockArray(ql(X).L, fill(2,2), fill(2,3))
X = [c a b; z P[Block(2,2)] P[Block(2,3)]]
X
d,e = c,a


X = [b ; e]
F  = ql!(X)
P = PseudoBlockArray(F.Q'*[c a; z d], [2,2], [2,2])
P[Block(1,1)] == d && P[Block(1,2)] == e && return PseudoBlockArray([P X], fill(2,2), fill(2,3)), F.τ
d,e = P[Block(1,1)],P[Block(1,2)]

Q, L = ql(A[1:100,1:100])

Q'*


d,e = (Q'A[Block.(2:51), Block.(1:50)])[1:2,1:2],(Q'A[Block.(2:51), Block.(1:50)])[1:2,3:4]
σ = diagm(0=>[-1,1]); d = σ*d; e = σ*e;
z = zeros(2,2)
X = [c a b; z d e]
ql(X)

d
e

d

L


X = [c a b; z c a]

d,e = randn(2,2),randn(2,2)



F.factors

A

F = ql(A[1:300,1:300])

Q.τ


import MatrixFactorizations: QLPackedQ
Q2 = QLPackedQ(Q.factors[1:4,1:2], Q.τ[1:2])
Q2*[


sz = size.(A.d,1), size.(A.d,2)



sz[1]
cumsum(sz[1])

cumsum.(sz)