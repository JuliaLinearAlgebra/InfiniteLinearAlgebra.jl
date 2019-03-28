

using Revise, InfiniteBandedMatrices, BlockBandedMatrices, BandedMatrices, LazyArrays, FillArrays, MatrixFactorizations, Plots
using BlockBandedMatrices, BlockArrays, BandedMatrices
import MatrixFactorizations: reflectorApply!, QLPackedQ
import InfiniteBandedMatrices: blocktailiterate, _ql
import BandedMatrices: bandeddata,_BandedMatrix



###
# We have a fixed point to normalized Householder
#
#   [σ 0; 0 1] * (I - τ*[v,1]*[v,1]')
#  
#  But ∞-QL needs to combine this into a single τ
# 
# Note because we are multiplying by an ∞-number of times
# we can multiply each side by diagm(…conj(s),s,conj(s),s,…) without 
# changing the matrix. This freedom allows us to reduce the above
# to a single Householder.
#
# Through an annoying amount of algebra we get the following.
#
function combine_two_Q(σ, τ, v)
    α = σ*(1-τ*abs2(v))
    β = (1-τ)
    γ = (1-τ)*σ-σ*τ*abs2(v) + 1

    # companion matrix for α*conj(s)^2 + β*s^2 - γ. 
    # Why [1]??
    s2 = eigvals([0     1; -α/β   γ/β])[1]
    s = sqrt(s2)
    t = 1-s^2*(1-τ)
    ω = τ/t*σ*v
    s, t, ω
end


Z,A,B=2,2.1+0.01im,0.5
n = 100_000; T = Tridiagonal(Fill(Z,n-1), Fill(A,n), Fill(B,n-1)); Q,L = ql(T);
        


ñ1 = (A + sqrt(A^2-4B*Z))/2
ñ2 = (A - sqrt(A^2-4B*Z))/2
ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
# ñ = ñ1
(n,σ) = (abs(ñ),conj(sign(ñ)))
d = σ*e2*Z/n
e = sqrt(n^2 - abs2(B))
X = [Z A B; 0 d e]
F = ql!(X)
@test conj(F.τ[1]-1) ≈ σ

s,t, ω = combine_two_Q(σ, τ, v)
@test Q.factors[5,6] ≈ ω













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


reflectorApply!(F.factors[2:-1:1,2], F.τ[2], [0.0,0.5])

InfiniteBandedMatrices.rig_qltail(2.0,0.1,0.5)
F = ql(BandedMatrix(T'-(0.1+0.000001im)I)[1:1_000_000,1:1_000_000]);
plot(real.(F.factors[band(0)][1:100]))

InfiniteBandedMatrices.rig_qltail(2.0,2.2,0.5)
InfiniteBandedMatrices.rig_qltail(0.5,2.2,2.0)


InfiniteBandedMatrices.qltail(2.0,2.7,0.5)
InfiniteBandedMatrices.qltail(2.0,2.7,0.5)

X

F.factors

F.τ
x = -1.98
c,a,b = [0 2.0; 0 0],[0 0.5; 2.0 0],[0 0.0; 0.5 0]; 
T = BlockTridiagonal(Vcat([c], Fill(c,∞)), 
                Vcat([copy(a)], Fill(a,∞)), 
                Vcat([b], Fill(b,∞)))
A = deepcopy(T); A[1,1] =  2; A

function fsde(A, n=10_000)
    B = BandedMatrix(Tridiagonal(Fill(A[6,5],n-1), Fill(A[6,6],n), Fill(A[5,6],n-1)))
    Q,L = ql(B)
    d,e = (Q'*[[0 2.0; 0 0]; zeros(size(Q,1)-2,2)])[1:2,1:2],L[1:2,1:2]
    d,e = QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:2])*d,QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:2])*e
end    

@time fsde(A-0.1I, 10_000_000)
A = deepcopy(T); B = A+(0.5+0.001im)*I; (F, d, e) = _ql(B, fsde(B,1_000_000)...)

B = A+(0.5+0.000009im)*I; (F, d, e) = _ql(B, d, e)

A = deepcopy(T); B = A+(0.001im)*I; (F, d, e) = _ql(B, fsde(B,1_000_000)...)


A = deepcopy(T); B = A+(0.0im)*I; (F, d, e) = _ql(B, fsde(B,1_000_000)...)


F.L
F.L[2,2]-F.L[3,3]
c,a,b

ñ1 = (A + sqrt(A^2-4B*Z))/2
ñ2 = (A - sqrt(A^2-4B*Z))/2
ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
ñ = ñ2
(n,σ) = (abs(ñ),conj(sign(ñ)))
if n^2 < abs2(B)
    throw(ContinuousSpectrumError())
end
e = sqrt(n^2 - abs2(B))
d = σ*e*Z/n

A = deepcopy(T); B = A+(2.3)*I; (F, d, e) = _ql(B, fsde(B,1_000_000)...)

d,e

_ql(T +2.5I)

T+2.5I

Z,A,B=2,2.1+0.01im,0.5
T = Tridiagonal(Fill(2+0.0im,∞), Fill(A,∞), Fill(0.5+0.0im,∞)); Q,L = ql(T)

n = 10; Q[1:n,1:n+2]*L[1:n+2,1:n] - T[1:n,1:n] |> norm



ñ1 = (A + sqrt(A^2-4B*Z))/2
ñ2 = (A - sqrt(A^2-4B*Z))/2
ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
# ñ = ñ1
(n,σ) = (abs(ñ),conj(sign(ñ)))
e = sqrt(n^2 - abs2(B))

if n^2 < abs2(B)
    throw(ContinuousSpectrumError())
end

ql(T)
d,e = qltail(Z,A,B)

@which qltail(Z,A,B)

1/sign(e*A-B*d)

d,e

σ = 1/sign(e*A-B*d)
n = sqrt(abs(e)^2+abs(B)^2)
e2 = sqrt(n^2 - abs2(B))
d2 = σ*e2*Z/n

d,e

d2-d

e2 + e
(abs(ñ),conj(sign(ñ)))
ql([Z A B; 0 d e]).τ

Q = ql([A B;d e]).Q
Q'*[Z A B; 0 d e]
τ
Q = Q2*Q1

Qm = (Q,m,n) -> [Eye(m) Zeros(m,2) Zeros(m,n); Zeros(2,m) Q zeros(2,n); Zeros(n,m) Zeros(n,2) Eye(n)]
Qm(0,3)*Qm(1,2)*Qm(2,1)*Qm(3,0) |> Matrix
n
n = 100_000; ql(BandedMatrix(T)[1:n,1:n]).L
Q
abs(τ)
 |> sign

τ
d,e
1/σ

τ = 1+e/sqrt(abs2(e)+abs2(B))
v = B/(τ*sqrt(abs2(e)+abs2(B)))
σ = -1/sign(e*A-B*d)

Q2*(I - τ*[v,1]*[conj(v) 1]) - Q


[σ*(1-τ*abs2(v)) -τ*σ*v;
  -τ*conj(v)      1-τ      ]
Q2
σ
H'

Q2*(I - τ*[v,1]*[conj(v) 1])*[B; e]

Q1

Q1*[Z; 0]

H'

[σ 0; 0 1] * ( I-τ*[conj(v),1]*[v 1]) * [1 0; 0 σ]

Qm(2,1)*Qm(3,0) |> Matrix
v
Q
( I-τ*[conj(v),1]*[v 1]) * [1 0; 0 sqrt(σ)]
Q
Q

abs2(F.factors[1,2])
abs2(v)

F = ql(BandedMatrix(T)[1:n,1:n]);
ω = F.factors[1,2]
t = conj(F.τ[3])
H = I-t*[ω,1]*[ω,1]'
Ht = I-t*[ω,1]*[ω,1]'
Qm(Q,0,3)*Qm(Q,1,2)*Qm(Q,2,1)*Qm(Q,3,0) -Qm(H',0,3)*Qm(H',1,2)*Qm(H',2,1)*Qm(H',3,0) |> Matrix


conj(s)^2*σ*(1-τ*abs2(v)) - (1-t*abs2(ω))
s^2 *(1-τ) - (1-t)
-τ*σ*v - (-t*ω)
-τ*conj(v) - (-t*conj(ω))
t^2*abs2(ω) - τ^2*σ*abs2(v)

abs2(ω) - τ^2*σ*abs2(v)/t^2
abs2(ω) - τ^2*σ*abs2(v)/(1-s^2*(1-τ))^2

t - (1-s^2*(1-τ))


conj(s)^2*σ*(1-τ*abs2(v)) - (1- (1-s^2*(1-τ))*abs2(ω))
conj(s)^2*σ*(1-τ*abs2(v)) - (1- (1-s^2*(1-τ))*τ^2*σ*abs2(v)/(1-s^2*(1-τ))^2)
(1-s^2*(1-τ))*conj(s)^2*σ*(1-τ*abs2(v)) - (1-s^2*(1-τ)) + τ^2*σ*abs2(v)



α*conj(s)^2 + β*s^2 - γ


z = s^2
α/β + z^2 - γ/β*z

[0     1; -α/β   γ/β] * [1,z] - z * [1,z]

 - s^2

using ApproxFun


s2 = (rs2 + im*qs2); s2/abs(s2)

M = [real(α)+real(β) imag(α)-imag(β) ; imag(α)+imag(β) -real(α)+real(β)]
sol = -M  \ [real(γ); imag(γ)]
κ = nullspace(M)

(sol + c * κ)*
M*(vcat(reim.(s^2)...) - sol)



M* [real(s2); imag(s2)] - [-real(γ); -imag(γ)]

sol = [rs2,qs2]






[real(α)+real(β) imag(α)-imag(β) ;
  imag(α)+imag(β) -real(α)+real(β)] |> nullspace

s^2

conj(s)^2 * σ*(1-τ*abs2(v))-(1-τ)*σ+σ*τ*abs2(v) - 1 + s^2*(1-τ)
α*conj(s)^2 + β*s^2 + γ

cond([real(α)+real(β) imag(α)-imag(β) ; imag(α)+imag(β) -real(α)+real(β)])
[real(α)+real(β) imag(α)-imag(β) ;
  imag(α)+imag(β) -real(α)+real(β)] * [real(s^2); imag(s^2)] - [-real(γ); -imag(γ)]

real(β)*real(s^2)

real(α)*real(s^2)+imag(α)*imag(s^2)



α*conj(s)^2 + β*s^2 + γ + conj(α)*s^2 + conj(β)*conj(s)^2 + conj(γ)
α*conj(s)^2 + β*s^2 + γ + conj(α)*s^2 + conj(β)*conj(s)^2 + conj(γ)

α

conj(s)^2 * σ*(1-τ*abs2(v))-(1-τ)*σ+σ*τ*abs2(v) - 1 + s^2*(1-τ) + s^2 * conj(σ)*conj(1-τ*abs2(v))-(1-τ)*σ+σ*τ*abs2(v) - 1 + s^2*(1-τ)


s^2

τ
conj(t)

Ht
s = sqrt((1-t)/(1-τ))

[conj(s) 0;  0 s] * Q * [conj(s) 0 ; 0 s] - Ht

Ht

abs2(s)

1-τ

Ht-H'
t
ω




Qm(Q,0,3)*Qm(Q,1,2)*Qm(Q,2,1)*Qm(Q,3,0) |> Matrix
Q*[B; e]

abs.(H')

abs.(Q)
σ


Q2
H'

H

Q

1-τ2 == σ*(1-τ)

1-σ*(1-τ)

(1-τ)


Q1'

1-τ



1-τ = -conj(e)/...
e/sqrt(abs2(e)+abs2(B))
Q1'
Q1'


τ = conj(e)/sqrt(abs2(e)+abs2(B))

I - τ*[v2 v;
      

τ = ql([B; e]).τ[1]; v = 


I-τ*[

abs(τ)

-1/sign(e*A-B*d)

1-σ

Q1 = [e -B; -conj(B) -conj(e)]/sqrt(abs2(e)+abs2(B));  Q2 = diagm(0 => [-1/sign(e*A-B*d),1]); 
X = [Z A B; 0 d e]; X = Q2*Q1*X; d,e = X[1,1:2]

Q = [e -B; -conj(B) -conj(e)]/sqrt(abs2(e)+abs2(B));  Q2 = diagm(0 => [1,1]); 
    X = [Z A B; 0 d e]; X = Q2*Q*X; d,e = X[1,1:2]

d,e

d,e = InfiniteBandedMatrices.tailiterate(Z,A,B).L[1,1:2]

Q
n = 2000; Q,L = ql(BandedMatrix(T)[1:n,1:n]); 
    e = L[1,1]; d = (Q'*[2; Zeros(size(Q,1)-1)])[1,1]



n = 1000; ql(BandedMatrix(T)[1:n,1:n]).τ
ql([Z A B; 0 d e]).L[2,3]

L

e
X

A

X = Q'*X; d,e = X[1,1:2]

Q2,L2 = ql(X)

Q2'X

Q*X


    X = Q2'*X; d,e = X[1,1:2]; X

X = [Z A B; 0 d e]; ql(X)

ql([B; e])

X    

n= 2000; ql(BandedMatrix(T)[1:n,1:n])

X

Q2




Z,A,B    
d,e
X
det(Q)


B


T

ñ
n

InfiniteBandedMatrices.qltail(2,A,0.5)[1] 

X = InfiniteBandedMatrices.qltail(2,A,0.5)[1]

Q,L = ql(T+1.9I)

X


L
d, e = X[1,1:2]
X = [2 A 0.5; 0 d    e] 
x = view(X, 2:-1:1, 3)
τ = LinearAlgebra.reflector!(x)
reflectorApply!(x, τ, view(X,2:-1:1,1:2))
d,e = X[1,1:2]


B = BandedMatrix(T,(2,1))

data = bandeddata(B).arrays[1]
B̃ = _BandedMatrix(data, size(data,2), 2,1)
B̃[end,end-1:end] .= (X[1,1], X[1,2])
F = ql!(B̃)
B̃.data[3:end,end] .= (X[2,2], X[2,1]) # fill in L
B̃.data[4,end-1] = X[2,1] # fill in L
H = Hcat(B̃.data, [X[1,3], X[2,3], X[2,2], X[2,1]] * Ones{eltype(X)}(1,∞))
Q, L = QL(_BandedMatrix(H, ∞, 2, 1), Vcat(F.τ,Fill(τ,∞)))

_BandedMatrix(H, ∞, 2, 1)


typeof(H)

d,e


Q,L = ql!(X)
d,e = (Q'*[2.0; zeros(size(Q,1)-1)])[1],L[1,1]

(1-Q.τ[1])*d
(1-Q.τ[1])*e

d,e = X[1,1:2]

d,e = QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:1])*[d;],QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:1])*e
d,e


d
d


Q, L = ql(Tridiagonal(Fill(2+0.0im,∞), Fill(0.5+0.0im,∞), Fill(0.5+0.0im,∞)))

Q[1:10,1:10]*L[1:10,1:10]

Q.factors
Q.τ

X

B

A[1:100,1:100]

e
F

d,e

n = 100_000_000;  B = A+(0.5+0.0000001im)*I;  @time ql(BandedMatrix(Tridiagonal(Fill(B[6,5],n-1), Fill(B[6,6],n), Fill(B[5,6],n-1)))).L

n = 10000; B = BandedMatrix(Tridiagonal(Fill(A[6,5],∞), Fill(A[6,6],∞), Fill(A[5,6],∞))); @time B[1:n,1:n]

import IntervalArithmetic: Interval
import MatrixFactorizations: QLPackedQ, reflector!, reflectorApply!
import InfiniteBandedMatrices: _ql
using DualNumbers

n = 100
B = BandedMatrix(Tridiagonal(Fill(A[6,5],∞), Fill(A[6,6],∞), Fill(A[5,6],∞)))
n = 10_000; @time QL = ql(B[1:n,1:n]);

@which MatrixFactorizations.getL(QL, axes(QL.factors))
n = 1_000_000; @time tril!(B[1:n,1:n])
@which

B = A-1E-9.5I; F, d, e = _ql(B, d,e)

100000eps()

F.L

d
e



Base.copysign(a::Dual, b::Dual) = abs(a)*sign(b)

A

exp(Interval(0,1) + im*Interval(0,1))

xx = range(-3,3,length=40)
yy = range(-3,3,length=40)
λ = 0.001im
B = A -λ*I
_, d, e = _ql(B, fsde(B)...)
d = [0 2; 0 0]
e = [-λ 0.5; (1.718)+im*(0.21)  dual(0.1,1.0)+im*(-0.86)]
X = [c a-λ*I b; zero(c) d e]
ql!(X)
X

_, d, e = _ql(B, fsde(B)...)
Z,A,B = c, a - 0.001im*I, b
d = [0 2; 0 0]
e = [-λ 0.5; Interval(1.718,1.719)+im*Interval(0.21,0.22)  Interval(0.1,0.11)+im*Interval(-0.86,-0.85)]
lo = x -> x isa Interval ? x.lo : x
hi = x -> x isa Interval ? x.hi : x
e = [-λ 0.5; Interval(1.718,1.719)+im*Interval(0.21,0.22)  Interval(0.1,0.11)+im*Interval(-0.86,-0.85)]
import IntervalArithmetic: ±

reim.(e[2,1]) .± 0.0001

h = 1E-7; e = [-λ 0.5; complex(real(e[2,1])±h,imag(e[2,1])±h)  complex(real(e[2,2])±h,imag(e[2,2])±h)]


_, d, e = _ql(B, fsde(B)...)
x0,x1 = e[2,1:2]
(x0,x1) = fixed(c,a-0.001im*I,b,x0,x1) 

rt = (x0,x1) -> fixed(c,a-0.001im*I,b,x0,x1) .- (x0,x1)
rt2 = (x,y,z,w) -> vcat(SVector{2}.(reim.(rt(complex(x,y), complex(z,w))))...)
rt3 = xyzw -> rt2(xyzw...)

roots(rt3, x × y × z × w, Newton, 1E-5)
rt3((x,y,z,w))

x,y,z,w = (1.718..1.719) , (0.21..0.22) , (0.1..0.11) , ((-0.86)..(-0.85))
rt(x+im*y,z+im*w)

(x,y,z,w) = (reim(x0)..., reim(x1)...)

x0,x1 = x0+0.01,x1+0.01
J = epsilon.([rt(dual(x0,1),x1) rt(x0,dual(x1,1))])
x0,x1 = [x0,x1] - (J \ rt(x0,x1))
rt(x0,x1)


rt = (x,y,s,t) -> fixed(c,a-0.001im*I,b,complex(x,y),complex(s,t)) .- (complex(x,y),complex(s,t))
x0 = x0+0.001
(x,y),(s,t) = reim(x0),reim(x1)
rt(x,y,s,t)


B
a

function fixed(c,a,b,x0,x1)
    e = [a[1,1] 0.5; x0 x1]
    d = [0 2; 0 0]
    A = [c a b; zero(c) d e]
    m,n = size(A)
    τ = zeros(eltype(A), min(m,n))

    for k = 6:-1:5
        μ = m+k-n
        x = view(A, μ:-1:1, k)
        τk = reflector!(x)
        τ[k-n+min(m,n)] = τk
        reflectorApply!(x, τk, view(A, μ:-1:1, 1:k-1))
    end
    A[2,3:4]
end
real(A[2,3]).hi
real(e[2,1]).hi

A = X
k = 5
μ = m+k-n
x = view(A, μ:-1:1, k)
τk = reflector!(x)
τ[k-n+min(m,n)] = τk
reflectorApply!(x, τk, view(A, μ:-1:1, 1:k-1))
A
x

A
e


x

    @which ql!(X)
x = X[end:-1:1,end]
reflector!(x)
x

d̃,ẽ = F.L[1:2,1:2], F.L[1:2,3:4]

d̃,ẽ = QLPackedQ(F.factors[1:2,3:4],F.τ[1:2])*d̃,QLPackedQ(F.factors[1:2,3:4],F.τ[1:2])*ẽ  # undo last rotation
ql!(X)

@which _ql(B, fsde(B)...)
λ = 0.0
d = [0 2; 0 0]
e = [-λ 0.5; Interval(1.7,1.8)+im*Interval(0.2,0.3)  Interval(0.1,0.2)+im*Interval(-0.9,-0.8)]
X = [c a-λ*I b;
    zero(c) d e]

ql!(X)

X = [c a b; zero(c) d 
e


q = (x,y) -> begin 
    @show x,y
    B = A - (x+im*y)*I; 
    F,d,e = _ql(B, fsde(B)...);
    F.L[1,1]
end

q(0.1,0.2)

z = q.(xx', yy)

contourf(xx, yy, abs.(z); nlevels=100, contours=false)

contourf(xx, yy, abs.(z); nlevels=100, contours=false)

collect(yy)


B = A -(-1.98+0.0001im)*I;  F,d,e = _ql(B, fsde(B)...); F.L[1,1]


xx = -4:0.01:-2

y = q.(xx, 0.0)
plot(xx,real.(y); legend=false)
xx = 2:0.01:4
y = q.(xx, 0.0)
plot!(xx,real.(y); legend=false)

B = A -(-1.98+0.00000048im)*I;
F,d,e = _ql(B, d, e); F.L[1,1]
_ql

_ql(B, d, e)

_ql(B,d,e)

d,e

fsde(B)[2] -e 

ql(B[1:200,1:200])

function it(c, a, b, d, e)
    z = zero(c)
    X = [c a b; z d e]
    F = ql!(X)
    d̃,ẽ = F.L[1:2,1:2], F.L[1:2,3:4]
    d̃,ẽ = QLPackedQ(F.factors[1:2,3:4],F.τ[1:2])*d̃,QLPackedQ(F.factors[1:2,3:4],F.τ[1:2])*ẽ  # undo last rotation
end

it(c,a,b,0.,0.2)

B = BandedMatrix(Tridiagonal(Fill(2.0,∞), Fill(-x,∞), Fill(0.5,∞)))
Q,L = ql(B[1:100_000,1:100_000]);
d,e = (Q'*[[0 2.0; 0 0]; zeros(size(Q,1)-2,2)])[1:2,1:2],L[1:2,1:2]
d,e = QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:2])*d,QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:2])*e
f = (x,y) -> it(c,a,b,[0 2.; 0 0], [1.98 0.5; x y])[2][2,:] .- (x,y)

x,y = 1.98,0.9
x,y = it(c,a,b,[0 2.; 0 0], [1.98 0.5; x y])[2][2,:]
it(c,a,b,[0 2.; 0 0], [1.98 0.5; x y]) 
y = 1.0
q = Fun(x -> norm(f(x,y)), 0..2, 1000)
x = findmin(q)[2]
q = Fun(y -> norm(f(x,y)), 0.5..2, 1000)
y = findmin(q)[2]
q = Fun(x -> norm(f(x,y)), 1.5..2, 1000)
x = findmin(q)[2]
q(x)


U, Σ, V = svd(A[1:100,1:100])

U*diagm(0=>Σ)*V'

V[:,end]

plot(abs.(U[:,end]) .+ eps(); yscale=:log10)

ql(B[1:1000,1:1000]')

x,y = it(c,a,b,[0 2.; 0 0], [1.98 0.5; x y])[2][2,:]

x,y

plot(q)

f(1.5,1.0)
xx = range(1.7,stop=1.72,length=100)
yy = range(0.85,stop=0.9,length=100)
contour(xx, yy, norm.(f.(xx', yy));nlevels=200)


xx = range(-3,stop=3,length=101)
yy = range(-3,stop=3,length=100)
contour(xx, yy, norm.(f.(xx', yy));nlevels=200)
f(0,0)
f(0.0,0.0)
f(1.9,0.0)

f(-2.0,0.0)
[1,2].-(3,4)


c

B
ql((A+1.98I)[1:151,1:151]).L[1,1]
ql((A)[1:300,1:300]).L[1,1]
ql((T)[1:305,1:305]).L[1,1]



ql(A')


@time ql(A-0.1im*I).L[1,1]


ql(A-3.5*I).L[1,1]
import InfiniteBandedMatrices: _ql
F, d, e = _ql(A-3.5*I, randn(2,2), randn(2,2))

xx = range(-3,stop=3,length=101)
yy = Vector{Float64}()

for x in xx
    @show x
    B = BandedMatrix(Tridiagonal(Fill(2.0,∞), Fill(-x,∞), Fill(0.5,∞)))
    Q,L = ql(B[1:100_000,1:100_000]);
    d,e = (Q'*[[0 2.0; 0 0]; zeros(size(Q,1)-2,2)])[1:2,1:2],L[1:2,1:2]
    d,e = QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:2])*d,QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:2])*e
    F,d,e = _ql(A'-x*I, d, e)
    push!(yy, F.L[1,1])
end






eigvals((A'-x*I)[1:100,1:100])

BandedMatrix(B[1:10000,1:10000]')
x = -1.98

plot(xx,yy)

ql(A')

ql((A')[1:100,1:100])
A
A'

yy

xx
ql(A'-3.0*I)

x = 1.92
x = -1.98
c,a,b = let A =  (A'-x*I)
    N = max(length(A.blocks.du.arrays[1])+1,length(A.blocks.d.arrays[1]),length(A.blocks.dl.arrays[1]))
    c,a,b = A[Block(N+1,N)],A[Block(N,N)],A[Block(N-1,N)]
end



ql((A'-x*I)[1:100,1:100])



ql(A'-x*I)

plot(xx, yy)





ql(A'+1.95*I)


A = A'+1.95*I




_ql(A, d, e)

N = max(length(A.blocks.du.arrays[1])+1,length(A.blocks.d.arrays[1]),length(A.blocks.dl.arrays[1]))
c,a,b = A[Block(N+1,N)],A[Block(N,N)],A[Block(N-1,N)]


[c a b; z d e]

ql(A[1:100,1:100])

_ql(A, d, e)

ql(A[1:200,1:00]'+1.95*I)

x = -1.1608040201005025 + 1.0im

c,a,b

a = a-x*I
A
A-x*I

ql(A-x*I)

d,e
x = -2.819095477386935
_ql(A-x*I, randn(2,2), randn(2,2))
_ql(A-x*I,)

_ql(A-x*I

ql(A-(-2.8I))

_ql(A-3.2*I, d, e)
F.L[1,1]

ql(A).Q[1,1]



ql(A[1:100,1:100])
(10_000^2)



scatter(reim(eigvals(A[1:1000,1:1000]))...)


ql(A)

ql(A[1:102,1:102])


N = max(length(A.blocks.du.arrays[1])+1,length(A.blocks.d.arrays[1]),length(A.blocks.dl.arrays[1]))
c,a,b = A[Block(N+1,N)],A[Block(N,N)],A[Block(N-1,N)]
z = zero(c)

function blocktailiterate2(c,b,a)
    z = zero(c)
    d,e = c,a
    for _=1:1000
        X = [b ; e]
        F  = ql!(X)
        P = PseudoBlockArray(F.Q'*[c a; z d], [2,2], [2,2])
        P[Block(1,1)] == d && P[Block(1,2)] == e && return PseudoBlockArray([P X], fill(2,2), fill(2,3)), F.τ
        d,e = P[Block(1,1)],P[Block(1,2)]
    end
    error("Did not converge")
end
z = zeros(2,2)
X = [c a b z;
     z c a b;
     z z d e]

ql!(X).τ

c,a,b = [0 0.5; 0 0],[0 2.0; 0.5 0],[0 0.0; 2.0 0]; 
A = BlockTridiagonal(Vcat([c], Fill(c,∞)), 
                Vcat([a], Fill(a,∞)), 
                Vcat([b], Fill(b,∞)))
A = A' + 0.1im*I
N = max(length(A.blocks.du.arrays[1])+1,length(A.blocks.d.arrays[1]),length(A.blocks.dl.arrays[1]))
c,a,b = A[Block(N+1,N)],A[Block(N,N)],A[Block(N-1,N)]

Q,L = ql(A[1:100,1:100])

d,e = (Q'*[c; zeros(100-2,2)])[1:2,1:2],L[1:2,1:2]
d,e = QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:2])*d,QLPackedQ(Q.factors[1:2,1:2],Q.τ[1:2])*e


blocktailiterate(c,a,b)

[d e]
[c a b;
 z d e]


blocktailiterate(c,a,b,d,e)
A
Q,L = ql(A)



ql(A[1:100,1:100])


c,b,a

X = [c a b; z d e]
F = ql!(X)
d̃,ẽ = F.L[1:2,1:2], F.L[1:2,3:4]
d̃,ẽ = QLPackedQ(F.factors[1:2,3:4],F.τ[1:2])*d̃,QLPackedQ(F.factors[1:2,3:4],F.τ[1:2])*ẽ # undo last rotation


d,e

d̃ , d


&& ẽ == e 
X

F.τ[3:end]
X




X

F.L[1:2,1:2] == d

ql(A[1:100,1:100]).factors





z

 m

X, τ = blocktailiterate(c,a,b)
X, τ = blocktailiterate2(c,a,b)
X

# reflector in two steps
import MatrixFactorizations: reflector!
x = randn(3)
z = copy(x); reflector!(view(z,1:2)); reflector!(view(z, [1,3])); z


y = copy(x); reflector!(y), y
x

Q,L = ql(A[1:100,1:100])



l,m = L[3:4,1:2],L[3:4,3:4]
L

ql([b; e]).τ

b
e

ql([c a b; z d e]).τ

F = ql(BandedMatrix(T-3I)[1:100,1:100])
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