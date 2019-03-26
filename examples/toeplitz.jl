using Revise, InfiniteBandedMatrices, BlockBandedMatrices, LazyArrays, FillArrays, Plots

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

using BlockBandedMatrices, BlockArrays, BandedMatrices
import MatrixFactorizations: reflectorApply!, QLPackedQ
import InfiniteBandedMatrices: blocktailiterate
reflectorApply!(F.factors[2:-1:1,2], F.τ[2], [0.0,0.5])

X

F.factors

F.τ
x = -1.98
c,a,b = [0 2.0; 0 0],[0 0.5; 2.0 0] - x*I,[0 0.0; 0.5 0]; 
T = BlockTridiagonal(Vcat([c], Fill(c,∞)), 
                Vcat([copy(a)], Fill(a,∞)), 
                Vcat([b], Fill(b,∞)))
A = deepcopy(T); A[1,1] =  2+1.98; A
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