using Revise, InfiniteBandedMatrices, BandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test, DualNumbers, Plots
import InfiniteBandedMatrices: qltail, toeptail, tailiterate , tailiterate!

Base.copysign(a::Dual, b::Dual) = abs(a)*sign(b)

function symbolplot!(c,a,b; label="")
    θθ = 2π*(0:0.0001:1)
    zz = exp.(im*θθ)
    ii = b./zz  .+ c.*zz
    plot!(real.(ii), imag.(ii); label=label)
end

c,a,b
c,a,b = 2-0.2im,0.0+0im,0.5+im
c,a,b = 0.5+0im,0.0+0im,0.5+0im
c,a,b = ComplexF64.((0.5,0,2))
J = Tridiagonal(Vcat(ComplexF64[], Fill(c,∞)), 
                    Vcat(ComplexF64[1.5], Fill(0.0im,∞)),
                    Vcat(ComplexF64[], Fill(b,∞)))

xx = -5:0.001:5; plot(xx, (λ -> real(ql(J'-λ*I).L[1,1])).(xx); label="abs(L[1,1])", xlabel="x")
                plot(xx, (λ -> real(ql(J'-λ*I).L[1,1])).(xx); label="abs(L[1,1])", xlabel="x")


ql(J'-2.49999*I)

ql(J-2.50001*I)
eigbvalBandedMatrix(J)[1:100,1:100]


λ = eigvals(Matrix(J[1:2000,1:2000]))
scatter(real.(λ), imag.(λ); label="finite section with n = $(length(λ))")
    symbolplot!(c,a,b; label="essential spectrum")




ql(J'-0I)


symbolplot!(c,a,b)

@which ql(J'-λ[1]*I).Q[3,1]

ql(J'-λ[1]*I).factors
a
tailiterate(b,a-λ[1],c)
qltail(b,a-λ[1],c)

ql(J'-λ[1]*I).τ
ql(J'-λ[1]*I).factors

ql(J'-0.1*I)

Q.factors

Q.factors
Q.τ

Q,L = ((BandedMatrix(J'-λ[1]*I)[1:200,1:200]) |> ql)

(BandedMatrix(J)[1:200,1:200]-λ[1]*I) * Q[:,1]

Q[1000,1]
n = 150; κ = Q[1:n,1]
(J-λ[1]*I)[1:n,1:n]*κ



(J-λ[1]*I)*Vcat(Q[1:150,1],Zeros(∞))

Q*Vcat(1,Zeros(∞))

Q[:,1]

J'-λ[1]*I

ql(J-2.500001*I)

ql(BandedMatrix(J-3*I)[1:1000,1:1000])


Q,L = ql(BandedMatrix(J)[1:1000,1:1000])

Q[:,1]

M = zeros(ComplexF64,n,n); M[band(1)] .= b; M[band(-1)] .= c; 
M = zeros(ComplexF64,n,n); M[band(1)] .= conj(c); M[band(-1)] .= conj(b); 
ql(M)
M

plot(1:5)

qr(M)

svdvals(Matrix(F.L[1:1000,1:1000]))

Q,L = F;

@which Matrix(Q)

typeof(Q')
function Base.Matrix(::Adjoint{<:Any,<:QLPackedQ}) 
Base.Matrix{T}(Q::Adjoint{<:Any,<:QLPackedQ}) where {T} = lmul!(Q, Matrix{T}(I, size(Q, 1), min(Q,2)))
@time Q'*[1; zeros(999)]

n = 1000; Q = F.Q; Qt = lmul!(Q', Matrix{T}(I, n,n)); Q = Qt';  svdvals(Q[1:501,1:501])

Q[2:2:200,1:2:200]

Q[2:2:400,1:2:200] \ [Zeros(200-1);1] |> norm

plot(abs.(Q*[Zeros(n-1);1]))
svdvals(Q[2:2:400,1:2:200])

k = zeros(eltype(Q),n)
k[1] = 1
k[3] = -Q[2,1]/Q[2,3]
for m = 4:2:20 
    k[m+1] = -(Q*k)[m]/Q[m,m+1]
end


plot(abs.(k))

plot(abs.(Q*k))


plot(abs.(nullspace(Matrix(L'))); yscale=:log10)

Q*k

Q[2:2:end,1:2:end]

svdvals(Q[2:2:end,1:2:end])


Q*k




k

Q[1:505,1:501]

[1; zeros(999)]

F.τ

F.factors

F.τ
norm(Q[1,2])
Q[2,1]
heatmap(sparse(abs.(Q)))

svdvals(Q[:,1:10])

plot(abs.(Q[:,51]))

size(Q)

svdvals(Q[:,[1,3]])

svdvals(Q)

spy(abs.(Q))
norm(Q[3,4])
plot(abs.(Q[:,2])) 

plot(real.(F.L[band(-2)])[1:100])

function it(Z,A,B,d,e)
    X = [Z A B; 0 d e]
    ql!(X)
    X[1,1:2]
end
fxd(Z,A,B,d,e) = it(Z,A,B,d,e) - [d,e]

A = (J-0.0im*I)
ql(A)
@which qltail(c,0.0,b)
Z,A,B = c,0.9im,b
T = promote_type(eltype(Z),eltype(A),eltype(B))
ñ1 = (A + sqrt(A^2-4B*Z))/2
ñ2 = (A - sqrt(A^2-4B*Z))/2
ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
(n,σ) = (abs(ñ),conj(sign(ñ)))
if n^2 < abs2(B)
    throw(ContinuousSpectrumError())
end

e = sqrt(n^2 - abs2(B))
d = σ*e*Z/n
epsilon.(fxd(Z,A,B,dual(d,1), e))
A = 1.1im
Z,A,B = c,1.1im,b
tailiterate(Z,A,B)
J + A*I


(BandedMatrix(J - 0.0im*I)[1:10000,1:10000] |> ql).L

ql(J - 0.9im*I).L

J

A = -1.0001im
d,e = tailiterate(Z,A,B)[1][1,1:2]
sqrt(e^2 + abs2(B))

B = BandedMatrix(J+A*I,(2,1))

opnorm(Matrix(ql(BandedMatrix(J)[1:1000,1:1000]).L))

Matrix(ql(BandedMatrix(J)[1:100,1:100]).Q)




norm(J[1:1000,1:1000])

Matrix(J[1:2000,1:2000]) |> svdvals
opnorm(Matrix(J[1:2000,1:2000]))

b,a,c,_ = toeptail(B)

B
@which toeptail(B)



b,a,c

A

Z,A,B
J


T = promote_type(eltype(Z),eltype(A),eltype(B))
ñ1 = (A + sqrt(A^2-4B*Z))/2
ñ2 = (A - sqrt(A^2-4B*Z))/2
ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
(n,σ) = (abs(ñ),conj(sign(ñ)))
if n^2 < abs2(B)
    throw(ContinuousSpectrumError())
end

e = sqrt(n^2 - abs2(B))
d = σ*e*Z/n

X = [Z A B;
        0 d e]
QL = ql!(X)

# two iterations to correct for sign
X[2,:] .= (zero(T), X[1,1], X[1,2]);
X[1,:] .= (Z,A,B);
QL = ql!(X)

X, QL.τ[end]  
Z,A,B
c,a,b
tailiterate(c,a,b)
qltail(c,a,b)


A

it(d,e) - [d,e]

it(d,e) - [d,e]


X1 = [Z A B; 0 dual(d,1) e]; ql!(X1); epsilon.(X1[1,1:2]-[d,e])
X2 = [Z A B; 0 d dual(e,1)]
ql!(X2)

[epsilon.(X1[1,1:2]) epsilon.(X2[1,1:2])] |> eigvals

 X2

X = [Z A B; 0 d e+10eps()]
tailiterate!(X)
QL = ql!(X)

# two iterations to correct for sign
X[2,:] .= (zero(T), X[1,1], X[1,2]);
X[1,:] .= (Z,A,B);
QL = ql!(X)

X, QL.τ[end]      
c,a,b
a = 0.0
tailiterate(c,a,b)

tailiterate!(X)



Q,L= ql(J)

Q[1:100,1:100]*L[1:100,1:100]

Q,L = ql(BandedMatrix(J)[1:100,1:100])
plot(abs.(Q[:,1]))
Q'Q

J

abs.(eigvals(Matrix(J[1:1000,1:1000])))|> minimum

Q*L - J[1:100,1:100] |> norm

B = BandedMatrix(A, (2,1))
 b,a,c,_ = toeptail(B)

toeptail(B)

qltail(c,a,b)

Z,A,B = c,-1.001im,b
qltail(Z,A,B)

using Plots    

xx = -4:0.1:4
L = (λ -> try 
        real(ql(J-λ*I).L[1,1] )
    catch ContinuousSpectrumError 
        NaN
    end).(xx)

plot(xx,L)    


θθ = 2π*(0:0.0001:1)
    zz = exp.(im*θθ)
    plot(b./zz  .+ c.*zz)
θθ = 2π*(0:0.0001:0.5)
    zz = exp.(im*θθ)
    plot!(b./zz  .+ c.*zz)

b,c

c,a,b

plot(θθ, abs.(c./zz .+ a .+ b.*zz))


Z,A,B = c,-1.0001im,b

Bandw(J+A*I)
qltail(Z,A,B)


b,a,c,_ = toeptail(B)
X, τ = qltail(c,a,b)

T = promote_type(eltype(Z),eltype(A),eltype(B))
ñ1 = (A + sqrt(A^2-4B*Z))/2
ñ2 = (A - sqrt(A^2-4B*Z))/2
ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
(n,σ) = (abs(ñ),conj(sign(ñ)))
if n^2 < abs2(B)
    throw(ContinuousSpectrumError())
end
n^2 < abs2(B)

ql(J-1.007im*I)
###

import InfiniteBandedMatrices: tailiterate, tailiterate!, reflector!, reflectorApply!

c,a,b = 0.6,2,0.5
T = Float64
X, τ = tailiterate(c,a,b)
        X[2,:] .= (zero(T), X[1,1], X[1,2])
        X[1,:] .= (c,a,b)

tailiterate(c,a,b)

function givenstail(Z::Real, A::Real, B::Real)
    s = (-A + sqrt(A^2-4B*Z))/(2Z)

    if s^2 > 1
        X, τ = givenstail(-Z,-A,-B)
        X[1,3] = -X[1,3]
        return -X, τ
    end

    c = -sqrt(1-s^2)
    γ¹ = Z*c
    γ⁰ = c*A + s*γ¹
    X = [Z A B;
         0 -γ¹ -γ⁰]
    QL = ql!(X)
    X, QL.τ[end]         
end

function givenstail(Z, A, B)
    ñ = (A + sign(real(A))*sqrt(A^2-4B*Z))/2
    (n,σ) = (abs(ñ),conj(sign(ñ)))
    e = sqrt(n^2 - abs2(B))
    d = σ*e*Z/n

    X = [Z A B;
         0 d e]
    QL = ql!(X)

    # two iterations to correct for sign
    X[2,:] .= (zero(T), X[1,1], X[1,2]);
    X[1,:] .= (Z,A,B);
    QL = ql!(X)

    X, QL.τ[end]         
end


Z,A,B = 2-0.2im,0,0.5+im
    givenstail(Z,A,B) .≈  tailiterate(Z,A,B)

ñ = (A - sqrt(A^2-4B*Z))/2
(n,σ) = (abs(ñ),conj(sign(ñ)))
e = sqrt(n^2 - abs2(B))
d = σ*e*Z/n

d,e = X[1,1:2]

X = [Z A B;
        0 d e]
QL = ql!(X)
X, QL.τ[end]         

tailiterate(c,a,b)

all(givenstail(c,a,b) .≈ tailiterate(c,a,b))


Z,A,B = -1+0.2im,-2+2im,-0.5+0.1im
givenstail(Z,A,B)


Z,A,B = -1,-2im,0.5
X, τ = tailiterate(Z,A,B)

d,e = X[1,1:2]

n = sqrt(e^2+abs2(B))
σ = conj(sign(e*A-B*d))
H = [σ*e -σ*B; -conj(B) -conj(e)]/n
H*[Z A B; 0 d e]

σ*e*Z/n - d


(A + sqrt(A^2-4B*Z))/2
ñ = (A - sqrt(A^2-4B*Z))/2
(n,σ) .== (abs(ñ),conj(sign(ñ)))

sqrt(n^2 - abs2(B))
e
e^2

(n,σ)
(abs(ñ),conj(sign(ñ)))
(n,σ)

σ
n

n/σ

σ|>abs
d,e

H2 = diagm(0 => [1/sign(e*A-B*d),1])

d,e

H2*H2'

x = [e,B]
    τ = reflector!(x)
    v = [x[2],1]
    H = I - τ*v*v' 
   
[e -B; -conj(B) -conj(e)]/n    
    
ql([Z A B;
    0 d e]).τ

H*[B,e]

H'H
e*Z/n -d

e^2

e^4 + 2*(abs2(B)+B*Z-A^2)*e^2 + (abs2(B)+B*Z)^2 - A^2*abs2(B)



s, t = -X[1,3]/norm(X[:,3]), -X[2,3]/norm(X[:,3])
H = [t -s; conj(s) conj(t)]
H'H
H*X



(-A + sqrt(abs2(A)-4B*conj(Z)))/(2conj(Z))


H

X

Z,A,B

X

J = SymTridiagonal(Vcat(Float64[], Fill(b,∞)), 
                    Vcat(Float64[], Fill(a,∞)),
                    Vcat(Float64[], Fill(c,∞)))


givenstail(-c,-a,-b)

tailiterate(c,a,b)

X


X

x = X[2:-1:1,3]
    τ = reflector!(x)
    v = [1; x[2]]
    H = I - τ*v*v' 
    H*X[2:-1:1,3]
x = X[2:-1:1,3]
s, t = -x[1]/sqrt(x[1]^2+x[2]^2), -x[2]/sqrt(x[1]^2+x[2]^2)
[s t;t -s] * x

s, t = x[2]/sqrt(x[1]^2+x[2]^2), x[1]/sqrt(x[1]^2+x[2]^2)
H = [t -s; s t]
H*X




v
τ

1/H[1,1]

X

function givenstail2(Z, A, B)
    # @assert a^2-4c*b ≥ 0
    s = (-A + sqrt(A^2-4B*Z))/(2Z)
    l⁰ = (A + sqrt(A^2-4B*Z))/2
    # if s∞^2 > 1
    #     s∞ = (t₀ + sqrt(t₀^2-4t₁^2))/(2t₁)
    #     l0 = (t₀ - sqrt(t₀^2-4t₁^2))/2
    # end
    c = -sqrt(1-s^2)
    γ¹ = Z*c
    γ⁰ = c*A + s*γ¹
    l¹ = B+Z  # = c*γ¹ - st₁
    l² = -Z*s
    c,s,l⁰,l¹,l²,γ¹,γ⁰
end

givenstail2(c,a,b)

v

s,t


H
 
[s t;t -s]

H*x

[s t;t -s]

x
x1 = X[2,3]
v1, v2 = x
X[2,2]
X
reflectorApply!(x, τ, X[2:-1:1,2])
A = X[2:-1:1,1]
x2 = X[2,1]

vAj = conj(τ)*v2'*a
A[1] = - vAj
A[2] = a -  v2*vAj
x2
function it(x1, x2)
    ν = sqrt(abs2(x1) + abs2(b))
    ξ1 = x1 + ν * sign(real(x1))
    v1 = -ν 
    v2 = b/ξ1
    τ = ξ1/ν

    a -  v2*conj(τ)*(x2 + v2'*a), c -  v2*conj(τ)*v2'*c
end

x2,x1 = X[2,2:3]
it(x1,x2)  .≈ (x1,x2)

# simplify
function it(x1, x2, c, a, b)
    ν = sqrt(abs2(x1) + abs2(b))
    ξ1 = x1 + ν * sign(real(x1))
    v2 = b/ξ1
    τ = ξ1/ν
    a -  v2*conj(τ)*(x2 + v2*a), c -  v2^2*conj(τ)*c
end

ν = sqrt(abs2(x1) + abs2(b))
ξ1 = x1 + ν * sign(real(x1))
v2 = b/ξ1
τ = ξ1/ν
a -  v2*conj(τ)*(x2 + v2'*a), c -  v2*conj(τ)*v2'*c


v = copy(x); reflector!(v); v


X

x1

x1
x2
A[1]
ql(X).factors

X
x1 = x[1]
ξ1 = x[1] 
normu = abs2(ξ1) + b^2
normu = sqrt(normu)
ν = normu * sign(real(ξ1))
ξ1 += ν



ξ1/ν


ql(X)

###


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


T = Tridiagonal(Vcat(ComplexF64[], Fill(1/2+0im,∞)), 
                Vcat(ComplexF64[], Fill(0.0im,∞)), 
                Vcat(ComplexF64[], Fill(2.0+0im,∞)))



scatter((z -> 1/(2z) + 2z).(exp.(im*(0:0.01:2π))))
ql(J-1.6im*I).L[1,1]

eigvals(Matrix(J[1:100,1:100]))

scatter(eigvals(J[1:100,1:100]+eps()*randn(100,100)))

using Plots

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