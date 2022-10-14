using Revise, InfiniteLinearAlgebra, MatrixFactorizations
import IntervalArithmetic
import IntervalArithmetic: Interval, emptyinterval
import MatrixFactorizations: add_mul_signed, reflector!

Δ = SymTridiagonal(Vcat([3.0,4.0],Fill(0.0,∞)), Vcat([0.5],Fill(0.5,∞)))
λ = Interval(4.1,4.2); ql(Δ - λ*I).L[1,1]
m = (λ.hi + λ.lo)/2
λ1 = Interval(λ.lo,m)
λ = 0 in ql(Δ - λ1*I).L[1,1] ? λ1 : Interval(m,λ.hi)
ql(Δ - λ*I).L[1,1]


@inline function add_mul_signed(normu::Interval, ξ1::Interval)
    ν = copysign(normu, real(ξ1))
    ξ1 + ν, ν
end

λ2 = Interval(m,λ.hi)
ql(Δ - λ1.hi*I).L[1,1]


ql(Δ - Interval(4.2)I).L[1,1]


rig_qltail(0.5,-1.98,2.0)

A = (Δ-λ*I)
import InfiniteLinearAlgebra: _qlfactUnblocked!
InfiniteLinearAlgebra.qltail(0.5,Interval(-4.1,-4.0), 0.5)

InfiniteLinearAlgebra.qltail(0.5,-4.1, 0.5)

InfiniteLinearAlgebra.qltail(0.5,-4.0, 0.5)

Z,A,B = 0.5,-4.0,0.5


(Interval(0,1) ∩ Interval(2,3)).lo


A = 4
rig_qltail(-2,10,B)

d,e = d2,e2

rig_qltail(Z,A,B,d,e)


InfiniteLinearAlgebra.qltail(Z,A,B)

@which reflector!(X[2:-1:1,end])

@which ql!(X)
A = X



import MatrixFactorizations: reflector!, reflectorApply!

m, n = size(A)
τ = zeros(eltype(A), min(m,n))
for k = n:-1:max((n - m + 1 + (T<:Real)),1)
    k = n
    μ = m+k-n
    x = view(A, μ:-1:1, k)
    τk = reflector!(x)
    τ[k-n+min(m,n)] = τk
    reflectorApply!(x, τk, view(A, μ:-1:1, 1:k-1))
end
x

@which reflector!(x)
QL(A, τ)




sign(Interval(-10,10))

length(d)

InfiniteLinearAlgebra.qltail(Z,A,B)



Z,A,B = 0.5,Interval(-4.1,-4.0),0.5
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
QL = _qlfactUnblocked!(X)

# two iterations to correct for sign
X[2,:] .= (zero(T), X[1,1], X[1,2]);
X[1,:] .= (Z,A,B);
QL = _qlfactUnblocked!(X)

X, QL.τ[end]
