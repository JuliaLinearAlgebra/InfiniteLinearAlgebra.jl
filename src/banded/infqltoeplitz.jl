
# this gives the d and e so that
# [Z A B;
#  0 d e]
#
# is the fixed point
function tail_de(a::AbstractVector{T}; branch=findmax) where T<:Real
    m = length(a)
    C = [view(a,m-1:-1:1) Vcat(-a[end]*Eye(m-2), Zeros{T}(1,m-2))]
    λ, V = eigen(C)
    n2, j = branch(abs2.(λ))
    isreal(λ[j]) || throw(DomainError(a, "Real-valued QL factorization does not exist. Try ql(complex(A)) to see if a complex-valued QL factorization exists."))
    n2 ≥ a[end]^2 || throw(DomainError(a, "QL factorization does not exist. This could indicate that the operator is not Fredholm or that the dimension of the kernel exceeds that of the co-kernel. Try again with the adjoint."))
    c = sqrt((n2 - a[end]^2)/real(V[1,j])^2)
    c*real(V[end:-1:1,j])
end

function tail_de(a::AbstractVector{T}; branch=findmax) where T
    m = length(a)
    C = [view(a,m-1:-1:1) Vcat(-a[end]*Eye(m-2), Zeros{T}(1,m-2))]
    λ, V = eigen(C)::Eigen{T,T,Matrix{T},Vector{T}}
    n2, j = branch(abs2.(λ))
    n2 ≥ abs2(a[end]) || throw(DomainError(a, "QL factorization does not exist. This could indicate that the operator is not Fredholm or that the dimension of the kernel exceeds that of the co-kernel. Try again with the adjoint."))
    c_abs = sqrt((n2 - abs2(a[end]))/abs2(V[1,j]))
    c_sgn = -sign(λ[j])/sign(V[1,j]*a[end-1] - V[2,j]*a[end])
    c_sgn*c_abs*V[end:-1:1,j]    
end


# this calculates the QL decomposition of X and corrects sign
function ql_X!(X)
    s = sign(real(X[2,end])) 
    F = ql!(X)
    if s ≠ sign(real(X[1,end-1])) # we need to normalise the sign if ql! flips it
        F.τ[1] = 2 - F.τ[1] # 1-F.τ[1] is the sign so this flips it
        X[1,1:end-1] *= -1
    end
    F
end




function ql(Op::TriToeplitz{T}; kwds...) where T<:Real
    Z,A,B = Op.dl.value, Op.d.value, Op.du.value
    d,e = tail_de([Z,A,B]; kwds...) # fixed point of QL but with two Qs, one that changes sign
    X = [Z A B; zero(T) d e]
    F = ql_X!(X)
    t,ω = F.τ[2],X[1,end]
    QL(_BandedMatrix(Hcat([zero(T), e, X[2,2], X[2,1]], [ω, X[2,3], X[2,2], X[2,1]] * Ones{T}(1,∞)), ∞, 2, 1), Vcat(F.τ[1],Fill(t,∞)))
end

ql(Op::TriToeplitz{T}) where T = ql(InfToeplitz(Op))

function ql(A::InfToeplitz{T}; kwds...) where T
    l,u = bandwidths(A)
    @assert u == 1
    a = reverse(A.data.args[1])
    de = tail_de(a; kwds...)
    X = [transpose(a); zero(T) transpose(de)]::Matrix{T}
    F = ql_X!(X) # calculate data for fixed point
    factors = _BandedMatrix(Hcat([zero(T); X[1,end-1]; X[2,end-1:-1:1]], [0; X[2,end:-1:1]] * Ones{T}(1,∞)), ∞, l+u, 1)
    QLHessenberg(factors, Fill(F.Q,∞))
end
