
# this gives the d and e so that
# [Z A B;
#  0 d e]
#
# is the fixed point
function tail_de(a::AbstractVector{T}) where T<:Real
    m = length(a)
    C = [view(a,m-1:-1:1) Vcat(-a[end]*Eye(m-2), Zeros{T}(1,m-2))]
    λ, V = eigen(C)
    n, j = findmax(real.(λ))
    isreal(λ[j]) || throw(DomainError(a, "Real-valued QL factorization does not exist. Try ql(complex(A)) to see if a complex-valued QL factorization exists."))
    n^2 ≥ a[end]^2 || throw(DomainError(a, "QL factorization does not exist. This could indicate that the operator is not Fredholm or that the dimension of the kernel exceeds that of the co-kernel. Try again with the adjoint."))
    c = sqrt((n^2 - a[end]^2)/real(V[1,j])^2)
    c*real(V[end:-1:1,j])
end

function tail_de(a::AbstractVector{T}) where T
    m = length(a)
    C = [view(a,m-1:-1:1) Vcat(-a[end]*Eye(m-2), Zeros{T}(1,m-2))]
    λ, V = eigen(C)
    n2, j = findmax(abs2.(λ))
    n2 ≥ abs2(a[end]) || throw(DomainError(a, "QL factorization does not exist. This could indicate that the operator is not Fredholm or that the dimension of the kernel exceeds that of the co-kernel. Try again with the adjoint."))
    c_abs = sqrt((n2 - abs2(a[end]))/abs2(V[1,j]))
    c_sgn = -sign(λ[j])/sign(V[1,j]*a[end-1] - V[2,j]*a[end])
    c_sgn*c_abs*V[end:-1:1,j]    
end

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

    # companion matrix for β*z^2  - γ*z + α for z = s^2
    # Why sign??
    s2 = (γ - sqrt(γ^2-4α*β))/(2β)
    s = sqrt(s2)
    t = 1-s^2*(1-τ)
    ω = τ/t*σ*v
    conj(t), -ω
end


# this calculates the QL decomposition of X and corrects sign
function ql_X!(X)
    s = sign(real(X[2,end])) 
    F = ql!(X)
    if s ≠ sign(real(X[1,2])) # we need to normalise the sign if ql! flips it
        F.τ[1] = 2 - F.τ[1] # 1-F.τ[1] is the sign so this flips it
        X[1,1:end-1] *= -1
    end
    F
end


# this gives the parameters of the QL decomposition tail
function tail_stω!(F)
    σ = conj(F.τ[1]-1)
    τ = F.τ[2]
    v = F.factors[1,end]
    combine_two_Q(σ, τ, v)
end

function ql(Op::TriToeplitz{T}) where T<:Real
    Z,A,B = Op.dl.value, Op.d.value, Op.du.value
    d,e = tail_de([Z,A,B]) # fixed point of QL but with two Qs, one that changes sign
    X = [Z A B; zero(T) d e]
    F = ql_X!(X)
    t,ω = F.τ[2],X[1,end]
    QL(_BandedMatrix(Hcat([zero(T), e, X[2,2], X[2,1]], [ω, X[2,3], X[2,2], X[2,1]] * Ones{T}(1,∞)), ∞, 2, 1), Vcat(zero(T),Fill(t,∞)))
end

function ql(Op::TriToeplitz{T}) where T
    Z,A,B = Op.dl.value, Op.d.value, Op.du.value
    d,e = tail_de([Z,A,B]) # fixed point of QL but with two Qs, one that changes sign
    X = [Z A B; zero(T) d e]
    t,ω = tail_stω!(ql_X!(X))    # combined two Qs into one, these are the parameteris
    Q∞11 = 1 - ω*t*conj(ω)  # Q[1,1] without the callowing correction
    τ1 = 1 - (A -t*ω * X[2,2])/(Q∞11 * e) # Choose τ[1] so that (Q*L)[1,1] = A
    QL(_BandedMatrix(Hcat([zero(T), e, -X[2,2], -X[2,1]], [ω, -X[2,3], -X[2,2], -X[2,1]] * Ones{T}(1,∞)), ∞, 2, 1), Vcat(τ1,Fill(t,∞)))
end

function ql(A::InfToeplitz{T}) where T<:Real
    l,u = bandwidths(A)
    @assert u == 1
    a = reverse(A.data.applied.args[1])
    de = tail_de(a)
    X = [transpose(a); zero(T) transpose(de)]
    F = ql!(X)
    # second row of X contains L, first row contains factors. 
    ω = X[1,end] # reflector v
    QL(_BandedMatrix(Hcat([zero(T); X[1,end-1]; X[2,end-1:-1:1]], [ω; X[2,end:-1:1]] * Ones{T}(1,∞)), ∞, l+u, 1), 
        Vcat(zero(T), Fill(F.τ[2], ∞)))
end

function ql(A::InfToeplitz{T}) where T
    l,u = bandwidths(A)
    @assert u == 1
    a = reverse(A.data.applied.args[1])
    de = tail_de(a)
    X = [transpose(a); zero(T) transpose(de)]
    t,ω = tail_stω!(X)    # combined two Qs into one, these are the parameteris
    Q∞11 = 1 - ω*t*conj(ω)  # Q[1,1] without the callowing correction
    τ1 = 1 - (a[end-1] -t*ω * X[2,end-1])/(Q∞11 * de[end]) # Choose τ[1] so that (Q*L)[1,1] = A
    # second row of X contains L, first row contains factors. 
    # TODO: Why -X?
    QL(_BandedMatrix(Hcat([zero(T); X[1,end-1]; -X[2,end-1:-1:1]], [ω; -X[2,end:-1:1]] * Ones{T}(1,∞)), ∞, l+u, 1), 
        Vcat(τ1, Fill(t, ∞)))
end