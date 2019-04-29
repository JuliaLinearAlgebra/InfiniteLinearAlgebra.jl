
# this gives the d and e so that
# [Z A B;
#  0 d e]
#
# is the fixed point
function tail_de(a::AbstractVector{T}) where T<:Real
    m = length(a)
    C = [view(a,m-1:-1:1) Vcat(-a[end]*Eye(m-2), Zeros{T}(1,m-2))]
    λ, V = eigen(C)
    n2, j = findmax(abs2.(λ))
    isreal(λ[j]) || throw(DomainError(a, "Real-valued QL factorization does not exist. Try ql(complex(A)) to see if a complex-valued QL factorization exists."))
    n2 ≥ a[end]^2 || throw(DomainError(a, "QL factorization does not exist. This could indicate that the operator is not Fredholm or that the dimension of the kernel exceeds that of the co-kernel. Try again with the adjoint."))
    c = sqrt((n2 - a[end]^2)/real(V[1,j])^2)
    c*real(V[end:-1:1,j])
end

function tail_de(a::AbstractVector{T}) where T
    m = length(a)
    C = [view(a,m-1:-1:1) Vcat(-a[end]*Eye(m-2), Zeros{T}(1,m-2))]
    λ, V = eigen(C)::Eigen{T,T,Matrix{T},Vector{T}}
    n2, j = findmax(abs2.(λ))
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




function ql(Op::TriToeplitz{T}) where T<:Real
    Z,A,B = Op.dl.value, Op.d.value, Op.du.value
    d,e = tail_de([Z,A,B]) # fixed point of QL but with two Qs, one that changes sign
    X = [Z A B; zero(T) d e]
    F = ql_X!(X)
    t,ω = F.τ[2],X[1,end]
    QL(_BandedMatrix(Hcat([zero(T), e, X[2,2], X[2,1]], [ω, X[2,3], X[2,2], X[2,1]] * Ones{T}(1,∞)), ∞, 2, 1), Vcat(F.τ[1],Fill(t,∞)))
end

ql(Op::TriToeplitz{T}) where T = ql(InfToeplitz(Op))

function ql(A::InfToeplitz{T}) where T
    l,u = bandwidths(A)
    @assert u == 1
    a = reverse(A.data.applied.args[1])
    de = tail_de(a)
    X = [transpose(a); zero(T) transpose(de)]::Matrix{T}
    F = ql_X!(X) # calculate data for fixed point
    factors = _BandedMatrix(Hcat([zero(T); X[1,end-1]; X[2,end-1:-1:1]], [0; X[2,end:-1:1]] * Ones{T}(1,∞)), ∞, l+u, 1)
    QLHessenberg(factors, Fill(F.Q,∞))
end


###
# We have a fixed point to normalized Householder
#
#   [σ 0; 0 1] * (I - τ*[v,1]*[v,1]')
#  
#  But ∞-QL needs to combine this into a single τ
# 
# Note because we are multiplying by an ∞-number of times
# we can multiply each side by 
#       diagm(…conj(s),s,conj(s),s,…) 
# or
#       diagm(…conj(s),-s,conj(s),-s,…) 
# without 
# changing the matrix. This freedom allows us to reduce the above
# to a single Householder.
#
# Through an annoying amount of algebra we get the following, see tests
#
function combine_two_Q(σ, τ, v)
    # two possibilities 
    β =σ-τ*σ-σ*τ*abs2(v)-1; γ = abs2(v)*σ*τ^2;
    tt = (-β + sqrt(β^2 - 4γ))/2
    s = 1
    if !(abs2((1-τ)/(1-tt)) ≈ 1)
         β = -(σ*(1-τ*abs2(v))-σ*τ+1); γ = -abs2(v)*σ*τ^2;
         tt = (-β + sqrt(β^2 - 4γ))/2
         s = -1
         if !(abs2((1-τ)/(1-tt)) ≈ 1)
            tt *= NaN
         end
    end
    t = tt'
    ω = σ*τ*v/t'
    s, t, ω
end

function periodic_combine_two_Q(σ,τ,v)
    α = -σ * (1-τ*abs2(v)) 
    β =  (α*(1-τ)+σ*(τ*v)^2)
    a,b,c = (1-τ)*(β-1), (1-τ)*2α-(β^2+1), 2α- (β+1)*α
    s2 = (-b - sqrt(b^2-4a*c))/(2a) 
    s1 = (2α + (β-1)*s2)/(β+1)
    if !(abs(s2) ≈ 1) || !(abs(s1) ≈ 1)
        return (NaN*α,NaN*α),(NaN*α,NaN*α)
    end
    t1 = 1-(1-τ')*s1'
    ω1 = -τ'*v'/t1
    t2 = 1+(1-τ')*s2'
    ω2 = τ'*v' / t2
    (t1,t2), (ω1,ω2)
end


function tri_periodic_combine_two_Q(σ,τ,v)
    α = σ * (1-τ*abs2(v))
    β = α - τ*σ
    s1 = -((1 + 3α + 3*α*β + β^3 - 3*α*τ - 3*α*β*τ - sqrt((1 + β^3 - 3*α*(1 + β)*(-1 + τ))^2 + 4*α*(-1 + τ)*(1 + α + β + β^2 - α*τ)^2))/(2*(-1 + τ)*(1 + α + β + β^2 - α*τ)))
    s2 = -((1 - α - α*β + β^3 + α*τ + α*β*τ - sqrt((1 + β^3 - 3*α*(1 + β)*(-1 + τ))^2 + 4*α*(-1 + τ)*(1 + α + β + β^2 - α*τ)^2))/(2*(-1 + β + β^2 + α*(-1 + τ))*(-1 + τ)))
    s3 = (1 - α - α*β + β^3 + α*τ + α*β*τ - sqrt((1 + β^3 - 3*α*(1 + β)*(-1 + τ))^2 + 4*α*(-1 + τ)*(1 + α + β + β^2 - α*τ)^2))/(2*(1 + β - β^2 + α*(-1 + τ))*(-1 + τ))
    @assert abs(s1) ≈ 1
    @assert abs(s2) ≈ 1
    @assert abs(s3) ≈ 1

    t2 = 1-(1-τ')*(-s2') # (1)
    t1 = 1-(1-τ')*s1' # (2)
    t3 = 1+(1-τ')*s3' # (3)

    ω1 = τ'*v'/t1  # (11)
    ω2 = -τ'*v' / t2
    ω3 = -τ'*v' / t3

    (t1,t2,t3),(ω1,ω2,ω3)
end

# gives parameters for 2x2 Householder Q'
function householderparams(F)
    σ = conj(1 - F.τ[1])
    τ = conj(F.τ[2])
    v = F.factors[1,end]
    σ, τ, v
end

# Convert to LAPAck format
# There are several cases, it may not always work
function QL(QLin::QLHessenberg{T,<:InfBandedMatrix{T}}) where T
    F = QL(QLin.q[1].factors, QLin.q[1].τ) # The q stores the original iterated factorization
    σ,τ,v = householderparams(F) # parameters for fixed point householder
    s,t,ω = combine_two_Q(σ,τ,v) # combined two Qs into one, these are the parameteris
    if isnan(t) || isnan(ω) # NaN is returned if can't be combined as Toeplitz, so we try periodic Toeplitz
        (t1,t2),(ω1,ω2) = periodic_combine_two_Q(σ,τ,v)

        if isnan(t1) || isnan(t2) || isnan(ω1) || isnan(ω2)
            (t1,t2,t3),(ω1,ω2,ω3) = tri_periodic_combine_two_Q(σ,τ,v)
            Q∞11 = 1 - ω1*t1*conj(ω1)  # Q[1,1] without the callowing correction
            μ = mortar(Fill([1 -1 -1], 1, ∞))
            μ0 = mortar(Fill([1 0 0], 1, ∞))
            μ1 = mortar(Fill([0 1 0], 1, ∞))
            μ2 = mortar(Fill([0 0 1], 1, ∞))
            m = size(X,2)
            data = Hcat([X[1,end-1]; μ[1,1:m-1] .* X[2,end-1:-1:1]], 
                        ApplyArray(*, [μ[1,1:m].*X[2,end:-1:1] μ[1,2:m+1].*X[2,end:-1:1] μ[1,3:m+2].*X[2,end:-1:1]], Vcat(μ0,μ1,μ2)))
            data2 = Vcat(Hcat(zero(T), mortar(Fill([ω1 ω2 ω3],1,∞))), data)
            factors = _BandedMatrix(data2,∞,m-1,1)
            L∞11 = X[1,end-1]
            L∞21 = X[2,end-1]
            Q∞12 = -t1*ω1
            τ1 = (Q∞12 * L∞21 - A[1,1])/(L∞11*Q∞11) + 1
            QL(factors, Vcat(τ1, mortar(Fill([t1,t2,t3],∞))))
        else # 2-periodic
            Q∞11 = 1 - ω1*t1*conj(ω1)  # Q[1,1] without the callowing correction
            data = Hcat([-X[1,end-1]; (-1).^(1:l+u) .* X[2,end-1:-1:1]], 
                        ApplyArray(*, ((-1).^(1:l+u+1) .* X[2,end:-1:1]), ((-1).^(0:∞))'))
            
            
            data2 = Vcat(Hcat(zero(T), mortar(Fill([ω1 ω2],1,∞))), data)
            factors = _BandedMatrix(data2,∞,l+u,1)
            L∞11 = -X[1,end-1]
            L∞21 = -X[2,end-1]
            Q∞12 = -t1*ω1
            τ1 = (Q∞12 * L∞21 - A[1,1])/(L∞11*Q∞11) + 1
            QL(factors, Vcat(τ1, mortar(Fill([t1,t2],∞))))
        end
    else # non-periodic
        Q∞11 = 1 - ω*t*conj(ω)  # Q[1,1] without the callowing correction
        τ1 = if s == 1
            (a[end-1] +t*ω * X[2,end-1])/(Q∞11 * de[end])+1
        else
            1 - (a[end-1] -t*ω * X[2,end-1])/(Q∞11 * de[end]) # Choose τ[1] so that (Q*L)[1,1] = A
        end
        # second row of X contains L, first row contains factors. 
        factors = _BandedMatrix(Hcat([zero(T); -s*X[1,end-1]; s*X[2,end-1:-1:1]], [ω; s*X[2,end:-1:1]] * Ones{T}(1,∞)), ∞, l+u, 1)
        QL(factors, Vcat(τ1, Fill(t, ∞)))
    end
end