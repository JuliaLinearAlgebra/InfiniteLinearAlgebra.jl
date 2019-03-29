
# this gives the d and e so that
# [Z A B;
#  0 d e]
#
# is the fixed point
function tail_de(Z,A,B)
    ñ1 = (A + sqrt(A^2-4B*Z))/2
    ñ2 = (A - sqrt(A^2-4B*Z))/2
    ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
    # ñ = ñ1
    (n,σ) = (abs(ñ),conj(sign(ñ)))
    e = -sqrt(n^2 - abs2(B))
    d = σ*e*Z/n
    d,e
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
    # Why [1]??
    s2 = (γ - sqrt(γ^2-4α*β))/(2β)
    s = sqrt(s2)
    t = 1-s^2*(1-τ)
    ω = τ/t*σ*v
    conj(t), -ω
end




# this gives the parameters of the QL decomposition tail
function tail_stω!(X)
    F = ql!(X)
    σ = conj(F.τ[1]-1)
    τ = F.τ[2]
    v = F.factors[1,3]
    combine_two_Q(σ, τ, v)
end

function ql(Op::TriToeplitz{T}) where T
    Z,A,B = Op.dl.value, Op.d.value, Op.du.value
    d,e = tail_de(Z,A,B) # fixed point of QL but with two Qs, one that changes sign
    X = [Z A B; zero(T) d e]
    t,ω = tail_stω!(X)    # combined two Qs into one, these are the parameteris
    Q∞11 = 1 - ω*t*conj(ω)  # Q[1,1] without the callowing correction
    τ1 = 1 - (A -t*ω * X[2,2])/(Q∞11 * e) # Choose τ[1] so that (Q*L)[1,1] = A
    QL(_BandedMatrix(Hcat([zero(T), e, -X[2,2], -X[2,1]], [ω, -X[2,3], -X[2,2], -X[2,1]] * Ones{T}(1,∞)), ∞, 2, 1), Vcat(τ1,Fill(t,∞)))
end
