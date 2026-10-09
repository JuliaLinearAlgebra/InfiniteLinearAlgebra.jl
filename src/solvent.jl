"""
    matrixroot(A₂, A₁, A₀)

computes the minimal solvent of the quadratic matrix equation

    A₂*W^2 + A₁*W + A₀ == 0,

that is, the solution `W` whose eigenvalues are the `n` eigenvalues of smallest modulus of the
quadratic eigenvalue problem `det(A₂*λ^2 + A₁*λ + A₀) == 0`, where `n == size(A₀,1)`.

This uses an ordered generalized Schur decomposition of the linearization
`[0 I; -A₀ -A₁] - λ*[I 0; 0 A₂]`. `A₂` may be singular, in which case the corresponding
eigenvalues are infinite and never selected. The selection is well-defined when there is a gap
`|λₙ| < |λₙ₊₁|` between the selected and the remaining eigenvalues, and the solvent then exists
provided the top `n×n` block of a basis of the selected deflating subspace is invertible.
The result is always complex, even for real coefficients.
"""
function matrixroot(A₂, A₁, A₀)
    n = LinearAlgebra.checksquare(A₀)
    T = complex(float(promote_type(eltype(A₀), eltype(A₁), eltype(A₂))))
    Z = zeros(T,n,n); Iₙ = Matrix{T}(I,n,n)
    F = schur([Z Iₙ; -T.(A₀) -T.(A₁)], [Iₙ Z; Z T.(A₂)])
    s = falses(2n)
    s[sortperm(abs.(ifelse.(iszero.(F.β), Inf, F.α ./ F.β)))[1:n]] .= true
    F = ordschur(F, s)
    U = @view F.Z[:,1:n]
    U₁ = @view U[1:n,:]
    U₁ * (UpperTriangular(@view F.T[1:n,1:n]) \ @view(F.S[1:n,1:n])) / U₁
end
