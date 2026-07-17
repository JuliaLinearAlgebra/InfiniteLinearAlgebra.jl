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
