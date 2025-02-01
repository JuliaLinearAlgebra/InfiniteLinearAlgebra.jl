"""
upper_mul_tri_triview(U, X) == Tridiagonal(U*X) where U is Upper triangular BandedMatrix and X is Tridiagonal
"""
function upper_mul_tri_triview(U::BandedMatrix, X::Tridiagonal)
    T = promote_type(eltype(U), eltype(X))
    n = size(U,1)
    UX = Tridiagonal(Vector{T}(undef, n-1), Vector{T}(undef, n), Vector{T}(undef, n-1))
    
    upper_mul_tri_triview!(UX, U, X)
end

function upper_mul_tri_triview!(UX::Tridiagonal, U::BandedMatrix, X::Tridiagonal)
    n = size(UX,1)

    
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    UXdl, UXd, UXdu = UX.dl, UX.d, UX.du
    Udat = U.data
    
    l,u = bandwidths(U)

    @assert size(U) == (n,n)
    @assert l == 0 && u ≥ 2
    # Tridiagonal bands can be resized
    @assert length(Xdl)+1 == length(Xd) == length(Xdu)+1 == length(UXdl)+1 == length(UXd) == length(UXdu)+1 == n

    UX, bⱼ, aⱼ, cⱼ, cⱼ₋₁ = initiate_upper_mul_tri_triview!(UX, U, X)
    UX, bⱼ, aⱼ, cⱼ, cⱼ₋₁ = main_upper_mul_tri_triview!(UX, U, X, 2:n-2, bⱼ, aⱼ, cⱼ, cⱼ₋₁)
    finalize_upper_mul_tri_triview!(UX, U, X, n-1, bⱼ, aⱼ, cⱼ, cⱼ₋₁)
end


function initiate_upper_mul_tri_triview!(UX, U, X)
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    UXdl, UXd, UXdu = UX.dl, UX.d, UX.du
    Udat = U.data

    l,u = bandwidths(U)

    j = 1
    aⱼ, cⱼ = Xd[1], Xdl[1]
    Uⱼⱼ, Uⱼⱼ₊₁, Uⱼⱼ₊₂ =  Udat[u+1,1], Udat[u,2],  Udat[u-1,3] # U[j,j], U[j,j+1], U[j,j+2]
    UXd[1] = Uⱼⱼ*aⱼ +  Uⱼⱼ₊₁*cⱼ  # UX[j,j] = U[j,j]*X[j,j] + U[j,j+1]*X[j+1,j]
    bⱼ, aⱼ, cⱼ, cⱼ₋₁ = Xdu[1], Xd[2], Xdl[2], cⱼ  # X[j,j+1], X[j+1,j+1], X[j+2,j+1], X[j+1,j]
    UXdu[1] = Uⱼⱼ*bⱼ + Uⱼⱼ₊₁*aⱼ + Uⱼⱼ₊₂*cⱼ # UX[j,j+1] = U[j,j]*X[j,j+1] + U[j,j+1]*X[j+1,j+1] + U[j,j+1]*X[j+1,j]

    UX, bⱼ, aⱼ, cⱼ, cⱼ₋₁
end


function main_upper_mul_tri_triview!(UX, U, X, jr, bⱼ, aⱼ, cⱼ, cⱼ₋₁)
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    UXdl, UXd, UXdu = UX.dl, UX.d, UX.du
    Udat = U.data
    l,u = bandwidths(U)

    @inbounds for j = jr
        Uⱼⱼ, Uⱼⱼ₊₁, Uⱼⱼ₊₂ =  Udat[u+1,j], Udat[u,j+1],  Udat[u-1,j+2] # U[j,j], U[j,j+1], U[j,j+2]
        UXdl[j-1] = Uⱼⱼ*cⱼ₋₁ # UX[j,j-1] = U[j,j]*X[j,j-1]
        UXd[j] = Uⱼⱼ*aⱼ +  Uⱼⱼ₊₁*cⱼ  # UX[j,j] = U[j,j]*X[j,j] + U[j,j+1]*X[j+1,j]
        bⱼ, aⱼ, cⱼ, cⱼ₋₁ = Xdu[j], Xd[j+1], Xdl[j+1], cⱼ  # X[j,j+1], X[j+1,j+1], X[j+2,j+1], X[j+1,j]
        UXdu[j] = Uⱼⱼ*bⱼ + Uⱼⱼ₊₁*aⱼ + Uⱼⱼ₊₂*cⱼ # UX[j,j+1] = U[j,j]*X[j,j+1] + U[j,j+1]*X[j+1,j+1] + U[j,j+2]*X[j+2,j+1]
    end

    UX, bⱼ, aⱼ, cⱼ, cⱼ₋₁
end

function finalize_upper_mul_tri_triview!(UX, U, X, j, bⱼ, aⱼ, cⱼ, cⱼ₋₁)
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    UXdl, UXd, UXdu = UX.dl, UX.d, UX.du
    Udat = U.data
    l,u = bandwidths(U)

    Uⱼⱼ, Uⱼⱼ₊₁ =  Udat[u+1,j], Udat[u,j+1] # U[j,j], U[j,j+1]
    UXdl[j-1] = Uⱼⱼ*cⱼ₋₁ # UX[j,j-1] = U[j,j]*X[j,j-1]
    UXd[j] = Uⱼⱼ*aⱼ +  Uⱼⱼ₊₁*cⱼ  # UX[j,j] = U[j,j]*X[j,j] + U[j,j+1]*X[j+1,j]
    bⱼ, aⱼ, cⱼ₋₁ = Xdu[j], Xd[j+1], cⱼ  # X[j,j+1], X[j+1,j+1], X[j+2,j+1], X[j+1,j]
    UXdu[j] = Uⱼⱼ*bⱼ + Uⱼⱼ₊₁*aⱼ # UX[j,j+1] = U[j,j]*X[j,j+1] + U[j,j+1]*X[j+1,j+1] + U[j,j+2]*X[j+2,j+1]

    j += 1
    Uⱼⱼ =  Udat[u+1,j] # U[j,j]
    UXdl[j-1] = Uⱼⱼ*cⱼ₋₁ # UX[j,j-1] = U[j,j]*X[j,j-1]
    UXd[j] = Uⱼⱼ*aⱼ  # UX[j,j] = U[j,j]*X[j,j] + U[j,j+1]*X[j+1,j]

    UX
end


# X*R^{-1} = X*[1/R₁₁ -R₁₂/(R₁₁R₂₂)  -R₁₃/R₂₂ …
#               0       1/R₂₂   -R₂₃/R₃₃
#                               1/R₃₃

tri_mul_invupper_triview(X::Tridiagonal, R::BandedMatrix) = tri_mul_invupper_triview!(similar(X, promote_type(eltype(X), eltype(R))), X, R)

function tri_mul_invupper_triview!(Y::Tridiagonal, X::Tridiagonal, R::BandedMatrix)
    n = size(X,1)
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    Ydl, Yd, Ydu = Y.dl, Y.d, Y.du
    Rdat = R.data
    
    l,u = bandwidths(R)
    
    @assert size(R) == (n,n)
    @assert l == 0 && u ≥ 2
    # Tridiagonal bands can be resized
    @assert length(Xdl)+1 == length(Xd) == length(Xdu)+1 == length(Ydl)+1 == length(Yd) == length(Ydu)+1 == n
    
    
    k = 1
    aₖ,bₖ = Xd[k], Xdu[k]
    Rₖₖ,Rₖₖ₊₁ = Rdat[u+1,k], Rdat[u,k+1] # R[1,1], R[1,2]
    Yd[k] = aₖ/Rₖₖ
    Ydu[k] = bₖ - aₖ * Rₖₖ₊₁/Rₖₖ

    @inbounds for k = 2:n-1
        cₖ₋₁,aₖ,bₖ = Xdl[k-1], Xd[k], Xdu[k]
        Ydl[k-1] = cₖ₋₁/Rₖₖ
        Yd[k] = aₖ-cₖ₋₁*Rₖₖ₊₁/Rₖₖ
        Ydu[k] = cₖ₋₁/Rₖₖ
        Rₖₖ,Rₖₖ₊₁,Rₖ₋₁ₖ₊₁,Rₖ₋₁ₖ = Rdat[u+1,k], Rdat[u,k+1],Rdat[u-1,k+1],Rₖₖ₊₁ # R[2,2], R[2,3], R[1,3]
        Yd[k] /= Rₖₖ
        Ydu[k-1] /= Rₖₖ
        Ydu[k] *= Rₖ₋₁ₖ*Rₖₖ₊₁/Rₖₖ - Rₖ₋₁ₖ₊₁
        Ydu[k] += bₖ - aₖ * Rₖₖ₊₁ / Rₖₖ
    end

    k = n
    cₖ₋₁,aₖ = Xdl[k-1], Xd[k]
    Ydl[k-1] = cₖ₋₁/Rₖₖ
    Yd[k] = aₖ-cₖ₋₁*Rₖₖ₊₁/Rₖₖ
    Rₖₖ = Rdat[u+1,k] # R[2,2], R[2,3], R[1,3]
    Yd[k] /= Rₖₖ
    Ydu[k-1] /= Rₖₖ

    Y
end
"""
    TridiagonalConjugationData(U, X, V, Y)

caches the infinite dimensional Tridiagonal(U*X/V)
in the tridiagonal matrix `Y`
"""

mutable struct TridiagonalConjugationData{T}
    const U::AbstractMatrix{T}
    const X::AbstractMatrix{T}
    const V::AbstractMatrix{T}

    const UX::Tridiagonal{T,Vector{T}} # cache Tridiagonal(U*X)
    const Y::Tridiagonal{T,Vector{T}} # cache Tridiagonal(U*X/V)

    datasize::Int
end

function TridiagonalConjugationData(U, X, V, uplo::Char)
    T = promote_type(typeof(inv(V[1, 1])), eltype(U), eltype(C)) # include inv so that we can't get Ints
    return BidiagonalConjugationData(U, X, V, Tridiagonal(T[], T[], T[]), Tridiagonal(T[], T[], T[]), 0)
end

copy(data::TridiagonalConjugationData) = TridiagonalConjugationData(copy(data.U), copy(data.X), copy(data.V), copy(data.UX), copy(data.Y), data.datasize)


function resizedata!(data::TridiagonalConjugationData, n)
    n ≤ 0 && return data
    n = max(v, n)
    dv, ev = data.dv, data.ev
    if n > length(ev) # Avoid O(n²) growing. Note min(length(dv), length(ev)) == length(ev)
        resize!(data.UX.dl, 2n)
        resize!(data.UX.d, 2n + 1)
        resize!(data.UX.du, 2n)
    
        resize!(data.Y.dl, 2n)
        resize!(data.Y.d, 2n + 1)
        resize!(data.Y.du, 2n)
    end


end

