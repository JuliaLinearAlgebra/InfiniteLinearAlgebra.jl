
# upper_mul_tri_triview(U, X) == Tridiagonal(U*X) where U is Upper triangular BandedMatrix and X is Tridiagonal
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

    l,u = bandwidths(U)

    @assert size(U) == (n,n)
    @assert l == 0 && u ≥ 2
    # Tridiagonal bands can be resized
    @assert length(Xdl)+1 == length(Xd) == length(Xdu)+1 == length(UXdl)+1 == length(UXd) == length(UXdu)+1 == n

    UX, bₖ, aₖ, cₖ, cₖ₋₁ = initiate_upper_mul_tri_triview!(UX, U, X)
    UX, bₖ, aₖ, cₖ, cₖ₋₁ = main_upper_mul_tri_triview!(UX, U, X, 2:n-2, bₖ, aₖ, cₖ, cₖ₋₁)
    finalize_upper_mul_tri_triview!(UX, U, X, n-1, bₖ, aₖ, cₖ, cₖ₋₁)
end

# populate first row of UX with UX
function initiate_upper_mul_tri_triview!(UX, U, X)
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    UXdl, UXd, UXdu = UX.dl, UX.d, UX.du
    Udat = U.data

    l,u = bandwidths(U)

    k = 1
    aₖ, cₖ = Xd[1], Xdl[1]
    Uₖₖ, Uₖₖ₊₁, Uₖₖ₊₂ =  Udat[u+1,1], Udat[u,2],  Udat[u-1,3] # U[k,k], U[k,k+1], U[k,k+2]
    UXd[1] = Uₖₖ*aₖ +  Uₖₖ₊₁*cₖ  # UX[k,k] = U[k,k]*X[k,k] + U[k,k+1]*X[k+1,k]
    bₖ, aₖ, cₖ, cₖ₋₁ = Xdu[1], Xd[2], Xdl[2], cₖ  # X[k,k+1], X[k+1,k+1], X[k+2,k+1], X[k+1,k]
    UXdu[1] = Uₖₖ*bₖ + Uₖₖ₊₁*aₖ + Uₖₖ₊₂*cₖ # UX[k,k+1] = U[k,k]*X[k,k+1] + U[k,k+1]*X[k+1,k+1] + U[k,k+1]*X[k+1,k]

    UX, bₖ, aₖ, cₖ, cₖ₋₁
end

# fills in the rows kr of UX
function main_upper_mul_tri_triview!(UX, U, X, kr, bₖ=X.du[kr[1]-1], aₖ=X.d[kr[1]], cₖ=X.dl[kr[1]], cₖ₋₁=X.du[kr[1]-1])
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    UXdl, UXd, UXdu = UX.dl, UX.d, UX.du
    Udat = U.data
    l,u = bandwidths(U)

    for k = kr
        Uₖₖ, Uₖₖ₊₁, Uₖₖ₊₂ =  Udat[u+1,k], Udat[u,k+1],  Udat[u-1,k+2] # U[k,k], U[k,k+1], U[k,k+2]
        UXdl[k-1] = Uₖₖ*cₖ₋₁ # UX[k,k-1] = U[k,k]*X[k,k-1]
        UXd[k] = Uₖₖ*aₖ +  Uₖₖ₊₁*cₖ  # UX[k,k] = U[k,k]*X[k,k] + U[k,k+1]*X[k+1,k]
        bₖ, aₖ, cₖ, cₖ₋₁ = Xdu[k], Xd[k+1], Xdl[k+1], cₖ  # X[k,k+1], X[k+1,k+1], X[k+2,k+1], X[k+1,k]
        UXdu[k] = Uₖₖ*bₖ + Uₖₖ₊₁*aₖ + Uₖₖ₊₂*cₖ # UX[k,k+1] = U[k,k]*X[k,k+1] + U[k,k+1]*X[k+1,k+1] + U[k,k+2]*X[k+2,k+1]
    end

    UX, bₖ, aₖ, cₖ, cₖ₋₁
end

# populate rows k and k+1 of UX, assuming we are at the bottom-right
function finalize_upper_mul_tri_triview!(UX, U, X, k, bₖ, aₖ, cₖ, cₖ₋₁)
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    UXdl, UXd, UXdu = UX.dl, UX.d, UX.du
    Udat = U.data
    l,u = bandwidths(U)

    Uₖₖ, Uₖₖ₊₁ =  Udat[u+1,k], Udat[u,k+1] # U[k,k], U[k,k+1]
    UXdl[k-1] = Uₖₖ*cₖ₋₁ # UX[k,k-1] = U[k,k]*X[k,k-1]
    UXd[k] = Uₖₖ*aₖ +  Uₖₖ₊₁*cₖ  # UX[k,k] = U[k,k]*X[k,k] + U[k,k+1]*X[k+1,k]
    bₖ, aₖ, cₖ₋₁ = Xdu[k], Xd[k+1], cₖ  # X[k,k+1], X[k+1,k+1], X[k+2,k+1], X[k+1,k]
    UXdu[k] = Uₖₖ*bₖ + Uₖₖ₊₁*aₖ # UX[k,k+1] = U[k,k]*X[k,k+1] + U[k,k+1]*X[k+1,k+1] + U[k,k+2]*X[k+2,k+1]

    k += 1
    Uₖₖ =  Udat[u+1,k] # U[k,k]
    UXdl[k-1] = Uₖₖ*cₖ₋₁ # UX[k,k-1] = U[k,k]*X[k,k-1]
    UXd[k] = Uₖₖ*aₖ  # UX[k,k] = U[k,k]*X[k,k] + U[k,k+1]*X[k+1,k]

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

    l,u = bandwidths(R)

    @assert size(R) == (n,n)
    @assert l == 0 && u ≥ 2
    # Tridiagonal bands can be resized
    @assert length(Xdl)+1 == length(Xd) == length(Xdu)+1 == length(Ydl)+1 == length(Yd) == length(Ydu)+1 == n

    UX, Rₖₖ, Rₖₖ₊₁ = initiate_tri_mul_invupper_triview!(Y, X, R)
    UX, Rₖₖ, Rₖₖ₊₁ = main_tri_mul_invupper_triview!(Y, X, R, 2:n-1, Rₖₖ, Rₖₖ₊₁)
    finalize_tri_mul_invupper_triview!(Y, X, R, n, Rₖₖ, Rₖₖ₊₁)
end

# populate first row of X/R
function initiate_tri_mul_invupper_triview!(Y, X, R)
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    Ydl, Yd, Ydu = Y.dl, Y.d, Y.du
    Rdat = R.data

    l,u = bandwidths(R)

    k = 1
    aₖ,bₖ = Xd[k], Xdu[k]
    Rₖₖ,Rₖₖ₊₁ = Rdat[u+1,k], Rdat[u,k+1] # R[1,1], R[1,2]
    Yd[k] = aₖ/Rₖₖ
    Ydu[k] = bₖ - aₖ * Rₖₖ₊₁/Rₖₖ

    Y, Rₖₖ, Rₖₖ₊₁
end


# populate rows kr of X/R
function main_tri_mul_invupper_triview!(Y::Tridiagonal, X::Tridiagonal, R::BandedMatrix, kr, Rₖₖ=R[first(kr),first(kr)], Rₖₖ₊₁=R[first(kr),first(kr)+1])
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    Ydl, Yd, Ydu = Y.dl, Y.d, Y.du
    Rdat = R.data
    l,u = bandwidths(R)

    @inbounds for k = kr
        cₖ₋₁,aₖ,bₖ = Xdl[k-1], Xd[k], Xdu[k]
        Ydl[k-1] = cₖ₋₁/Rₖₖ
        Yd[k] = aₖ-cₖ₋₁*Rₖₖ₊₁/Rₖₖ
        Ydu[k] = cₖ₋₁/Rₖₖ
        Rₖₖ,Rₖₖ₊₁,Rₖ₋₁ₖ₊₁,Rₖ₋₁ₖ = Rdat[u+1,k], Rdat[u,k+1],Rdat[u-1,k+1],Rₖₖ₊₁ # R[k,k], R[k,k+1], R[k-1,k]
        Yd[k] /= Rₖₖ
        Ydu[k-1] /= Rₖₖ
        Ydu[k] *= Rₖ₋₁ₖ*Rₖₖ₊₁/Rₖₖ - Rₖ₋₁ₖ₊₁
        Ydu[k] += bₖ - aₖ * Rₖₖ₊₁ / Rₖₖ
    end
    Y, Rₖₖ, Rₖₖ₊₁
end


# populate row k of X/R, assuming we are at the bottom-right
function finalize_tri_mul_invupper_triview!(Y::Tridiagonal, X::Tridiagonal, R::BandedMatrix, k, Rₖₖ=R[k-1,k-1], Rₖₖ₊₁=R[k-1,k])
    Xdl, Xd, Xdu = X.dl, X.d, X.du
    Ydl, Yd, Ydu = Y.dl, Y.d, Y.du
    Rdat = R.data  
    l,u = bandwidths(R)
    cₖ₋₁,aₖ = Xdl[k-1], Xd[k]
    Ydl[k-1] = cₖ₋₁/Rₖₖ
    Yd[k] = aₖ-cₖ₋₁*Rₖₖ₊₁/Rₖₖ
    Rₖₖ = Rdat[u+1,k] # R[k,k]
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

function TridiagonalConjugationData(U, X, V)
    T = promote_type(typeof(inv(V[1, 1])), eltype(U), eltype(X)) # include inv so that we can't get Ints
    n_init = 100
    UX = Tridiagonal(Vector{T}(undef, n_init-1), Vector{T}(undef, n_init), Vector{T}(undef, n_init-1))
    Y = Tridiagonal(Vector{T}(undef, n_init-1), Vector{T}(undef, n_init), Vector{T}(undef, n_init-1))
    initiate_upper_mul_tri_triview!(UX, U, X) # fill-in 1st row
    return TridiagonalConjugationData(U, X, V, UX, Y, 1)
end

TridiagonalConjugationData(U, X) = TridiagonalConjugationData(U, X, U)

copy(data::TridiagonalConjugationData) = TridiagonalConjugationData(copy(data.U), copy(data.X), copy(data.V), copy(data.UX), copy(data.Y), data.datasize)


function resizedata!(data::TridiagonalConjugationData, n)
    n ≤ data.datasize && return data

    if n > length(data.UX.d) # Avoid O(n²) growing. Note min(length(dv), length(ev)) == length(ev)
        resize!(data.UX.dl, 2n)
        resize!(data.UX.d, 2n + 1)
        resize!(data.UX.du, 2n)

        resize!(data.Y.dl, 2n)
        resize!(data.Y.d, 2n + 1)
        resize!(data.Y.du, 2n)
    end

    main_upper_mul_tri_triview!(data.UX, data.U, data.X, data.datasize+1:n)

    data.datasize = n
end

