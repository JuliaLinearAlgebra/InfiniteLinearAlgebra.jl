const MAX_TRIDIAG_CHOL_N = 2^21 - 1 # Maximum allowable size of Cholesky factor before terminating to prevent OutOfMemory errors without convergence
mutable struct LazySymTridiagonalReverseCholeskyFactors{T,M1,M2} <: LazyMatrix{T}
    const A::M1 # original matrix
    const L::M2 # LN
    const ε::T  # adaptive tolerance
    N::Int      # size of L used for approximating Ln     
    n::Int      # size of approximated finite section
end # this should behave like a lower Bidiagonal matrix
function LazySymTridiagonalReverseCholeskyFactors(A, N, n, L, ε)
    require_one_based_indexing(A)
    M1, M2 = typeof(A), typeof(L)
    T = eltype(L)
    return LazySymTridiagonalReverseCholeskyFactors{T,M1,M2}(A, L, convert(T, ε), N, n)
end

function getproperty(C::ReverseCholesky{<:Any,<:LazySymTridiagonalReverseCholeskyFactors}, d::Symbol) # mimic getproperty(::ReverseCholesky{<:Any, <:Bidiagonal}, ::Symbol)
    Cfactors = getfield(C, :factors)
    #Cuplo    = 'L' 
    if d == :U
        return Cfactors'
    elseif d == :L || d == :UL
        return Cfactors
    else
        return getfield(C, d)
    end
end
MemoryLayout(::Type{<:LazySymTridiagonalReverseCholeskyFactors}) = BidiagonalLayout{LazyLayout,LazyLayout}()

size(L::LazySymTridiagonalReverseCholeskyFactors) = size(L.A)
axes(L::LazySymTridiagonalReverseCholeskyFactors) = axes(L.A)
Base.eltype(L::Type{LazySymTridiagonalReverseCholeskyFactors}) = eltype(L.L)

copy(L::LazySymTridiagonalReverseCholeskyFactors) = LazySymTridiagonalReverseCholeskyFactors(copy(L.A), copy(L.L), L.ε, L.N, L.n)
copy(U::Adjoint{T,<:LazySymTridiagonalReverseCholeskyFactors}) where {T} = copy(parent(U))'

LazyBandedMatrices.bidiagonaluplo(L::LazySymTridiagonalReverseCholeskyFactors) = 'L'

"""
    InfiniteBoundsAccessError <: Exception 

Struct for defining an error when accessing a `LazySymTridiagonalReverseCholeskyFactors` object outside of the 
maximum allowable finite section of size `$MAX_TRIDIAG_CHOL_N × $MAX_TRIDIAG_CHOL_N`.
"""
struct InfiniteBoundsAccessError <: Exception
    i::Int
    j::Int
end
function Base.showerror(io::IO, err::InfiniteBoundsAccessError)
    print(io, "InfiniteBoundsAccessError: Tried to index reverse Cholesky factory at index (", err.i, ", ", err.j)
    print(io, "), outside of the maximum allowable finite section of size (", MAX_TRIDIAG_CHOL_N, " × ", MAX_TRIDIAG_CHOL_N, ")")
end

function getindex(L::LazySymTridiagonalReverseCholeskyFactors, i::Int, j::Int)
    max(i, j) > MAX_TRIDIAG_CHOL_N && throw(InfiniteBoundsAccessError(i, j))
    T = eltype(L)
    if j > i
        return zero(T)
    elseif max(i, j) > L.n
        _expand_factor!(L, max(i, j))
        return L.L[i, j]
    else
        return L.L[i, j]
    end
end

function reversecholesky_layout(::SymTridiagonalLayout, ::NTuple{2,OneToInf{Int}}, A, ::NoPivot; kwds...)
    a, b = A.dv, A.ev
    T = promote_type(eltype(a), eltype(b), eltype(b[1] / a[1])) # could also use promote_op(/, eltype(a), eltype(b)), but promote_op is fragile apparently 
    tol = eps(real(T)) # no good way to pass this as a keyword currently, so just hardcode it
    L = Bidiagonal([zero(T)], T[], :L)
    chol = LazySymTridiagonalReverseCholeskyFactors(A, 1, 1, L, tol)
    _expand_factor!(chol, 2^4) # initialise with 2^4 
    return ReverseCholesky(chol, 'L', 0)
end

function _expand_factor!(L::LazySymTridiagonalReverseCholeskyFactors, n)
    L.n ≥ n && return L
    return __expand_factor!(L::LazySymTridiagonalReverseCholeskyFactors, n)
end

function compute_ξ(LL::LazySymTridiagonalReverseCholeskyFactors)
    #=
    We can show that ||LN' Pn inv(LN') Vb|| = |bN| ||ξ||, where 
        ξₙ = LN[n, n]νₙ,
        ξᵢ = LN[i, i]νᵢ + LN[i+1, i]νᵢ₊₁, i = 1, 2, …, n-1, where 
        νN = 1/LN[N, N] 
        νᵢ = -(LN[i+1, i] / LN[i, i]) νᵢ₊₁, i = 1, 2, …, N-1.
    =#
    L, N, n = LL.L, LL.N, LL.n
    ν = inv(L[N, N])
    for i in (N-1):-1:n
        ν *= -(L[i+1, i] / L[i, i])
    end
    ξ = (L[n, n] * ν)^2
    for i in (n-1):-1:1
        ξ′ = L[i+1, i] * ν
        ν *= -(L[i+1, i] / L[i, i])
        ξ′ += L[i, i] * ν
        ξ += ξ′^2
    end
    bN = LL.A[N, N+1]
    scale = iszero(bN) ? one(bN) : abs(bN)
    return scale * sqrt(ξ) # could maybe just return sqrt(ξ), but maybe bN helps for scaling?
end

function has_converged(LL::LazySymTridiagonalReverseCholeskyFactors)
    ξ = compute_ξ(LL)
    return ξ ≤ LL.ε
end

function _resize_factor!(L::LazySymTridiagonalReverseCholeskyFactors, N=2L.N)
    L.N = N
    resize!(L.L.dv, L.N)
    resize!(L.L.ev, L.N - 1)
    return L
end

function _finite_revchol!(L::LazySymTridiagonalReverseCholeskyFactors)
    # Computes the reverse Cholesky factorisation of L.A[1:L.N, 1:L.N]
    N = L.N
    a, b = L.A.dv, L.A.ev
    ℓa, ℓb = L.L.dv, L.L.ev
    ℓa[N] = sqrt(a[N])
    for i in (N-1):-1:1
        ℓb[i] = b[i] / ℓa[i+1]
        ℓa[i] = sqrt(a[i] - ℓb[i]^2)
    end
    return L
end

function __expand_factor!(L::LazySymTridiagonalReverseCholeskyFactors, n)
    L.N > MAX_TRIDIAG_CHOL_N && return L
    L.n = n
    L.N < L.n && _resize_factor!(L, 2n)
    while !has_converged(L) && L.N ≤ MAX_TRIDIAG_CHOL_N
        _resize_factor!(L)
        _finite_revchol!(L)
    end
    !has_converged(L) && L.N > MAX_TRIDIAG_CHOL_N && @warn "Reverse Cholesky algorithm failed to converge. Returned results may not be accurate." maxlog = 1
    return L
end