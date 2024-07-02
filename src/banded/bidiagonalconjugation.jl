"""
    BidiagonalConjugation{T, MU, MC} <: AbstractCachedMatrix{T}

Struct for efficiently projecting the matrix product `inv(U)XV` onto a 
bidiagonal matrix.

# Fields 
- `U`: The matrix `U`.
- `C`: The matrix product `C = XV`.
- `data::Bidiagonal{T,Vector{T}}`: The cached data of the projection. 
- `datasize::Int`: The size of the finite section computed thus far.
- `dv::BandWrapper`: The diagonal part.
- `ev::BandWrapper`: The offdiagonal part.

# Constructor
To construct the projection, use the constructor 

    BidiagonalConjugation(U, X, V, uplo)

where `uplo` specifies whether the projection is upper (`U`) or lower (`L`).
"""
mutable struct BidiagonalConjugation{T,MU,MC} <: AbstractCachedMatrix{T}
    const U::MU
    const C::MC
    const data::Bidiagonal{T,Vector{T}}
    datasize::Int # also compute up to a square finite section 
    function BidiagonalConjugation(U::MU, C::MC, data::Bidiagonal{T}, datasize::Int) where {T,MU,MC}
        return new{T,MU,MC}(U, C, data, datasize)
    end # disambiguate with the next constructor
end
function BidiagonalConjugation(U::MU, X, V, uplo) where {MU}
    C = X * V
    T = promote_type(typeof(inv(U[1, 1])), eltype(U), eltype(C)) # include inv so that we can't get Ints
    data = Bidiagonal(T[], T[], uplo)
    return BidiagonalConjugation(U, C, data, 0)
end
get_uplo(A::BidiagonalConjugation) = A.data.uplo

MemoryLayout(::Type{<:BidiagonalConjugation}) = BidiagonalLayout{LazyLayout,LazyLayout}()
diagonaldata(A::BidiagonalConjugation) = A.dv
supdiagonaldata(A::BidiagonalConjugation) = get_uplo(A) == 'U' ? A.ev : throw(ArgumentError(LazyString(A, " is lower-bidiagonal")))
subdiagonaldata(A::BidiagonalConjugation) = get_uplo(A) == 'L' ? A.ev : throw(ArgumentError(LazyString(A, " is upper-bidiagonal")))

bandwidths(A::BidiagonalConjugation) = bandwidths(A.data)
size(A::BidiagonalConjugation) = (ℵ₀, ℵ₀)
axes(A::BidiagonalConjugation) = (OneToInf(), OneToInf())

copy(A::BidiagonalConjugation) = BidiagonalConjugation(copy(A.U), copy(A.C), copy(A.data), A.datasize)
copy(A::Adjoint{T,<:BidiagonalConjugation}) where {T} = copy(parent(A))'

LazyBandedMatrices.bidiagonaluplo(A::BidiagonalConjugation) = get_uplo(A)
LazyBandedMatrices.Bidiagonal(A::BidiagonalConjugation) = LazyBandedMatrices.Bidiagonal(A.dv, A.ev, get_uplo(A))

# We could decouple this from parent since the computation of each vector is 
# independent of the other, but it would be slower since they get computed 
# in tandem. Thus, we instead use a wrapper that captures both. 
# This is needed so that A.dv and A.ev can be defined (and are useful, instead of 
# just returning A.data.dv and A.data.ev which are finite and have no knowledge 
# of the parent).
struct BandWrapper{T,P<:BidiagonalConjugation{T}} <: LazyVector{T}
    parent::P
    diag::Bool # true => main diagonal, false => off diagonal 
end
size(wrap::BandWrapper) = (size(wrap.parent, 1),)
function getindex(wrap::BandWrapper, i::Int)
    parent = wrap.parent
    uplo = get_uplo(parent)
    if wrap.diag
        return parent[i, i]
    elseif uplo == 'U'
        return parent[i, i+1]
    else # uplo == 'L' 
        return parent[i+1, i]
    end
end

function getproperty(A::BidiagonalConjugation, d::Symbol)
    if d == :dv
        return BandWrapper(A, true)
    elseif d == :ev
        return BandWrapper(A, false)
    else
        return getfield(A, d)
    end
end

function _compute_column_up!(A::BidiagonalConjugation, i)
    U, C = A.U, A.C
    data = A.data
    dv, ev = data.dv, data.ev
    if i == 1
        dv[i] = C[1, 1] / U[1, 1]
    else
        uᵢ₋₁ᵢ₋₁, uᵢᵢ₋₁, uᵢ₋₁ᵢ, uᵢᵢ = U[i-1, i-1], U[i, i-1], U[i-1, i], U[i, i]
        cᵢ₋₁ᵢ, cᵢᵢ = C[i-1, i], C[i, i]
        Uᵢ⁻¹ = inv(uᵢ₋₁ᵢ₋₁ * uᵢᵢ - uᵢ₋₁ᵢ * uᵢᵢ₋₁)
        dv[i] = Uᵢ⁻¹ * (uᵢ₋₁ᵢ₋₁ * cᵢᵢ - uᵢᵢ₋₁ * cᵢ₋₁ᵢ)
        ev[i-1] = Uᵢ⁻¹ * (uᵢᵢ * cᵢ₋₁ᵢ - uᵢ₋₁ᵢ * cᵢᵢ)
    end
    return A
end

function _compute_column_lo!(A::BidiagonalConjugation, i)
    U, C = A.U, A.C
    data = A.data
    dv, ev = data.dv, data.ev
    uᵢᵢ, uᵢ₊₁ᵢ, uᵢᵢ₊₁, uᵢ₊₁ᵢ₊₁ = U[i, i], U[i+1, i], U[i, i+1], U[i+1, i+1]
    cᵢᵢ, cᵢ₊₁ᵢ = C[i, i], C[i+1, i]
    Uᵢ⁻¹ = inv(uᵢᵢ * uᵢ₊₁ᵢ₊₁ - uᵢᵢ₊₁ * uᵢ₊₁ᵢ)
    dv[i] = Uᵢ⁻¹ * (uᵢ₊₁ᵢ₊₁ * cᵢᵢ - uᵢᵢ₊₁ * cᵢ₊₁ᵢ)
    ev[i] = Uᵢ⁻¹ * (uᵢᵢ * cᵢ₊₁ᵢ - uᵢ₊₁ᵢ * cᵢᵢ)
    return A
end

function _compute_columns!(A::BidiagonalConjugation, i)
    ds = A.datasize
    up = get_uplo(A) == 'U'
    for j in (ds+1):i
        up ? _compute_column_up!(A, j) : _compute_column_lo!(A, j)
    end
    A.datasize = i
    return A
end

function resizedata!(A::BidiagonalConjugation, i::Int, j::Int)
    ds = A.datasize
    j ≤ ds && return A
    data = A.data
    dv, ev = data.dv, data.ev
    resize!(dv, 2j + 1)
    resize!(ev, 2j)
    return _compute_columns!(A, 2j)
end