@inline function _to_uplo(char::Symbol)
    if char == :U
        'U'
    elseif char == :L
        'L'
    else
        _throw_uplo()
    end
end
@inline function _to_uplo(char::Char)
    if char ∈ ('L', 'U')
        char
    else
        _throw_uplo()
    end
end
@noinline _throw_uplo() = throw(ArgumentError("uplo argument must be either :U (upper) or :L (lower)"))

mutable struct BidiagonalConjugationData{T}
    const U::AbstractMatrix{T} # Typing these concretely prevents the use of Bidiagonal, unless we want LazyBandedMatrices.Bidiagonal
    const C::AbstractMatrix{T} # Function barriers help to minimise the penalty from this when resizing anyway.
    const dv::Vector{T}
    const ev::Vector{T}
    const uplo::Char
    datasize::Int # Number of columns
end
function BidiagonalConjugationData(U, X, V, uplo::Char)
    C = X * V
    T = promote_type(typeof(inv(U[1, 1])), eltype(U), eltype(C)) # include inv so that we can't get Ints
    dv, ev = T[], T[]
    return BidiagonalConjugationData(U, C, dv, ev, uplo, 0)
end

function copy(data::BidiagonalConjugationData)
    U, C, dv, ev, uplo, datasize = data.U, data.C, data.dv, data.ev, data.uplo, data.datasize
    return BidiagonalConjugationData(copy(U), copy(C), copy(dv), copy(ev), uplo, datasize)
end

function _compute_column_up!(data::BidiagonalConjugationData, i)
    U, C = data.U, data.C
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
    return data
end

function _compute_column_lo!(data::BidiagonalConjugationData, i)
    U, C = data.U, data.C
    dv, ev = data.dv, data.ev
    uᵢᵢ, uᵢ₊₁ᵢ, uᵢᵢ₊₁, uᵢ₊₁ᵢ₊₁ = U[i, i], U[i+1, i], U[i, i+1], U[i+1, i+1]
    cᵢᵢ, cᵢ₊₁ᵢ = C[i, i], C[i+1, i]
    Uᵢ⁻¹ = inv(uᵢᵢ * uᵢ₊₁ᵢ₊₁ - uᵢᵢ₊₁ * uᵢ₊₁ᵢ)
    dv[i] = Uᵢ⁻¹ * (uᵢ₊₁ᵢ₊₁ * cᵢᵢ - uᵢᵢ₊₁ * cᵢ₊₁ᵢ)
    ev[i] = Uᵢ⁻¹ * (uᵢᵢ * cᵢ₊₁ᵢ - uᵢ₊₁ᵢ * cᵢᵢ)
    return data
end

function _compute_columns!(data::BidiagonalConjugationData, i)
    ds = data.datasize
    up = data.uplo == 'U'
    for j in (ds+1):i
        up ? _compute_column_up!(data, j) : _compute_column_lo!(data, j)
    end
    data.datasize = i
    return data
end

function resizedata!(data::BidiagonalConjugationData, n)
    ds = data.datasize
    n ≤ ds && return data
    dv, ev = data.dv, data.ev
    resize!(dv, 2n + 1)
    resize!(ev, 2n)
    return _compute_columns!(data, 2n)
end

struct BidiagonalConjugationBand{T} <: AbstractCachedVector{T}
    data::BidiagonalConjugationData{T}
    diag::Bool # true => diagonal, false => offdiagonal 
end
@inline size(A::BidiagonalConjugationBand) = (ℵ₀,)
@inline resizedata!(A::BidiagonalConjugationBand, n) = resizedata!(A.data, n)

function _bcb_getindex(band::BidiagonalConjugationBand, I)
    resizedata!(band, maximum(I) + 1)
    if band.diag 
        return band.data.dv[I]
    else 
        return band.data.ev[I]
    end
end

@inline cache_getindex(band::BidiagonalConjugationBand, I::Integer) = _bcb_getindex(band, I) # needed for show
@inline getindex(band::BidiagonalConjugationBand, I::Integer) = cache_getindex(band, I) # also needed for show because of isassigned
@inline getindex(band::BidiagonalConjugationBand, I::AbstractVector) = _bcb_getindex(band, I)
#@inline getindex(band::BidiagonalConjugationBand, I::AbstractInfUnitRange{<:Integer}) = view(band, I)
#@inline getindex(band::SubArray{<:Any, 1, <:BidiagonalConjugationBand}, I::AbstractInfUnitRange{<:Integer}) = view(band, I)

copy(band::BidiagonalConjugationBand) = band

function convert(::Type{BidiagonalConjugationBand{T}}, ::Zeros{V, 1, Tuple{OneToInf{Int}}}) where {T, V}
    # without this method, we can't do e.g. B[Band(-1)] for upper bidiagonal
    
end

const BidiagonalConjugation{T} = Bidiagonal{T, BidiagonalConjugationBand{T}}

"""
    BidiagonalConjugation(U, X, V, uplo)

Efficiently compute the projection of the matrix product
`inv(U)XV` onto a bidiagonal matrix. The `uplo` argument 
specifies whether the projection is upper (`uplo = 'U'`)
or lower (`uplo = 'L'`) bidiagonal. 

The computation is returned as a `Bidiagonal` matrix whose 
diagonal and off-diagonal vectors are computed lazily.
"""
function BidiagonalConjugation(U, X, V, uplo)
    _uplo = _to_uplo(uplo)
    data = BidiagonalConjugationData(U, X, V, _uplo)
    return _BidiagonalConjugation(data, _uplo)
end

function _BidiagonalConjugation(data, uplo) # need uplo argument so that we can take transposes
    dv = BidiagonalConjugationBand(data, true)
    ev = BidiagonalConjugationBand(data, false)
    return Bidiagonal(dv, ev, uplo)
end

function copy(A::BidiagonalConjugation) # need a new method so that the data remains aliased between dv and ev 
    data = copy(A.dv.data)
    return _BidiagonalConjugation(data, A.uplo)
end

LazyBandedMatrices.Bidiagonal(A::BidiagonalConjugation) = LazyBandedMatrices.Bidiagonal(A.dv, A.ev, A.uplo)