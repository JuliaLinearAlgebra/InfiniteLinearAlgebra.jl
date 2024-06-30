"""
    BidiagonalConjugationData{T, MU, MC} <: LazyMatrix{T}

Struct for efficiently representing the matrix product `A = inv(U)XV`,
assuming that 

- `A` is upper bidiagonal, 
- `U` is upper Hessenberg,
- `X` is banded,
- `V` is banded.

None of these properties are checked internally. It is the user's responsibility
to ensure these properties hold and that the product `inv(U)XV` is indeed bidiagonal.

# Fields 
- `U`: The upper Hessenberg matrix. 
- `C`: The matrix product `XV`.
- `dv`: A vector giving the diagonal of `A`. 
- `ev`: A vector giving the superdiagonal of `A`.

The vectors `dv` and `ev` grow on demand from `getindex` and should not be 
used directly. Simply treat 

    A = BidiagonalConjugationData(U, X, V)

as you would an upper bidiagonal matrix.
"""
struct BidiagonalConjugationData{T,MU,MC} <: LazyMatrix{T}
    U::MU
    C::MC
    dv::Vector{T}
    ev::Vector{T}
end
function BidiagonalConjugationData(U::MU, X::MX, V::MV) where {MU,MX,MV}
    C = X * V
    T = promote_type(typeof(inv(U[begin])), eltype(U), eltype(C)) # include inv so that we can't get Ints
    dv, ev = T[], T[]
    return BidiagonalConjugationData{T,MU,typeof(C)}(U, C, dv, ev)
end
MemoryLayout(::Type{<:BidiagonalConjugationData}) = BidiagonalLayout{LazyLayout,LazyLayout}()
bandwidths(A::BidiagonalConjugationData) = (0, 1)
size(A::BidiagonalConjugationData) = (ℵ₀, ℵ₀)
axes(A::BidiagonalConjugationData) = (OneToInf(), OneToInf())
Base.eltype(A::Type{<:BidiagonalConjugationData{T}}) where {T} = T

copy(A::BidiagonalConjugationData) = BidiagonalConjugationData(copy(A.U), copy(A.C), copy(A.dv), copy(A.ev))
copy(A::Adjoint{T,<:BidiagonalConjugationData}) where {T} = copy(parent(A))'

LazyBandedMatrices.bidiagonaluplo(A::BidiagonalConjugationData) = 'U'
LazyBandedMatrices.Bidiagonal(A::BidiagonalConjugationData) = LazyBandedMatrices.Bidiagonal(A[band(0)], A[band(1)], 'U')

_colsize(A::BidiagonalConjugationData) = length(A.dv)

function _compute_column!(A::BidiagonalConjugationData, i)
    # computes A[i, i] and A[i-1, i]
    i ≤ _colsize(A) && return A
    dv, ev = A.dv, A.ev
    U, C = A.U, A.C
    resize!(dv, i)
    resize!(ev, i - 1)
    if i == 1
        dv[i] = C[1, 1] / U[1, 1]
    else
        uᵢ₋₁ᵢ₋₁, uᵢ₋₁ᵢ, uᵢᵢ₋₁, uᵢᵢ = U[i-1, i-1], U[i-1, i], U[i, i-1], U[i, i]
        cᵢ₋₁ᵢ, cᵢᵢ = C[i-1, i], C[i, i]
        Uᵢ⁻¹ = inv(uᵢ₋₁ᵢ₋₁ * uᵢᵢ - uᵢ₋₁ᵢ * uᵢᵢ₋₁)
        dv[i] = Uᵢ⁻¹ * (uᵢ₋₁ᵢ₋₁ * cᵢᵢ - uᵢᵢ₋₁ * cᵢ₋₁ᵢ)
        ev[i-1] = Uᵢ⁻¹ * (uᵢᵢ * cᵢ₋₁ᵢ - uᵢ₋₁ᵢ * cᵢᵢ)
    end
    return A
end

function getindex(A::BidiagonalConjugationData, i::Int, j::Int)
    i ≤ 0 || j ≤ 0 && throw(BoundsError(A, (i, j)))
    T = eltype(A)
    in_band = i == j || i == j - 1
    if !in_band
        return zero(T)
    elseif j > _colsize(A)
        _compute_column!(A, j)
        return i == j ? A.dv[i] : A.ev[i]    
    else 
        return i == j ? A.dv[i] : A.ev[i]
    end
end