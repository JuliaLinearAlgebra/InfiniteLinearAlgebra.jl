"""
    _ultailL1(C, A, B)

Return the stationary UL tail pivot for a block-tridiagonal Toeplitz operator
with subdiagonal C, diagonal A, and superdiagonal B.
"""

# Seek a stationary UL factorisation of the semi-infinite block Toeplitz matrix:
#
#       [ A  B  0  ... ]   [ I  U  0  ... ] [ L  0  0  ... ]
#   J = [ C  A  B  ... ] = [ 0  I  U  ... ] [ C  L  0  ... ]
#       [ 0  C  A  ... ]   [ 0  0  I  ... ] [ 0  C  L  ... ]
#       [ :  :  :  ... ]   [ :  :  :  ... ] [ :  :  :  ... ]
#
# The product gives A = L + U*C and B = U*L, hence L = A - B*inv(L)*C
# for nonsingular L. Equivalently, local elimination must reproduce L:
#
#   [I -B*inv(L); 0 I] * [C A B; 0 C L] = [C L 0; 0 C L].
#
# Scalars give L^2 - a*L + b*c = 0. For real nonzero a and nonnegative
# discriminant, the sign below selects the root of larger magnitude.
_ultailL1(c::Number, a::Number, b::Number) = (a + sign(a)*sqrt(a^2-4b*c))/2

# Set R = inv(L)*C. Then L = A - B*R and L*R = C give B*R^2 - A*R + C = 0.
# Apply Theorem 3 of Higham--Kim (IMAJNA, 2000) with coefficients (B, -A, C):
# https://doi.org/10.1093/imanum/20.4.499.
function _ultailL1(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    d = size(A,1)
    # Linearise the quadratic as F*[I; R] = G*[I; R]*R.
    F = [zeros(d,d) I; -C A]; G = [I zeros(d,d); zeros(d,d) B]
    # Julia's convention is F = Q*S*Z' and G = Q*T*Z'.
    schurfact = schur(F, G)

    # Require the infinite lower factor to have a bounded inverse on ℓ².
    # Its formal inverse has (i,j) block (-R)^(i-j)*inv(L), i ≥ j, and is
    # bounded iff every eigenvalue of R has |λ| < 1. The ≤ test also admits
    # unit-circle boundary cases, where bounded invertibility fails.
    # Select exactly d generalized eigenvalues for the d-by-d R and move
    # their Schur blocks first.
    select = abs.(schurfact.values) .≤ 1
    @assert count(select) == d
    ordschur!(schurfact, select)

    # With d-by-d blocks, Theorem 3 gives R = Z21/Z11. From G*Z = Q*T,
    # Z11 = Q11*T11 and B*Z21 = Q21*T11. Finite selected roots make T11
    # invertible, so B*R = Q21/Q11 and L = A - Q21/Q11. This requires
    # nonsingular Q11 (equivalently Z11), but allows singular R.
    A - schurfact.Q[d+1:2d, 1:d] / schurfact.Q[1:d, 1:d]
end

function ul_layout(::TridiagonalToeplitzLayout, J::AbstractMatrix, ::NoPivot; check::Bool = true)
    C = getindex_value(subdiagonaldata(J))
    A = getindex_value(diagonaldata(J))
    B = getindex_value(supdiagonaldata(J))
    L = _ultailL1(C, A, B)
    U = B/L
    UL(Tridiagonal(Fill(convert(typeof(L),C),∞), Fill(L,∞), Fill(U,∞)), OneToInf(), 0)
end

function ul_layout(::TridiagonalToeplitzLayout, J::AbstractMatrix, ::RowMaximum; check::Bool = true)
    C = getindex_value(subdiagonaldata(J))
    A = getindex_value(diagonaldata(J))
    B = getindex_value(supdiagonaldata(J))
    A^2 ≥ 4B*C || error("Pivotting not implemented")
    ul(J, NoPivot())
end

function ul_layout(::BlockTridiagonalToeplitzLayout, J::AbstractMatrix, ::NoPivot; check::Bool = true)
    C = getindex_value(subdiagonaldata(blocks(J)))
    A = getindex_value(diagonaldata(blocks(J)))
    B = getindex_value(supdiagonaldata(blocks(J)))
    # Factor the dense tail block and absorb its upper factor into the adjacent blocks.
    F = ul!(_ultailL1(C, A, B), NoPivot(); check=check)
    U = UnitUpperTriangular(F.factors)
    L = LowerTriangular(F.factors)
    UL(mortar(Tridiagonal(Fill(U \ C,∞), Fill(F.factors,∞), Fill(B/L,∞))), OneToInf(), F.info)
end

function _ul_perturbed(J, pivot; check::Bool = true)
    C, A, B = subdiagonaldata(J), diagonaldata(J), supdiagonaldata(J)
    c, c∞ = _data_tail(C)
    a, a∞ = _data_tail(A)
    b, b∞ = _data_tail(B)
    F∞ = ul(Tridiagonal(Fill(c∞,∞), Fill(a∞,∞), Fill(b∞,∞)), pivot; check=check)
    L∞, U∞ = F∞.factors.d[1], F∞.factors.du[1]
    n = max(length(c), length(a), length(b))
    L = Vector{eltype(F∞)}(undef, n)
    U = similar(L)
    info = F∞.info

    # Complete the finite prefix backwards from the exact tail pivot.
    nextL = L∞
    for k = n:-1:1
        U[k] = B[k]/nextL
        L[k] = A[k] - U[k]*C[k]
        if iszero(L[k])
            check && throw(SingularException(k))
            iszero(info) && (info = k)
        end
        nextL = L[k]
    end
    UL(Tridiagonal(Vcat(Vector{eltype(F∞)}(C[1:n]), Fill(convert(eltype(F∞),c∞),∞)),
                   Vcat(L, Fill(L∞,∞)), Vcat(U, Fill(U∞,∞))), OneToInf(), info)
end

ul_layout(::TridiagonalLayout, J::InfiniteArrays.TriPertToeplitz, pivot::Union{Val{false},Val{true}}; check::Bool = true) =
    _ul_perturbed(J, pivot; check=check)
ul_layout(::PertTridiagonalToeplitzLayout, J::AbstractMatrix, pivot::Union{Val{false},Val{true}}; check::Bool = true) =
    _ul_perturbed(J, pivot; check=check)

function ul_layout(::BlockLayout{<:TridiagonalLayout}, J::BlockTriPertToeplitz, ::Val{false}; check::Bool = true)
    C, A, B = subdiagonaldata(blocks(J)), diagonaldata(blocks(J)), supdiagonaldata(blocks(J))
    c, c∞ = _data_tail(C)
    a, a∞ = _data_tail(A)
    b, b∞ = _data_tail(B)
    F∞ = ul(mortar(Tridiagonal(Fill(c∞,∞), Fill(a∞,∞), Fill(b∞,∞))), NoPivot(); check=check)
    C∞, A∞, B∞ = F∞.factors.blocks.dl[1], F∞.factors.blocks.d[1], F∞.factors.blocks.du[1]
    n = max(length(c), length(a), length(b))
    d = Vector{typeof(A∞)}(undef, n)
    dl, du = similar(d), similar(d)
    info = F∞.info

    # Eliminate each block against the next block's triangular factors.
    nextD = A∞
    for k = n:-1:1
        dl[k] = UnitUpperTriangular(nextD) \ C[k]
        du[k] = B[k]/LowerTriangular(nextD)
        F = ul!(A[k] - du[k]*dl[k], NoPivot(); check=check)
        d[k] = nextD = F.factors
        if iszero(info) && !iszero(F.info)
            info = sum(j -> size(A[j],1), 1:k-1; init=0) + F.info
        end
    end
    UL(mortar(Tridiagonal(Vcat(dl, Fill(C∞,∞)), Vcat(d, Fill(A∞,∞)), Vcat(du, Fill(B∞,∞)))), OneToInf(), info)
end

ul_layout(::BlockLayout{<:TridiagonalLayout}, ::BlockTriPertToeplitz, ::Val{true}; check::Bool = true) =
    error("Pivoting not implemented; use ul(J, NoPivot())")


_inf_getU(::Union{TridiagonalToeplitzLayout,TridiagonalLayout}, F::UL) = Bidiagonal(one.(F.factors.d),F.factors.du, :U)
_inf_getL(::Union{TridiagonalToeplitzLayout,TridiagonalLayout}, F::UL) = Bidiagonal(F.factors.d,F.factors.dl, :L)


function _inf_getU(::Union{BlockTridiagonalToeplitzLayout,BlockLayout{<:TridiagonalLayout}}, F::UL)
    U = Matrix.(UnitUpperTriangular.(F.factors.blocks.d))
    mortar(Bidiagonal(U,F.factors.blocks.du, :U))
end

function _inf_getL(::Union{BlockTridiagonalToeplitzLayout,BlockLayout{<:TridiagonalLayout}}, F::UL)
    L = Matrix.(LowerTriangular.(F.factors.blocks.d))
    mortar(Bidiagonal(L,F.factors.blocks.dl, :L))
end

# The product of infinite upper and lower bidiagonal Toeplitz matrices U*L has no boundary
# effects, hence is (block) tridiagonal Toeplitz.
function _bidiagonaltoeplitz_mul(U::Bidiagonal, L::Bidiagonal)
    Ud, Uu = getindex_value(U.dv), getindex_value(U.ev)
    Ld, Ll = getindex_value(L.dv), getindex_value(L.ev)
    Tridiagonal(Fill(Ud*Ll,∞), Fill(Ud*Ld + Uu*Ll,∞), Fill(Uu*Ld,∞))
end

_isupperlower(U, L) = false
_isupperlower(U::Bidiagonal, L::Bidiagonal) = U.uplo == 'U' && L.uplo == 'L'

function copy(M::Mul{BidiagonalToeplitzLayout,BidiagonalToeplitzLayout})
    _isupperlower(M.A, M.B) || return simplify(M)
    _bidiagonaltoeplitz_mul(M.A, M.B)
end

const BlockBidiagonalToeplitzLayout = BlockLayout{BidiagonalToeplitzLayout,DenseColumnMajor}

function copy(M::Mul{BlockBidiagonalToeplitzLayout,BlockBidiagonalToeplitzLayout})
    U, L = blocks(M.A), blocks(M.B)
    _isupperlower(U, L) || return ApplyArray(*, M.A, M.B)
    mortar(_bidiagonaltoeplitz_mul(U, L))
end


getU(F::UL, ::NTuple{2,InfiniteCardinal{0}}) = _inf_getU(MemoryLayout(F.factors), F)
getL(F::UL, ::NTuple{2,InfiniteCardinal{0}}) = _inf_getL(MemoryLayout(F.factors), F)

getU(F::UL{T,<:Tridiagonal}, ::NTuple{2,InfiniteCardinal{0}}) where T = _inf_getU(MemoryLayout(F.factors), F)
getL(F::UL{T,<:Tridiagonal}, ::NTuple{2,InfiniteCardinal{0}}) where T = _inf_getL(MemoryLayout(F.factors), F)
