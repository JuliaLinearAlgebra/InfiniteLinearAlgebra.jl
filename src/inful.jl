"""
_ultailL1(C, A, B)

gives L[Block(1,1)] of a block-tridiagonal Toeplitz operator. Based on Delvaux and Dette 2012.
"""
function _ultailL1(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    d = size(A,1)
    λs = filter!(λ -> abs(λ) ≤ 1, eigvals([zeros(2,2) -B; -C -A], [B zeros(2,2); zeros(2,2) -B]))
    @assert length(λs) == d
    V = Matrix{eltype(λs)}(undef, d, d)
    for (j,λ) in enumerate(λs)
        V[:,j] = svd(A-λ*B - C/λ).V[:,end] # nullspace(A-λ*B - C/λ)
    end
    C*(V*Diagonal(inv.(λs))/V)
end

function _ul(::BlockTriToeplitzLayout, J)
    C = getindex_value(subdiagonaldata(blocks(J)))
    A = getindex_value(diagonaldata(blocks(J)))
    B = getindex_value(supdiagonaldata(blocks(J)))
    L = _ultailL1(C, A, B)
    U = B/L
    II = convert(typeof(U), I(size(A,1)))
    mortar(Bidiagonal(Fill(II,∞), Fill(U,∞), :U)),mortar(Bidiagonal(Fill(L,∞), Fill(convert(typeof(L),C),∞), :L))
end

ul(A::AbstractMatrix) = _ul(MemoryLayout(A), A)
