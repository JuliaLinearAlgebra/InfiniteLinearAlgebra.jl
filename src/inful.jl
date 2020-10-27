"""
_ultailL1(C, A, B)

gives L[Block(1,1)] of a block-tridiagonal Toeplitz operator. Based on Delvaux and Dette 2012.
"""

# need to solve [1 -B*inv(L); 0 1] * [C A B; 0 C L] == [C L 0; 0 C L]
# that is 
# A - B*inv(L)*C == L
# In the scalar case this is quadratic
# L^2 - A*L + B*C == 0
# so that 
# = L = (A ± sqrt(A^2 - 4B*C))/2
# we choose sign(A) to maximise the magnitude as we know inv(T)[1,1] -> 0, hence
# inv(L) -> 0
_ultailL1(c::Number, a::Number, b::Number) = (a + sign(a)*sqrt(a^2-4b*c))/2

# In the matrix case we write inv(L)*C = R (double check) to make it quadratic
# A - B*R == inv(C)inv(R)
# C*A*R - C*B*R^2 == I
# and do eigen decomposition of R to reduce this to a companion matrix problem.

function _ultailL1(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    d = size(A,1)
    λs = filter!(λ -> abs(λ) ≤ 1, eigvals([zeros(d,d) -B; -C -A], [B zeros(d,d); zeros(d,d) -B]))
    @assert length(λs) == d
    V = Matrix{eltype(λs)}(undef, d, d)
    j = 1
    for λ in union(λs)
        c = count(==(λ), λs) # multiplicities
        V[:,j:j+c-1] = svd(A-λ*B - C/λ).V[:,end-c+1:end] # nullspace(A-λ*B - C/λ)
        j += c
    end
    C*(V*Diagonal(inv.(λs))/V)
end

function _ul(::TridiagonalToeplitzLayout, J::AbstractMatrix, ::Val{false}; check::Bool = true)
    C = getindex_value(subdiagonaldata(J))
    A = getindex_value(diagonaldata(J))
    B = getindex_value(supdiagonaldata(J))
    L = _ultailL1(C, A, B)
    U = B/L
    UL(Tridiagonal(Fill(convert(typeof(L),C),∞), Fill(L,∞), Fill(U,∞)), OneToInf(), 0)
end

function _ul(::TridiagonalToeplitzLayout, J::AbstractMatrix, ::Val{true}; check::Bool = true)
    C = getindex_value(subdiagonaldata(J))
    A = getindex_value(diagonaldata(J))
    B = getindex_value(supdiagonaldata(J))
    A^2 ≥ 4B*C || error("Pivotting not implemented")
    ul(J, Val(false))
end

function _ul(::BlockTridiagonalToeplitzLayout, J::AbstractMatrix, ::Val{false}; check::Bool = true)
    C = getindex_value(subdiagonaldata(blocks(J)))
    A = getindex_value(diagonaldata(blocks(J)))
    B = getindex_value(supdiagonaldata(blocks(J)))
    L = _ultailL1(C, A, B)
    U = B/L
    UL(mortar(Tridiagonal(Fill(convert(typeof(L),C),∞), Fill(L,∞), Fill(U,∞))), OneToInf(), 0)
end


_inf_getU(::TridiagonalToeplitzLayout, F::UL) = Bidiagonal(Fill(one(eltype(F)),∞),F.factors.du, :U)
_inf_getL(::TridiagonalToeplitzLayout, F::UL) = Bidiagonal(F.factors.d,F.factors.dl, :L)


function _inf_getU(::BlockTridiagonalToeplitzLayout, F::UL)
    d = size(F.factors.blocks.du[1],1)
    II = convert(eltype(F.factors.blocks.du), I(d))
    mortar(Bidiagonal(Fill(II,∞),F.factors.blocks.du, :U))
end

_inf_getL(::BlockTridiagonalToeplitzLayout, F::UL) = mortar(Bidiagonal(F.factors.blocks.d,F.factors.blocks.dl, :L))


getU(F::UL, ::NTuple{2,Infinity}) = _inf_getU(MemoryLayout(F.factors), F)
getL(F::UL, ::NTuple{2,Infinity}) = _inf_getL(MemoryLayout(F.factors), F)

getU(F::UL{T,<:Tridiagonal}, ::NTuple{2,Infinity}) where T = _inf_getU(MemoryLayout(F.factors), F)
getL(F::UL{T,<:Tridiagonal}, ::NTuple{2,Infinity}) where T = _inf_getL(MemoryLayout(F.factors), F)