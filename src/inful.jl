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
# = L = (A ± sqrt(A^2 - 4B*C))/2
# we choose sign(A) to maximise the magnitude as we know inv(T)[1,1] -> 0, hence
# inv(L) -> 0
_ultailL1(c::Number, a::Number, b::Number) = (a + sign(a)*sqrt(a^2-4b*c))/2

# In the matrix case we write inv(L)*C = R (double check) to make it quadratic
# A - B*R == inv(C)inv(R)
# C*A*R - C*B*R^2 == I
# and do eigen decomposition of R to reduce this to a companion matrix problem.
function _ultailL1(C::AbstractMatrix, A::AbstractMatrix, B::AbstractMatrix)
    d = size(A,1)    
    F = eigen([zeros(d,d) B; C A], [-B zeros(d,d); zeros(d,d) B])
    inds = findall(λ -> abs(λ) ≤ 1, F.values)
    @assert length(inds) == d
    λ, V = F.values[inds], F.vectors[d+1:2d,inds]
    C*(V*Diagonal(inv.(λ))/V) # L = C inv(R)
end

function ul_layout(::TridiagonalToeplitzLayout, J::AbstractMatrix, ::Val{false}; check::Bool = true)
    C = getindex_value(subdiagonaldata(J))
    A = getindex_value(diagonaldata(J))
    B = getindex_value(supdiagonaldata(J))
    L = _ultailL1(C, A, B)
    U = B/L
    UL(Tridiagonal(Fill(convert(typeof(L),C),∞), Fill(L,∞), Fill(U,∞)), OneToInf(), 0)
end

function ul_layout(::TridiagonalToeplitzLayout, J::AbstractMatrix, ::Val{true}; check::Bool = true)
    C = getindex_value(subdiagonaldata(J))
    A = getindex_value(diagonaldata(J))
    B = getindex_value(supdiagonaldata(J))
    A^2 ≥ 4B*C || error("Pivotting not implemented")
    ul(J, Val(false))
end

function ul_layout(::BlockTridiagonalToeplitzLayout, J::AbstractMatrix, ::Val{false}; check::Bool = true)
    C = getindex_value(subdiagonaldata(blocks(J)))
    A = getindex_value(diagonaldata(blocks(J)))
    B = getindex_value(supdiagonaldata(blocks(J)))
    # Factor the dense tail block and absorb its upper factor into the adjacent blocks.
    F = ul!(_ultailL1(C, A, B), Val(false); check=check)
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
    F∞ = ul(mortar(Tridiagonal(Fill(c∞,∞), Fill(a∞,∞), Fill(b∞,∞))), Val(false); check=check)
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
        F = ul!(A[k] - du[k]*dl[k], Val(false); check=check)
        d[k] = nextD = F.factors
        if iszero(info) && !iszero(F.info)
            info = sum(j -> size(A[j],1), 1:k-1; init=0) + F.info
        end
    end
    UL(mortar(Tridiagonal(Vcat(dl, Fill(C∞,∞)), Vcat(d, Fill(A∞,∞)), Vcat(du, Fill(B∞,∞)))), OneToInf(), info)
end

ul_layout(::BlockLayout{<:TridiagonalLayout}, ::BlockTriPertToeplitz, ::Val{true}; check::Bool = true) =
    error("Pivoting not implemented; use ul(J, Val(false))")


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


getU(F::UL, ::NTuple{2,InfiniteCardinal{0}}) = _inf_getU(MemoryLayout(F.factors), F)
getL(F::UL, ::NTuple{2,InfiniteCardinal{0}}) = _inf_getL(MemoryLayout(F.factors), F)

getU(F::UL{T,<:Tridiagonal}, ::NTuple{2,InfiniteCardinal{0}}) where T = _inf_getU(MemoryLayout(F.factors), F)
getL(F::UL{T,<:Tridiagonal}, ::NTuple{2,InfiniteCardinal{0}}) where T = _inf_getL(MemoryLayout(F.factors), F)
