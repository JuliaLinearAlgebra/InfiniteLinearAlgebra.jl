function reversecholesky_layout(::TridiagonalToeplitzLayout, ::NTuple{2,OneToInf{Int}}, A, ::NoPivot; kwds...)
    # [a b     = [α β       * [α
    #  b a ⋱        ⋱ ⋱]     β α
    #    ⋱ ⋱]
    #  [a       b'      [a    b'        [1  b'inv(U)'   ]   [a - b'inv(U)'inv(U)*b      [1
    #   b    A   ] =     b U*U']    =       U           ]                           I]   inv(U)b    U']

    # since inv(U) = [inv(α) …] we have inv(U)*(b*e₁) = b/α
    # we also assume Toeplitz structure so that α^2 = a - b'inv(U)'inv(U)*b = a - b^2/α^2
    # Thus we have a Quadratic equation:
    #
    #   α^4 - a*α^2 + b^2 = 0
    #
    # i.e. α^2 = (a ± sqrt(a^2 - 4b^2))/2.
    # We also have αβ = b.

    a = diagonalconstant(A)
    b = supdiagonalconstant(A)
    @assert b == subdiagonalconstant(A)

    α² = (a + sqrt(a^2 - 4b^2))/2
    # (a - sqrt(a^2 - 4b^2))/2 other branch gives non-invertible U
    
    α = sqrt(α²)
    β = b/α

    ReverseCholesky(Bidiagonal(Fill(α,∞), Fill(β,∞), :U), 'U', 0)
end

function reversecholesky_layout(::PertTridiagonalToeplitzLayout, ::NTuple{2,OneToInf{Int}}, A, ::NoPivot; kwds...)
    a = diagonaldata(A)
    aₙ,a∞ = arguments(vcat, a)
    b = supdiagonaldata(A)
    bₙ,b∞ = arguments(vcat, b)
    U∞, = reversecholesky(SymTridiagonal(a∞, b∞))

    n = max(length(aₙ), length(bₙ)+1)
    Aₙ = SymTridiagonal([aₙ; float(a∞[1:(n-length(aₙ))])], [bₙ; float(b∞[1:(n-length(bₙ)-1)])])
    α = U∞[1,1]
    b = getindex_value(b∞)
    Aₙ[end,end] -= b^2/α^2
    Uₙ, = reversecholesky(Aₙ)
    ReverseCholesky(Bidiagonal([Uₙ.dv; U∞.dv], [Uₙ.ev; U∞.ev], :U), 'U', 0)
end

"""
    reversecholesky_layout(::BlockTridiagonalToeplitzLayout, ...)

computes the reverse Cholesky factorisation `A == U*U'` of a symmetric block-tridiagonal
Toeplitz operator, where `U` is block-bidiagonal Toeplitz.
"""
function reversecholesky_layout(::BlockTridiagonalToeplitzLayout, ::NTuple{2,BlockedOneTo{Int,<:InfStepRange}}, A, ::NoPivot; kwds...)
    # Write the block-Toeplitz reverse Cholesky factor as
    #
    #   U = [α β         so that   U*U' = [α*α' + β*β'  β*α'
    #          α β  ⋱                      α*β'         α*α' + β*β'  ⋱
    #            ⋱ ⋱]                                    ⋱            ⋱]
    #
    # Matching against A, whose diagonal block is D and subdiagonal block is B, gives
    #
    #   α*α' + β*β' == D    and    α*β' == B.
    #
    # Hence β' == inv(α)*B and, writing X = α*α', we get the quadratic matrix equation
    #
    #   X == D - B'*inv(X)*B.
    #
    # Setting W = inv(X)*B this becomes B'*W^2 - D*W + B == 0, i.e. W is the solvent
    # returned by `matrixroot(B', -D, B)`. Choosing the minimal solvent (as `matrixroot`
    # does) is the analogue of taking the `+` branch in the scalar case, and is what makes
    # X positive-definite. We then recover X == D - B'*W.

    D = getindex_value(diagonaldata(blocks(A)))
    B = getindex_value(subdiagonaldata(blocks(A)))
    @assert getindex_value(supdiagonaldata(blocks(A))) ≈ B'

    X = _realifreal(eltype(A), D - B'*matrixroot(B', -D, B))

    α = Matrix(reversecholesky(Hermitian((X + X')/2)).U)
    β = Matrix((α \ B)')

    ReverseCholesky(mortar(Bidiagonal(Fill(α,∞), Fill(β,∞), :U)), 'U', 0)
end

_realifreal(::Type{<:Real}, X) = real(X)
_realifreal(_, X) = X

# the factors are already block-upper-triangular so we avoid the `UpperTriangular` wrapper,
# mimicking `getproperty(::ReverseCholesky{<:Any,<:Bidiagonal}, ::Symbol)`.
function getproperty(C::ReverseCholesky{<:Any,<:BlockMatrix{<:Any,<:Bidiagonal}}, d::Symbol)
    Cfactors = getfield(C, :factors)
    @assert getfield(C, :uplo) === 'U'
    if d === :U || d === :UL
        return Cfactors
    elseif d === :L
        return Cfactors'
    else
        return getfield(C, d)
    end
end
