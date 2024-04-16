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