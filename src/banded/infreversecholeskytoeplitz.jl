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

"""
    \\(F::ReverseCholesky{<:Any,<:BlockMatrix{<:Any,<:Bidiagonal}}, b; tolerance)

solves `U*U'*x == b` where `U` is block-bidiagonal Toeplitz and `b` has finite support.
The solution has infinite support but decays geometrically: it is truncated once a block
has all entries at most `tolerance` in absolute value.
"""
function \(F::ReverseCholesky{<:Any,<:BlockMatrix{<:Any,<:Bidiagonal}}, b::AbstractVector; tolerance=floatmin(real(promote_type(eltype(F), eltype(b)))))
    #   U = [α β          U' = [α'
    #          α β  ⋱          β' α'
    #            ⋱ ⋱]             ⋱  ⋱]
    #
    # Solving U*y == b by back-substitution, the minimal solvent guarantees that inv(α)*β has
    # spectral radius below one so that y is supported on the same blocks as b. Then U'*x == y
    # by forward-substitution, where beyond the support of y we have x_{k+1} == -inv(α')*β'*x_k.
    U = F.U
    ax = axes(U, 2)
    α = getindex_value(blocks(U).dv)
    β = getindex_value(blocks(U).ev)
    m = size(α, 1)
    T = promote_type(eltype(F), eltype(b))

    cs = colsupport(b, 1)
    n = isempty(cs) ? 0 : last(cs)
    N = iszero(n) ? 0 : Int(findblock(ax, n))
    Y = zeros(T, m, N)
    Y[1:n] = b[1:n]

    αl = lu(α)
    for k = N:-1:1
        k < N && mul!(view(Y,:,k), β, view(Y,:,k+1), -one(T), one(T))
        ldiv!(αl, view(Y,:,k))
    end

    αl = lu(α')
    X = Y # overwrite in place
    for k = 1:N
        k > 1 && mul!(view(X,:,k), β', view(X,:,k-1), -one(T), one(T))
        ldiv!(αl, view(X,:,k))
    end

    if !iszero(N)
        Γ = -(αl \ β') # x_{k+1} == Γ*x_k
        tail = Vector{T}[]
        x = Γ*X[:,N]
        while maximum(abs, x) > tolerance
            push!(tail, x)
            x = Γ*x
        end
        X = [X reduce(hcat, tail; init=zeros(T,m,0))]
    end

    BlockedVector(Vcat(vec(X), Zeros{T}(∞)), (ax,))
end
