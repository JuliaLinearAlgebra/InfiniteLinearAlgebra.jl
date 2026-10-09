####
# This file shows how the ∞-dimensional QL decomposition can be used
# for spectral theory of Toeplitz operators. We investigate the
# QL decomposition on each of the examples in Trefethen & Embree.
#


using InfiniteLinearAlgebra, BandedMatrices, MatrixFactorizations
using PyPlot

###
# Basic routines for plotting
###

function ℓ11(A,λ; kwds...)
    try
        abs(ql(A-λ*I; kwds...).L[1,1])
    catch DomainError
        -1.0
    end
end

function branch(k)
    function(λ)
        j = sortperm(λ)[end-k+1]
        λ[j], j
    end
end

qlplot(A::AbstractMatrix; kwds...) = qlplot(BandedMatrix(A); kwds...)
function qlplot(A::BandedMatrix; branch=findmax, x=range(-4,4; length=200), y=range(-4,4;length=200), kwds...)
    z = ℓ11.(Ref(A), x' .+ y.*im; branch=branch)
    contourf(x,y,z; kwds...)
end



toepcoeffs(A::BandedMatrix) = InfiniteLinearAlgebra.rightasymptotics(A.data).args[1]
toepcoeffs(A::Adjoint) = reverse(toepcoeffs(A'))

function symbolplot(A::BandedMatrix; kwds...)
    l,u = bandwidths(A)
    a = toepcoeffs(A)
    θ = range(0,2π; length=1000)
    i = map(t -> dot(exp.(im.*(u:-1:-l).*t),a),θ)
    plot(real.(i), imag.(i); kwds...)
end


###
# Trefethen & Embree example
###

A = BandedMatrix(1 => Fill(2im,∞), 2 => Fill(-1,∞), 3 => Fill(2,∞), -2 => Fill(-4,∞), -3 => Fill(-2im,∞))
clf(); qlplot(A; x=range(-10,7; length=100), y=range(-7,8;length=100)); symbolplot(A; color=:black); title("Trefethen & Embree")
clf(); qlplot(transpose(A); x=range(-10,7; length=200), y=range(-7,8;length=200)); symbolplot(A; color=:black); title("Trefethen & Embree, transpose")

###
# limaçon
###

A = BandedMatrix(-2 => Fill(1.0,∞), -1 => Fill(1.0,∞), 1 => Fill(eps(),∞))
qlplot(A; x=range(-2,3; length=100), y=range(-2.5,2.5;length=100)); symbolplot(A; color=:black); title("Limacon")
clf(); qlplot(A; branch=findsecond, x=range(-2,3; length=100), y=range(-2.5,2.5;length=100)); symbolplot(A; color=:black); title("Limacon, second branch")
clf(); qlplot(transpose(A); x=range(-2,3; length=100), y=range(-2.5,2.5;length=100)); symbolplot(A; color=:black); title("Limacon, transpose")


###
# bull-head
###

A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
clf(); qlplot(A; x=range(-10,7; length=100), y=range(-7,8;length=100)); symbolplot(A; color=:black); title("Bull-head")
clf(); qlplot(transpose(A); x=range(-10,7; length=100), y=range(-7,8;length=100)); symbolplot(A; color=:black); title("Bull-head, transpose")

At = BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), -1 => Fill(2im,∞))

###
# Grcar
###

A = BandedMatrix(-3 => Fill(1,∞), -2 => Fill(1,∞), -1 => Fill(1,∞), 0 => Fill(1,∞), 1 => Fill(-1,∞))
clf(); qlplot(A; x=range(-4,5; length=100), y=range(-6,5;length=100)); symbolplot(A; color=:black); title("Grcar")
clf(); qlplot(A; branch=branch(2), x=range(-4,5; length=100), y=range(-6,5;length=100)); symbolplot(A; color=:black); title("Grcar, branch 2")
clf(); qlplot(A; branch=branch(3), x=range(-4,5; length=100), y=range(-6,5;length=100)); symbolplot(A; color=:black); title("Grcar, branch 3")
clf(); qlplot(transpose(A); x=range(-4,5; length=100), y=range(-6,5;length=100)); symbolplot(A; color=:black); title("Grcar, transpose")

###
# Triangle
###

A = BandedMatrix(-2 => Fill(1/4,∞), 1 => Fill(1,∞))
clf(); qlplot(A; x=range(-2,2; length=100), y=range(-2,2;length=100)); symbolplot(A; color=:black); title("Triangle")
clf(); qlplot(transpose(A); x=range(-2,2; length=100), y=range(-2,2;length=100)); symbolplot(A; color=:black); title("Triangle, transpose")
clf(); qlplot(transpose(A); branch=branch(2), x=range(-2,2; length=100), y=range(-2,2;length=100)); symbolplot(A; color=:black); title("Triangle, transpose, branch 2")

###
# Whale
###

A = BandedMatrix(-4 => Fill(im,∞), -3 => Fill(4,∞), -2 => Fill(3+im,∞), -1 => Fill(10,∞),
                    1 => Fill(1,∞), 2 => Fill(im,∞), 3 => Fill(-(3+2im),∞), 4=>Fill(-1,∞))

clf(); qlplot(A; x=range(-15,20; length=100), y=range(-20,20;length=100)); symbolplot(A; color=:black); title("Whale")
clf(); qlplot(transpose(A); x=range(-15,20; length=100), y=range(-20,20;length=100)); symbolplot(A; color=:black); title("Whale, transpose")

###
# Butterfly
###

A = BandedMatrix(-2 => Fill(1,∞), -1 => Fill(-im,∞), 1 => Fill(im,∞), 2 => Fill(-1,∞))
clf(); qlplot(A; x=range(-3,3; length=100), y=range(-3,3;length=100)); symbolplot(A; color=:black); title("Butterfly")
clf(); qlplot(A; branch=branch(2), x=range(-3,3; length=100), y=range(-3,3;length=100)); symbolplot(A; color=:black); title("Butterfly, branch 2")
clf(); qlplot(transpose(A); x=range(-3,3; length=100), y=range(-3,3;length=100)); symbolplot(A; color=:black); title("Butterfly")
clf(); qlplot(transpose(A); branch=branch(2), x=range(-3,3; length=100), y=range(-3,3;length=100)); symbolplot(A; color=:black); title("Butterfly")



A = LazyBandedMatrices.Tridiagonal(Fill(2,∞), Zeros(∞), Fill(1/2,∞))
A = BandedMatrix(1 => Fill(1/2,∞), -1 => Fill(2,∞))

# symbol is 2z + z/2, an ellipse


Q,L = ql(complex(A))
Q₂,L₂ = ql(complex(A); branch=branch(2))

n = 1000
@test Q[1:n,1:n-10]'Q[1:n,1:n-10] ≈ I
@test Q[1:n-10,1:n]Q[1:n-10,1:n]' ≈ I

@test Q₂[1:n,1:n-10]'Q₂[1:n,1:n-10] ≈ I
@test Q₂[1:n-10,1:n]Q₂[1:n-10,1:n]' ≈ I


L'L
L₂'L₂

B = A - 0.1im * I
Q,L = ql(complex(B))
Q₂,L₂ = ql(complex(B); branch=branch(2))

eigvals((L'L)[1:100,1:100])

using Plots
λ = eigvals(Matrix(A[1:100,1:100]) .+ 0.0000001 .* randn.())
scatter(real.(λ), imag.(λ))


scatter(real(eigvals(B[1:100,1:100])), imag(eigvals(B[1:100,1:100])))

norm(inv(B[1:100,1:100]))
norm(inv((B'B)[1:100,1:100]))

L'L - (B'B)

L₂'L₂ - (B'B)

L₃ = reversecholesky((B'B)[1:100,1:100]).L

L₃'L₃ - (B'B)[1:100,1:100]
L₂
Q₃ = B[1:100,1:100] * inv(L₃[1:100,1:100])
Q₃'Q₃
Q₃*Q₃'


# B = QL => B'B 

inv(reversecholesky((B'B)[1:10,1:10]).U)


n = 100
norm(inv(L[1:n,1:n]))
norm(inv(L₂[1:n,1:n]))



########
#. walk through

A = BandedMatrix(1 => Fill(1/2,∞), -1 => Fill(2,∞))
# spectrum of A is ellipse
B = A - 0.1im * I # inside continuous spectrum

# our previous algorithm (in this case defined in terms of sqrts) gives us two solutions:

Q,L = ql(complex(B))
Q₂,L₂ = ql(complex(B); branch= branch(2))

# Now consider HPD Gram matrix 
K = B'B

# K is invertible!
# we have two reverse Choleskies for K (but unbounded factors):

@test K[1:n,1:n] ≈ (L'L)[1:n,1:n]
@test K[1:n,1:n] ≈ (L₂'L₂)[1:n,1:n]

# these are not invertible:

@test cond(L[1:n,1:n]) ≥ 1E20
@test cond(L₂[1:n,1:n]) ≥ 1E20

# But K is invertible (it's HPD) so we know it has it's own reversecholesky. We can compute it numerically:

L₃ = reversecholesky(K[1:n,1:n]).L
@test cond(Matrix(L₃)) ≤ 2
@test K[1:n,1:n] ≈ (L₃'L₃)[1:n,1:n]

# can we use this to form QL? We have
# B = QL => Q = B*inv(L) thus consider

Q₃ = B[1:100,1:100] * inv(L₃[1:100,1:100])

@test Q₃ * L₃ ≈ B[1:100,1:100]

@test (Q₃'Q₃)[1:n-20,1:n-20] ≈ I
@test !((Q₃*Q₃')[1:n-20,1:n-20] ≈ I)

# Let's repeat the experiment with a real B:

B = A - 0.5 * I # inside continuous spectrum

# our previous algorithm (in this case defined in terms of sqrts) gives us two solutions:

Q,L = ql(complex(B))
Q₂,L₂ = ql(complex(B); branch= branch(2))

# Now consider HPD Gram matrix 
K = B'B

# K is invertible!
# we have two reverse Choleskies for K (but unbounded factors):

@test K[1:n,1:n] ≈ (L'L)[1:n,1:n]
@test K[1:n,1:n] ≈ (L₂'L₂)[1:n,1:n]

# these are not invertible:

@test cond(L[1:n,1:n]) ≥ 1E20
@test cond(L₂[1:n,1:n]) ≥ 1E20

# But K is invertible (it's HPD) so we know it has it's own reversecholesky. We can compute it numerically:

L₃ = reversecholesky(K[1:n,1:n]).L
@test cond(Matrix(L₃)) ≤ 3
@test K[1:n,1:n] ≈ (L₃'L₃)[1:n,1:n]

# can we use this to form QL? We have
# B = QL => Q = B*inv(L) thus consider

Q₃ = B[1:100,1:100] * inv(L₃[1:100,1:100])

@test Q₃ * L₃ ≈ B[1:100,1:100]

@test (Q₃'Q₃)[1:n-20,1:n-20] ≈ I
@test !((Q₃*Q₃')[1:n-20,1:n-20] ≈ I)

# We can write the the tailo f the real K as a Block-Toeplitz:

using BlockArrays

B = [1 0; -1.25 1]
D = [4.5 -1.25; -1.25 4.5]
K = mortar(Tridiagonal(Fill(Matrix(B'), ∞), Fill(D, ∞), Fill(B, ∞)))

@test_broken K[1:n,1:n] ≈ (B'B)[1:n,1:n]
F = reversecholesky(K)
@test F.L[2:n-20,2:n-20] ≈ L₃[2:n-20,2:n-20]

@ent reversecholesky(K)

A₂,A₁,A₀ = B',-D, B
@test A₂ * Z₂^2 + A₁*Z₂ ≈ -A₀
@test A₂ * Z^2 + A₁*Z ≈ -A₀

Z₃ = let X = L₂[2:3,2:3], C = L₂[4:5,2:3]
    @test X'X + C'C ≈ D
    @test X'C ≈ B'
    @test C'X ≈ B
    Y = X'X
    @test Y + C'C ≈ D ≈ Y + B*inv(Y)*B'
    (B*inv(Y))'
end

Z₄ = let X = L[2:3,2:3], C = L[4:5,2:3]
    @test X'X + C'C ≈ D
    @test X'C ≈ B'
    @test C'X ≈ B
    Y = X'X
    @test Y + C'C ≈ D ≈ Y + B*inv(Y)*B'
    (B*inv(Y))'
end

@test (Z₃')^2*B' - (Z₃')*D ≈ -B
@test (Z₃')^2*B' - Z₃'*D ≈ -B
@test B*Z₃^2 - D*Z₃ ≈ -B'

function matrixroot_2(A₂, A₁, A₀, s_in)
    n = LinearAlgebra.checksquare(A₀)
    T = complex(float(promote_type(eltype(A₀), eltype(A₁), eltype(A₂))))
    Z = zeros(T,n,n); Iₙ = Matrix{T}(I,n,n)
    F = schur([Z Iₙ; -T.(A₀) -T.(A₁)], [Iₙ Z; Z T.(A₂)])
    s = falses(2n)
    s .= s_in
    F = ordschur(F, s)
    U = @view F.Z[:,1:n]
    U₁ = @view U[1:n,:]
    U₁ * (UpperTriangular(@view F.T[1:n,1:n]) \ @view(F.S[1:n,1:n])) / U₁
end

@test Z₃ ≈ 



Z = real(matrixroot(A₂, A₁, A₀))
@test Z ≈ matrixroot_2(A₂, A₁, A₀, [0,0,1,1])
Z₂ = real(matrixroot_2(A₂, A₁, A₀, [1,0,0,0]))
@test Z₂ ≈ matrixroot_2(A₂, A₁, A₀, [1,1,0,0]) ≈ matrixroot_2(A₂, A₁, A₀, [0,1,0,0]) ≈ matrixroot_2(A₂, A₁, A₀, [1,1,0,1]) ≈ matrixroot_2(A₂, A₁, A₀, [1,1,1,0]) ≈ matrixroot_2(A₂, A₁, A₀, [1,1,1,1])
@test Z₃ ≈  matrixroot_2(A₂, A₁, A₀, [0,0,0,1]) ≈ matrixroot_2(A₂, A₁, A₀, [1,0,0,1])
@test Z₄ ≈ matrixroot_2(A₂, A₁, A₀, [0,1,1,1])
Z₅ = matrixroot_2(A₂, A₁, A₀, [0,0,1,0])
@test Z₅ ≈ matrixroot_2(A₂, A₁, A₀, [1,0,1,0]) ≈ matrixroot_2(A₂, A₁, A₀, [1,0,1,1])
Z₆ = matrixroot_2(A₂, A₁, A₀, [0,1,0,1])


@test Z₆ ≈ conj(Z₅)

@test A₂ * Z^2 + A₁*Z ≈ -A₀
@test A₂ * Z₂^2 + A₁*Z₂ ≈ -A₀
@test A₂ * Z₃^2 + A₁*Z₃ ≈ -A₀
@test A₂ * Z₄^2 + A₁*Z₄ ≈ -A₀
@test A₂ * Z₅^2 + A₁*Z₅ ≈ -A₀

@test Z₃ ≈ conj(Z₄)


n = LinearAlgebra.checksquare(A₀)
T = complex(float(promote_type(eltype(A₀), eltype(A₁), eltype(A₂))))
Z = zeros(T,n,n); Iₙ = Matrix{T}(I,n,n)
F,F̃ = let A = [Z Iₙ; -T.(A₀) -T.(A₁)], B = [Iₙ Z; Z T.(A₂)]
    F = schur(A, B)
    s = falses(2n)
    @test F.Q * F.S * F.Z' ≈ A
    @test F.Q * F.T * F.Z' ≈ B
    s[sortperm(abs.(ifelse.(iszero.(F.β), Inf, F.α ./ F.β)))[1:n]] .= true
    F̃ = ordschur(F, s)
    @test F̃.Q * F̃.S * F̃.Z' ≈ A
    @test F̃.Q * F̃.T * F̃.Z' ≈ B
    F,F̃
end
U = @view F.Z[:,1:n]
U₁ = @view U[1:n,:]
U₁ * (UpperTriangular(@view F.T[1:n,1:n]) \ @view(F.S[1:n,1:n])) / U₁


inv(Z')*B
inv(Z₂')*B

inv(Z₃')*B
inv(Z₄')*B

inv(Z₅')*B
inv(Z₆')*B

B



Δ = BandedMatrix(1 => Fill(1/2,∞), -1 => Fill(1/2,∞))

ql(Δ - (0.1+0.00000001im)*I).L[1:3,1:3]

ql(Δ - (0.1+0.001im)*I).Q[1:10,1:10]

Q,L = ql(Δ - complex(2.0)*I)


Q₂,L₂ = ql(Δ - (0.1+0.0001im)*I; branch= function(λ)
        b = branch(1)(λ)
        complex(b[1]),b[2]
end)

Q₂*L₂


Q * L



using FillArrays, LazyArrays, BlockArrays

A = Tridiagonal(Vcat(Float64[], Fill(0.5,∞)),
                Vcat(Float64[2.0], Fill(0.0,∞)),
                Vcat(Float64[], Fill(0.5,∞)))

ql(A - 0.00001im * I).L[1,1]
ql(A - (0.1+0.00001im) * I).L[1,1]

ql(A - (0.1+0.00001im) * I; branch=branch(2)).L[1,1]


using Toe
B = [1.0 0; 1 1.0]
D = [5.0 1; 1 5]
J = mortar(Tridiagonal(Fill(Matrix(B'), ∞), Fill(D, ∞), Fill(B, ∞)))

reversecholesky(J)