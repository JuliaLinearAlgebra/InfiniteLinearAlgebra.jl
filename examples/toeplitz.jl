using Revise, InfiniteBandedMatrices, LazyArrays, FillArrays, Plots

# create a complex valued Toeplitz matrix
T = Tridiagonal(Vcat(Float64[], Fill(1/2,∞)), 
                Vcat(Float64[], Fill(0.0,∞)), 
                Vcat(Float64[], Fill(2.0,∞)))

# This operator has spectrum inside the image of 1/(2z) + 2z on the unit circle, which is a Bernstein ellipse through 1.
# But in finite-dimensions it is a conjugation of a symmetric Toeplitz matrix with symbol 1/z+z so the eigenvalues are on [-2,2]
n = 100; T_n = T[1:n,1:n]
λ = eigvals(Matrix(T_n))
scatter(real.(λ), imag.(λ)) # points near [-2,2]

# The QL decomposition gives a very good indicator of being inside the spectrum: if `L[1,1] == 0` we have a kernel. 
# Away from the borderline this works fine with finite-dimensional linear algebra
λ = 0.5im; n = 100; ql(BandedMatrix(T-λ*I)[1:n,1:n]).L[1,1] # == 0.0 exactly

# but as we approach the edge of the spectrum, we need increasingly large n to determine `L[1,1]`:
λ = 1.49im;  n = 1000; ql(BandedMatrix(T-λ*I)[1:n,1:n]).L[1,1] # ≈ 0.0032 
λ = 1.499im; n = 10_000; ql(BandedMatrix(T-λ*I)[1:n,1:n]).L[1,1] # ≈ 0.001
λ = 1.4999im; n = 100_000; ql(BandedMatrix(T-λ*I)[1:n,1:n]).L[1,1] # ≈ 0.0003

# Thus the simple test of "are we in the spectrum" is computationally challenging. 
# The infinite-dimensional QL decomposition avoids this issue because the cost is independent of λ
# 
λ = 1.49im; ql(T-λ*I).L[1,1] # ≈ 1E-13
λ = 1.499im; ql(T-λ*I).L[1,1] # ≈ 6E-13
λ = 1.4999im; ql(T-λ*I).L[1,1] # ≈ 1E-11

λ = 1.50001im; ql(T-λ*I).L[1,1] # ≈ 1E-11

# There are some round-off error as we approach 1.5, which we can overcome using Big Floats:

λ = BigFloat(1.49)im; ql(T-λ*I).L[1,1] # ≈ 1E-13


ql(T-λ*I)