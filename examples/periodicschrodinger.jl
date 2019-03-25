using InfiniteBandedMatrices, Plots

# Consider a rank-1 perturbation of periodic semi-infinite Schrödinger (Jacobi) operator.
# We can construct it as a block matrix as follows:

A = BlockTridiagonal(Vcat([[0. 1.; 0. 0.]],Fill([0. 1.; 0. 0.], ∞)), 
                       Vcat([[-1. 1.; 1. 1.]], Fill([-1. 1.; 1. 1.], ∞)), 
                       Vcat([[0. 0.; 1. 0.]], Fill([0. 0.; 1. 0.], ∞)))

A[1,1] = 2 # perturbation

# in this case finite-section works quite well, though higher-dimensional
# problems have spectal pollution:
n = 100; scatter(eigvals(A[1:n,1:n]), zeros(n); label="finite-section")

# In any case, we can also calculate the spectrum using ∞-QL:
xx = [-0.95:0.05:0.95; 2.25:0.125/4:4.0]
plot!(xx, (x -> ql(A-x*I).L[1,1]).(xx); label="L[1,1]")
xx = 
plot!(xx, (x -> ql(A-x*I).L[1,1]).(xx))


ql(A-2.25*I);




eigvals(A[1:n,1:n])
