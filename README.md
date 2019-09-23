# InfiniteLinearAlgebra.jl

A Julia repository for linear algebra with infinite banded and block-banded matrices


## Infinite-dimensional QR factorization

This currently supports the infinite-dimensional QR factorization for banded matrices, also known as the adaptive QR decomposition as the entries of the QR decomposition are determined lazily. 

## Infinite-dimensional QL factorization


This currently supports the infinite-dimensional QL factorization for perturbations of Toeplitz operators. Here is an example:
```julia
# Bull head matrix
A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
ql(A - 5*I)
```
The infinite-dimensional QL factorization is a subtly thing: its defined when the operator has non-positive Fredholm index, and if the Fredholm index is not zero, it may not be unique. For the Bull head matrix `A`, here are plots of `ql(A-λ*I).L[1,1]` alongside the image of the symbol `A`, which depicts the essential spectrum of `A` and where the Fredholm index changes. Note we have two plots as the regions with negative Fredholm index  have multiple QL factorizations. Where the Fredholm index is positive, the QL factorization doesn't exist and is depected in black.

<img src=https://github.com/JuliaMatrices/InfiniteLinearAlgebra.jl/raw/master/images/ql1.png width=500 height=400>
<img src=https://github.com/JuliaMatrices/InfiniteLinearAlgebra.jl/raw/master/images/ql2.png width=500 height=400>

