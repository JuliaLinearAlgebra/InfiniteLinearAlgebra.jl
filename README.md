# InfiniteLinearAlgebra.jl

A Julia repository for linear algebra with infinite banded and block-banded matrices


## Infinite-dimensional QR factorization

This currently supports the infinite-dimensional QR factorization for banded matrices, also known as the adaptive QR decomposition as the entries of the QR decomposition are determined lazily. 

As a simple example, consider the Bessel recurrence relationship:
$$
J_{n-1}(z)  - {2 n \over z} J_n(z) + J_{n+1}(z) = 0
$$
This can be recast as an infinite linear system:
```julia
julia> using InfiniteLinearAlgebra, InfiniteArrays, BandedMatrices, FillArrays, SpecialFunctions

julia> z = 10_000; # the bigger z the longer before we see convergence

julia> A = BandedMatrix(0 => -2*(0:∞)/z, 1 => Ones(∞), -1 => Ones(∞))
∞×∞ BandedMatrix{Float64,ApplyArray{Float64,2,typeof(*),Tuple{Array{Float64,2},ApplyArray{Float64,2,typeof(vcat),Tuple{Transpose{Float64,InfiniteArrays.InfStepRange{Float64,Float64}},Ones{Float64,2,Tuple{Base.OneTo{Int64},InfiniteArrays.OneToInf{Int64}}},Ones{Float64,2,Tuple{Base.OneTo{Int64},InfiniteArrays.OneToInf{Int64}}}}}}},InfiniteArrays.OneToInf{Int64}} with indices OneToInf()×OneToInf():
 0.0   1.0       ⋅        ⋅        ⋅        ⋅       ⋅        ⋅        ⋅      …  
 1.0  -0.0002   1.0       ⋅        ⋅        ⋅       ⋅        ⋅        ⋅         
  ⋅    1.0     -0.0004   1.0       ⋅        ⋅       ⋅        ⋅        ⋅         
  ⋅     ⋅       1.0     -0.0006   1.0       ⋅       ⋅        ⋅        ⋅         
  ⋅     ⋅        ⋅       1.0     -0.0008   1.0      ⋅        ⋅        ⋅         
  ⋅     ⋅        ⋅        ⋅       1.0     -0.001   1.0       ⋅        ⋅      …  
  ⋅     ⋅        ⋅        ⋅        ⋅       1.0    -0.0012   1.0       ⋅         
  ⋅     ⋅        ⋅        ⋅        ⋅        ⋅      1.0     -0.0014   1.0        
  ⋅     ⋅        ⋅        ⋅        ⋅        ⋅       ⋅       1.0     -0.0016     
  ⋅     ⋅        ⋅        ⋅        ⋅        ⋅       ⋅        ⋅       1.0        
  ⋅     ⋅        ⋅        ⋅        ⋅        ⋅       ⋅        ⋅        ⋅      …  
  ⋅     ⋅        ⋅        ⋅        ⋅        ⋅       ⋅        ⋅        ⋅         
  ⋅     ⋅        ⋅        ⋅        ⋅        ⋅       ⋅        ⋅        ⋅         
  ⋅     ⋅        ⋅        ⋅        ⋅        ⋅       ⋅        ⋅        ⋅         
  ⋅     ⋅        ⋅        ⋅        ⋅        ⋅       ⋅        ⋅        ⋅         
 ⋮                                         ⋮                                 ⋱  
```
The first row corresponds to specifying an initial condition. Thus we can determine $J_n(z)$ via solving the recurrence:
```julia
julia> A \ Vcat([besselj(1,z)], Zeros(∞)) 
∞-element LazyArrays.CachedArray{Float64,1,Array{Float64,1},Zeros{Float64,1,Tuple{InfiniteArrays.OneToInf{Int64}}}} with indices OneToInf():
 -0.007096160353406478 
  0.0036474507555295833
  0.007096889843557584 
 -0.0036446119995921654
 -0.007099076610757337 
  0.0036389327383035616
  0.00710271554349564  
 -0.003630409479651369 
 -0.007107798116767152 
  0.0036190370026645442
  0.007114312383371949 
 -0.0036048083778978026
 -0.0071222429618033245
  0.0035877149947894766
  0.007131571020789777 
  ⋮                    

julia> J[1000] - besselj(999,z) # matches besselj to high (relative) accuracy
-6.8252695162307475e-15

julia> J[11_000] - besselj(11_000-1, z)
3.3730094946097293e-143
```
We're even faster than SpecialFunctions.jl for constructing a range of Bessel functions:
```julia
julia> @time [besselj(k-1, z) for k=0:11_000-1];
  0.188690 seconds (77.20 k allocations: 3.295 MiB)

julia> @time J = A \ Vcat([besselj(1,z)], Zeros(∞));
  0.036701 seconds (406.40 k allocations: 46.552 MiB, 24.43% gc time)
```


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

