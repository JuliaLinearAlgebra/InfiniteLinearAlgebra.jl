using InfiniteLinearAlgebra, MatrixFactorizations, LinearAlgebra, Test

A = SymTridiagonal(Fill(3, ∞), Fill(1, ∞))

reversecholesky(A)