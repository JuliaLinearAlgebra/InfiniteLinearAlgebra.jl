using InfiniteLinearAlgebra, BlockBandedMatrices, BandedMatrices, BlockArrays, InfiniteArrays, FillArrays, LazyArrays, Test

A = BlockTridiagonal(Vcat([fill(1.0,2,1),Matrix(1.0I,2,2),Matrix(1.0I,2,2),Matrix(1.0I,2,2)],Fill(Matrix(1.0I,2,2), ∞)),
                       Vcat([zeros(1,1)], Fill(zeros(2,2), ∞)),
                       Vcat([fill(1.0,1,2),Matrix(1.0I,2,2)], Fill(Matrix(1.0I,2,2), ∞)))


