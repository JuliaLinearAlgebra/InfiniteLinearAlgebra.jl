using Revise, InfiniteBandedMatrices, BlockBandedMatrices, BandedMatrices, BlockArrays, InfiniteArrays, FillArrays, LazyArrays, Test
import BlockBandedMatrices: _BlockSkylineMatrix

A = BlockTridiagonal(Vcat([fill(1.0,2,1)],Fill(Matrix(1.0I,2,2), ∞)), 
                       Vcat([zeros(1,1)], Fill(zeros(2,2), ∞)), 
                       Vcat([fill(1.0,1,2)], Fill(Matrix(1.0I,2,2), ∞)))

data = mortar(Fill(vec(A[Block.(3:5),Block(4)]), ∞))

bs = Vcat(1, Fill(2,∞))