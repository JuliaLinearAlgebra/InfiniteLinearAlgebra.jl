using InfiniteLinearAlgebra, BlockBandedMatrices, BlockArrays, BandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test,
    MatrixFactorizations, ArrayLayouts, LinearAlgebra, Random, LazyBandedMatrices, StaticArrays
import InfiniteLinearAlgebra: qltail, toeptail, tailiterate, tailiterate!, tail_de, ql_X!,
    InfToeplitz, PertToeplitz, TriToeplitz, InfBandedMatrix,
    rightasymptotics, QLHessenberg, PertConstRows, chop, chop!, pad,
    PertToeplitzLayout, TridiagonalToeplitzLayout,
    BidiagonalConjugation
import Base: BroadcastStyle, oneto
import BlockArrays: _BlockArray, blockcolsupport, findblock
import BlockBandedMatrices: isblockbanded, _BlockBandedMatrix
import MatrixFactorizations: QLPackedQ
import BandedMatrices: bandeddata, _BandedMatrix, BandedStyle
import LazyArrays: colsupport, MemoryLayout, ApplyLayout, LazyArrayStyle, arguments, paddeddata, PaddedColumns, LazyLayout
import InfiniteArrays: OneToInf, oneto, RealInfinity
import LazyBandedMatrices: BroadcastBandedBlockBandedLayout, BroadcastBandedLayout, LazyBandedLayout, BlockVec
import InfiniteRandomArrays: InfRandTridiagonal, InfRandBidiagonal
import ArrayLayouts: diagonaldata, supdiagonaldata, subdiagonaldata

using Aqua
@testset "Project quality" begin
    Aqua.test_all(InfiniteLinearAlgebra, ambiguities=false, unbound_args=false, piracies=false)
end

@testset "chop" begin
    a = randn(5)
    b = [a; zeros(5)]
    chop!(b, eps())
    @test b == a

    @test isempty(chop!([0]))

    A = randn(5, 5)
    @test chop([A zeros(5, 2); zeros(2, 5) zeros(2, 2)], eps()) == A

    c = BlockedArray([randn(5); zeros(10)], (blockedrange(1:5),))
    d = chop!(c, 0)
    @test length(d) == 6

    @test pad(1:3, 5) == [1:3; 0; 0]
    @test pad(1:3, oneto(∞)) isa Vcat
    @test pad(1:3, :) ≡ 1:3
    X = Matrix(reshape(1:6, 3, 2))
    # TODO: replace ≈ with ==
    @test pad(X, oneto(3), ∞) ≈ pad(X, oneto(3), oneto(∞)) ≈ pad(X, 3, oneto(∞)) ≈ pad(X, 3, ∞) ≈ pad(X, :, ∞)
    @test pad(X, oneto(3), oneto(∞)) isa PaddedArray
    @test pad(X, :, oneto(∞)) isa PaddedArray
    @test pad(X, :, :) isa Matrix
    @test pad(X, oneto(10), :) isa PaddedArray
    P = pad(BlockVec(X), blockedrange(Fill(3,∞)))
    @test P isa BlockVec
    @test MemoryLayout(P) isa PaddedColumns
    @test paddeddata(P) isa BlockVec
    @test colsupport(P) == 1:6
    P = pad(BlockVec(X'), blockedrange(Fill(3,∞)))
    @test P isa BlockVec{Int,<:Adjoint}
    @test MemoryLayout(P) isa PaddedColumns
    @test pad(BlockVec(transpose(X)), blockedrange(Fill(3,∞))) isa BlockVec{Int,<:Transpose}
end



include("test_hessenbergq.jl")
include("test_infql.jl")
include("test_infqr.jl")
include("test_inful.jl")
include("test_infcholesky.jl")
include("test_periodic.jl")
include("test_infreversecholesky.jl")
include("test_bidiagonalconjugation.jl")