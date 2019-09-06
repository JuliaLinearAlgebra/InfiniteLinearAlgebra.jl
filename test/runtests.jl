using InfiniteLinearAlgebra, BlockBandedMatrices, BlockArrays, BandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test, MatrixFactorizations, LinearAlgebra, Random
import InfiniteLinearAlgebra: qltail, toeptail, tailiterate , tailiterate!, tail_de, ql_X!,
                    InfToeplitz, PertToeplitz, TriToeplitz, InfBandedMatrix, 
                    rightasymptotics, QLHessenberg
import BlockBandedMatrices: isblockbanded, _BlockBandedMatrix
import MatrixFactorizations: QLPackedQ
import BandedMatrices: bandeddata, _BandedMatrix

@testset "Algebra" begin 
    A = BlockTridiagonal(Vcat([fill(1.0,2,1),Matrix(1.0I,2,2),Matrix(1.0I,2,2),Matrix(1.0I,2,2)],Fill(Matrix(1.0I,2,2), ∞)), 
                        Vcat([zeros(1,1)], Fill(zeros(2,2), ∞)), 
                        Vcat([fill(1.0,1,2),Matrix(1.0I,2,2)], Fill(Matrix(1.0I,2,2), ∞)))
                        
    @test A isa InfiniteLinearAlgebra.BlockTriPertToeplitz                       
    @test isblockbanded(A)

    @test A[Block.(1:2),Block(1)] == A[1:3,1:1] == reshape([0.,1.,1.],3,1)

    @test BlockBandedMatrix(A)[1:100,1:100] == BlockBandedMatrix(A,(2,1))[1:100,1:100] == BlockBandedMatrix(A,(1,1))[1:100,1:100] == A[1:100,1:100]

    @test (A - I)[1:100,1:100] == A[1:100,1:100]-I
    @test (A + I)[1:100,1:100] == A[1:100,1:100]+I
    @test (I + A)[1:100,1:100] == I+A[1:100,1:100]
    @test (I - A)[1:100,1:100] == I-A[1:100,1:100]

    A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
    Ac = BandedMatrix(A')
    At = BandedMatrix(transpose(A))
    @test Ac[1:10,1:10] ≈ (A')[1:10,1:10] ≈ A[1:10,1:10]'
    @test At[1:10,1:10] ≈ transpose(A)[1:10,1:10] ≈ transpose(A[1:10,1:10])

    A = BandedMatrix(-1 => Vcat(Float64[], Fill(1/4,∞)), 0 => Vcat([1.0+im],Fill(0,∞)), 1 => Vcat(Float64[], Fill(1,∞)))
    Ac = BandedMatrix(A')
    At = BandedMatrix(transpose(A))
    @test Ac[1:10,1:10] ≈ (A')[1:10,1:10] ≈ A[1:10,1:10]'
    @test At[1:10,1:10] ≈ transpose(A)[1:10,1:10] ≈ transpose(A[1:10,1:10])

    A = BandedMatrix(-2 => Vcat(Float64[], Fill(1/4,∞)), 0 => Vcat([1.0+im,2,3],Fill(0,∞)), 1 => Vcat(Float64[], Fill(1,∞)))
    Ac = BandedMatrix(A')
    At = BandedMatrix(transpose(A))
    @test Ac[1:10,1:10] ≈ (A')[1:10,1:10] ≈ A[1:10,1:10]'
    @test At[1:10,1:10] ≈ transpose(A)[1:10,1:10] ≈ transpose(A[1:10,1:10])
end
include("test_hessenbergq.jl")
include("test_infql.jl")

