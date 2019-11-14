using InfiniteLinearAlgebra, BlockBandedMatrices, BlockArrays, BandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test, MatrixFactorizations, LinearAlgebra, Random
import InfiniteLinearAlgebra: qltail, toeptail, tailiterate , tailiterate!, tail_de, ql_X!,
                    InfToeplitz, PertToeplitz, TriToeplitz, InfBandedMatrix, 
                    rightasymptotics, QLHessenberg, ConstRows, PertConstRows, BandedToeplitzLayout, PertToeplitzLayout
import Base: BroadcastStyle
import BlockArrays: _BlockArray                    
import BlockBandedMatrices: isblockbanded, _BlockBandedMatrix
import MatrixFactorizations: QLPackedQ
import BandedMatrices: bandeddata, _BandedMatrix, BandedStyle
import LazyArrays: colsupport, ApplyStyle, MemoryLayout, ApplyLayout, LazyArrayStyle

@testset "∞-Toeplitz and Pert-Toeplitz" begin
    A = BandedMatrix(1 => Fill(2im,∞), 2 => Fill(-1,∞), 3 => Fill(2,∞), -2 => Fill(-4,∞), -3 => Fill(-2im,∞))
    @test A isa InfToeplitz
    @test MemoryLayout(typeof(A.data)) == ConstRows()
    @test MemoryLayout(typeof(A)) == BandedToeplitzLayout()
    V = view(A,:,3:∞)
    @test MemoryLayout(typeof(bandeddata(V))) == ConstRows()
    @test MemoryLayout(typeof(V)) == BandedToeplitzLayout()

    @test BandedMatrix(V) isa InfToeplitz
    @test A[:,3:end] isa InfToeplitz

    A = BandedMatrix(-2 => Vcat(Float64[], Fill(1/4,∞)), 0 => Vcat([1.0+im,2,3],Fill(0,∞)), 1 => Vcat(Float64[], Fill(1,∞)))
    @test A isa PertToeplitz
    @test MemoryLayout(typeof(A)) == PertToeplitzLayout()
    V = view(A,2:∞,2:∞)
    @test MemoryLayout(typeof(V)) == PertToeplitzLayout()
    @test BandedMatrix(V) isa PertToeplitz
    @test A[2:∞,2:∞] isa PertToeplitz
end


@testset "Algebra" begin 
    @testset "BandedMatrix" begin
        A = BandedMatrix(-3 => Fill(7/10,∞), -2 => 1:∞, 1 => Fill(2im,∞))
        @test A isa BandedMatrix{ComplexF64}
        @test A[1:10,1:10] == diagm(-3 => Fill(7/10,7), -2 => 1:8, 1 => Fill(2im,9))

        A = BandedMatrix(0 => Vcat([1,2,3],Zeros(∞)), 1 => Vcat(1, Zeros(∞)))
        @test A[1,2] == 1

        A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
        Ac = BandedMatrix(A')
        At = BandedMatrix(transpose(A))
        @test Ac[1:10,1:10] ≈ (A')[1:10,1:10] ≈ A[1:10,1:10]'
        @test At[1:10,1:10] ≈ transpose(A)[1:10,1:10] ≈ transpose(A[1:10,1:10])

        A = BandedMatrix(-1 => Vcat(Float64[], Fill(1/4,∞)), 0 => Vcat([1.0+im],Fill(0,∞)), 1 => Vcat(Float64[], Fill(1,∞)))
        @test MemoryLayout(typeof(view(A.data,:,1:10))) == ApplyLayout{typeof(hcat)}()
        Ac = BandedMatrix(A')
        At = BandedMatrix(transpose(A))
        @test Ac[1:10,1:10] ≈ (A')[1:10,1:10] ≈ A[1:10,1:10]'
        @test At[1:10,1:10] ≈ transpose(A)[1:10,1:10] ≈ transpose(A[1:10,1:10])

        A = BandedMatrix(-2 => Vcat(Float64[], Fill(1/4,∞)), 0 => Vcat([1.0+im,2,3],Fill(0,∞)), 1 => Vcat(Float64[], Fill(1,∞)))
        Ac = BandedMatrix(A')
        At = BandedMatrix(transpose(A))
        @test Ac[1:10,1:10] ≈ (A')[1:10,1:10] ≈ A[1:10,1:10]'
        @test At[1:10,1:10] ≈ transpose(A)[1:10,1:10] ≈ transpose(A[1:10,1:10])

        A = _BandedMatrix(Fill(1,4,∞),∞,1,2)
        @test A*A isa ApplyArray
        @test (A^2)[1:10,1:10] == (A*A)[1:10,1:10] == (A[1:100,1:100]^2)[1:10,1:10]
        @test (A^3)[1:10,1:10] == (A*A*A)[1:10,1:10] == (A[1:100,1:100]^3)[1:10,1:10]
    end

    @testset "BlockTridiagonal" begin
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
    end
    

    @testset "Fill" begin
        A = _BandedMatrix(Ones(1,∞),∞,-1,1)
        @test 1.0 .* A isa BandedMatrix{Float64,<:Fill}
        @test_skip Ones(∞) .* A
        @test 2.0 .* A isa BandedMatrix{Float64,<:Fill}
        @test A .* 2.0 isa BandedMatrix{Float64,<:Fill}
        @test Eye(∞)*A isa BandedMatrix{Float64,<:Fill}
        @test A*Eye(∞) isa BandedMatrix{Float64,<:Fill}
    end

    @testset "Banded Broadast" begin
        A = _BandedMatrix((1:∞)',∞,-1,1)
        @test 2.0 .* A isa BandedMatrix{Float64,<:Adjoint}
        @test A .* 2.0 isa BandedMatrix{Float64,<:Adjoint}
        @test Eye(∞)*A isa BandedMatrix{Float64,<:Adjoint}
        @test A*Eye(∞) isa BandedMatrix{Float64,<:Adjoint}
        A = _BandedMatrix(Vcat((1:∞)',Ones(1,∞)),∞,0,1)
        @test 2.0 .* A isa BandedMatrix
        @test A .* 2.0 isa BandedMatrix
        @test Eye(∞) * A isa BandedMatrix
        @test A * Eye(∞) isa BandedMatrix
        b = 1:∞
        @test BroadcastStyle(typeof(b)) isa LazyArrayStyle{1}
        @test BroadcastStyle(typeof(A)) isa BandedStyle
        @test BroadcastStyle(LazyArrayStyle{1}(), BandedStyle()) isa LazyArrayStyle{2}
        @test BroadcastStyle(LazyArrayStyle{2}(), BandedStyle()) isa LazyArrayStyle{2}
        @test bandwidths(b .* A) == (0,1)

        @test colsupport(b.*A, 1) == 1:1
        @test Base.replace_in_print_matrix(b.*A, 2,1,"0.0") == " ⋅ "
        @test bandwidths(A .* b) == (0,1)
        @test A .* b' isa BroadcastArray
        @test bandwidths(A .* b') == bandwidths(A .* b')
        @test colsupport(A .* b', 3) == 2:3

        A = _BandedMatrix(Ones{Int}(1,∞),∞,0,0)'
        B = _BandedMatrix((-2:-2:-∞)', ∞,-1,1)
        C = Diagonal( 2 ./ (1:2:∞))
        @test A*(B*C) isa MulMatrix
        @test bandwidths(A*(B*C)) == (-1,1)
    end
    
    @testset "Triangle OP recurrences" begin
        # mortar((n -> 1:n).(1:∞))
    end
    # Multivariate OPs Corollary (3)
    # n = 5
    # BlockTridiagonal(Zeros.(1:∞,2:∞),
    #         (n -> Diagonal(((n+2).+(0:n)))/ (2n + 2)).(0:∞),
    #         Zeros.(2:∞,1:∞))
end

include("test_hessenbergq.jl")
include("test_infql.jl")
include("test_infqr.jl")