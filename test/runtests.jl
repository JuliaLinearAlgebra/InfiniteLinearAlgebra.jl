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

@testset "PertTriToeplitz QL" begin
    A = Tridiagonal(Vcat(Float64[], Fill(2.0,∞)), 
                    Vcat(Float64[2.0], Fill(0.0,∞)), 
                    Vcat(Float64[], Fill(0.5,∞)))
    for λ in (-2.1-0.01im,-2.1+0.01im,-2.1+eps()im,-2.1-eps()im,-2.1+0.0im,-2.1-0.0im,-1.0+im,-3.0+im,-3.0-im)
        Q, L = ql(A - λ*I)
        @test Q[1:10,1:12]*L[1:12,1:10] ≈ A[1:10,1:10] - λ*I
    end
end

@testset "Pert Hessenberg Toeplitz" begin
    a = [1,2,5,0.5]
    Random.seed!(0)
    A = _BandedMatrix(Hcat(randn(4,2), reverse(a) * Ones(1,∞)), ∞, 2, 1)
    @test A isa PertToeplitz
    @test BandedMatrix(A, (3,1))[1:10,1:10] == A[1:10,1:10]
    Q,L = ql(A)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ A[1:10,1:10]

    a = [0.1,1,2,3,0.5]
    A = _BandedMatrix(Hcat([0.5 0.5; -1 3; 2 2; 1 1; 0.1 0.1], reverse(a) * Ones(1,∞)), ∞, 3, 1)
    @test A isa PertToeplitz
    @test BandedMatrix(A, (3,1))[1:10,1:10] == A[1:10,1:10]

    B = BandedMatrix(A, (3,1))
    @test B[1:10,1:10] == A[1:10,1:10]
    Q,L = ql(A)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ A[1:10,1:10]

    A = BandedMatrix(-2 => Vcat([1], Fill(1,∞)), 
                      0 => Vcat([0.0], Fill(1/2,∞)),
                      1 => Vcat([1/4], Fill(1/4,∞)))
    Q,L = ql(A)                      
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ A[1:10,1:10]
    a = [-0.1,0.2,0.3]
    A = BandedMatrix(-2 => Vcat([1], Fill(1,∞)),  0 => Vcat(a, Fill(0,∞)), 1 => Vcat([1/4], Fill(1/4,∞)))
    λ = 0.5+0.1im

    B = BandedMatrix(A-λ*I, (3,1))
    T = toeptail(B) 
    Q,L = ql(T)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]

    Q,L = ql(A-λ*I)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ (A-λ*I)[1:10,1:10]

    a = [-0.1,0.2,0.3]
    A = BandedMatrix(-2 => Vcat([1], Fill(1,∞)),  0 => Vcat(a, Fill(0,∞)), 1 => Vcat([1/4], Fill(1/4,∞)))
    for λ in (0.1+0im,0.1-0im, 3.0, 3.0+1im, 3.0-im, -0.1+0im, -0.1-0im)
        Q, L = ql(A-λ*I)
        @test Q[1:10,1:11]*L[1:11,1:10] ≈ (A-λ*I)[1:10,1:10]
    end
end

@testset "Pert faux-periodic QL" begin
    a = [0.5794879759059747 + 0.0im,0.538107104952824 - 0.951620830938543im,-0.19352887774167749 - 0.3738926065520737im,0.4314153362874331,0.0]
    T = _BandedMatrix(a*Ones{ComplexF64}(1,∞), ∞, 3,1)
    Q,L = ql(T)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
    Qn,Ln = ql(T[1:1001,1:1001])
    @test Qn[1:10,1:10] * diagm(0=>[-1; (-1).^(1:9)]) ≈ Q[1:10,1:10]
    @test diagm(0=>[-1; (-1).^(1:9)]) * Ln[1:10,1:10] ≈ L[1:10,1:10]

    f =  [0.0+0.0im        0.522787+0.0im     ; 0.59647-1.05483im    0.538107-0.951621im; -0.193529-0.373893im  -0.193529-0.373893im;
               0.431415+0.0im        0.431415+0.0im; 0.0+0.0im             0.0+0.0im]
    A = _BandedMatrix(Hcat(f, a*Ones{ComplexF64}(1,∞)), ∞, 3,1)
    Q,L = ql(A)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ A[1:10,1:10]
    Qn,Ln = ql(A[1:1000,1:1000])
    @test Qn[1:10,1:10] * diagm(0 => [Ones(5); -(-1).^(1:5)]) ≈ Q[1:10,1:10]
    @test diagm(0 => [Ones(5); -(-1).^(1:5)]) * Ln[1:10,1:10] ≈ L[1:10,1:10]
end
