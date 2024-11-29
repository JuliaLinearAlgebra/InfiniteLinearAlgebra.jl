using InfiniteLinearAlgebra, BlockBandedMatrices, BlockArrays, BandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test,
    MatrixFactorizations, ArrayLayouts, LinearAlgebra, Random, LazyBandedMatrices, StaticArrays
import InfiniteLinearAlgebra: qltail, toeptail, tailiterate, tailiterate!, tail_de, ql_X!,
    InfToeplitz, PertToeplitz, TriToeplitz, InfBandedMatrix, InfBandCartesianIndices,
    rightasymptotics, QLHessenberg, ConstRows, PertConstRows, chop, chop!, pad,
    BandedToeplitzLayout, PertToeplitzLayout, TridiagonalToeplitzLayout, BidiagonalToeplitzLayout,
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
    X = Matrix(reshape(1:6, 3, 2))
    P = pad(X, oneto(3), oneto(∞))
    @test P isa PaddedArray
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


@testset "∞-block arrays" begin
    @testset "fixed block size" begin
        k = Base.OneTo.(oneto(∞))
        n = Fill.(oneto(∞), oneto(∞))
        @test broadcast(length, k) ≡ map(length, k) ≡ OneToInf()
        @test broadcast(length, n) ≡ map(length, n) ≡ OneToInf()

        b = mortar(Fill([1, 2], ∞))
        @test blockaxes(b, 1) ≡ Block.(OneToInf())
        @test b[Block(5)] == [1, 2]
        @test b[Block.(2:∞)][Block.(2:10)] == b[Block.(3:11)]
        @test exp.(b)[Block.(2:∞)][Block.(2:10)] == exp.(b[Block.(3:11)])

        @test blockedrange(Vcat(2, Fill(3, ∞))) isa BlockedOneTo{<:Any,<:InfiniteArrays.InfStepRange}

        c = BlockedArray(1:∞, Vcat(2, Fill(3, ∞)))
        @test c[Block.(2:∞)][Block.(2:10)] == c[Block.(3:11)]

        @test length(axes(b, 1)) ≡ ℵ₀
        @test last(axes(b, 1)) ≡ ℵ₀
        @test Base.BroadcastStyle(typeof(b)) isa LazyArrayStyle{1}

        @test unitblocks(oneto(∞)) ≡ blockedrange(Ones{Int}(∞))
        @test unitblocks(2:∞) == 2:∞

        @test unitblocks(oneto(∞))[Block.(2:∞)] == 2:∞
    end

    @testset "1:∞ blocks" begin
        a = blockedrange(oneto(∞))
        @test axes(a, 1) == a
        o = Ones((a,))
        @test Base.BroadcastStyle(typeof(a)) isa LazyArrayStyle{1}
        b = exp.(a)
        @test axes(b, 1) == a
        @test o .* b isa typeof(b)
        @test b .* o isa typeof(b)
    end

    @testset "padded" begin
        c = BlockedArray([1; zeros(∞)], Vcat(2, Fill(3, ∞)))
        @test c + c isa BlockedVector
    end

    @testset "concat" begin
        a = unitblocks(1:∞)
        b = exp.(a)
        c = BlockBroadcastArray(vcat, a, b)
        @test length(c) == ∞
        @test blocksize(c) == (∞,)
        @test c[Block(5)] == [a[5], b[5]]

        A = unitblocks(BandedMatrix(0 => 1:∞, 1 => Fill(2.0, ∞), -1 => Fill(3.0, ∞)))
        B = BlockBroadcastArray(hvcat, 2, A, Zeros(axes(A)), Zeros(axes(A)), A)
        @test B[Block(3, 3)] == [A[3, 3] 0; 0 A[3, 3]]
        @test B[Block(3, 4)] == [A[3, 4] 0; 0 A[3, 4]]
        @test B[Block(3, 5)] == [A[3, 5] 0; 0 A[3, 5]]
        @test blockbandwidths(B) == (1, 1)
        @test subblockbandwidths(B) == (0, 0)
        @test B[Block.(1:10), Block.(1:10)] isa BlockSkylineMatrix

        C = BlockBroadcastArray(hvcat, 2, A, A, A, A)
        @test C[Block(3, 3)] == fill(A[3, 3], 2, 2)
        @test C[Block(3, 4)] == fill(A[3, 4], 2, 2)
        @test C[Block(3, 5)] == fill(A[3, 5], 2, 2)
        @test blockbandwidths(C) == (1, 1)
        @test subblockbandwidths(C) == (1, 1)
        @test B[Block.(1:10), Block.(1:10)] isa BlockSkylineMatrix
    end

    @testset "DiagTrav" begin
        C = zeros(∞,∞);
        C[1:3,1:4] .= [1 2 3 4; 5 6 7 8; 9 10 11 12]
        A = DiagTrav(C)
        @test blockcolsupport(A) == Block.(1:6)
        @test A[Block.(1:7)] == [1; 5; 2; 9; 6; 3; 0; 10; 7; 4; 0; 0; 11; 8; 0; 0; 0; 0; 12; zeros(9)]

        C = zeros(∞,4);
        C[1:3,1:4] .= [1 2 3 4; 5 6 7 8; 9 10 11 12]
        A = DiagTrav(C)
        @test A[Block.(1:7)] == [1; 5; 2; 9; 6; 3; 0; 10; 7; 4; 0; 0; 11; 8; 0; 0; 0; 12; zeros(4)]

        C = zeros(3,∞);
        C[1:3,1:4] .= [1 2 3 4; 5 6 7 8; 9 10 11 12]
        A = DiagTrav(C)
        @test A[Block.(1:7)] == [1; 5; 2; 9; 6; 3; 10; 7; 4; 11; 8; 0; 12; zeros(5)]
    end

    @testset "KronTrav" begin
        Δ = BandedMatrix(1 => Ones(∞), -1 => Ones(∞)) / 2
        A = KronTrav(Δ - 2I, Eye(∞))
        @test axes(A, 1) isa InfiniteLinearAlgebra.OneToInfBlocks
        V = view(A, Block.(Base.OneTo(3)), Block.(Base.OneTo(3)))

        @test MemoryLayout(A) isa InfiniteLinearAlgebra.InfKronTravBandedBlockBandedLayout
        @test MemoryLayout(V) isa LazyBandedMatrices.KronTravBandedBlockBandedLayout

        @test A[Block.(Base.OneTo(3)), Block.(Base.OneTo(3))] isa KronTrav

        u = A * [1; zeros(∞)]
        @test u[1:3] == A[1:3, 1]
        @test bandwidths(view(A, Block(1, 1))) == (1, 1)

        @test A*A isa KronTrav
        @test (A*A)[Block.(Base.OneTo(3)), Block.(Base.OneTo(3))] ≈ A[Block.(1:3), Block.(1:4)]A[Block.(1:4), Block.(1:3)]
    end

    @testset "triangle recurrences" begin
        @testset "n and k" begin
            n = mortar(Fill.(oneto(∞), oneto(∞)))
            k = mortar(Base.OneTo.(oneto(∞)))

            @test n[Block(5)] ≡ layout_getindex(n, Block(5)) ≡ view(n, Block(5)) ≡ Fill(5, 5)
            @test k[Block(5)] ≡ layout_getindex(k, Block(5)) ≡ view(k, Block(5)) ≡ Base.OneTo(5)
            @test Base.BroadcastStyle(typeof(n)) isa LazyArrays.LazyArrayStyle{1}
            @test Base.BroadcastStyle(typeof(k)) isa LazyArrays.LazyArrayStyle{1}

            N = 1000
            v = view(n, Block.(Base.OneTo(N)))
            @test view(v, Block(2)) ≡ Fill(2, 2)
            @test axes(v) isa Tuple{BlockedOneTo{Int,ArrayLayouts.RangeCumsum{Int64,Base.OneTo{Int64}}}}
            @test @allocated(axes(v)) ≤ 40

            dest = BlockedArray{Float64}(undef, axes(v))
            @test copyto!(dest, v) == v
            @test @allocated(copyto!(dest, v)) ≤ 40

            v = view(k, Block.(Base.OneTo(N)))
            @test view(v, Block(2)) ≡ Base.OneTo(2)
            @test axes(v) isa Tuple{BlockedOneTo{Int,ArrayLayouts.RangeCumsum{Int64,Base.OneTo{Int64}}}}
            @test @allocated(axes(v)) ≤ 40
            @test copyto!(dest, v) == v

            @testset "stack overflow" begin
                i = Base.to_indices(k, (Block.(2:∞),))[1].indices
                last(i)
            end

            v = view(k, Block.(2:∞))
            @test Base.BroadcastStyle(typeof(v)) isa LazyArrayStyle{1}
            @test v[Block(1)] == 1:2
            @test v[Block(1)] ≡ k[Block(2)] ≡ Base.OneTo(2)

            @test axes(n, 1) isa BlockedOneTo{Int,ArrayLayouts.RangeCumsum{Int64,OneToInf{Int64}}}
        end

        @testset "BlockHcat copyto!" begin
            n = mortar(Fill.(oneto(∞), oneto(∞)))
            k = mortar(Base.OneTo.(oneto(∞)))

            a = b = c = 0.0
            dat = BlockHcat(
                BroadcastArray((n, k, b, bc1) -> (k + b - 1) * (n + k + bc1) / (2k + bc1), n, k, b, b + c - 1),
                BroadcastArray((n, k, abc, bc, bc1) -> (n + k + abc) * (k + bc) / (2k + bc1), n, k, a + b + c, b + c, b + c - 1)
            )
            N = 1000
            KR = Block.(Base.OneTo(N))
            V = view(dat, Block.(Base.OneTo(N)), :)
            @test MemoryLayout(V) isa LazyArrays.ApplyLayout{typeof(hcat)}
            @test BlockedArray(V)[Block.(1:5), :] == dat[Block.(1:5), :]
            V = view(dat', :, Block.(Base.OneTo(N)))
            @test MemoryLayout(V) isa LazyArrays.ApplyLayout{typeof(vcat)}
            a = dat.arrays[1]'
            N = 100
            KR = Block.(Base.OneTo(N))
            v = view(a, :, KR)
            @time r = BlockedArray(v)
            @test v == r
        end

        @testset "BlockBanded" begin
            a = b = c = 0.0
            n = mortar(Fill.(oneto(∞), oneto(∞)))
            k = mortar(Base.OneTo.(oneto(∞)))
            Dy = BlockBandedMatrices._BandedBlockBandedMatrix((k .+ (b + c))', axes(k, 1), (-1, 1), (-1, 1))
            N = 100
            @test Dy[Block.(1:N), Block.(1:N)] == BlockBandedMatrices._BandedBlockBandedMatrix((k.+(b+c))[Block.(1:N)]', axes(k, 1)[Block.(1:N)], (-1, 1), (-1, 1))
            @test colsupport(Dy, axes(Dy,2)) == 1:∞
            @test rowsupport(Dy, axes(Dy,1)) == 2:∞
        end

        @testset "Symmetric" begin
            k = mortar(Base.OneTo.(oneto(∞)))
            n = mortar(Fill.(oneto(∞), oneto(∞)))

            dat = BlockHcat(
                BlockBroadcastArray(hcat, float.(k), Zeros((axes(n, 1),)), float.(n)),
                Zeros((axes(n, 1), Base.OneTo(3))),
                Zeros((axes(n, 1), Base.OneTo(3))))
            M = BlockBandedMatrices._BandedBlockBandedMatrix(dat', axes(k, 1), (1, 1), (1, 1))
            Ms = Symmetric(M)
            @test blockbandwidths(M) == (1, 1)
            @test blockbandwidths(Ms) == (1, 1)
            @test Ms[Block.(1:5), Block.(1:5)] == Symmetric(M[Block.(1:5), Block.(1:5)])
            @test Ms[Block.(1:5), Block.(1:5)] isa BandedBlockBandedMatrix

            b = [ones(10); zeros(∞)]
            @test (Ms * b)[Block.(1:6)] == Ms[Block.(1:6), Block.(1:4)]*ones(10)
            @test ((Ms * Ms) * b)[Block.(1:6)] == (Ms * (Ms * b))[Block.(1:6)]
            @test ((Ms + Ms) * b)[Block.(1:6)] == (2*(Ms * b))[Block.(1:6)]

            dat = BlockBroadcastArray(hcat, float.(k), Zeros((axes(n, 1),)), float.(n))
            M = BlockBandedMatrices._BandedBlockBandedMatrix(dat', axes(k, 1), (-1, 1), (1, 1))
            Ms = Symmetric(M)
            @test Symmetric((M+M)[Block.(1:10), Block.(1:10)]) == (Ms+Ms)[Block.(1:10), Block.(1:10)]
        end
    end

    @testset "blockdiag" begin
        D = Diagonal(mortar(Fill.((-(0:∞) - (0:∞) .^ 2), 1:2:∞)))
        x = [randn(5); zeros(∞)]
        x̃ = BlockedArray(x, (axes(D, 1),))
        @test (D*x)[1:10] == (D*x̃)[1:10]
    end

    @testset "sortedunion" begin
        a = cumsum(1:2:∞)
        @test BlockArrays.sortedunion(a, a) ≡ a
        @test BlockArrays.sortedunion([∞], a) ≡ BlockArrays.sortedunion(a, [∞]) ≡ a
        @test BlockArrays.sortedunion([∞], [∞]) == [∞]

        b = Vcat([1, 2], 3:∞)
        c = Vcat(1, 3:∞)
        @test BlockArrays.sortedunion(b, b) ≡ b
        @test BlockArrays.sortedunion(c, c) ≡ c
    end
end



@testset "Algebra" begin
    @testset "BlockTridiagonal" begin
        A = BlockTridiagonal(Vcat([fill(1.0, 2, 1), Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2), Matrix(1.0I, 2, 2)], Fill(Matrix(1.0I, 2, 2), ∞)),
            Vcat([zeros(1, 1)], Fill(zeros(2, 2), ∞)),
            Vcat([fill(1.0, 1, 2), Matrix(1.0I, 2, 2)], Fill(Matrix(1.0I, 2, 2), ∞)))

        @test A isa InfiniteLinearAlgebra.BlockTriPertToeplitz
        @test isblockbanded(A)

        @test A[Block.(1:2), Block(1)] == A[1:3, 1:1] == reshape([0.0, 1.0, 1.0], 3, 1)

        @test BlockBandedMatrix(A)[1:100, 1:100] == BlockBandedMatrix(A, (2, 1))[1:100, 1:100] == BlockBandedMatrix(A, (1, 1))[1:100, 1:100] == A[1:100, 1:100]

        @test (A-I)[1:100, 1:100] == A[1:100, 1:100] - I
        @test (A+I)[1:100, 1:100] == A[1:100, 1:100] + I
        @test (I+A)[1:100, 1:100] == I + A[1:100, 1:100]
        @test (I-A)[1:100, 1:100] == I - A[1:100, 1:100]

        @test (A-im*I)[1:100, 1:100] == A[1:100, 1:100] - im * I
        @test (A+im*I)[1:100, 1:100] == A[1:100, 1:100] + im * I
        @test (im*I+A)[1:100, 1:100] == im * I + A[1:100, 1:100]
        @test (im*I-A)[1:100, 1:100] == im * I - A[1:100, 1:100]

        T = mortar(LazyBandedMatrices.Tridiagonal(Fill([1 2; 3 4], ∞), Fill([1 2; 3 4], ∞), Fill([1 2; 3 4], ∞)));
        #TODO: copy BlockBidiagonal code from BlockBandedMatrices to LazyBandedMatrices
        @test T[Block(2, 2)] == [1 2; 3 4]
        @test_broken T[Block(1, 3)] == Zeros(2, 2)
    end

    @testset "BlockBidiagonal" begin
        B = mortar(LazyBandedMatrices.Bidiagonal(Fill([1 2; 3 4], ∞), Fill([1 2; 3 4], ∞), :U));
        #TODO: copy BlockBidiagonal code from BlockBandedMatrices to LazyBandedMatrices
        @test B[Block(2, 3)] == [1 2; 3 4]
        @test_broken B[Block(1, 3)] == Zeros(2, 2)
    end

    @testset "Triangle OP recurrences" begin
        k = mortar(Base.OneTo.(1:∞))
        n = mortar(Fill.(1:∞, 1:∞))
        @test k[Block.(2:3)] isa BlockArray
        @test n[Block.(2:3)] isa BlockArray
        @test k[Block.(2:3)] == [1, 2, 1, 2, 3]
        @test n[Block.(2:3)] == [2, 2, 3, 3, 3]
        @test blocksize(BroadcastVector(exp, k)) == (ℵ₀,)
        @test BroadcastVector(exp, k)[Block.(2:3)] == exp.([1, 2, 1, 2, 3])
        # BroadcastVector(+,k,n)
    end
    # Multivariate OPs Corollary (3)
    # n = 5
    # BlockTridiagonal(Zeros.(1:∞,2:∞),
    #         (n -> Diagonal(((n+2).+(0:n)))/ (2n + 2)).(0:∞),
    #         Zeros.(2:∞,1:∞))

    @testset "KronTrav" begin
        Δ = BandedMatrix(1 => Ones(∞) / 2, -1 => Ones(∞))
        A = KronTrav(Δ, Eye(∞))
        @test A[Block(100, 101)] isa BandedMatrix
        @test A[Block(100, 100)] isa BandedMatrix
        @test A[Block.(1:5), Block.(1:5)] isa BandedBlockBandedMatrix
        B = KronTrav(Eye(∞), Δ)
        @test B[Block(100, 101)] isa BandedMatrix
        @test B[Block(100, 100)] isa BandedMatrix
        V = view(A + B, Block.(1:5), Block.(1:5))
        @test MemoryLayout(typeof(V)) isa BroadcastBandedBlockBandedLayout{typeof(+)}
        @test arguments(V) == (A[Block.(1:5), Block.(1:5)], B[Block.(1:5), Block.(1:5)])
        @test (A+B)[Block.(1:5), Block.(1:5)] == A[Block.(1:5), Block.(1:5)] + B[Block.(1:5), Block.(1:5)]

        @test blockbandwidths(A + B) == (1, 1)
        @test blockbandwidths(2A) == (1, 1)
        @test blockbandwidths(2 * (A + B)) == (1, 1)

        @test subblockbandwidths(A + B) == (1, 1)
        @test subblockbandwidths(2A) == (1, 1)
        @test subblockbandwidths(2 * (A + B)) == (1, 1)
    end
end

@testset "findblock at +∞, HarmonicOrthogonalPolynomials#88" begin
    @test findblock(blockedrange(1:2:∞), RealInfinity()) == Block(ℵ₀)
end

include("test_hessenbergq.jl")
include("test_infql.jl")
include("test_infqr.jl")
include("test_inful.jl")
include("test_infcholesky.jl")
include("test_periodic.jl")
include("test_infreversecholesky.jl")
include("test_bidiagonalconjugation.jl")