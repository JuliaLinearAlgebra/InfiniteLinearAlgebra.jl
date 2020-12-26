using InfiniteLinearAlgebra, BlockBandedMatrices, BlockArrays, BandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test,
        MatrixFactorizations, ArrayLayouts, LinearAlgebra, Random, LazyBandedMatrices
import InfiniteLinearAlgebra: qltail, toeptail, tailiterate , tailiterate!, tail_de, ql_X!,
                    InfToeplitz, PertToeplitz, TriToeplitz, InfBandedMatrix, InfBandCartesianIndices,
                    rightasymptotics, QLHessenberg, ConstRows, PertConstRows, BandedToeplitzLayout, PertToeplitzLayout
import Base: BroadcastStyle
import BlockArrays: _BlockArray
import BlockBandedMatrices: isblockbanded, _BlockBandedMatrix
import MatrixFactorizations: QLPackedQ
import BandedMatrices: bandeddata, _BandedMatrix, BandedStyle
import LazyArrays: colsupport, ApplyStyle, MemoryLayout, ApplyLayout, LazyArrayStyle, arguments
import InfiniteArrays: OneToInf
import LazyBandedMatrices: BroadcastBandedBlockBandedLayout, BroadcastBandedLayout


@testset "∞-banded" begin
    D = Diagonal(Fill(2,∞))

    B = D[1:∞,2:∞]
    @test B isa BandedMatrix
    @test B[1:10,1:10] == diagm(-1 => Fill(2,9))
    @test B[1:∞,2:∞] isa BandedMatrix

    A = BandedMatrix(0 => 1:∞, 1=> Fill(2.0,∞), -1 => Fill(3.0,∞))
    x = [1; 2; zeros(∞)]
    @test A*x isa Vcat
    @test (A*x)[1:10] == A[1:10,1:10]*x[1:10]

    @test InfBandCartesianIndices(0)[1:5] == CartesianIndex.(1:5,1:5)
    @test InfBandCartesianIndices(1)[1:5] == CartesianIndex.(1:5,2:6)
    @test InfBandCartesianIndices(-1)[1:5] == CartesianIndex.(2:6,1:5)

    @test D[band(0)] ≡ Fill(2,∞)
    @test D[band(1)] ≡ Fill(0,∞)
    @test A[band(0)][2:10] == 2:10
end

@testset "∞-block arrays" begin
    @testset "fixed block size" begin
        k = Base.OneTo.(Base.OneTo(∞))
        n = Fill.(Base.OneTo(∞),Base.OneTo(∞))
        @test broadcast(length,k) ≡ map(length,k) ≡ OneToInf()
        @test broadcast(length,n) ≡ map(length,n) ≡ OneToInf()
        b = mortar(Fill([1,2],∞))
        @test blockaxes(b,1) ≡ Block.(OneToInf())
        @test b[Block(5)] == [1,2]
        @test length(axes(b,1)) ≡ last(axes(b,1)) ≡ ∞
    end

    @testset "1:∞ blocks" begin
        a = blockedrange(Base.OneTo(∞))
        @test axes(a,1) == a
        o = Ones((a,))
        @test Base.BroadcastStyle(typeof(a)) isa LazyArrayStyle{1}
        b = exp.(a)
        @test axes(b,1) == a
        @test o .* b isa typeof(b)
        @test b .* o isa typeof(b)
    end

    @testset "concat" begin
        unitblocks(a::AbstractArray) = PseudoBlockArray(a, Ones{Int}.(axes(a))...)
        a = unitblocks(1:∞)
        b = exp.(a)
        c = BlockBroadcastArray(vcat,a,b)
        @test length(c) == ∞
        @test blocksize(c) == (∞,)
        @test c[Block(5)] == [a[5],b[5]]

        A = unitblocks(BandedMatrix(0 => 1:∞, 1=> Fill(2.0,∞), -1 => Fill(3.0,∞)))
        B = BlockBroadcastArray(hvcat, 2, A, Zeros(axes(A)), Zeros(axes(A)), A)
        @test B[Block(3,3)] == [A[3,3] 0; 0 A[3,3]]
        @test B[Block(3,4)] == [A[3,4] 0; 0 A[3,4]]
        @test B[Block(3,5)] == [A[3,5] 0; 0 A[3,5]]
        @test blockbandwidths(B) == (1,1)
        @test subblockbandwidths(B) == (2,2)
        @test B[Block.(1:10), Block.(1:10)] isa BlockSkylineMatrix
    end

    @testset "KronTrav" begin
        Δ = BandedMatrix(1 => Ones(∞), -1 => Ones(∞))/2
        A = KronTrav(Δ - 2I, Eye(∞))
        @test axes(A,1) isa InfiniteLinearAlgebra.OneToInfBlocks
        V = view(A, Block.(Base.OneTo(3)), Block.(Base.OneTo(3)))
        @test MemoryLayout(V) isa BlockBandedMatrices.BandedBlockBandedLayout

        u = A * [1; zeros(∞)]
        @test u[1:3] == A[1:3,1]
        @test bandwidths(view(A, Block(1,1))) == (1,1)
    end

    @testset "triangle recurrences" begin
        n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
        k = mortar(Base.OneTo.(Base.OneTo(∞)))

        @test n[Block(5)] ≡ layout_getindex(n, Block(5)) ≡ Fill(5,5)
        @test Base.BroadcastStyle(typeof(n)) isa LazyArrays.LazyArrayStyle{1}
        @test Base.BroadcastStyle(typeof(k)) isa LazyArrays.LazyArrayStyle{1}

        N = 1000
        v = view(n,Block.(Base.OneTo(N)))
        @test axes(v) isa Tuple{BlockedUnitRange{InfiniteArrays.RangeCumsum{Int64,Base.OneTo{Int64}}}}
        @test @allocated(axes(v)) ≤ 40

        dest = PseudoBlockArray{Float64}(undef, axes(v))
        @test copyto!(dest, v) == v
        @test @allocated(copyto!(dest, v)) ≤ 40

        v = view(k,Block.(Base.OneTo(N)))
        @test axes(v) isa Tuple{BlockedUnitRange{InfiniteArrays.RangeCumsum{Int64,Base.OneTo{Int64}}}}
        @test @allocated(axes(v)) ≤ 40
        @test copyto!(dest, v) == v


        @testset "BlockHcat copyto!" begin
            n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
            k = mortar(Base.OneTo.(Base.OneTo(∞)))

            a = b = c = 0.0
            dat = BlockHcat(
                BroadcastArray((n,k,b,bc1) -> (k + b-1) * (n + k + bc1) / (2k + bc1), n, k, b, b+c-1),
                BroadcastArray((n,k,abc,bc,bc1) -> (n + k + abc) * (k + bc) / (2k + bc1), n, k, a+b+c,b+c,b+c-1)
                )
            N = 1000
            KR = Block.(Base.OneTo(N))
            V = view(dat,Block.(Base.OneTo(N)),:)
            @test MemoryLayout(V) isa LazyArrays.ApplyLayout{typeof(hcat)}
            @test PseudoBlockArray(V)[Block.(1:5),:] == dat[Block.(1:5),:]
            V = view(dat',:,Block.(Base.OneTo(N)))
            @test MemoryLayout(V) isa LazyArrays.ApplyLayout{typeof(vcat)}
            a = dat.arrays[1]'
            N = 100
            KR = Block.(Base.OneTo(N))
            v = view(a,:,KR); @time r = PseudoBlockArray(v)

            b = LazyArrays._broadcastarray2broadcasted(v).args[1]'
            r = BlockArray(b)
            @time copyto!(r, b);
            
            @time ArrayLayouts._copyto!(r, b);
            @ent ArrayLayouts.copyto!(r, view(n,KR));
            @time n[KR];
            MemoryLayout(b)

            @ent ArrayLayouts._copyto!(r, v)
            @time ArrayLayouts._copyto!(r', v')
            @time PseudoBlockArray(V)
            @ent (dat.arrays[1]')[:,Block.(1:N)];
            N = 10
            MemoryLayout(dat.arrays[1]')
        end
        
        import ArrayLayouts: MemoryLayout, _copyto!
        struct UnitRangeLayout <: MemoryLayout end
        MemoryLayout(::Type{<:AbstractUnitRange}) = UnitRangeLayout()
        @test MemoryLayout(view(k,Block(5))) isa UnitRangeLayout
        first(view(k,Block(5)))
        function _copyto!(_, ::UnitRangeLayout, dest::AbstractVector, src::AbstractVector)
            @inbounds for k in axes(dest,1)
                dest[k] = k
            end
            dest
        end

        @ent dest .= view(cos.(n), Block.(1:N))

        v = view(exp.(n), Block.(1:N))
        @ent Base.BroadcastStyle(typeof(v))
        @time copyto!(dest, Base.broadcasted(BlockArrays.BlockStyle{1}(), cos, view(k,Block.(Base.OneTo(N)))));
        @time copyto!(dest, Base.broadcasted(BlockArrays.BlockStyle{1}(), cos, view(n,Block.(Base.OneTo(N)))));

        @time copyto!(dest, Base.broadcasted(BlockArrays.BlockStyle{1}(), /, view(k,Block.(Base.OneTo(N))), view(n,Block.(Base.OneTo(N)))));

        v = view(k,Block.(Base.OneTo(N)))
        w = view(n,Block.(Base.OneTo(N)))

        @time atan.(v, w)

        @test Base.BroadcastStyle(typeof(view(k,Block.(1:N)))) isa BlockArrays.BlockStyle{1}

        v = view(k,Block.(2:∞))
        @test Base.BroadcastStyle(typeof(v)) isa LazyArrayStyle{1}
        @test v[Block(1)] == 1:2
        @test v[Block(1)] ≡ k[Block(2)] ≡ Base.OneTo(2)




        @time k[Block.(Base.OneTo(N))]
        MemoryLayout()
        exp.(view(n, Block(5)))

        @test axes(n,1) isa BlockedUnitRange{InfiniteArrays.RangeCumsum{Int64,OneToInf{Int64}}}


        a = b = c =0.0
        d = (k .+ (c-1)) .* ( k .- n .- 1 ) ./ (2k .+ (b+c-1))
        N = 1000
        @time d[Block.(Base.OneTo(N))];
        v = view(d,Block.(Base.OneTo(N)));
        dest = PseudoBlockArray{Float64}(undef, axes(v));
        @time copyto!(dest, v);

        @time (d')[1,Block.(Base.OneTo(N))];
        @ent (d')[:,Block.(Base.OneTo(N))];
        @ent layout_getindex(d',:,Block.(Base.OneTo(N)));

        copyto!(dest', view(d',:,Block.(Base.OneTo(N))))




        v = view(k',:,Block.(Base.OneTo(N)))'
        MemoryLayout(v.parent.parent)

        v = view(d',:,Block.(Base.OneTo(N)))'
        dest = PseudoBlockArray{Float64}(undef, axes(v))
        @ent ArrayLayouts._copyto!(dest, v);



        b = ß

        v = view(b,Block.(1:N))
        MemoryLayout(v)

        a = b = c = 0.0
        dat = BlockVcat(
            ((k .+ (c-1)) .* ( k .- n .- 1 ) ./ (2k .+ (b+c-1)))',
            (k .* (k .- n .- a) ./ (2k .+ (b+c-1)))'
            )
        @testset "BlockHcat" begin
            a = b = c = 0.0
            n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
            k = mortar(Base.OneTo.(Base.OneTo(∞)))
            D = BlockHcat(((k .+ (c-1)) .* ( k .- n .- 1 ) ./ (2k .+ (b+c-1))),
                            k .* (k .- n .- a) ./ (2k .+ (b+c-1)))


            N = 1000
            V = view(D,Block.(Base.OneTo(N)), :)
            @test axes(V) ≡ (BlockArrays._BlockedUnitRange(1, InfiniteArrays.RangeCumsum(Base.OneTo(N))), blockedrange(SVector(1,1)))
            @test @allocated(axes(V)) ≤ 50
            dest = PseudoBlockArray{Float64}(undef, axes(V))
            @ent copyto!(dest, V);

            w = view(D.arrays[1],Block.(Base.OneTo(N)));
            @time copyto!(view(dest,Block(N),1), view(w,Block(N)));
            bc = LazyArrays._broadcastarray2broadcasted(w)
            @test axes(bc)[1] ≡ axes(w,1)
            @test @allocated(axes(bc)) ≤ 40
            dest = PseudoBlockArray{Float64}(undef, axes(w));
            @time copyto!(dest, v);
            @time copyto!(dest, w);

            dest = PseudoBlockArray{Float64}(undef, axes(V));
            @time copyto!(view(dest,:,1), w);

            @time copyto!(view(dest,:,1), w);
            @ent copyto!(view(dest,:,1), w)

            dest = PseudoBlockArray{Float64}(undef, axes(w))
            @time copyto!(dest,w);

            u = view(D.arrays[2],Block.(Base.OneTo(N)));
            @time copyto!(view(dest,:,2), u);

            # u = view(k .* (k .- n .- a) ./ (2k .+ (b+c-1)), Block.(Base.OneTo(N)))
            bc1 = LazyArrays._broadcastarray2broadcasted((k .+ 0) .* (k .- n .- a) ./ (2k .+ (b+c-1)))
            bc2 = LazyArrays._broadcastarray2broadcasted((k) .* (k .- n .- a) ./ (2k .+ (b+c-1)))


            axes(bc2)[1]
            axes(bc1)[1]


            n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
            k = mortar(Base.OneTo.(Base.OneTo(∞)))
            a = b = c = 0.0
            N = 100
            u = view((k .+ 0) .* (k .- n .- a) ./ (2k .+ (b+c-1)), Block.(Base.OneTo(N)))

            @code_warntype arguments(view(u, Block(4)))
            @code_warntype(arguments(u))
            arg = arguments(u)[1]
            @ent MemoryLayout(typeof(parent(arg)))
            @code_native LazyArrays._broadcastarray2broadcasted(MemoryLayout(arg),arg)
            u = view((k) .* (k .- n .- a) ./ (2k .+ (b+c-1)), Block.(Base.OneTo(N)))
            dest = PseudoBlockArray{Float64}(undef, (axes(u,1),blockedrange(SVector(1,1))))
            dest = view(dest,:,2)
            bc = Base.broadcasted(u)
            BS = BlockArrays.BlockStyle{1}
            import BlockArrays: combine_blockaxes
            bc = Base.Broadcast.instantiate(Base.Broadcast.Broadcasted{BS}(bc.f, bc.args, combine_blockaxes.(axes(dest),axes(bc))))
            @code_warntype copyto!(dest, bc)
            @inferred LazyArrays._broadcastarray2broadcasted(u)

            axes(bc1)[1]
            axes(bc2)[1]

            n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
            k = mortar(Base.OneTo.(Base.OneTo(∞)))
            N = 1000
            u = view((k) .* (k) ./ (k), Block.(Base.OneTo(N)))
            @time bc1 = LazyArrays._broadcastarray2broadcasted(view(u, Block(5)));
            @test @allocated(axes(bc1)) ≤ 20
            @time v = view(u, Block(5));
            a = arguments(v)[1]
            @time lay = MemoryLayout(a)
            @inferred LazyArrays._broadcastarray2broadcasted(a)

            a = 0.0
            N
            u = view((k) .* (k .- n .- a) ./ (k), Block.(Base.OneTo(N)));
            using StaticArrays
            

            @time v = view(u, Block(5));
            @code_warntype LazyArrays._broadcastarray2broadcasted(v)
            a = arguments(v)[1]
            @time lay = MemoryLayout(a)
            @code_warntype LazyArrays._broadcastarray2broadcasted(a)
            b = arguments(lay, a)[2]
            @code_warntype LazyArrays._broadcastarray2broadcasted(b)
            lay = MemoryLayout(typeof(b))
            @code_warntype LazyArrays._broadcastarray2broadcasted(lay, b)
            lay = MemoryLayout(a)
            bcs = map(LazyArrays._broadcastarray2broadcasted, arguments(lay, a))
            @time LazyArrays.call(lay, a)
            @time Base.broadcasted(*, bcs)

            @time bc2 = LazyArrays._broadcastarray2broadcasted(v);
            @test @allocated(axes(bc2)) ≤ 20
            @time copyto!(view(dest,:,2), u)
        end

        @testset "BlockBanded" begin
            a = b = c = 0.0
            n = mortar(Fill.(Base.OneTo(∞),Base.OneTo(∞)))
            k = mortar(Base.OneTo.(Base.OneTo(∞)))
            Dy = BlockBandedMatrices._BandedBlockBandedMatrix((k .+ (b+c))', axes(k,1), (-1,1), (-1,1))
            N = 1000; @time Dy[Block.(1:N), Block.(1:N)];
            c = BroadcastVector((k,bc) -> k + bc, k, b+c)
            c = BroadcastVector(+, k, b+c)
            @time c[Block.(1:N)];
            @ent (Dy.data')[Block.(1:N)];

            v = view(Dy.data',Block.(1:N))
            MemoryLayout(v)

            N = 1000;
            k = mortar(Base.OneTo.(Base.OneTo(N)))
            @time k .+ 1


            dat = Hcat(
                BroadcastVector((n,k,a,b,c) -> ((k+b-1) * (n+k+b+c-1)) / (2k+b+c-1), n, k, a, b, c),
                BroadcastVector((n,k,a,b,c) -> (n+k+a+b+c) * (k+b+c) / (2k+b+c-1), n, k, a, b, c)
                )
            Dx = BlockBandedMatrices._BandedBlockBandedMatrix(dat', axes(k,1), (-1,1), (0,1))
        end
    end
end

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

    @testset "InfBanded" begin
        A = _BandedMatrix(Fill(2,4,∞),∞,2,1)
        B = _BandedMatrix(Fill(3,2,∞),∞,-1,2)
        @test mul(A,A) isa PertToeplitz
        @test A*A isa PertToeplitz
        @test (A*A)[1:20,1:20] == A[1:20,1:23]*A[1:23,1:20]
        @test (A*B)[1:20,1:20] == A[1:20,1:23]*B[1:23,1:20]
    end
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
        @test A^2 isa BandedMatrix
        @test (A^2)[1:10,1:10] == (A*A)[1:10,1:10] == (A[1:100,1:100]^2)[1:10,1:10]
        @test A^3 isa ApplyMatrix{<:Any,typeof(*)}
        @test (A^3)[1:10,1:10] == (A*A*A)[1:10,1:10]  == ((A*A)*A)[1:10,1:10]  == (A*(A*A))[1:10,1:10] == (A[1:100,1:100]^3)[1:10,1:10]

        @testset "∞ x finite" begin
            A = BandedMatrix(1 => 1:∞) + BandedMatrix(-1 => Fill(2,∞))
            B = _BandedMatrix(randn(3,5), ∞, 1,1)

            @test lmul!(2.0,copy(B)')[:,1:10] ==  (2B')[:,1:10]

            @test_throws ArgumentError BandedMatrix(A)
            @test A*B isa MulMatrix
            @test B'A isa MulMatrix

            @test all(diag(A[1:6,1:6]) .=== zeros(6))

            @test (A*B)[1:7,1:5] ≈ A[1:7,1:6] * B[1:6,1:5]
            @test (B'A)[1:5,1:7] ≈ (B')[1:5,1:6] * A[1:6,1:7]
        end
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

        @test (A - im*I)[1:100,1:100] == A[1:100,1:100]-im*I
        @test (A + im*I)[1:100,1:100] == A[1:100,1:100]+im*I
        @test (im*I + A)[1:100,1:100] == im*I+A[1:100,1:100]
        @test (im*I - A)[1:100,1:100] == im*I-A[1:100,1:100]
    end

    @testset "Fill" begin
        A = _BandedMatrix(Ones(1,∞),∞,-1,1)
        @test 1.0 .* A isa BandedMatrix{Float64,<:Fill}
        @test Zeros(∞) .* A ≡ Zeros(∞,∞) .* A ≡ A .* Zeros(1,∞) ≡ A .* Zeros(∞,∞) ≡ Zeros(∞,∞)
        @test Ones(∞) .* A isa BandedMatrix{Float64,<:Ones}
        @test A .* Ones(1,∞) isa BandedMatrix{Float64,<:Ones}
        @test 2.0 .* A isa BandedMatrix{Float64,<:Fill}
        @test A .* 2.0 isa BandedMatrix{Float64,<:Fill}
        @test Eye(∞)*A isa BandedMatrix{Float64,<:Ones}
        @test A*Eye(∞) isa BandedMatrix{Float64,<:Ones}

        @test A*A isa BandedMatrix
        @test (A*A)[1:10,1:10] == BandedMatrix(2 => Ones(8))

        Ã = _BandedMatrix(Fill(1,1,∞), ∞, -1,1)
        @test A*Ã isa BandedMatrix
        @test Ã*A isa BandedMatrix
        @test Ã*Ã isa BandedMatrix

        B = _BandedMatrix(Ones(1,10),∞,-1,1)
        C = _BandedMatrix(Ones(1,10),10,-1,1)
        D = _BandedMatrix(Ones(1,∞),10,-1,1)

        @test (A*B)[1:10,1:10] == (B*C)[1:10,1:10] == (D*A)[1:10,1:10] == D*B == (C*D)[1:10,1:10] == BandedMatrix(2 => Ones(8))
    end

    @testset "Banded Broadcast" begin
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
        @test bandwidths(A*(B*C)) == (-1,1)
        @test bandwidths((A*B)*C) == (-1,1)

        A = _BandedMatrix(Ones{Int}(1,∞),∞,0,0)'
        B = _BandedMatrix((-2:-2:-∞)', ∞,-1,1)
        @test MemoryLayout(A+B) isa BroadcastBandedLayout{typeof(+)}
        @test MemoryLayout(2*(A+B)) isa BroadcastBandedLayout{typeof(*)}
        @test bandwidths(A+B) == (0,1)
        @test bandwidths(2*(A+B)) == (0,1)
    end

    @testset "Triangle OP recurrences" begin
        k = mortar(Base.OneTo.(1:∞))
        n = mortar(Fill.(1:∞, 1:∞))
        @test k[Block.(2:3)] isa BlockArray
        @test n[Block.(2:3)] isa BlockArray
        @test k[Block.(2:3)] == [1,2,1,2,3]
        @test n[Block.(2:3)] == [2,2,3,3,3]
        @test blocksize(BroadcastVector(exp,k)) == (∞,)
        @test BroadcastVector(exp,k)[Block.(2:3)] == exp.([1,2,1,2,3])
        # BroadcastVector(+,k,n)
    end
    # Multivariate OPs Corollary (3)
    # n = 5
    # BlockTridiagonal(Zeros.(1:∞,2:∞),
    #         (n -> Diagonal(((n+2).+(0:n)))/ (2n + 2)).(0:∞),
    #         Zeros.(2:∞,1:∞))

    @testset "KronTrav" begin
        Δ = BandedMatrix(1 => Ones(∞)/2, -1 => Ones(∞))
        A = KronTrav(Δ, Eye(∞))
        @test A[Block(100,101)] isa BandedMatrix
        @test A[Block(100,100)] isa BandedMatrix
        @test A[Block.(1:5), Block.(1:5)] isa BandedBlockBandedMatrix
        B = KronTrav(Eye(∞), Δ)
        @test B[Block(100,101)] isa BandedMatrix
        @test B[Block(100,100)] isa BandedMatrix
        V = view(A+B, Block.(1:5), Block.(1:5))
        @test MemoryLayout(typeof(V)) isa BroadcastBandedBlockBandedLayout{typeof(+)}
        @test arguments(V) == (A[Block.(1:5),Block.(1:5)],B[Block.(1:5),Block.(1:5)])
        @test (A+B)[Block.(1:5), Block.(1:5)] == A[Block.(1:5), Block.(1:5)] + B[Block.(1:5), Block.(1:5)]

        @test blockbandwidths(A+B) == (1,1)
        @test blockbandwidths(2A) == (1,1)
        @test blockbandwidths(2*(A+B)) == (1,1)

        @test subblockbandwidths(A+B) == (1,1)
        @test subblockbandwidths(2A) == (1,1)
        @test subblockbandwidths(2*(A+B)) == (1,1)
    end
end

include("test_hessenbergq.jl")
include("test_infql.jl")
include("test_infqr.jl")
include("test_inful.jl")
