using InfiniteLinearAlgebra, InfiniteArrays, Random, BandedMatrices, LazyArrays, FillArrays, ArrayLayouts, LinearAlgebra, LazyBandedMatrices, Test
using InfiniteLinearAlgebra: LowerHessenbergQ, tail_de, toeptail, InfToeplitz, PertToeplitz
using LazyBandedMatrices: LazyBandedLayout
using BandedMatrices: _BandedMatrix, BandedLayout
using ArrayLayouts: TriangularLayout, UnknownLayout

@testset "Inf QL" begin
    @testset "Toeplitz" begin
        @testset "Toeplitz QLHessenberg" begin
            @testset "Tridiagonal Toeplitz" begin
                for (Z,A,B) in ((2.0,5.1,0.5), (2.0,2.2,0.5), (2.0,-2.2,0.5), (2.0,-5.1,0.5),
                            (0.5,5.1,2.0),  (0.5,-5.1,2.0))
                    n = 100_000; T = Tridiagonal(Fill(Z,∞), Fill(A,∞), Fill(B,∞)); Qn,Ln = ql(BandedMatrix(T)[1:n,1:n]);
                    Q,L = ql(T)
                    @test L[1:10,1:10] ≈ Ln[1:10,1:10]
                    @test Q[1:10,1:10] ≈ Qn[1:10,1:10]
                    @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
                end

                for (Z,A,B)  in ((2,5.0im,0.5),
                            (2,2.1+0.1im,0.5),
                            (2,2.1-0.1im,0.5),
                            (2,1.5+0.1im,0.5),
                            (2,1.5+0.0im,0.5),
                            (2,1.5-0.0im,0.5),
                            (2,0.0+0.0im,0.5),
                            (2,-0.1+0.1im,0.5),
                            (2,-1.1+0.1im,0.5),
                            (2,-0.1-0.1im,0.5))
                    T = Tridiagonal(Fill(ComplexF64(Z),∞), Fill(ComplexF64(A),∞), Fill(ComplexF64(B),∞))
                    Q,L = ql(T)
                    @test Q[1:10,1:12] * L[1:12,1:10] ≈ T[1:10,1:10]
                end

                for (Z,A,B) in ((0.5,2.1,2.0),(0.5,-1.1,2.0))
                    @test_throws DomainError ql(Tridiagonal(Fill(Z,∞), Fill(A,∞), Fill(B,∞)))
                end
            end

            @testset "Hessenberg Toeplitz" begin
                a = [1,2,3,0.5]
                T = _BandedMatrix(reverse(a) * Ones(1,∞), ℵ₀, 2, 1)
                F = ql(T)
                @test F.Q[1:10,1:11]*F.L[1:11,1:10] ≈ T[1:10,1:10]

                a = [1,2,3+im,0.5]
                T = _BandedMatrix(reverse(a) * Ones{eltype(a)}(1,∞), ℵ₀, 2, 1)
                Q,L = ql(T)
                @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
                @test T isa InfToeplitz

                T = BandedMatrix(-2 => Fill(1,∞), 0 => Fill(0.5+eps()im,∞), 1 => Fill(0.25,∞))
                Q,L = ql(T)
                @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]

                for T in (BandedMatrix(-2 => Fill(1,∞), 0 => Fill(0.5,∞), 1 => Fill(0.25,∞)),
                            BandedMatrix(-2 => Fill(1/4,∞), 1 => Fill(1,∞))-im*I)
                    Q,L = ql(T)
                    @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
                end

                a =    [ -2.531640004434771-0.0im , 0.36995310821558014+2.5612894011525276im, -0.22944284364953327+0.39386202384951985im, -0.2700241133710857 + 0.8984628598798804im, 4.930380657631324e-32 + 0.553001215633963im ]
                T = _BandedMatrix(a * Ones{ComplexF64}(1,∞), ℵ₀, 3, 1)
                Q,L = ql(T)
                @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
            end
        end

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
            A = _BandedMatrix(Hcat(randn(4,2), reverse(a) * Ones(1,∞)), ℵ₀, 2, 1)
            @test A isa PertToeplitz
            @test BandedMatrix(A, (3,1))[1:10,1:10] == A[1:10,1:10]
            Q,L = ql(A)
            @test Q[1:10,1:11]*L[1:11,1:10] ≈ A[1:10,1:10]

            a = [0.1,1,2,3,0.5]
            A = _BandedMatrix(Hcat([0.5 0.5; -1 3; 2 2; 1 1; 0.1 0.1], reverse(a) * Ones(1,∞)), ℵ₀, 3, 1)
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
            T = _BandedMatrix(a*Ones{ComplexF64}(1,∞), ℵ₀, 3,1)
            Q,L = ql(T)
            @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
            Qn,Ln = ql(T[1:1001,1:1001])
            @test Qn[1:10,1:10] * diagm(0=>[-1; (-1).^(1:9)]) ≈ Q[1:10,1:10]
            @test diagm(0=>[-1; (-1).^(1:9)]) * Ln[1:10,1:10] ≈ L[1:10,1:10]

            f =  [0.0+0.0im        0.522787+0.0im     ; 0.59647-1.05483im    0.538107-0.951621im; -0.193529-0.373893im  -0.193529-0.373893im;
                    0.431415+0.0im        0.431415+0.0im; 0.0+0.0im             0.0+0.0im]
            A = _BandedMatrix(Hcat(f, a*Ones{ComplexF64}(1,∞)), ℵ₀, 3,1)
            Q,L = ql(A)
            @test Q[1:10,1:11]*L[1:11,1:10] ≈ A[1:10,1:10]
            Qn,Ln = ql(A[1:1000,1:1000])
            @test Qn[1:10,1:10] * diagm(0 => [Ones(5); -(-1).^(1:5)]) ≈ Q[1:10,1:10]
            @test diagm(0 => [Ones(5); -(-1).^(1:5)]) * Ln[1:10,1:10] ≈ L[1:10,1:10]
        end

        @testset "solve with QL" begin
            A = BandedMatrix(-1 => Fill(2,∞), 0 => Fill(5,∞), 1 => Fill(0.5,∞))
            @test (qr(A)\Vcat(1.0,Zeros(∞)))[1:1000] ≈ (ql(A)\Vcat(1.0,Zeros(∞)))[1:1000]

            J = BandedMatrix(0 => Vcat([1.0], Fill(0.0,∞)), 1 => Vcat(Float64[],Fill(0.5,∞)), -1 => Vcat(Float64[],Fill(0.5,∞)))
            z = 3.5
            A = J - z*I
            F = ql(A)
            Q,L = F
            b = [1; zeros(∞)]
            @test (Q'b)[1] ≈ 0.9892996329463546
            @test L[1] == L[1,1]

            @test L[1:5,1:5] isa BandedMatrix
            @test (L*(L \ b))[1:10] ≈ [1; zeros(9)]
            u = F \ b
            @test (A*u)[1:10] ≈ [1; zeros(9)]
        end

        @testset "Derivation" begin
            A = BandedMatrix(-1 => Fill(2,∞), 0 => Fill(5,∞), 1 => Fill(0.5,∞))
            a = reverse(A.data.args[1])
            d,e = tail_de(a)
            X = [transpose(a); 0 d e]
            Q = LowerHessenbergQ(Fill(ql!(X).Q,∞))
            L = _BandedMatrix(Hcat([e; X[2,2]; X[2,1]], X[2,end:-1:1] * Ones{Float64}(1,∞)), ℵ₀, 2, 0)
            Qn,Ln = ql(A[1:1000,1:1000])
            @test Q[1:10,1:10] ≈ Qn[1:10,1:10]
            @test Q'A isa MulMatrix
            @test Array((Q'A)[1:10,1:10]) ≈ [(Q'A)[k,j] for k=1:10,j=1:10]
            @test (Q'A)[1:10,1:10] ≈ Ln[1:10,1:10] ≈ L[1:10,1:10]

            A = BandedMatrix(-1 => Fill(2,∞), 0 => Fill(5+im,∞), 1 => Fill(0.5,∞))
            a = reverse(A.data.args[1])
            d,e = tail_de(a)
            X = [transpose(a); 0 d e]
            q = ql!(X).Q
            Q = LowerHessenbergQ(Fill(q,∞))
            L = _BandedMatrix(Hcat([e; X[2,2]; X[2,1]], X[2,end:-1:1] * Ones{Float64}(1,∞)), ℵ₀, 2, 0)
            Qn,Ln = ql(A[1:1000,1:1000])
            @test Q[1:10,1:10] ≈ Qn[1:10,1:10] * diagm(0 => [1; -Ones(9)] )
            @test (Q'A)[1:10,1:10] ≈ diagm(0 => [1; -Ones(9)] ) * Ln[1:10,1:10] ≈ L[1:10,1:10]
        end

        @testset "Tridiagonal QL" begin
            for A in (LinearAlgebra.SymTridiagonal([[1,2]; Fill(3,∞)], [[1, 2]; Fill(1,∞)]),
                    LinearAlgebra.Tridiagonal([[1, 2]; Fill(1,∞)], [[1,2]; Fill(3,∞)], [[1, 2]; Fill(1,∞)]),
                    LazyBandedMatrices.SymTridiagonal([[1,2]; Fill(3,∞)], [[1, 2]; Zeros(∞)]),
                    LazyBandedMatrices.SymTridiagonal([[1,2]; Fill(3,∞)], [[1, 2]; zeros(∞)]),
                    LazyBandedMatrices.SymTridiagonal([[1,2]; fill(3,∞)], [[1, 2]; zeros(∞)]))
                @test abs.(ql(A).L[1:10,1:10]) ≈ abs.(ql(A[1:1000,1:1000]).L[1:10,1:10])
            end

            A = LazyBandedMatrices.SymTridiagonal([[1,2]; Fill(3,∞)], [[1, 2]; Fill(1,∞)])
            Q,L = ql(A)
            @test (Q*L)[1:10,1:10] ≈ A[1:10,1:10]

            @test (L*Q)[1:10,1:10] ≈ LazyBandedMatrices.SymTridiagonal(L*Q)[1:10,1:10]
        end

        @test_throws ErrorException ql(zeros(∞,∞))
    end

    @testset "Adaptive finite-section-based QL" begin
        @testset "Basic properties" begin
            A = _BandedMatrix(Vcat(2*Ones(1,∞), ((1 ./(1:∞)).+1/4)', Ones(1,∞)./3), ℵ₀, 1, 1)
            Q, L = ql(A)
            b = [[1, 2, 3]; zeros(∞)]
            @test MemoryLayout(L) isa TriangularLayout{'L', 'N'}
            @test MemoryLayout(L') isa TriangularLayout{'U', 'N'}
            @test (Q'*b)[1:2] == ApplyArray(*,Q',b)[1:2] == [-0.,-1.]
            @test (L*b)[1:6] == ApplyArray(*,L,b)[1:6] == [0. , -5.25,  -7.833333333333333, -2.4166666666666666, -1., 0.]
            @test size(ql(A).τ) == (ℵ₀, )
        end
        @testset "Explicit tolerance tests" begin
            Asym = LinearAlgebra.SymTridiagonal([[1.,2.]; Fill(3.,∞)], [[1., 2.]; Fill(1.,∞)])
            Aplain = LinearAlgebra.Tridiagonal([[1., 2.]; Fill(1.,∞)], [[1.,2.]; Fill(3.,∞)], [[1., 2.]; Fill(1.,∞)])
            Qsym, Lsym = ql(Asym, 1e-10)
            Qplain, Lplain = ql(Aplain, 1e-10)
            
            @test size(Qsym) == (ℵ₀, ℵ₀)
            @test size(Lsym) == (ℵ₀, ℵ₀)
            @test size(Qplain) == (ℵ₀, ℵ₀)
            @test size(Lplain) == (ℵ₀, ℵ₀)
            @test Qsym[1:100,1:100] ≈ Qplain[1:100,1:100]
            @test Lsym[1:100,1:100] ≈ Lplain[1:100,1:100]
            @test Qsym[101,1:110] ≈ Qplain[101,1:110]
            @test Qsym[1:101,110] ≈ Qplain[1:101,110]
            @test Qsym[Vector(1:100),Vector(1:100)] ≈ Qplain[Vector(1:100),Vector(1:100)]
            @test Lsym[101,1:110] ≈ Lplain[101,1:110]
        end
        @testset "compare with Toeplitz QL" begin
            for Tri in (LazyBandedMatrices.Tridiagonal, LinearAlgebra.Tridiagonal)
                A = Tri([[1., 2.]; Fill(1.,∞)], [[1.,2.]; Fill(3.,∞)], [[1., 2.]; Fill(1.,∞)])
                Abanded = _BandedMatrix(Hcat(Vcat(1.,A.du),A.d,A.dl)', ℵ₀, 1, 1)
                F = ql(Abanded)
                G = ql(A)
                @test LowerTriangular(F.factors[1:300,1:300])[1:300,1:200] ≈ F.L[1:300,1:200] ≈ G.L[1:300,1:200]
                @test MemoryLayout(F.L.data) == LazyBandedLayout()
                @test bandwidths(F.L) == (2,0)
                @test (F.Q*[ones(200) ; zeros(∞)])[1:200] ≈ (G.Q*[ones(200) ; zeros(∞)])[1:200]
                @test (F.L*[ones(200) ; zeros(∞)])[1:200] ≈ (G.L*[ones(200) ; zeros(∞)])[1:200]
                @test (F.Q\[ones(200) ; zeros(∞)])[1:200] ≈ (F.Q'*[ones(200) ; zeros(∞)])[1:200]
            end
        end
        @testset "Adaptive QL with complex entries" begin
            A = im * LinearAlgebra.Tridiagonal([[1., 2.]; Fill(1.,∞)], [[1.,2.]; Fill(3.,∞)], [[1., 2.]; Fill(1.,∞)])
            Abanded = _BandedMatrix(conj.(Hcat(Vcat(1.,A.du),A.d,A.dl)'), ℵ₀, 1, 1)
            F = ql(Abanded)
            @test (F.Q[1:51,1:51]*F.L[1:51,1:51])[1:50,1:50] ≈ A[1:50,1:50] 
            @test MemoryLayout(F.L.data) == LazyBandedLayout()
            @test bandwidths(F.L) == (2,0)
        end
        @testset "non-tridiagonal" begin
            A = _BandedMatrix(Vcat(2*Ones(2,∞), ((1 ./(1:∞)).+4)', Ones(1,∞)./3, Ones(1,∞)./3), ℵ₀, 2, 2)
            Q,L = ql(A)
            @test ql(A[1:100,1:100]).Q[1:10,1:10] ≈ Q[1:10,1:10]

            @test (Q' * [1; 2; zeros(∞)])[1:10] ≈ Q[1:10,1:10]' * [1; 2; zeros(8)]
            B = zeros(∞,2); B[1:2,1:2] = [1 2; 3 4];
            @test (Q' * B)[1:10,:] ≈ Q[1:10,1:10]'*B[1:10,:]

            @test (Q * [1; 2; zeros(∞)])[1:10] ≈ Q[1:10,1:10] * [1; 2; zeros(8)]
            @test (Q * B)[1:10,:] ≈ Q[1:10,1:10]*B[1:10,:]
        end
    end
end