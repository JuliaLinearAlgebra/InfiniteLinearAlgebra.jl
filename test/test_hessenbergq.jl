using InfiniteLinearAlgebra, BandedMatrices, LazyArrays, Test
import InfiniteLinearAlgebra: UpperHessenbergQ, LowerHessenbergQ, tail_de, _BandedMatrix, QL, InfToeplitz

@testset "HessenbergQ" begin
    @testset "finite UpperHessenbergQ" begin
        c,s = cos(0.1), sin(0.1); q = [c s; s -c]; 
        Q = UpperHessenbergQ(Fill(q,1))
        @test size(Q) == (size(Q,1),size(Q,2)) == (2,2)
        @test Q ≈ q
        Q = UpperHessenbergQ(Fill(q,2))
        @test bandwidths(Q) == (1,2)
        @test Q ≈ [q zeros(2); zeros(1,2) 1] * [1 zeros(1,2); zeros(2) q]
        @test Q' isa LowerHessenbergQ
        @test Q' ≈ [1 zeros(1,2); zeros(2) q'] * [q' zeros(2); zeros(1,2) 1] 

        q = qr(randn(2,2) + im*randn(2,2)).Q
        Q = UpperHessenbergQ(Fill(q,1))
        @test Q ≈ q
        Q = UpperHessenbergQ(Fill(q,2))
        @test bandwidths(Q) == (1,2)
        @test Q ≈ [q zeros(2); zeros(1,2) 1] * [1 zeros(1,2); zeros(2) q] 
        @test Q' isa LowerHessenbergQ
        @test Q' ≈ [1 zeros(1,2); zeros(2) q'] * [q' zeros(2); zeros(1,2) 1] 
    end

    @testset "finite LowerHessenbergQ" begin
        c,s = cos(0.1), sin(0.1); q = [c s; s -c]; 
        Q = LowerHessenbergQ(Fill(q,1))
        @test size(Q) == (size(Q,1),size(Q,2)) == (2,2)
        @test Q ≈ q
        Q = LowerHessenbergQ(Fill(q,2))
        @test bandwidths(Q) == (2,1)
        @test Q ≈ [1 zeros(1,2); zeros(2) q] * [q zeros(2); zeros(1,2) 1] 
        @test Q' isa UpperHessenbergQ
        @test Q' ≈ [q' zeros(2); zeros(1,2) 1]  * [1 zeros(1,2); zeros(2) q'] 

        q = qr(randn(2,2) + im*randn(2,2)).Q
        Q = LowerHessenbergQ(Fill(q,1))
        @test Q ≈ q
        Q = LowerHessenbergQ(Fill(q,2))
        @test bandwidths(Q) == (2,1)
        @test Q ≈ [1 zeros(1,2); zeros(2) q] * [q zeros(2); zeros(1,2) 1] 
        @test Q' isa UpperHessenbergQ
        @test Q' ≈ [q' zeros(2); zeros(1,2) 1]  * [1 zeros(1,2); zeros(2) q'] 
    end

    @testset "infinite-Q" begin
        c,s = cos(0.1), sin(0.1); q = [c s; s -c]; 
        Q = LowerHessenbergQ(Fill(q,∞))
        @test Q[1,1] ≈ 0.9950041652780258
        Q = UpperHessenbergQ(Fill(q,∞))
        @test Q[1,1] ≈ 0.9950041652780258
    end

    @testset "QRPackedQ,QLPackedQ -> LowerHessenbergQ,UpperHessenbergQ" begin
        for T in (Float64, ComplexF64)
            A = brand(T,10,10,1,1)
            Q,R = qr(A)
            @test UpperHessenbergQ(Q) ≈ Q
            Q,L = ql(A)
            @test LowerHessenbergQ(Q) ≈ Q            
        end
    end
end 


@testset "Toeplitz QLHessenberg" begin
    @testset "Derivation" begin
        A = BandedMatrix(-1 => Fill(2,∞), 0 => Fill(5,∞), 1 => Fill(0.5,∞))
        a = reverse(A.data.applied.args[1])
        d,e = tail_de(a)
        X = [transpose(a); 0 d e]
        Q = LowerHessenbergQ(Fill(ql!(X).Q,∞))
        L = _BandedMatrix(Hcat([e; X[2,2]; X[2,1]], X[2,end:-1:1] * Ones{Float64}(1,∞)), ∞, 2, 0)
        Qn,Ln = ql(A[1:1000,1:1000])
        @test Q[1:10,1:10] ≈ Qn[1:10,1:10]
        @test (Q'A)[1:10,1:10] ≈ Ln[1:10,1:10] ≈ L[1:10,1:10]

        A = BandedMatrix(-1 => Fill(2,∞), 0 => Fill(5+im,∞), 1 => Fill(0.5,∞))
        a = reverse(A.data.applied.args[1])
        d,e = tail_de(a)
        X = [transpose(a); 0 d e]
        q = ql!(X).Q
        Q = LowerHessenbergQ(Fill(q,∞))
        L = _BandedMatrix(Hcat([e; X[2,2]; X[2,1]], X[2,end:-1:1] * Ones{Float64}(1,∞)), ∞, 2, 0)
        Qn,Ln = ql(A[1:1000,1:1000])
        @test Q[1:10,1:10] ≈ Qn[1:10,1:10] * diagm(0 => [1; -Ones(9)] )
        @test (Q'A)[1:10,1:10] ≈ diagm(0 => [1; -Ones(9)] ) * Ln[1:10,1:10] ≈ L[1:10,1:10]
    end

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
        T = _BandedMatrix(reverse(a) * Ones(1,∞), ∞, 2, 1)
        F = ql(T)
        @test F.Q[1:10,1:11]*F.L[1:11,1:10] ≈ T[1:10,1:10]

        a = [1,2,3+im,0.5] 
        T = _BandedMatrix(reverse(a) * Ones{eltype(a)}(1,∞), ∞, 2, 1)
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
        T = _BandedMatrix(a * Ones{ComplexF64}(1,∞), ∞, 3, 1)
        Q,L = ql(T)
        @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
    end
end