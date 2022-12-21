using InfiniteLinearAlgebra, BandedMatrices, LazyArrays, Test
import InfiniteLinearAlgebra: UpperHessenbergQ, LowerHessenbergQ, tail_de, _BandedMatrix, QL, InfToeplitz

@testset "HessenbergQ" begin
    @testset "finite UpperHessenbergQ" begin
        c,s = cos(0.1), sin(0.1); q = [c s; s -c];
        Q = UpperHessenbergQ(Fill(q,1))
        @test size(Q) == (size(Q,1),size(Q,2)) == (2,2)
        @test all(j -> Q[:,j] ≈ q[:,j], axes(Q, 2))
        Q = UpperHessenbergQ(Fill(q,2))
        @test bandwidths(Q) == (1,2)
        QM = [q zeros(2); zeros(1,2) 1] * [1 zeros(1,2); zeros(2) q]
        @test all(j -> Q[:,j] ≈ QM[:,j], axes(Q, 2))
        @test Q' isa LowerHessenbergQ
        QM = [1 zeros(1,2); zeros(2) q'] * [q' zeros(2); zeros(1,2) 1]
        @test all(j -> Q'[:,j] ≈ QM[:,j], axes(Q, 2))

        q = qr(randn(2,2) + im*randn(2,2)).Q
        Q = UpperHessenbergQ(Fill(q,1))
        @test all(j -> Q[:,j] ≈ q[:,j], axes(Q, 2))
        Q = UpperHessenbergQ(Fill(q,2))
        @test bandwidths(Q) == (1,2)
        QM = [Matrix(q) zeros(2); zeros(1,2) 1] * [1 zeros(1,2); zeros(2) Matrix(q)]
        @test all(j -> Q[:,j] ≈ QM[:,j], axes(Q, 2))
        @test Q' isa LowerHessenbergQ
        QM = [1 zeros(1,2); zeros(2) Matrix(q')] * [Matrix(q') zeros(2); zeros(1,2) 1]
        @test all(j -> Q'[:,j] ≈ QM[:,j], axes(Q, 2))
    end

    @testset "finite LowerHessenbergQ" begin
        c,s = cos(0.1), sin(0.1); q = [c s; s -c];
        Q = LowerHessenbergQ(Fill(q,1))
        @test size(Q) == (size(Q,1),size(Q,2)) == (2,2)
        @test all(j -> Q[:,j] ≈ q[:,j], axes(Q, 2))
        Q = LowerHessenbergQ(Fill(q,2))
        @test bandwidths(Q) == (2,1)
        QM = [1 zeros(1,2); zeros(2) q] * [q zeros(2); zeros(1,2) 1]
        @test all(j -> Q[:,j] ≈ QM[:,j], axes(Q, 2))
        @test Q' isa UpperHessenbergQ
        QM = [q' zeros(2); zeros(1,2) 1]  * [1 zeros(1,2); zeros(2) q']
        @test all(j -> Q'[:,j] ≈ QM[:,j], axes(Q, 2))

        q = qr(randn(2,2) + im*randn(2,2)).Q
        Q = LowerHessenbergQ(Fill(q,1))
        @test all(j -> Q[:,j] ≈ q[:,j], axes(Q, 2))
        Q = LowerHessenbergQ(Fill(q,2))
        @test bandwidths(Q) == (2,1)
        QM = [1 zeros(1,2); zeros(2) Matrix(q)] * [Matrix(q) zeros(2); zeros(1,2) 1]
        @test all(j -> Q[:,j] ≈ QM[:,j], axes(Q, 2))
        @test Q' isa UpperHessenbergQ
        QM = [Matrix(q') zeros(2); zeros(1,2) 1] * [1 zeros(1,2); zeros(2) Matrix(q')]
        @test all(j -> Q'[:,j] ≈ QM[:,j], axes(Q, 2))
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
            @test all(j -> UpperHessenbergQ(Q)[:,j] ≈ Q[:,j], axes(Q, 2))
            Q,L = ql(A)
            @test all(j -> LowerHessenbergQ(Q)[:,j] ≈ Q[:,j], axes(Q, 2))
        end
    end
end
