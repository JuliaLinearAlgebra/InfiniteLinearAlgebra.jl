using InfiniteBandedMatrices, BandedMatrices, Test
import InfiniteBandedMatrices: UpperHessenbergQ, LowerHessenbergQ

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
end 