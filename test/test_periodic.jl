using InfiniteLinearAlgebra, BlockBandedMatrices, LinearAlgebra, Test

@testset "periodic" begin
    @testset "single-∞" begin
        A = BlockTridiagonal(Vcat([[0. 1.; 0. 0.]],Fill([0. 1.; 0. 0.], ∞)),
                            Vcat([[-1. 1.; 1. 1.]], Fill([-1. 1.; 1. 1.], ∞)),
                            Vcat([[0. 0.; 1. 0.]], Fill([0. 0.; 1. 0.], ∞)))


        Q,L = ql(A);
        @test parent(L) isa InfiniteLinearAlgebra.InfBlockBandedMatrix
        Q̃,L̃ = ql(BlockBandedMatrix(A)[Block.(1:100),Block.(1:100)])

        # the QL factorisation is unique up to the signs of the rows of L
        S = Diagonal(sign.(diag(L[1:100,1:100])))
        S̃ = Diagonal(sign.(diag(L̃[1:100,1:100])))
        @test S*L[1:100,1:100] ≈ S̃*L̃[1:100,1:100]
        @test Q[1:10,1:10]*S[1:10,1:10] ≈ Q̃[1:10,1:10]*S̃[1:10,1:10]
        @test Q[1:10,1:12]*L[1:12,1:10] ≈ A[1:10,1:10]
        @test (Q*(Q'*[1; 2; zeros(∞)]))[1:10] ≈ [1; 2; zeros(8)]

        # complex non-selfadjoint
        c,a,b = [0 0.5; 0 0],[0 2.0; 0.5 0],[0 0.0; 2.0 0];
        A = BlockTridiagonal(Vcat([c], Fill(c,∞)),
                        Vcat([a], Fill(a,∞)),
                        Vcat([b], Fill(b,∞))) - 5im*I
        Q,L = ql(A)
        @test Q[1:10,1:12]*L[1:12,1:10] ≈ A[1:10,1:10]

        c,a,b = [0 0.5; 0 0],[0 2.0; 0.5 0],[0 0.0; 2.0 0];
        A = BlockTridiagonal(Vcat([c], Fill(c,∞)),
                        Vcat([a], Fill(a,∞)),
                        Vcat([b], Fill(b,∞)))
        Q,L = ql(A)
        @test Q[1:10,1:12]*L[1:12,1:10] ≈ A[1:10,1:10]
        @test abs(L[1,1] ) ≤ 1E-11 # degenerate


        Q,L = ql(A')
        @test Q[1:10,1:12]*L[1:12,1:10] ≈ A[1:10,1:10]'
        @test L[1,1]  ≠ 0 # non-degenerate

        # perturbations of different lengths
        c,a,b = [1 2; 3 4]/10, [5.0 1; 2 6], [1 0; 2 1]/5
        A = BlockTridiagonal(Vcat([[1 0; 0 1.0]], Fill(c,∞)),
                             Vcat([[1 2; 3 4.0], [2 1; 1 2.0], [7 0; 1 3.0], [4 1; 0 5.0]], Fill(a,∞)),
                             Vcat([[0 1; 1 0.0], [1 1; 0 1.0]], Fill(b,∞)))
        Q,L = ql(A)
        @test Q[1:20,1:22]*L[1:22,1:20] ≈ A[1:20,1:20]
    end

    @testset "bi" begin
        B  = [ 0    0    1    0;
       0    0    0    1;
       0    0    0    0;
       0    0    0    0]/2
        A₀ = [ 1   1/2  1/2   0 ;
            1/2  -1    0   1/2;
            1/2   0   -1    0 ;
            0   1/2   0    1]
        A  = [ 1    0   1/2   0 ;
            0   -1    0  1/2 ;
            1/2   0   -1   0  ;
            0   1/2   0   1 ]

        A = BlockTridiagonal(Vcat([B],Fill(B, ∞)),
                            Vcat([A₀], Fill(A, ∞)),
                            Vcat([copy(B')], Fill(copy(B'), ∞)))
        Q,L = ql(A)
        @test Q[1:10,1:12]*L[1:12,1:10] ≈ A[1:10,1:10]
    end
end
