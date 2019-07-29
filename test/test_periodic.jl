@testset "periodic" begin
    A = BlockTridiagonal(Vcat([[0. 1.; 0. 0.]],Fill([0. 1.; 0. 0.], ∞)), 
                        Vcat([[-1. 1.; 1. 1.]], Fill([-1. 1.; 1. 1.], ∞)), 
                        Vcat([[0. 0.; 1. 0.]], Fill([0. 0.; 1. 0.], ∞)))


    Q,L = ql(A);                       
    @test Q.factors isa InfiniteLinearAlgebra.InfBlockBandedMatrix
    Q̃,L̃ = ql(BlockBandedMatrix(A)[Block.(1:100),Block.(1:100)])

    @test Q̃.factors[1:100,1:100] ≈ Q.factors[1:100,1:100]
    @test Q̃.τ[1:100] ≈ Q.τ[1:100]
    @test L[1:100,1:100] ≈ L̃[1:100,1:100]
    @test Q[1:10,1:10] ≈ Q̃[1:10,1:10]
    @test Q[1:10,1:12]*L[1:12,1:10] ≈ A[1:10,1:10]

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
    @test L[1,1] == 0 # degenerate


    Q,L = ql(A')                
    @test Q[1:10,1:12]*L[1:12,1:10] ≈ A[1:10,1:10]'
    @test L[1,1]  ≠ 0 # non-degenerate
end
