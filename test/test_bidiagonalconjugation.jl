@testset "BidiagonalConjugationData" begin
    for _ in 1:10
        V1 = InfRandTridiagonal()
        A1 = InfRandBidiagonal('U')
        X1 = brand(∞, 0, 2)
        U1 = X1 * V1 * ApplyArray(inv, A1)
        B1 = BidiagonalConjugation(U1, X1, V1, 'U')

        V2 = brand(∞, 0, 1)
        A2 = LazyBandedMatrices.Bidiagonal(Fill(0.2, ∞), 2.0 ./ (1.0 .+ (1:∞)), 'L') # LinearAlgebra.Bidiagonal not playing nice for this case
        X2 = InfRandBidiagonal('L')
        U2 = X2 * V2 * ApplyArray(inv, A2) # U2 isn't upper Hessenberg (actually it's lower, oh well), doesn't seem to matter for computing A
        B2 = BidiagonalConjugation(U2, X2, V2, 'L')

        for (A, B, uplo) in ((A1, B1, 'U'), (A2, B2, 'L'))
            @test get_uplo(B) == uplo
            @test MemoryLayout(B) isa BidiagonalLayout{LazyLayout,LazyLayout}
            @test diagonaldata(B) === B.dv
            if uplo == 'U'
                @test supdiagonaldata(B) === B.ev
                @test_throws ArgumentError subdiagonaldata(B)
                @test bandwidths(B) == (0, 1)
            else
                @test subdiagonaldata(B) === B.ev
                @test_throws ArgumentError supdiagonaldata(B)
                @test bandwidths(B) == (1, 0)
            end
            @test size(B) == (ℵ₀, ℵ₀)
            @test axes(B) == (OneToInf(), OneToInf())
            @test eltype(B) == Float64
            for _B in (B, B')
                BB = copy(_B)
                @test parent(BB).datasize == parent(_B).datasize
                @test !(BB === B) && !(parent(BB).data === parent(B).data)
                @test BB[1:10, 1:10] == _B[1:10, 1:10]
            end
            @test LazyBandedMatrices.bidiagonaluplo(B) == uplo
            @test LazyBandedMatrices.Bidiagonal(B) === LazyBandedMatrices.Bidiagonal(B.dv, B.ev, Symbol(uplo))
            @test B[1:10, 1:10] ≈ A[1:10, 1:10]
            @test B[230, 230] ≈ A[230, 230]
            @test B[102, 102] ≈ A[102, 102] # make sure we compute intermediate columns correctly when skipping 
            @test B[band(0)][1:100] == B.dv[1:100]
            if uplo == 'U'
                @test B[band(1)][1:100] == B.ev[1:100]
                @test B[band(-1)][1:100] == zeros(100)
            else
                @test B[band(-1)][1:100] == B.ev[1:100]
                @test B[band(1)][1:100] == zeros(100)
            end
            @test B.dv[500] == B.data.dv[500]
            @test B.datasize == 1000 # make sure wrapper is working correctly 
            @test B.ev[1005] == B.data.ev[1005]
            @test B.datasize == 2010 + 2 * (uplo == 'U')
            # @test_broken ApplyArray(inv, B)[1:100, 1:100] ≈ ApplyArray(inv, A)[1:100, 1:100] # need to somehow let inv (or even ApplyArray(inv, )) work
            @test (B+B)[1:100, 1:100] ≈ 2A[1:100, 1:100] ≈ 2B[1:100, 1:100]
            @test (B*I)[1:100, 1:100] ≈ B[1:100, 1:100]
            @test (B*Diagonal(1:∞))[1:100, 1:100] ≈ B[1:100, 1:100] * Diagonal(1:100)
        end
    end
end