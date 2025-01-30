using InfiniteLinearAlgebra, InfiniteRandomArrays, BandedMatrices, LazyArrays, LazyBandedMatrices, InfiniteArrays, ArrayLayouts, Test
using InfiniteLinearAlgebra: BidiagonalConjugation, OneToInf
using ArrayLayouts: supdiagonaldata, subdiagonaldata, diagonaldata
using LinearAlgebra
using LazyArrays: LazyLayout

@testset "BidiagonalConjugation" begin
    @test InfiniteLinearAlgebra._to_uplo('U') == 'U'
    @test InfiniteLinearAlgebra._to_uplo('L') == 'L'
    @test_throws ArgumentError InfiniteLinearAlgebra._to_uplo('a')
    @test InfiniteLinearAlgebra._to_uplo(:U) == 'U'
    @test InfiniteLinearAlgebra._to_uplo(:L) == 'L'
    @test_throws ArgumentError InfiniteLinearAlgebra._to_uplo(:a)

    for _ in 1:3
        V1 = InfRandTridiagonal()
        A1 = InfRandBidiagonal('U')
        X1 = brand(∞, 0, 2)
        U1 = X1 * V1 * ApplyArray(inv, A1)
        B1 = BidiagonalConjugation(U1, X1, V1, 'U');

        V2 = brand(∞, 0, 1)
        A2 = LazyBandedMatrices.Bidiagonal(Fill(0.2, ∞), 2.0 ./ (1.0 .+ (1:∞)), 'L') # LinearAlgebra.Bidiagonal not playing nice for this case
        X2 = InfRandBidiagonal('L')
        U2 = X2 * V2 * ApplyArray(inv, A2) 
        B2 = BidiagonalConjugation(U2, X2, V2, :L);

        for (A, B, uplo) in ((A1, B1, 'U'), (A2, B2, 'L'))
            @test B.dv.data === B.ev.data
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
                @test BB.dv.data === BB.ev.data
                @test parent(BB).dv.data.datasize == parent(_B).dv.data.datasize
                # @test !(BB === B) && !(parent(BB).dv.data === parent(B).dv.data) # copy is a no-op
                @test BB[1:100, 1:100] == _B[1:100, 1:100]
                @test BB[1:2:50, 1:3:40] == _B[1:2:50, 1:3:40]
                @test view(BB, [1, 3, 7, 10], 1:10) == _B[[1, 3, 7, 10], 1:10]
            end
            @test LazyBandedMatrices.bidiagonaluplo(B) == uplo
            @test LazyBandedMatrices.Bidiagonal(B) === LazyBandedMatrices.Bidiagonal(B.dv, B.ev, Symbol(uplo))
            @test B[1:10, 1:10] ≈ A[1:10, 1:10]
            @test B[230, 230] ≈ A[230, 230]
            @test B[102, 102] ≈ A[102, 102] # make sure we compute intermediate columns correctly when skipping 
            @test B[band(0)][1:100] == B.dv[1:100]
            if uplo == 'U'
                @test B[band(1)][1:100] == B.ev[1:100]
                # @test B[band(-1)][1:100] == zeros(100) # This test requires that we define a 
                # convert(::Type{BidiagonalConjugationBand{T}}, ::Zeros{V, 1, Tuple{OneToInf{Int}}}) where {T, V} method, 
                # which we probably don't need beyond this test
            else
                @test B[band(-1)][1:100] == B.ev[1:100]
                # @test B[band(1)][1:100] == zeros(100)
            end
            @test B.dv[500] == B.dv.data.dv[500]
            @test B.dv.data.datasize == 501
            @test B.ev[1005] == B.ev.data.ev[1005]
            @test B.ev.data.datasize == 1006
            @test ApplyArray(inv, B)[1:100, 1:100] ≈ ApplyArray(inv, A)[1:100, 1:100] # need to somehow let inv (or even ApplyArray(inv, )) work
            @test (B+B)[1:100, 1:100] ≈ 2A[1:100, 1:100] ≈ 2B[1:100, 1:100]
            @test (B*I)[1:100, 1:100] ≈ B[1:100, 1:100]
            # @test (B*Diagonal(1:∞))[1:100, 1:100] ≈ B[1:100, 1:100] * Diagonal(1:100) # Uncomment once https://github.com/JuliaLinearAlgebra/ArrayLayouts.jl/pull/241 is registered

            # Pointwise tests 
            for i in 1:10
                for j in 1:10
                    @test B[i, j] ≈ A[i, j]
                    @test B'[i, j] ≈ A[j, i]
                end
            end
            @inferred B[5, 5]

            # Make sure that, when indexing the transpose, B expands correctly
            @test B'[3000:3005, 2993:3006] ≈ A[2993:3006, 3000:3005]'
        end
    end

    @testset "Chebyshev" begin
        R0 = BandedMatrices._BandedMatrix(Vcat(-Ones(1,∞)/2,
                                    Zeros(1,∞),
                                    Hcat(Ones(1,1),Ones(1,∞)/2)), ℵ₀, 0,2)
        
        D0 = BandedMatrix(1 => 1:∞)
        R1 = BandedMatrix(0 => 1 ./ (1:∞), 2 => -1 ./ (3:∞))

        B = BidiagonalConjugation(R0', D0', R1', :L)'
    end
end


"""
upper_mul_tri_triview(A, B) == Tridiagonal(A*B) where A is Upper triangular BandedMatrix and B is 
"""
function upper_mul_tri_triview(U, X)
    T = promote_type(eltype(U), eltype(X))
    n = size(U,1)
    upper_mul_tri_triview!(Tridiagonal(Vector{T}(undef, n-1), Vector{T}(undef, n), Vector{T}(undef, n-1)), U, X)
end


# function upper_mul_tri_triview!(UX::Tridiagonal, U::BandedMatrix, X::Tridiagonal)
#     n = size(U,1)
#     @inbounds for j = 1:n-1
#         UX.d[j] = U.data[3,j]*X.d[j] + U.data[2,j]*X.dl[j]
#     end
#     UX.d[n] = U.data[3,n]*X.d[n]

#     @inbounds for j = 1:n-1
#         UX.dl[j] = U.data[3,j+1]*X.dl[j]
#     end

#     @inbounds for j = 1:n-2
#         UX.du[j] = U.data[3,j]*X.du[j] + U.data[2,j+1]*X.d[j+1] + U.data[1,j+2]*X.dl[j+1]
#     end

#     UX.du[n-1] = U.data[3,n-1]*X.du[n-1] + U.data[2,n]*X.d[n]

#     UX
# end

function upper_mul_tri_triview!(UX::Tridiagonal, U::BandedMatrix, X::Tridiagonal)
    n = size(U,1)
    j = 1
    Xⱼⱼ, Xⱼ₊₁ⱼ = X.d[1], X.dl[1]
    Uⱼⱼ, Uⱼⱼ₊₁, Uⱼⱼ₊₂ =  U.data[3,1], U.data[2,2],  U.data[1,3] # U[j,j], U[j,j+1], U[j,j+2]
    UX.d[1] = Uⱼⱼ*Xⱼⱼ +  Uⱼⱼ₊₁*Xⱼ₊₁ⱼ  # UX[j,j] = U[j,j]*X[j,j] + U[j,j+1]*X[j+1,j]
    Xⱼⱼ₊₁, Xⱼⱼ, Xⱼ₊₁ⱼ, Xⱼⱼ₋₁ = X.du[1], X.d[2], X.dl[2], Xⱼ₊₁ⱼ  # X[j,j+1], X[j+1,j+1], X[j+2,j+1], X[j+1,j]
    UX.du[1] = Uⱼⱼ*Xⱼⱼ₊₁ + Uⱼⱼ₊₁*Xⱼⱼ + Uⱼⱼ₊₂*Xⱼ₊₁ⱼ # UX[j,j+1] = U[j,j]*X[j,j+1] + U[j,j+1]*X[j+1,j+1] + U[j,j+1]*X[j+1,j]

    @inbounds for j = 2:n-2
        Uⱼⱼ, Uⱼⱼ₊₁, Uⱼⱼ₊₂ =  U.data[3,j], U.data[2,j+1],  U.data[1,j+2] # U[j,j], U[j,j+1], U[j,j+2]
        UX.dl[j-1] = Uⱼⱼ*Xⱼⱼ₋₁ # UX[j,j-1] = U[j,j]*X[j,j-1]
        UX.d[j] = Uⱼⱼ*Xⱼⱼ +  Uⱼⱼ₊₁*Xⱼ₊₁ⱼ  # UX[j,j] = U[j,j]*X[j,j] + U[j,j+1]*X[j+1,j]
        Xⱼⱼ₊₁, Xⱼⱼ, Xⱼ₊₁ⱼ, Xⱼⱼ₋₁ = X.du[j], X.d[j+1], X.dl[j+1], Xⱼ₊₁ⱼ  # X[j,j+1], X[j+1,j+1], X[j+2,j+1], X[j+1,j]
        UX.du[j] = Uⱼⱼ*Xⱼⱼ₊₁ + Uⱼⱼ₊₁*Xⱼⱼ + Uⱼⱼ₊₂*Xⱼ₊₁ⱼ # UX[j,j+1] = U[j,j]*X[j,j+1] + U[j,j+1]*X[j+1,j+1] + U[j,j+2]*X[j+2,j+1]
    end

    j = n-1
    Uⱼⱼ, Uⱼⱼ₊₁ =  U.data[3,j], U.data[2,j+1] # U[j,j], U[j,j+1]
    UX.dl[j-1] = Uⱼⱼ*Xⱼⱼ₋₁ # UX[j,j-1] = U[j,j]*X[j,j-1]
    UX.d[j] = Uⱼⱼ*Xⱼⱼ +  Uⱼⱼ₊₁*Xⱼ₊₁ⱼ  # UX[j,j] = U[j,j]*X[j,j] + U[j,j+1]*X[j+1,j]
    Xⱼⱼ₊₁, Xⱼⱼ, Xⱼⱼ₋₁ = X.du[j], X.d[j+1], Xⱼ₊₁ⱼ  # X[j,j+1], X[j+1,j+1], X[j+2,j+1], X[j+1,j]
    UX.du[j] = Uⱼⱼ*Xⱼⱼ₊₁ + Uⱼⱼ₊₁*Xⱼⱼ # UX[j,j+1] = U[j,j]*X[j,j+1] + U[j,j+1]*X[j+1,j+1] + U[j,j+2]*X[j+2,j+1]

    j = n
    Uⱼⱼ =  U.data[3,j] # U[j,j]
    UX.dl[j-1] = Uⱼⱼ*Xⱼⱼ₋₁ # UX[j,j-1] = U[j,j]*X[j,j-1]
    UX.d[j] = Uⱼⱼ*Xⱼⱼ  # UX[j,j] = U[j,j]*X[j,j] + U[j,j+1]*X[j+1,j]

    UX
end

tri_mul_invupper_triview(UX::Tridiagonal, R::BandedMatrix) = tri_mul_invupper_triview!(similar(UX, promote_type(eltype(UX), eltype(R))), UX, R)

function tri_mul_invupper_triview!(X, UX, R)
    @inbounds for j = 1:n-1
        UX.d[j] = UX.d[j]/ U.data[3,j]*X.d[j] + U.data[2,j]*X.dl[j]
    end
    UX.d[n] = U.data[3,n]*X.d[n]

    @inbounds for j = 1:n-1
        UX.dl[j] = U.data[3,j+1]*X.dl[j]
    end

    @inbounds for j = 1:n-2
        UX.du[j] = U.data[3,j]*X.du[j] + U.data[2,j+1]*X.d[j+1] + U.data[1,j+2]*X.dl[j+1]
    end

    UX.du[n-1] = U.data[3,n-1]*X.du[n-1] + U.data[2,n]*X.d[n]

    UX
end


@testset "TridiagonalConjugation" begin
    R0 = BandedMatrices._BandedMatrix(Vcat(-Ones(1,∞)/2,
                                    Zeros(1,∞),
                                    Hcat(Ones(1,1),Ones(1,∞)/2)), ℵ₀, 0,2)
    X_T = LazyBandedMatrices.Tridiagonal(Vcat(1.0, Fill(1/2,∞)), Zeros(∞), Fill(1/2,∞))

    n = 1000; @time U = V = R0[1:n,1:n];
    @time X = Tridiagonal(Vector(X_T.dl[1:n-1]), Vector(X_T.d[1:n]), Vector(X_T.du[1:n-1]));

    @test Tridiagonal(U*X) ≈  upper_mul_tri_triview(U, X)

end