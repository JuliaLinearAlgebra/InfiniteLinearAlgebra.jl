using Revise, InfiniteBandedMatrices, BlockBandedMatrices, BlockArrays, BandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test, DualNumbers, MatrixFactorizations
import InfiniteBandedMatrices: qltail, toeptail, tailiterate , tailiterate!, tail_de, ql_X!,
                    InfToeplitz, PertToeplitz, TriToeplitz, InfBandedMatrix, householderparams, combine_two_Q, periodic_combine_two_Q, householderparams,
                    rightasymptotics, QLHessenberg
import BlockBandedMatrices: isblockbanded, _BlockBandedMatrix
import MatrixFactorizations: QLPackedQ
import BandedMatrices: bandeddata, _BandedMatrix

@testset "Algebra" begin 
    A = BlockTridiagonal(Vcat([fill(1.0,2,1),Matrix(1.0I,2,2),Matrix(1.0I,2,2),Matrix(1.0I,2,2)],Fill(Matrix(1.0I,2,2), ∞)), 
                        Vcat([zeros(1,1)], Fill(zeros(2,2), ∞)), 
                        Vcat([fill(1.0,1,2),Matrix(1.0I,2,2)], Fill(Matrix(1.0I,2,2), ∞)))
                        
    @test A isa InfiniteBandedMatrices.BlockTriPertToeplitz                       
    @test isblockbanded(A)

    @test A[Block.(1:2),Block(1)] == A[1:3,1:1] == reshape([0.,1.,1.],3,1)

    @test BlockBandedMatrix(A)[1:100,1:100] == BlockBandedMatrix(A,(2,1))[1:100,1:100] == BlockBandedMatrix(A,(1,1))[1:100,1:100] == A[1:100,1:100]

    @test (A - I)[1:100,1:100] == A[1:100,1:100]-I
    @test (A + I)[1:100,1:100] == A[1:100,1:100]+I
    @test (I + A)[1:100,1:100] == I+A[1:100,1:100]
    @test (I - A)[1:100,1:100] == I-A[1:100,1:100]

    A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
    Ac = BandedMatrix(A')
    At = BandedMatrix(transpose(A))
    @test Ac[1:10,1:10] ≈ (A')[1:10,1:10] ≈ A[1:10,1:10]'
    @test At[1:10,1:10] ≈ transpose(A)[1:10,1:10] ≈ transpose(A[1:10,1:10])

    A = BandedMatrix(-1 => Vcat(Float64[], Fill(1/4,∞)), 0 => Vcat([1.0+im],Fill(0,∞)), 1 => Vcat(Float64[], Fill(1,∞)))
    Ac = BandedMatrix(A')
    At = BandedMatrix(transpose(A))
    @test Ac[1:10,1:10] ≈ (A')[1:10,1:10] ≈ A[1:10,1:10]'
    @test At[1:10,1:10] ≈ transpose(A)[1:10,1:10] ≈ transpose(A[1:10,1:10])

    A = BandedMatrix(-2 => Vcat(Float64[], Fill(1/4,∞)), 0 => Vcat([1.0+im,2,3],Fill(0,∞)), 1 => Vcat(Float64[], Fill(1,∞)))
    Ac = BandedMatrix(A')
    At = BandedMatrix(transpose(A))
    @test Ac[1:10,1:10] ≈ (A')[1:10,1:10] ≈ A[1:10,1:10]'
    @test At[1:10,1:10] ≈ transpose(A)[1:10,1:10] ≈ transpose(A[1:10,1:10])
end

include("test_hessenbergq.jl")


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
    A = _BandedMatrix(Hcat(randn(4,2), reverse(a) * Ones(1,∞)), ∞, 2, 1)
    @test A isa PertToeplitz
    @test BandedMatrix(A, (3,1))[1:10,1:10] == A[1:10,1:10]
    Q,L = ql(A)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ A[1:10,1:10]

    a = [0.1,1,2,3,0.5]
    A = _BandedMatrix(Hcat([0.5 0.5; -1 3; 2 2; 1 1; 0.1 0.1], reverse(a) * Ones(1,∞)), ∞, 3, 1)
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
    T = _BandedMatrix(a*Ones{ComplexF64}(1,∞), ∞, 3,1)
    Q,L = ql(T)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
    Qn,Ln = ql(T[1:1001,1:1001])
    @test Qn[1:10,1:10] * diagm(0=>[-1; (-1).^(1:9)]) ≈ Q[1:10,1:10]
    @test diagm(0=>[-1; (-1).^(1:9)]) * Ln[1:10,1:10] ≈ L[1:10,1:10]

    f =  [0.0+0.0im        0.522787+0.0im     ; 0.59647-1.05483im    0.538107-0.951621im; -0.193529-0.373893im  -0.193529-0.373893im;
               0.431415+0.0im        0.431415+0.0im; 0.0+0.0im             0.0+0.0im]
    A = _BandedMatrix(Hcat(f, a*Ones{ComplexF64}(1,∞)), ∞, 3,1)
    Q,L = ql(A)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ A[1:10,1:10]
    Qn,Ln = ql(A[1:1000,1:1000])
    @test Qn[1:10,1:10] * diagm(0 => [Ones(5); -(-1).^(1:5)]) ≈ Q[1:10,1:10]
    @test diagm(0 => [Ones(5); -(-1).^(1:5)]) * Ln[1:10,1:10] ≈ L[1:10,1:10]
end


function qdL(A)
    l,u = bandwidths(A)
    H = _BandedMatrix(A.data, ∞, l+u-1, 1)
    Q1,L1 = ql(H)
    D1, Q1, L1 = reduceband(A)
    T2 = _BandedMatrix(Lrightasymptotics(L1), ∞, l, u)
    l1 = L1[1,1]
    A2 = [[D1 l1 zeros(1,10-size(D1,2)-1)]; T2[1:10-1,1:10]] # TODO: remove
    B2 = _BandedMatrix(T2.data, ∞, l+u-2, 2)
    B2 = _BandedMatrix(T2.data, ∞, l+u-2, 2)
    D2, Q2, L2 = reduceband(B2)
    l2 = L2[1,1]
    # peroidic tail
    T3 = _BandedMatrix(Lrightasymptotics(L2), ∞, l+1, u-1)
    A3 = [[D2 l2 zeros(1,10-size(D2,2)-1)]; T3[1:10-1,1:10]] # TODO: remove

    Q3,L3 = ql( [A2[1,1] A2[1:1,2:3]; [Q2[1:3,1:1]' * T2[1:3,1]  A3[1:1,1:2] ]])

    fd_data = hcat([0; L3[:,1]; Q2[1:3,2:3]' * T2[1:3,1]], [L3[:,2]; T3[1:3,1]], [L3[2,3]; T3[1:4,2]])
    B3 = _BandedMatrix(Hcat(fd_data, T3.data), ∞, l+u-1, 1)

    ql(B3).L
end

@testset "quick-and-dirty L" begin
    for λ in (5,1,0.1+0.1im,-0.5-0.1im), A in (BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(-λ,∞), -1 => Fill(2im,∞)),
                                        BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(-conj(λ),∞), -1 => Fill(-2im,∞)))
        L∞ = qdL(A)[1:10,1:10]
        Ln = ql(A[1:1000,1:1000]).L[1:10,1:10]
        @test L∞ .* sign.(diag(L∞)) ≈ Matrix(Ln) .* sign.(diag(Ln))
    end
    for λ in (-3-0.1im, 0.0, -1im)
        A = BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(-conj(λ),∞), -1 => Fill(-2im,∞))
        @test abs(qdL(A)[1,1]) ≈ abs(ql(A[1:10000,1:10000]).L[1,1])
    end
    for λ in (1+2im,)
        A = BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(-λ,∞), -1 => Fill(2im,∞))
        @test_throws DomainError qdL(A)
    end
end



function tail_de_j(a::AbstractVector{T}, j) where T
    m = length(a)
    C = [view(a,m-1:-1:1) Vcat(-a[end]*Eye(m-2), Zeros{T}(1,m-2))]
    λ, V = eigen(C)::Eigen{T,T,Matrix{T},Vector{T}}
    n2 = abs2.(λ[j])
    n2 ≥ abs2(a[end]) || throw(DomainError(a, "QL factorization does not exist. This could indicate that the operator is not Fredholm or that the dimension of the kernel exceeds that of the co-kernel. Try again with the adjoint."))
    c_abs = sqrt((n2 - abs2(a[end]))/abs2(V[1,j]))
    c_sgn = -sign(λ[j])/sign(V[1,j]*a[end-1] - V[2,j]*a[end])
    c_sgn*c_abs*V[end:-1:1,j]    
end

@testset "non-uniqueness" begin
    a =    [10.774290245267503 - 2.01600077393353im   , -0.21512211005762638 + 0.4071609685512763im , -1.1744464421530598 + 0.6046364065537878im , 0.9690771351593747 + 0.24407852288135806im,
                -0.17679826119222275 - 1.0449912257889253im ,  1.350321850620113 + 0.1195877826052787im , -0.7557518148047799 - 0.809927665736972im  , -0.24869467464627973 + 0.06062801043516876im,
        -0.83619838577036 + 1.053001604590783im  ,1.0 + 0.0im ]

    A = _BandedMatrix(reverse(a) * Ones{ComplexF64}(1,∞), ∞, length(a)-2, 1)    
    l,u = bandwidths(A)
    Q,L = ql(A)
    @test Q[1:10,1:11] * L[1:11,1:10] ≈ A[1:10,1:10]

    de = tail_de(a)
    de2 = tail_de_j(a, 2)

    @test !(de ≈ de2)

    X = [transpose(a); 0 transpose(de2)]
    F = ql_X!(X)
    @test X[1,1:end-1] ≈ de2
    factors = _BandedMatrix(Hcat([zero(T); X[1,end-1]; X[2,end-1:-1:1]], [0; X[2,end:-1:1]] * Ones{T}(1,∞)), ∞, l+u, 1)
    Q2,L2 = QLHessenberg(factors, Fill(F.Q,∞))
    @test Q2[1:10,1:11] * L2[1:11,1:10] ≈ A[1:10,1:10]
    @test Q2[1:10,1:11] * (Q2')[1:11,1:10] ≈ I
    @test (Q2')[1:10,1:100] * (Q2)[1:100,1:10] ≈ I
    @test (Q')[1:10,1:100] * (Q)[1:100,1:10] ≈ I

    @test !(abs(L2[1,1]) ≈ abs(L[1,1]))
end