using Revise, InfiniteBandedMatrices, BlockBandedMatrices, BlockArrays, BandedMatrices, InfiniteArrays, FillArrays, LazyArrays, Test, DualNumbers, MatrixFactorizations
import InfiniteBandedMatrices: qltail, toeptail, tailiterate , tailiterate!, tail_de, ql_X!,
                    InfToeplitz, PertToeplitz, TriToeplitz, InfBandedMatrix, householderparams, combine_two_Q, periodic_combine_two_Q, householderparams,
                    rightasymptotics
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
end

include("test_hessenbergq.jl")

function reduceband(A)
    l,u = bandwidths(A)
    H = _BandedMatrix(A.data, ∞, l+u-1, 1)
    Q1,L1 = ql(H)
    D = Q1[1:l+u+1,1:1]'A[1:l+u+1,1:u-1]
    D, Q1, L1
end

function householderiterate(Q::AbstractMatrix{T}, n) where T
    ret = Matrix{T}(I,n,n)
    for j = n-1:-1:1
        ret[j:j+1,:] = Q*ret[j:j+1,:]
    end
    ret
end

function householderiterate(Q1::AbstractMatrix{T}, Q2::AbstractMatrix{T}, n) where T
    ret = Matrix{T}(I,n,n)
    for j = n-1:-1:1
        ret[j:j+1,:] = (iseven(j) ? Q1 : Q2)*ret[j:j+1,:]
    end
    ret
end

function householderiterate(Q3::AbstractMatrix{T}, Q1::AbstractMatrix{T}, Q2::AbstractMatrix{T}, n) where T
    ret = Matrix{T}(I,n,n)
    for j = n-1:-1:1
        if mod(j,3) == 0
            ret[j:j+1,:] = Q1*ret[j:j+1,:]
        elseif mod(j,3) == 1
            ret[j:j+1,:] = Q2*ret[j:j+1,:]
        else
            ret[j:j+1,:] = Q3*ret[j:j+1,:]
        end
    end
    ret
end

@testset "Householder combine" begin
    @testset "sign = -1" begin
        σ = -0.9998783597231515 + 0.01559697910943707im
        v = -0.18889614060709453 + 7.427756729736341e-19im
        τ = 1.9310951421717215 - 7.593434620701001e-18im
        Z,A,B = 2,2.1+0.01im,0.5
        a = [Z,A,B]
        n = 1_000; T = Tridiagonal(Fill(ComplexF64(Z),∞), Fill(ComplexF64(A),∞), Fill(ComplexF64(B),∞)); Qn,Ln = ql(BandedMatrix(T)[1:n,1:n]);
        @test T isa TriToeplitz
        @test InfToeplitz(T) isa InfToeplitz

        de = d,e =tail_de([Z,A,B])
        X =  [Z A B; 0 d e]
        F = ql_X!(copy(X))
        H = I - τ *[v,1]*[v,1]'   
        @test H ≈ QLPackedQ([NaN v; NaN NaN], [0,τ])
        Qt = [σ 0; 0 1] * H
        @test Qt ≈ F.Q'
        Qt2 = [1 0; 0 -1] * Qt
        @test (householderiterate(Qt2, 11)')[1:10,1:10] ≈ Qn[1:10,1:10]
        @test Qt2*X  ≈ [1 0; 0 -1] * ql(X).L 
        
        t, ω = Qn.τ[2], Qn.factors[1,2]
        Q̃ = QLPackedQ([NaN ω; NaN NaN], [0, t])
        @test I - t*[ω,1]*[ω,1]' ≈ Q̃
        @test I - t'*[ω,1]*[ω,1]' ≈ Q̃'
        @test Q̃'[1,2] ≈ Qt2[1,2]
        @test Q̃'[2,1] ≈ Qt2[2,1]
        # (1-τ)/z == Qt[2,2]
        z = (1-τ)/(1-t')
        s = sqrt(z)
        @test abs(z) ≈ 1
        @test [-s 0; 0 1/s] * Qt2 * [s 0 ; 0 -1/s]  ≈ Q̃' ≈ I - t'*[ω,1]*[ω,1]'

        # [-σ*(1-τ*abs2(v)) -σ*τ*v; τ*conj(v) 1-τ]
        # [-z*σ*(1-τ*abs2(v)) -σ*τ*v; τ*conj(v)*(1-τ)/z]

        @test (1-τ) ≈ (1-t')*z
        @test -z*σ*(1-τ*abs2(v)) ≈ 1-t'*abs2(ω)
        @test -σ*τ*v ≈ -t'*ω
        @test τ*v' ≈ -t'*ω'

        @test (1-τ) ≈ (1-t')*z

        @test -z*σ*(1-τ*abs2(v)) ≈ 1-σ*τ*v*ω'
        @test -z*σ*(1-τ*abs2(v))*t' ≈ t'-σ*τ*v*ω'*t'
        @test -z*σ*(1-τ*abs2(v))*t' ≈ t'+τ*conj(v)*σ*τ*v
        @test (-(1-t')*z*σ*(1-τ*abs2(v))-(1-t'))*t' ≈ (1-t')*τ*conj(v)*σ*τ*v
        @test -t'*(1-τ)*σ*(1-τ*abs2(v))-(1-t')*t' ≈ (1-t')*abs2(v)*σ*τ^2
        @test -t'*((1-τ)*σ*(1-τ*abs2(v))+1-abs2(v)*σ*τ^2) + t'*t' ≈ abs2(v)*σ*τ^2

        β = -(σ*(1-τ*abs2(v))-σ*τ+1); γ = -abs2(v)*σ*τ^2;
        @test t' ≈ (-β + sqrt(β^2 - 4γ))/2
        @test ω ≈ σ*τ*v/t'
        @test abs2((1-τ)/(1-t')) ≈ 1

        # Check other sign choice fails
        β2 =σ-τ*σ-σ*τ*abs2(v)-1; γ2 = abs2(v)*σ*τ^2;    
        t2t = (-β2 + sqrt(β2^2 - 4γ2))/2
        @test !(abs2((1-τ)/(1-t2t)) ≈ 1)

        @test all(householderparams(ql_X!(copy(X))) .≈ (σ,τ,v))
        @test all(combine_two_Q(σ,τ,v) .≈ (-1,t,ω))

        X = [transpose(a); 0 d e]
        ql_X!(X)
        s,t,ω = combine_two_Q(σ,τ,v) # combined two Qs into one, these are the parameteris
        Q∞11 = 1 - ω*t*conj(ω)  # Q[1,1] without the callowing correction
        Q̃n = QLPackedQ(Qn.factors, [0; Qn.τ[2:end]])
        @test Q∞11 ≈ Q̃n[1,1]
        τ1 = 1 - (a[end-1] -t*ω * X[2,end-1])/(Q∞11 * de[end]) # Choose τ[1] so that (Q*L)[1,1] = A
        @test τ1 ≈ Qn.τ[1]

        M = BandedMatrix(0 => Fill(A,∞), -1 => Fill(Z,∞), 1 => Fill(B,∞))
        Q∞, L∞ = ql(M)
        @test Q∞[1:10,1:10] ≈ Qn[1:10,1:10]
        @test L∞[1:10,1:10] ≈ Ln[1:10,1:10]
        @test Q∞[1:10,1:11]*L∞[1:11,1:10] ≈ M[1:10,1:10]
    end

    @testset "sign = +1" begin
        n = 1000
        T = BandedMatrix(-2 => Fill(1,∞), 0 => Fill(-0.5-0.1im, ∞), 1 => Fill(0.25, ∞));  Qn,Ln = ql(T[1:n,1:n]);
        a = reverse(T.data.applied.args[1])
        de = tail_de(a)
        X =  [transpose(a); 0 transpose(de)]
        F = ql_X!(copy(X))
        @test F.factors[1,1:3] ≈ de
        @test F.factors[2,:] ≈ Ln[4,1:4]
        (σ,τ,v) = householderparams(F)
        H = I - τ *[v,1]*[v,1]'   
        @test H ≈ QLPackedQ([NaN v; NaN NaN], [0,τ])
        Qt = [σ 0; 0 1] * H
        @test Qt ≈ F.Q'
        @test ((diagm(0 => [-1; Ones(10)]) * householderiterate(Qt, 11))')[1:10,1:10] ≈ Qn[1:10,1:10]

        t, ω = Qn.τ[2], Qn.factors[1,2]
        Q̃ = QLPackedQ([NaN ω; NaN NaN], [0, t])

        z = (1-τ)/(1-t')
        @test z ≈ (1-τ)/conj(Q̃[2,2])
        s = sqrt(z)
        @test [s 0; 0 1/s] * Qt * [s 0 ; 0 1/s]  ≈ Q̃' ≈ I - t'*[ω,1]*[ω,1]'
        @test z*σ*(1-τ*abs2(v)) ≈ 1-t'*abs2(ω)
        @test τ*v' ≈ t'*ω'

        # this derivation reduces to quadratic equation for t' in terms of knowns v, σ, and τ
        @test z*σ*(1-τ*abs2(v)) ≈ 1-σ*τ*v*ω'               ≈  Q̃[1,1]'
        @test z*σ*(1-τ*abs2(v))*t' ≈ t'-σ*τ*v*ω'*t'         ≈  Q̃[1,1]' * t'
        @test z*σ*(1-τ*abs2(v))*t' ≈ t'-σ*τ*v*τ*v'          ≈ Q̃[1,1]' * t'
        @test (1-τ)*σ*(1-τ*abs2(v))*t' ≈ (1-t')*(t'-σ*τ*v*τ*v')          ≈ (1-t')*Q̃[1,1]' * t'
        @test (t')^2 + ((1-τ)*σ*(1-τ*abs2(v))-1-σ*τ*v*τ*v')*t' + σ*τ^2*abs2(v) ≈  0  atol=1E-14  # simplify to stabndard form

        β =σ-τ*σ-σ*τ*abs2(v)-1; γ = abs2(v)*σ*τ^2;
        @test t' ≈ (-β + sqrt(β^2 - 4γ))/2
        @test ω ≈ σ*τ*v/t'
        @test abs2((1-τ)/(1-t')) ≈ 1

        β1 = -(σ*(1-τ*abs2(v))-σ*τ+1); γ1 = -abs2(v)*σ*τ^2;
        t1t = (-β1 + sqrt(β1^2 - 4γ1))/2
        @test !(abs2((1-τ)/(1-t1t)) ≈ 1)

        @test all(householderparams(ql_X!(copy(X))) .≈ (σ,τ,v))
        @test all(combine_two_Q(σ,τ,v) .≈ (1,t,ω))    
        Q∞11 = 1 - ω*t*conj(ω)  # Q[1,1] without the callowing correction
        Q̃n = QLPackedQ(Qn.factors, [0; Qn.τ[2:end]])
        @test Q∞11 ≈ Q̃n[1,1]
        @test  -t*ω ≈ Q̃n[1,2]
        @test de[end] ≈ -Ln[1,1]
        @test (Q̃n*diagm(0 => [1-Qn.τ[1]; Ones(n-1)]))[1:10,1:10] ≈ Qn[1:10,1:10]
        @test Qn[1,1] * Ln[1,1] + Qn[1,2] * Ln[2,1] ≈ T[1,1]
        @test Q∞11 * (Qn.τ[1]-1) * de[end] - t*ω * F.factors[2,end-1] ≈ T[1,1]

        τ1 = (a[end-1] +t*ω * F.factors[2,end-1])/(Q∞11 * de[end])+1 # Choose τ[1] so that (Q*L)[1,1] = A
        @test τ1 ≈ Qn.τ[1]

        Q, L = ql(T)
        @test L[1,1] ≈ Ln[1,1] ≈ -de[end]
        @test L[1:10,1:10] ≈  Ln[1:10,1:10]
        @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
    end
    @testset "periodic 1" begin
        A = BandedMatrix(-2 => Fill(1/4,∞), 1 => Fill(1,∞))-im*I
        n = 1000; Qn,Ln = ql(A[1:n,1:n])
        a = reverse(A.data.applied.args[1])
        de = tail_de(a)
        X =  [transpose(a); 0 transpose(de)]
        F = ql_X!(copy(X))
        @test F.factors[1,1:3] ≈ de
        @test F.factors[2,:] ≈ -Ln[4,1:4]
        @test F.factors[2,:] ≈ Ln[5,2:5]

        # @test F.factors[2,:] ≈ Ln[4,1:4]
        # @test F.factors[2,:] ≈ -Ln[5,2:5]

        (σ,τ,v) = householderparams(F)
        H = I - τ *[v,1]*[v,1]'   
        @test H ≈ QLPackedQ([NaN v; NaN NaN], [0,τ])
        Qt = [σ 0; 0 1] * H
        @test Qt ≈ F.Q'
        @test  (householderiterate(Qt, 11)' * diagm(0 => [-1; (-1).^(1:10)]))[1:10,1:10] ≈ Qn[1:10,1:10]
        # @test  (householderiterate(Qt, 11)' * diagm(0 => [1; 1; (-1).^(1:9)]))[1:10,1:10] ≈ Qn[1:10,1:10]

        t1, ω1 = Qn.τ[2], Qn.factors[1,2]
        Q1 = QLPackedQ([NaN ω1; NaN NaN], [0, t1])
        t2, ω2 = Qn.τ[3], Qn.factors[2,3]
        Q2 = QLPackedQ([NaN ω2; NaN NaN], [0, t2])

        @test abs.(householderiterate(Q2', Q1', 11)')[1:10,1:10] ≈ abs.(Qn[1:10,1:10])

        Q̃n = QLPackedQ(Qn.factors, [0; Qn.τ[2:end]])
        @test Q̃n[1:10,1:10] ≈ (householderiterate(Q2', Q1', 11)')[1:10,1:10]
        
        S1 = [-1 0; 0 1] * Qt 
        @test S1 ≈ [-σ 0; 0 1] * H
        S2 = [1 0; 0 -1] * Qt * [1 0; 0 -1]
        @test S2 ≈ [σ 0; 0 1] * (I - τ *[-v,1]*[-v,1]')
        @test (diagm(0 => [-1; Ones(10)]) * householderiterate(S1, S2, 11))'[1:10,1:10] ≈ Qn[1:10,1:10]
        # @test (diagm(0 => [1; -Ones(10)]) * householderiterate(S1, S2, 11))'[1:10,1:10] ≈ Qn[1:10,1:10]

        s1 = sign(S1[1,1]/Q2[1,1]')
        s2 = sign(Q2[2,2]')
        @test [s1 0; 0 1] * Q2' * [1 0; 0 -1/s2] ≈ S1
        @test [-s2 0; 0 1] * Q1' * [1 0 ; 0 1/s1] ≈ S2

        @test (1-τ)*(-s2) ≈ 1-t2'  # [2,2] entry
        @test (1-τ)*s1 ≈ 1-t1'
        @test 1/s1 * (-σ) * (1-τ*abs2(v)) ≈ 1-t2' * abs2(ω2) # [1,1] entry
        @test -1/s2 * (σ) * (1-τ*abs2(v)) ≈ 1-t1' * abs2(ω1) # [1,1] entry
        @test 1/s1 * (-s2) * (σ*τ*v) ≈ -t2' * ω2
        @test τ*v ≈ t2'*ω2'
        @test -1/s2 * (s1) * (σ*τ*v) ≈ -t1' * ω1
        @test τ*v ≈ -t1'*ω1'

        @test (-σ) * (1-τ*abs2(v)) ≈ s1 - s2 * (σ*τ*v) * ω2' # remove abs2
        @test -σ * (1-τ*abs2(v)) ≈ s2- (s1) * (σ*τ*v) * ω1'
        @test -σ * (1-τ*abs2(v)) * t2' ≈ s1 * t2' - s2 * (σ*τ*v) * τ*v  # remove abs2
        @test -σ * (1-τ*abs2(v)) * t1' ≈ s2 * t1' + (s1) * (σ*τ*v) * τ*v 

        @test -σ * (1-τ*abs2(v)) * (1-(1-τ)*(-s2)) ≈ s1 * (1-(1-τ)*(-s2)) - s2 * (σ*τ*v) * τ*v  # remove abs2
        @test -σ * (1-τ*abs2(v)) * (1-(1-τ)*s1) ≈ s2 * (1-(1-τ)*s1 ) + (s1) * (σ*τ*v) * τ*v 

        α = -σ * (1-τ*abs2(v)) 
        β =  (α*(1-τ)+σ*(τ*v)^2)
        @test α + β*s2 ≈ s1 + (1-τ)*s1*s2   # remove abs2
        @test α - β*s1 ≈ s2  - (1-τ)*s1*s2

        @test α + β*s2 - s1 ≈ s2 - α + β*s1 
        @test 2α + (β-1)*s2  ≈  (β+1)*s1 

        @test (β+1)*α - 2α  + (β^2+1)*s2 ≈   (1-τ)*2α*s2 + (1-τ)*(β-1)*s2^2   # remove abs2
        a,b,c = (1-τ)*(β-1), (1-τ)*2α-(β^2+1), 2α- (β+1)*α
        @test a*s2^2 + b*s2 ≈ -c

        @test s2 ≈ (-b - sqrt(b^2-4a*c))/(2a) 
        @test s1 ≈ (2α + (β-1)*s2)/(β+1)
        @test t1' ≈ 1-(1-τ)*s1
        @test ω1' ≈ -τ*v/t1'
        @test t2' ≈ 1-(1-τ)*(-s2)
        @test ω2' ≈ τ*v / t2'

        X = F.factors
        m = size(X,2)
        data = Hcat([-X[1,end-1]; (-1).^(1:m-1) .* X[2,end-1:-1:1]], 
                    ApplyArray(*, ((-1).^(1:m) .* X[2,end:-1:1]), ((-1).^(0:∞))'))
        L∞ = _BandedMatrix(data,∞,m-1,0)
        @test L∞[1:10,1:10] ≈ Ln[1:10,1:10]

        Q∞11 = 1 - ω1*t1*conj(ω1)  # Q[1,1] without the callowing correction
        Q̃n = QLPackedQ(Qn.factors, [0; Qn.τ[2:end]])
        @test Q∞11 ≈ Q̃n[1,1]

        data2 = Vcat(Hcat(0.0+0im, mortar(Fill([ω1 ω2],1,∞))), data)
        factors = _BandedMatrix(data2,∞,m-1,1)
        Q̃∞ = QLPackedQ(factors, Vcat(0.0+0im,mortar(Fill([t1,t2],∞))))
        @test Q̃∞[1:10,1:10] ≈ Q̃n[1:10,1:10]

        @test Q̃∞[1:10,1:11]*L∞[1:11,2:10] ≈ A[1:10,2:10]

        @test (1-Qn.τ[1])*Q̃∞[1,1] * L∞[1,1] + Q̃∞[1,2] * L∞[2,1] ≈ A[1,1]
        L∞11 = -X[1,end-1]
        L∞21 = -X[2,end-1]
        Q∞12 = -t1*ω1
        @test Qn.τ[1] ≈ (Q∞12 * L∞21 - A[1,1])/(L∞11*Q∞11) + 1

        Q∞, L∞ = ql(A)
        @test Q∞[1:10,1:11] * L∞[1:11,1:10] ≈ A[1:10,1:10]
    end
    @testset "3-periodic" begin
        ain =    [ -2.531640004434771-0.0im , 0.36995310821558014+2.5612894011525276im, -0.22944284364953327+0.39386202384951985im, -0.2700241133710857 + 0.8984628598798804im, 4.930380657631324e-32 + 0.553001215633963im ]
        l = 3
        A = _BandedMatrix(ain * Ones{ComplexF64}(1,∞), ∞, l, 1)
        n = 1000; Qn,Ln = ql(A[1:n,1:n])
        a = reverse(A.data.applied.args[1])
        de = tail_de(a)
        X =  [transpose(a); 0 transpose(de)]
        F = ql_X!(copy(X))
        @test F.factors[1,1:end-1] ≈ de
        @test F.factors[2,:] ≈ Ln[l+2,1:l+2]
        @test F.factors[2,:] ≈ -Ln[l+3,2:l+3]

        (σ,τ,v) = householderparams(F)
        H = I - τ *[v,1]*[v,1]'   
        @test H ≈ QLPackedQ([NaN v; NaN NaN], [0,τ])
        Qt = [σ 0; 0 1] * H
        @test Qt ≈ F.Q'
        @test  (householderiterate(Qt, 11)' * diagm(0 => [1; mortar(Fill([1,-1,-1],4))[1:10]]))[1:10,1:10] ≈ Qn[1:10,1:10]


        t1, ω1 = Qn.τ[2], Qn.factors[1,2]
        Q1 = QLPackedQ([NaN ω1; NaN NaN], [0, t1])
        t2, ω2 = Qn.τ[3], Qn.factors[2,3]
        Q2 = QLPackedQ([NaN ω2; NaN NaN], [0, t2])
        t3, ω3 = Qn.τ[4], Qn.factors[3,4]
        Q3 = QLPackedQ([NaN ω3; NaN NaN], [0, t3])

        @test abs.(householderiterate(Q3', Q2', Q1', 11)')[1:10,1:10] ≈ abs.(Qn[1:10,1:10])
        @test abs.(householderiterate(Q1', Q2', Q3', 11)')[1:10,1:10] ≈ abs.(Qn[1:10,1:10])
        @test abs.(householderiterate(Q2', Q3', Q1', 11)')[1:10,1:10] ≈ abs.(Qn[1:10,1:10])

        # remove last 
        Q̃n = QLPackedQ(Qn.factors, [0; Qn.τ[2:end]])
        @test Q̃n[1:10,1:10] ≈ (householderiterate(Q2', Q3', Q1', 15)')[1:10,1:10]
        
        S1 = [1 0; 0 -1] * Qt * [1 0; 0 -1]
        @test S1 ≈ [σ 0; 0 1] * (I - τ *[-v,1]*[-v,1]')
        S2 = [-1 0; 0 -1] * Qt * [1 0; 0 -1]
        @test S2 ≈ [-σ 0; 0 1] * (I - τ *[-v,1]*[-v,1]')
        S3 = [-1 0; 0 1] * Qt 
        @test S3 ≈ [-σ 0; 0 1] * (I - τ *[v,1]*[v,1]')
        @test (diagm(0 => [-1; Ones(10)]) * householderiterate(S1, S2, S3, 11))'[1:10,1:10] ≈ Qn[1:10,1:10]

        s1 = sign(S1[1,1]/Q2[1,1]')
        s2 = sign(Q2[2,2]')
        s3 = sign(Q3[2,2]')
        @test [s1 0; 0 1] * Q2' * [1 0; 0 -1/s2] ≈ S1
        @test [-s2 0; 0 1] * Q3' * [1 0 ; 0 -1/s3] ≈ S2
        @test [-s3 0; 0 1] * Q1' * [1 0 ; 0 1/s1] ≈ S3


        @test (1-τ)*(-s2) ≈ 1-t2'  # [2,2] entry (1)
        @test (1-τ)*s1 ≈ 1-t1'
        @test (1-τ)*(-s3) ≈ 1-t3'
        @test -1/s1 * (-σ) * (1-τ*abs2(v)) ≈ 1-t2' * abs2(ω2) # [1,1] entry (4)
        @test 1/s2 * (σ) * (1-τ*abs2(v)) ≈ 1-t3' * abs2(ω3) # [1,1] entry
        @test 1/s3 * (σ) * (1-τ*abs2(v)) ≈ 1-t1' * abs2(ω1) # [1,1] entry
        @test 1/s1 * (-s2) * (σ*τ*v) ≈ -t2' * ω2 # (7)
        @test 1/s2 * (-s3) * (σ*τ*v) ≈ -t3' * ω3
        @test 1/s3 * (-s1) * (σ*τ*v) ≈ -t1' * ω1
        @test τ*v ≈ -t2'*ω2' # (10)
        @test τ*v ≈ t1'*ω1'
        @test τ*v ≈ -t3'*ω3'

        

        @test σ * (1-τ*abs2(v)) ≈ s1 - s2 * (σ*τ*v) * ω2'  # (13) = (4) & (7)
        @test σ * (1-τ*abs2(v)) ≈ s2 - s3 * (σ*τ*v) * ω3'  # (14) = (5) & (8)
        @test σ * (1-τ*abs2(v)) ≈ s3 - s1 * (σ*τ*v)  * ω1' # (15) = (6) & (9)

        @test (τ*v)/(-(1-(1-τ)*(-s2) )) ≈ ω2' # (16) = (10) & (1)
        @test (τ*v)/((1-(1-τ)*(s1) )) ≈ ω1' 
        @test (τ*v)/(-(1-(1-τ)*(-s3) )) ≈ ω3' 

        @test σ * (1-τ*abs2(v)) ≈ s1 - s2 * (σ*τ*v) * (τ*v)/(-(1-(1-τ)*(-s2) )) #  (19) = (16) & (13)
        @test σ * (1-τ*abs2(v)) ≈ s2 - s3 * (σ*τ*v) * (τ*v)/(-(1-(1-τ)*(-s3) )) 
        @test σ * (1-τ*abs2(v)) ≈ s3 - s1 * (σ*τ*v) * (τ*v)/((1-(1-τ)*(s1) )) 

        α = σ * (1-τ*abs2(v))
        β = α - τ*σ
        @test α  + β*s2  ≈ s1 + (1-τ)*s1*s2 
        @test  α + β*s3  ≈ s2 + (1-τ)*s2*s3  
        @test  α - β*s1  ≈ s3 - (1-τ)*s1*s3  

        # from Mathematica
        @test s1 ≈ -((1 + 3α + 3*α*β + β^3 - 3*α*τ - 3*α*β*τ - sqrt((1 + β^3 - 3*α*(1 + β)*(-1 + τ))^2 + 4*α*(-1 + τ)*(1 + α + β + β^2 - α*τ)^2))/(2*(-1 + τ)*(1 + α + β + β^2 - α*τ)))
        @test s2 ≈ -((1 - α - α*β + β^3 + α*τ + α*β*τ - sqrt((1 + β^3 - 3*α*(1 + β)*(-1 + τ))^2 + 4*α*(-1 + τ)*(1 + α + β + β^2 - α*τ)^2))/(2*(-1 + β + β^2 + α*(-1 + τ))*(-1 + τ)))
        @test s3 ≈ (1 - α - α*β + β^3 + α*τ + α*β*τ - sqrt((1 + β^3 - 3*α*(1 + β)*(-1 + τ))^2 + 4*α*(-1 + τ)*(1 + α + β + β^2 - α*τ)^2))/(2*(1 + β - β^2 + α*(-1 + τ))*(-1 + τ))

        @test t2' ≈ 1-(1-τ)*(-s2) # (1)
        @test t1' ≈ 1-(1-τ)*s1 # (2)
        @test t3' ≈ 1+(1-τ)*s3 # (3)
            
        @test ω1' ≈ τ*v/t1'  # (11)
        @test ω2' ≈ -τ*v / t2'
        @test ω3' ≈ -τ*v / t3'

        X = F.factors
        m = size(X,2)
        μ = mortar(Fill([1 -1 -1], 1, ∞))
        μ0 = mortar(Fill([1 0 0], 1, ∞))
        μ1 = mortar(Fill([0 1 0], 1, ∞))
        μ2 = mortar(Fill([0 0 1], 1, ∞))

        data = Hcat([X[1,end-1]; μ[1,1:m-1] .* X[2,end-1:-1:1]], ApplyArray(*, [μ[1,1:m].*X[2,end:-1:1] μ[1,2:m+1].*X[2,end:-1:1] μ[1,3:m+2].*X[2,end:-1:1]], Vcat(μ0,μ1,μ2)))
        L∞ = _BandedMatrix(data,∞,m-1,0)
        @test L∞[1:10,1:10] ≈ Ln[1:10,1:10]

        Q∞11 = 1 - ω1*t1*conj(ω1)  # Q[1,1] without the callowing correction
        Q̃n = QLPackedQ(Qn.factors, [0; Qn.τ[2:end]])
        @test Q∞11 ≈ Q̃n[1,1]

        data2 = Vcat(Hcat(0.0+0im, mortar(Fill([ω1 ω2 ω3],1,∞))), data)
        factors = _BandedMatrix(data2,∞,m-1,1)
        Q̃∞ = QLPackedQ(factors, Vcat(0.0+0im,mortar(Fill([t1,t2,t3],∞))))
        @test Q̃∞[1:10,1:10] ≈ Q̃n[1:10,1:10]

        @test Q̃∞[1:10,1:11]*L∞[1:11,2:10] ≈ A[1:10,2:10]

        @test (1-Qn.τ[1])*Q̃∞[1,1] * L∞[1,1] + Q̃∞[1,2] * L∞[2,1] ≈ A[1,1]
        L∞11 = X[1,end-1]
        @test L∞11 ≈ Ln[1,1]
        L∞21 = X[2,end-1]
        @test L∞21 ≈ Ln[2,1]
        Q∞12 = -t1*ω1
        @test Q∞12 ≈ Qn[1,2]
        @test Qn.τ[1] ≈ (Q∞12 * L∞21 - A[1,1])/(L∞11*Q∞11) + 1

        Q∞, L∞ = ql(A)
        @test Q∞[1:10,1:11] * L∞[1:11,1:10] ≈ A[1:10,1:10]
    end
end



@testset "Toeplitz" begin
    for (Z,A,B) in ((2,2.1+0.01im,0.5),) # (2,-0.1+0.1im,0.5))
        n = 1_000; T = Tridiagonal(Fill(ComplexF64(Z),∞), Fill(ComplexF64(A),∞), Fill(ComplexF64(B),∞)); Q,L = ql(BandedMatrix(T)[1:n,1:n]);
        d,e = tail_de([Z,A,B])
        @test L[1,1] ≈ e
        @test (Q'*[Z; zeros(n-1)])[1] ≈ d
        @test ql([Z A B; 0 d e]).L[1,1:2] ≈ [d;e]
        @test ql([Z A B; 0 d e]).L[2,:] ≈ -L[3,1:3]
        
        F = ql_X!([Z A B; 0 d e])
        @test F.Q'*[Z A B; 0 d e] ≈ F.L

        σ,τ,v = householderparams(F)
        @test [σ 0; 0 1] * (I - τ*[v,1]*[v,1]') ≈  F.Q'

        ql!([Z A B; 0 d e]).Q'
        householderiterate(F.Q', 100)[1:10,1:10]
        [σ/sqrt(σ) 0; 0 sqrt(σ)] * (I - τ*[v,1]*[v,1]') * [sqrt(σ) 0 ; 0 1/sqrt(σ)]

        s,t,ω = combine_two_Q(σ,τ,v)
        @test Q.τ[2] ≈ t
        @test Q.factors[1,2] ≈ ω

        Q∞,L∞ = F = ql(T)

        @test Q.τ[2] ≈ F.τ[2] ≈ t
        @test Q.factors[1,2] ≈ F.factors[1,2] ≈ ω
        @test Q∞[1:10,1:10] ≈ Q[1:10,1:10]
        @test L∞[1:10,1:10] ≈ L[1:10,1:10]
        @test Q∞[1:10,1:12] * L∞[1:12,1:10] ≈ T[1:10,1:10]
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

@testset "QL Construction" begin
    c,a,b =  (2,0.0+0.0im,0.5)
    J = Tridiagonal(Vcat(ComplexF64[], Fill(ComplexF64(c),∞)), 
                        Vcat(ComplexF64[2], Fill(ComplexF64(a),∞)),
                        Vcat(ComplexF64[], Fill(ComplexF64(b),∞)))
    A = J + (2.1+0.01im)I
    B = BandedMatrix(A,(2,1))
    T = toeptail(B)
    F∞ = ql(T)
    σ = 1- F∞.τ[1]
    F∞.factors[1,1] *= σ
    F∞ = QL(F∞.factors, Vcat(zero(ComplexF64),F∞.τ.arrays[2]))
    Q∞, L∞ = F∞
    @test Q∞[1:10,1:12] * L∞[1:12,1:10] ≈ T[1:10,1:10]

    n = 100_000; Q,L = ql(B[1:n,1:n]);
    @test Q.τ[3] ≈ F∞.τ[2]

    data = bandeddata(B).arrays[1]
    e = F∞.factors[1,1]
    d = conj(F∞.Q[1,1]*c)
    @test d == (F∞.Q'*Vcat(c,Zeros(∞)))[1]
    B̃ = _BandedMatrix(data, size(data,2), 2,1)
    B̃[end,end-1:end] .= (d,e)
    F = ql!(B̃)
    @test Q.τ[1] ≈ F.τ[1]
    B̃.data[3:end,end] .= (L∞[2,1], L∞[3,1]) # fill in L
    B̃.data[4,end-1] = L∞[3,1] # fill in L
    H = Hcat(B̃.data, F∞.factors[1:4,2] * Ones{eltype(F∞)}(1,∞))
    @test Q.τ[1:10] ≈ Vcat(F.τ, F∞.τ.arrays[2])[1:10]

    Q2,L2 = ql(A)

    @test Q.τ[1:10] ≈ Q2.τ[1:10]
    @test Q2[1:10,1:12]*L2[1:12,1:10] ≈ A[1:10,1:10]
end

@testset "Hessenberg Toeplitz" begin
    a = [1,2,3,0.5]
    T = _BandedMatrix(reverse(a) * Ones(1,∞), ∞, 2, 1)
    n = 10_000; Q, L = ql(T[1:n,1:n])

    de = tail_de(a)
    @test L[1,1] ≈ de[end]
    @test vec((Q')[1:1,1:2]*T[3:4,1:2]) ≈ de[1:end-1]

    X = [transpose(a); [0 transpose(de)]]
    @test ql(X).L[1,1:3] ≈ de
    @test ql(X).τ[1] == 0
    @test ql(X).τ[2] ≈ Q.τ[2]

    @test T isa InfToeplitz

    T = BandedMatrix(-2 => Fill(1,∞), 0 => Fill(0.5+eps()im,∞), 1 => Fill(0.25,∞))
    Q,L = QL(ql(T))
    Qn,Ln = ql(T[1:1000,1:1000])
    @test Qn.τ[1:10] ≈ Q.τ[1:10]
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ T[1:10,1:10]
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
    @test Qn[1:10,1:10] ≈ Q[1:10,1:10]
    @test Ln[1:10,1:10] ≈ L[1:10,1:10]

    f =  [0.0+0.0im        0.522787+0.0im     ; 0.59647-1.05483im    0.538107-0.951621im; -0.193529-0.373893im  -0.193529-0.373893im;
               0.431415+0.0im        0.431415+0.0im; 0.0+0.0im             0.0+0.0im]
    A = _BandedMatrix(Hcat(f, a*Ones{ComplexF64}(1,∞)), ∞, 3,1)
    Q,L = ql(A)
    @test Q[1:10,1:11]*L[1:11,1:10] ≈ A[1:10,1:10]
    Qn,Ln = ql(A[1:1000,1:1000])
    @test Qn[1:10,1:10] ≈ Q[1:10,1:10]
    @test Ln[1:10,1:10] ≈ L[1:10,1:10]
end

function Toep_L11(T)
    l,u = bandwidths(T)
    @assert u == 2
    # shift by one
    H = _BandedMatrix(T.data, ∞, l+1, 1)
    Q1,L1 = ql(H)

    d = Q1[1:3,1]'T[1:1+l,1]
    ℓ = Q1.factors.data.arrays[2].applied.args[1][2:end] # new L
    T2 = _BandedMatrix(Hcat([[zero(d); d; ℓ[3:end]] L1[1:5,1]], ℓ*Ones{eltype(T)}(1,∞)), ∞, 3, 1)
    Q2,L2 = ql(T2)
    D = (Q2')[1:5,1:4] * (Q1')[1:4,1:3] * T[3:5,1:3]
    X = [Matrix(T[3:4,1:6]); [zeros(2,2) [-1 0; 0 1]*D[1:2,:] [-1 0; 0 1]*L2[1:2,2]]]
    ql(X).L[1,3]
end

@testset "Pentadiagonal Toeplitz" begin
    a = [1,2,5+0im,0.5,0.2]
    T = _BandedMatrix(reverse(a) * Ones{eltype(a)}(1,∞), ∞, 2, 2)

    n = 1_000; Q, L = ql(T[1:n,1:n])
    H = _BandedMatrix(T.data, ∞, 3, 1)
    F1 = ql(H)
    Q1,L1 = F1
    @test Q1[1:10,1:11]*L1[1:11,1:10] ≈ H[1:10,1:10]

    d = Q1[1:3,1]'T[1:1+T.l,1]
    ℓ = F1.factors.data.arrays[2].applied.args[1][2:end]
    @test Vcat(zero(d), d, ℓ[3:end])[2:end] ≈ (Q1')[1:4,1:3]*T[1:3,1]

    T2 = _BandedMatrix(Hcat([[zero(d); d; ℓ[3:end]] L1[1:5,1]], ℓ*Ones{eltype(a)}(1,∞)), ∞, 3, 1)
    @test T2 isa PertToeplitz
    Q2,L2 = ql(T2)
    @test Q2[1:10,1:11]*L2[1:11,1:10] ≈ T2[1:10,1:10]
    @test Q1[1:10,1:11]*Q2[1:11,1:12]*L2[1:12,1:10] ≈ T[1:10,1:10]

    D = (Q2')[1:5,1:4] * (Q1')[1:4,1:3] * T[3:5,1:3]
    X = real([Matrix(T[3:4,1:6]); [zeros(2,2) [-1 0; 0 1]*D[1:2,:] [-1 0; 0 1]*L2[1:2,2]]])
    @test ql(X).L[1:2,1:4] ≈ X[3:4,3:end]

    @test L[1,1] ≈ -ql(X).L[1,3]
    
    @test abs(L[1,1]) ≈ abs(Toep_L11(T))


    a = [1,2,0,0.5,0.2]
    T = _BandedMatrix(reverse(a) * Ones{eltype(a)}(1,∞), ∞, 2, 2)
    A = T+(5+0im)I
    @test abs(ql((A)[1:1000,1:100]).L[1,1]) ≈ abs(Toep_L11(A))
    A = T+(-5+0im)I
    @test abs(ql((A)[1:1000,1:100]).L[1,1]) ≈ abs(Toep_L11(A))

    A = T+(-5+1im)I
    @test abs(ql((A)[1:1000,1:100]).L[1,1]) ≈ abs(Toep_L11(A))
end

@testset "3-diagonals" begin
    @testset "No periodic" begin
        A = BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(5,∞), -1 => Fill(2im,∞))
        l,u = bandwidths(A)
        H = _BandedMatrix(A.data, ∞, l+u-1, 1)
        Q1,L1 = ql(H)
        @test Q1[1:10,1:11] * L1[1:11,1:10] ≈ H[1:10,1:10]
        @test L1[1:10,1:10] ≈ Q1[1:13,1:10]'H[1:13,1:10]

        @test (Q1[1:13,1:10]'A[1:13,1:12])[1:10,u:10+u-1] ≈ L1[1:10,1:10]
        # to the left of H
        D1, Q1, L1 = reduceband(A)
        T2 = _BandedMatrix(rightasymptotics(parent(L1).data).applied.args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l, u)
        l1 = L1[1,1]
        

        A2 = [[D1 l1 zeros(1,10-size(D1,2)-1)]; T2[1:10-1,1:10]]
        @test Q1[1:13,1:10]'A[1:13,1:10] ≈ A2
        

        B2 = _BandedMatrix(T2.data, ∞, l+u-2, 2)
        D2, Q2, L2 = reduceband(B2)
        l2 = L2[1,1]
        T3 = _BandedMatrix(rightasymptotics(parent(L2).data).applied.args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l+1, u-1)
        A3 = [[D2 l2 zeros(1,10-size(D2,2)-1)]; T3[1:10-1,1:10]]
        @test Q2[1:13,1:10]'B2[1:13,1:10] ≈ A3

        n,m = 10,10
        Q̃2 = [1 zeros(1,m-1);  zeros(n-1,1) Q2[1:n-1,1:m-1]]
        @test norm((Q̃2'A2[:,1:end-2])[band(2)][2:end]) ≤ 10eps() # banded apart from (1,1) entry
        @test (Q̃2'A2[:,1:end-2])[2:end,2:end] ≈ A3[1:9,1:7]
        @test (Q̃2'A2[:,1:end-2])[3:end,1] ≈ A3[3:10,1]
        @test (Q̃2'A2[:,1:end-2])[1:5,1:2] ≈  Q̃2[1:4,1:5]' * [D1; T2[1:size(D1,2)+1,1:2]]
        @test (Q̃2'A2[:,1:end-2]) ≈ [A2[1,1] A2[1:1,2:8]; [Q2[1:3,1:3]' * T2[1:3,1]; Zeros(10-4)]  A3[1:end-1,1:7] ]

        # fix last entry
        @test (Q̃2'A2[:,1:3])[1:2,1:3] ≈ [A2[1,1] A2[1:1,2:3]; [Q2[1:3,1:1]' * T2[1:3,1]  A3[1:1,1:2] ]] 
        Q3,L3 = ql( [A2[1,1] A2[1:1,2:3]; [Q2[1:3,1:1]' * T2[1:3,1]  A3[1:1,1:2] ]])
        Q̃3 = [Q3 zeros(2, n-2); zeros(n-2,2) I]
        @test norm((Q̃3'Q̃2'A2[:,1:end-2])[band(2)]) ≤ 10eps()

        @test Q̃3'Q̃2'A2[:,1:end-2] ≈ [L3 zeros(2,8-3); [[Q2[1:3,2:3]' * T2[1:3,1]; Zeros(10-4)] A3[2:end-1,1:7] ] ]
        
        fd_data = hcat([0; L3[:,1]; Q2[1:3,2:3]' * T2[1:3,1]], [L3[:,2]; T3[1:3,1]], [L3[2,3]; T3[1:4,2]])
        B3 = _BandedMatrix(Hcat(fd_data, T3.data), ∞, l+u-1, 1)
        @test B3[1:10,1:8] ≈ Q̃3'Q̃2'A2[:,1:end-2]

        @test ql(B3).L[1,1] ≈ ql(A[1:1000,1:1000]).L[1,1]
    end
    @testset "periodic" begin
        A = BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(0.5-0.1im,∞), -1 => Fill(-2im,∞))
        l,u = bandwidths(A)
        H = _BandedMatrix(A.data, ∞, l+u-1, 1)
        Q1,L1 = ql(H)
        @test Q1[1:10,1:11] * L1[1:11,1:10] ≈ H[1:10,1:10]
        @test L1[1:10,1:10] ≈ Q1[1:13,1:10]'H[1:13,1:10]

        @test (Q1[1:13,1:10]'A[1:13,1:12])[1:10,u:10+u-1] ≈ L1[1:10,1:10]
        # to the left of H
        D1, Q1, L1 = reduceband(A)
        T2 = _BandedMatrix(rightasymptotics(parent(L1).data).applied.args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l, u)
        l1 = L1[1,1]
        
        A2 = [[D1 l1 zeros(1,10-size(D1,2)-1)]; T2[1:10-1,1:10]]
        @test Q1[1:13,1:10]'A[1:13,1:10] ≈ A2
        
        B2 = _BandedMatrix(T2.data, ∞, l+u-2, 2)
        D2, Q2, L2 = reduceband(B2)
        l2 = L2[1,1]

        # peroidic tail
        T3 = _BandedMatrix(rightasymptotics(parent(L2).data).arrays[2], ∞, l+1, u-1)
        A3 = [[D2 l2 zeros(1,10-size(D2,2)-1)]; T3[1:10-1,1:10]]
        @test Q2[1:13,1:10]'B2[1:13,1:10] ≈ A3

        n,m = 10,10
        Q̃2 = [1 zeros(1,m-1);  zeros(n-1,1) Q2[1:n-1,1:m-1]]
        @test norm((Q̃2'A2[:,1:end-2])[band(2)][2:end]) ≤ 20eps() # banded apart from (1,1) entry
        @test (Q̃2'A2[:,1:end-2])[2:end,2:end] ≈ A3[1:9,1:7]
        @test (Q̃2'A2[:,1:end-2])[3:end,1] ≈ -A3[3:10,1]
        @test (Q̃2'A2[:,1:end-2])[1:5,1:2] ≈  Q̃2[1:4,1:5]' * [D1; T2[1:size(D1,2)+1,1:2]]
        @test (Q̃2'A2[:,1:end-2]) ≈ [A2[1,1] A2[1:1,2:8]; [Q2[1:3,1:3]' * T2[1:3,1]; Zeros(10-4)]  A3[1:end-1,1:7] ]

        # fix last entry
        @test (Q̃2'A2[:,1:3])[1:2,1:3] ≈ [A2[1,1] A2[1:1,2:3]; [Q2[1:3,1:1]' * T2[1:3,1]  A3[1:1,1:2] ]] 
        Q3,L3 = ql( [A2[1,1] A2[1:1,2:3]; [Q2[1:3,1:1]' * T2[1:3,1]  A3[1:1,1:2] ]])
        Q̃3 = [Q3 zeros(2, n-2); zeros(n-2,2) I]
        @test norm((Q̃3'Q̃2'A2[:,1:end-2])[band(2)]) ≤ 50eps()

        @test Q̃3'Q̃2'A2[:,1:end-2] ≈ [L3 zeros(2,8-3); [[Q2[1:3,2:3]' * T2[1:3,1]; Zeros(10-4)] A3[2:end-1,1:7] ] ]
        
        fd_data = hcat([0; L3[:,1]; Q2[1:3,2:3]' * T2[1:3,1]], [L3[:,2]; T3[1:3,1]], [L3[2,3]; T3[1:4,2]])
        B3 = _BandedMatrix(Hcat(fd_data, T3.data), ∞, l+u-1, 1)
        @test B3[1:10,1:8] ≈ Q̃3'Q̃2'A2[:,1:end-2]

        # remove oscillation

        fd_data_s = diagm(0 => (-1).^(0:size(fd_data,1)-1)) * (fd_data * diagm(0 => (-1).^(1:size(fd_data,2))))
        T3_data_s = (-1)^size(fd_data,2) * (-1).^(1:u+l+1) .* T3.data[:,1]
        B3_s = _BandedMatrix(Hcat(fd_data_s, T3_data_s*Ones{ComplexF64}(1,∞)), ∞, l+u-1, 1)

        @test diagm(0 => (-1).^(0:9)) * B3[1:10,1:10] ≈ B3_s[1:10,1:10]

        @test ql(B3_s).L[1,1] ≈ ql(A[1:1000,1:1000]).L[1,1]
    end

    @testset "3rd case" begin
        A = BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(3-0.1im,∞), -1 => Fill(-2im,∞))
        l,u = bandwidths(A)
        H = _BandedMatrix(A.data, ∞, l+u-1, 1)
        Q1,L1 = ql(H)
        @test Q1[1:10,1:11] * L1[1:11,1:10] ≈ H[1:10,1:10]
        @test L1[1:10,1:10] ≈ Q1[1:13,1:10]'H[1:13,1:10]

        @test (Q1[1:13,1:10]'A[1:13,1:12])[1:10,u:10+u-1] ≈ L1[1:10,1:10]
        # to the left of H
        D1, Q1, L1 = reduceband(A)
        T2 = _BandedMatrix(rightasymptotics(parent(L1).data).applied.args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l, u)
        l1 = L1[1,1]
        
        A2 = [[D1 l1 zeros(1,10-size(D1,2)-1)]; T2[1:10-1,1:10]]
        @test Q1[1:13,1:10]'A[1:13,1:10] ≈ A2
        
        B2 = _BandedMatrix(T2.data, ∞, l+u-2, 2)
        D2, Q2, L2 = reduceband(B2)
        l2 = L2[1,1]

        # peroidic tail
        T3 = _BandedMatrix(rightasymptotics(parent(L2).data).applied.args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l+1, u-1)
        A3 = [[D2 l2 zeros(1,10-size(D2,2)-1)]; T3[1:10-1,1:10]]
        @test Q2[1:13,1:10]'B2[1:13,1:10] ≈ A3

        n,m = 10,10
        Q̃2 = [1 zeros(1,m-1);  zeros(n-1,1) Q2[1:n-1,1:m-1]]
        @test norm((Q̃2'A2[:,1:end-2])[band(2)][2:end]) ≤ 20eps() # banded apart from (1,1) entry
        @test (Q̃2'A2[:,1:end-2])[2:end,2:end] ≈ A3[1:9,1:7]
        @test (Q̃2'A2[:,1:end-2])[3:end,1] ≈ A3[3:10,1] # swapped sign
        @test (Q̃2'A2[:,1:end-2])[1:5,1:2] ≈  Q̃2[1:4,1:5]' * [D1; T2[1:size(D1,2)+1,1:2]]
        @test (Q̃2'A2[:,1:end-2]) ≈ [A2[1,1] A2[1:1,2:8]; [Q2[1:3,1:3]' * T2[1:3,1]; Zeros(10-4)]  A3[1:end-1,1:7] ]

        # fix last entry
        @test (Q̃2'A2[:,1:3])[1:2,1:3] ≈ [A2[1,1] A2[1:1,2:3]; [Q2[1:3,1:1]' * T2[1:3,1]  A3[1:1,1:2] ]] 
        Q3,L3 = ql( [A2[1,1] A2[1:1,2:3]; [Q2[1:3,1:1]' * T2[1:3,1]  A3[1:1,1:2] ]])
        Q̃3 = [Q3 zeros(2, n-2); zeros(n-2,2) I]
        @test norm((Q̃3'Q̃2'A2[:,1:end-2])[band(2)]) ≤ 50eps()

        @test Q̃3'Q̃2'A2[:,1:end-2] ≈ [L3 zeros(2,8-3); [[Q2[1:3,2:3]' * T2[1:3,1]; Zeros(10-4)] A3[2:end-1,1:7] ] ]
        
        fd_data = hcat([0; L3[:,1]; Q2[1:3,2:3]' * T2[1:3,1]], [L3[:,2]; T3[1:3,1]], [L3[2,3]; T3[1:4,2]])
        B3 = _BandedMatrix(Hcat(fd_data, T3.data), ∞, l+u-1, 1)
        @test B3[1:10,1:8] ≈ Q̃3'Q̃2'A2[:,1:end-2]

        # test tail
        T4 = _BandedMatrix(T3.data, ∞, l+u-1,1)
        Q4, L4 = ql(T4)
        @test Q4[1:10,1:11]*L4[1:11,1:10] ≈ T4[1:10,1:10]

        @test ql(B3).L[1,1] ≈ ql(B3[1:10000,1:10000]).L[1,1]
        @test abs(ql(B3).L[1,1]) ≈ abs(ql(A[1:10000,1:10000]).L[1,1])
    end
end

_Lrightasymptotics(D::Vcat) = D.arrays[2]
_Lrightasymptotics(D::ApplyArray) = D.applied.args[1][2:end] * Ones{ComplexF64}(1,∞)
Lrightasymptotics(L) = _Lrightasymptotics(rightasymptotics(parent(L).data))

function qdL(A)
    l,u = bandwidths(A)
    H = _BandedMatrix(A.data, ∞, l+u-1, 1)
    Q1,L1 = ql(H)
    D1, Q1, L1 = reduceband(A)
    T2 = _BandedMatrix(Lrightasymptotics(L1), ∞, l, u)
    l1 = L1[1,1]
    A2 = [[D1 l1 zeros(1,10-size(D1,2)-1)]; T2[1:10-1,1:10]] # TODO: remove
    B2 = _BandedMatrix(T2.data, ∞, l+u-2, 2)
    D2, Q2, L2 = reduceband(B2)
    l2 = L2[1,1]
    # peroidic tail
    T3 = _BandedMatrix(Lrightasymptotics(L2), ∞, l+1, u-1)
    A3 = [[D2 l2 zeros(1,10-size(D2,2)-1)]; T3[1:10-1,1:10]] # TODO: remove

    Q3,L3 = ql( [A2[1,1] A2[1:1,2:3]; [Q2[1:3,1:1]' * T2[1:3,1]  A3[1:1,1:2] ]])

    fd_data = hcat([0; L3[:,1]; Q2[1:3,2:3]' * T2[1:3,1]], [L3[:,2]; T3[1:3,1]], [L3[2,3]; T3[1:4,2]])
    B3 = _BandedMatrix(Hcat(fd_data, T3.data), ∞, l+u-1, 1)

    if T3 isa InfToeplitz
        ql(B3).L
    else
        # remove periodicity
        fd_data_s = diagm(0 => (-1).^(0:size(fd_data,1)-1)) * (fd_data * diagm(0 => (-1).^(1:size(fd_data,2))))
        T3_data_s = (-1)^size(fd_data,2) * (-1).^(1:u+l+1) .* T3.data[:,1]
        B3_s = _BandedMatrix(Hcat(fd_data_s, T3_data_s*Ones{ComplexF64}(1,∞)), ∞, l+u-1, 1)
        ql(B3_s).L
    end
end

@testset "quick-and-dirty L" begin
    for λ in (5,1,0.1+0.1im,-0.5-0.1im), A in (BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(-λ,∞), -1 => Fill(2im,∞)),
                                        BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(-conj(λ),∞), -1 => Fill(-2im,∞)))
        L∞ = qdL(A)[1:10,1:10]
        Ln = ql(A[1:1000,1:1000]).L[1:10,1:10]
        @test L∞ .* sign.(diag(L∞)) ≈ Matrix(Ln) .* sign.(diag(Ln))
    end
    for λ in (-3-0.1im, 0.0, -1im)
        @show λ
        A = BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(-conj(λ),∞), -1 => Fill(-2im,∞))
        @test abs(qdL(A)[1,1]) ≈ abs(ql(A[1:10000,1:10000]).L[1,1])
    end
    for λ in (1+2im,)
        A = BandedMatrix(3 => Fill(7/10,∞), 2 => Fill(1,∞), 0 => Fill(-λ,∞), -1 => Fill(2im,∞))
        @test_throws DomainError qdL(A)
    end
end

@testset "bi-infinite" begin
    Δ = BandedMatrix(-2 => Vcat(Float64[],Fill(1.0,∞)), -1 =>Vcat([1.0], Fill(0.0,∞)), 1 => Vcat([1.0], Fill(0.0,∞)), 2 => Vcat(Float64[],Fill(1.0,∞)))
    A = (Δ - 4I)
    B = BandedMatrix(A, (bandwidth(A,1)+bandwidth(A,2),bandwidth(A,2)))
    T = toeptail(B)
    H = _BandedMatrix(T.data, ∞, bandwidth(T,1)+bandwidth(T,2)-1, 1)
    Q,L = ql(H)

    A2 = Q'A
    L
end

# periodic
A = BlockTridiagonal(Vcat([[0. 1.; 0. 0.]],Fill([0. 1.; 0. 0.], ∞)), 
                       Vcat([[-1. 1.; 1. 1.]], Fill([-1. 1.; 1. 1.], ∞)), 
                       Vcat([[0. 0.; 1. 0.]], Fill([0. 0.; 1. 0.], ∞)))


Q,L = ql(A);                       
@test Q.factors isa InfiniteBandedMatrices.InfBlockBandedMatrix
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

B = (A')[1:100,1:100]; B[1,1] = 2
ql(B+0.00001I)

ql(A')

mortar(A.blocks')

A

(A.blocks')[1,2]
A.blocks[2,1]
ql(A')


BlockArray(A.blocks')

A'

T = Tridiagonal(randn(4), randn(5), randn(4))
T'


c,a,b = A[Block(N+1,N)],A[Block(N,N)],A[Block(N-1,N)]

z = zero(c)
d,e = c,a
ql(A-3*I)



F.factors


Block.(N-2:N-1)
BB
BB

BB
ql(A[1:100,1:100]).τ

ql(A[1:100,1:100]).factors - F∞[1:100,1:100]


X = [c a b; z d e]
ql(X)

QL = ql!(X)     
P = PseudoBlockArray(X, [2,2], [2,2,2])
P[Block(1,2)]


X

if h == X[1,3] 
    return QL
end
h = X[1,3]
X[2,:] .= (zero(T), X[1,1], X[1,2]);
X[1,:] .= (c,a,b);
end


X




function symbolplot!(c,a,b; label="")
    θθ = 2π*(0:0.0001:1)
    zz = exp.(im*θθ)
    ii = b./zz  .+ c.*zz
    plot!(real.(ii), imag.(ii); label=label)
end

c,a,b
c,a,b = 2-0.2im,0.0+0im,0.5+im
c,a,b = 0.5+0im,0.0+0im,0.5+0im
c,a,b = ComplexF64.((2,0,0.5))
J = Tridiagonal(Vcat(ComplexF64[], Fill(c,∞)), 
                    Vcat(ComplexF64[], Fill(0.0im,∞)),
                    Vcat(ComplexF64[], Fill(b,∞)))

c,a,b = (2.,0.,0.5)
J = Tridiagonal(Vcat(Float64[], Fill(c,∞)), 
                    Vcat(Float64[], Fill(a,∞)),
                    Vcat(Float64[], Fill(b,∞)))
xx = -5:0.001:5; plot(xx, (λ -> real(ql(J'-λ*I).L[1,1])).(xx); label="abs(L[1,1])", xlabel="x")
                plot(xx, (λ -> real(ql(J'-λ*I).L[1,1])).(xx); label="abs(L[1,1])", xlabel="x")


A = J+0.0000im*I
Q,L = ql(A)

X, τ =qltail(c,a+0im,b)
d,e = X[1:2,1]
X = [c a b; 0 d e]
F = ql(X)

a
Q,L = ql(A - 4im*I)


Matrix(Q)

A = J-(4im)*I
ql(BandedMatrix(A)[1:100,1:100])
Q,L = tailiterate(c,-4im,b)
X,τ = qltail(c,-4im,b)

d,e = X[1:2,1]
X = [c a b; 0 d e]
InfiniteBandedMatrices._qlfactUnblocked!(X)
X = [0 c a b 0 0;
     0 0 c a b 0;
     0 0 0 c a b;
     0 0 0 0 c a]

Q,L = tailiterate(b,a,c)



Q'Q-I

abs.(eigvals(Matrix(Q)))
det(Q)

ql(BandedMatrix(A)[1:100,1:100])


tailiterate(0.5,-4,0.5)


X[3:4,:]   .= 0
X[3,4:end] .= X[1,1:3]
X[4,5:end] .= X[2,2:3]
X[1:2,:] .= [0 c a b 0 0;
             0 0 c a b 0]
InfiniteBandedMatrices._qlfactUnblocked!(X)


ql(Matrix(A[1:100,1:100]))
X


d,e

Q,L= ql(A)
Q[1:5,1:7]*L[1:7,1:3]

Q[1:

tailiterate(c,a-4*im,b)
qltail(c,a-4*im,b)

1-ql(X).τ[1] 



ql(J'-2.49999*I)
a = -3.0
c,a,b
J = Tridiagonal(Vcat(ComplexF64[], Fill(complex(c),∞)), 
                    Vcat(ComplexF64[], Fill(complex(a),∞)),
                    Vcat(ComplexF64[], Fill(complex(b),∞)))
Tridiagonal(J')

A = J-3*I
ql(J-3*I)

ql(A)
Q,L = ql(J)

Q[1:3,1:4]*L[1:4,1:2]

A
J-3*I

d,e = qltail(c,a,b).factors[1,1:2]
F =  qltail(complex.((c,a,b))...)
qltail(complex.((c,a,b))...)
qltail(c,a,b)
F.τ

X = [c a b; 0 d e]
ql(X)

ql(complex.([c a b; 0 d e]))

ql(AbstractMatrix{ComplexF64}.(A)).factors

Q,L = ql(A)
a = -3
tailiterate(c,a,b)
tailiterate(c+0im,a+0im,b+0im)


L

Q[1:3,1:3]*L[1:3,1]

ql(BandedMatrix(A)[1:100,1:100])

X = [c a -b; 0 d e]
Q,L = ql(X)
Q.τ

X,τ = qltail(c,a,b)
d,e = X[1,1:2]
X = [c a b; 0 d e]
X = complex.(X)
F = ql!(X)


tailiterate(c,a,b)
ql(complex.(X)).factors
ql((X)).factors

ql([1+0im 1 0; 0 0 1]).τ
@which LinearAlgebra.qrfactUnblocked!([1 0 0; 0 1 1+0im])

x = [0,0.0,0,1]
    LinearAlgebra.reflector!(x)
x


LinearAlgebra.reflector!(x)
x
qr([1 0 0; 0 1 1])

Q*L-X

c,a,b
B = BandedMatrix(J-3*I,(2,1))
    ql!(B)

BandedMatrix(J-3*I)

B


Q2,L2 =     ql(BandedMatrix(J'-3*I)[1:1000,1:1000])
L

Q[1:10,1:10]*L[1:10,1:10]

Q[3:3,1:3]*L[1:3,1]
Q2[3:3,1:3]*L2[1:3,1]
Q[3:3,1:3],L[1:3,1]
Q2[3:3,1:3],L2[1:3,1]

Q2.τ
Q.τ
Q2.factors
Q.factors

B = BandedMatrix((J-3*I),(2,1))

b,a,c,_ = toeptail(B)
X, τ = qltail(c,a,b)
c,a,b
qltail(c,a,b)
d,e = tailiterate(c,a,b)[1][1,1:2]
tailiterate!([c a b; 0 d -e])
d,e
ql([c a b; 0 -d -e])

B[1,1,] = -3.0

ql!(B[1:100,1:100]).factors

B[1:100,1:100]
c,a,b

data = bandeddata(B).arrays[1]
B̃ = _BandedMatrix(data, size(data,2), 2,1)
B̃[end,end-1:end] .= (X[1,1], X[1,2])
F = ql!(B̃)
B̃.data[3:end,end] .= (X[2,2], X[2,1]) # fill in L
B̃.data[4,end-1] = X[2,1] # fill in L
H = Hcat(B̃.data, [X[1,3], X[2,3], X[2,2], X[2,1]] * Ones{T}(1,∞))
QL(_BandedMatrix(H, ∞, 2, 1), Vcat(F.τ,Fill(τ,∞)))
X

eigvalBandedMatrix(J)[1:100,1:100]


λ = eigvals(Matrix(J[1:2000,1:2000]))
scatter(real.(λ), imag.(λ); label="finite section with n = $(length(λ))")
    symbolplot!(c,a,b; label="essential spectrum")




ql(J'-0I)


symbolplot!(c,a,b)

@which ql(J'-λ[1]*I).Q[3,1]

ql(J'-λ[1]*I).factors
a
tailiterate(b,a-λ[1],c)
qltail(b,a-λ[1],c)

ql(J'-λ[1]*I).τ
ql(J'-λ[1]*I).factors

ql(J'-0.1*I)

Q.factors

Q.factors
Q.τ

Q,L = ((BandedMatrix(J'-λ[1]*I)[1:200,1:200]) |> ql)

(BandedMatrix(J)[1:200,1:200]-λ[1]*I) * Q[:,1]

Q[1000,1]
n = 150; κ = Q[1:n,1]
(J-λ[1]*I)[1:n,1:n]*κ



(J-λ[1]*I)*Vcat(Q[1:150,1],Zeros(∞))

Q*Vcat(1,Zeros(∞))

Q[:,1]

J'-λ[1]*I

ql(J-2.500001*I)

ql(BandedMatrix(J-3*I)[1:1000,1:1000])


Q,L = ql(BandedMatrix(J)[1:1000,1:1000])

Q[:,1]

M = zeros(ComplexF64,n,n); M[band(1)] .= b; M[band(-1)] .= c; 
M = zeros(ComplexF64,n,n); M[band(1)] .= conj(c); M[band(-1)] .= conj(b); 
ql(M)
M

plot(1:5)

qr(M)

svdvals(Matrix(F.L[1:1000,1:1000]))

Q,L = F;

@which Matrix(Q)

typeof(Q')
function Base.Matrix(::Adjoint{<:Any,<:QLPackedQ}) 
Base.Matrix{T}(Q::Adjoint{<:Any,<:QLPackedQ}) where {T} = lmul!(Q, Matrix{T}(I, size(Q, 1), min(Q,2)))
@time Q'*[1; zeros(999)]

n = 1000; Q = F.Q; Qt = lmul!(Q', Matrix{T}(I, n,n)); Q = Qt';  svdvals(Q[1:501,1:501])

Q[2:2:200,1:2:200]

Q[2:2:400,1:2:200] \ [Zeros(200-1);1] |> norm

plot(abs.(Q*[Zeros(n-1);1]))
svdvals(Q[2:2:400,1:2:200])

k = zeros(eltype(Q),n)
k[1] = 1
k[3] = -Q[2,1]/Q[2,3]
for m = 4:2:20 
    k[m+1] = -(Q*k)[m]/Q[m,m+1]
end


plot(abs.(k))

plot(abs.(Q*k))


plot(abs.(nullspace(Matrix(L'))); yscale=:log10)

Q*k

Q[2:2:end,1:2:end]

svdvals(Q[2:2:end,1:2:end])


Q*k




k

Q[1:505,1:501]

[1; zeros(999)]

F.τ

F.factors

F.τ
norm(Q[1,2])
Q[2,1]
heatmap(sparse(abs.(Q)))

svdvals(Q[:,1:10])

plot(abs.(Q[:,51]))

size(Q)

svdvals(Q[:,[1,3]])

svdvals(Q)

spy(abs.(Q))
norm(Q[3,4])
plot(abs.(Q[:,2])) 

plot(real.(F.L[band(-2)])[1:100])

function it(Z,A,B,d,e)
    X = [Z A B; 0 d e]
    ql!(X)
    X[1,1:2]
end
fxd(Z,A,B,d,e) = it(Z,A,B,d,e) - [d,e]

A = (J-0.0im*I)
ql(A)
@which qltail(c,0.0,b)
Z,A,B = c,0.9im,b
T = promote_type(eltype(Z),eltype(A),eltype(B))
ñ1 = (A + sqrt(A^2-4B*Z))/2
ñ2 = (A - sqrt(A^2-4B*Z))/2
ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
(n,σ) = (abs(ñ),conj(sign(ñ)))
if n^2 < abs2(B)
    throw(ContinuousSpectrumError())
end

e = sqrt(n^2 - abs2(B))
d = σ*e*Z/n
epsilon.(fxd(Z,A,B,dual(d,1), e))
A = 1.1im
Z,A,B = c,1.1im,b
tailiterate(Z,A,B)
J + A*I


(BandedMatrix(J - 0.0im*I)[1:10000,1:10000] |> ql).L

ql(J - 0.9im*I).L

J

A = -1.0001im
d,e = tailiterate(Z,A,B)[1][1,1:2]
sqrt(e^2 + abs2(B))

B = BandedMatrix(J+A*I,(2,1))

opnorm(Matrix(ql(BandedMatrix(J)[1:1000,1:1000]).L))

Matrix(ql(BandedMatrix(J)[1:100,1:100]).Q)




norm(J[1:1000,1:1000])

Matrix(J[1:2000,1:2000]) |> svdvals
opnorm(Matrix(J[1:2000,1:2000]))

b,a,c,_ = toeptail(B)

B
@which toeptail(B)



b,a,c

A

Z,A,B
J


T = promote_type(eltype(Z),eltype(A),eltype(B))
ñ1 = (A + sqrt(A^2-4B*Z))/2
ñ2 = (A - sqrt(A^2-4B*Z))/2
ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
(n,σ) = (abs(ñ),conj(sign(ñ)))
if n^2 < abs2(B)
    throw(ContinuousSpectrumError())
end

e = sqrt(n^2 - abs2(B))
d = σ*e*Z/n

X = [Z A B;
        0 d e]
QL = ql!(X)

# two iterations to correct for sign
X[2,:] .= (zero(T), X[1,1], X[1,2]);
X[1,:] .= (Z,A,B);
QL = ql!(X)

X, QL.τ[end]  
Z,A,B
c,a,b
tailiterate(c,a,b)
qltail(c,a,b)


A

it(d,e) - [d,e]

it(d,e) - [d,e]


X1 = [Z A B; 0 dual(d,1) e]; ql!(X1); epsilon.(X1[1,1:2]-[d,e])
X2 = [Z A B; 0 d dual(e,1)]
ql!(X2)

[epsilon.(X1[1,1:2]) epsilon.(X2[1,1:2])] |> eigvals

 X2

X = [Z A B; 0 d e+10eps()]
tailiterate!(X)
QL = ql!(X)

# two iterations to correct for sign
X[2,:] .= (zero(T), X[1,1], X[1,2]);
X[1,:] .= (Z,A,B);
QL = ql!(X)

X, QL.τ[end]      
c,a,b
a = 0.0
tailiterate(c,a,b)

tailiterate!(X)



Q,L= ql(J)

Q[1:100,1:100]*L[1:100,1:100]

Q,L = ql(BandedMatrix(J)[1:100,1:100])
plot(abs.(Q[:,1]))
Q'Q

J

abs.(eigvals(Matrix(J[1:1000,1:1000])))|> minimum

Q*L - J[1:100,1:100] |> norm

B = BandedMatrix(A, (2,1))
 b,a,c,_ = toeptail(B)

toeptail(B)

qltail(c,a,b)

Z,A,B = c,-1.001im,b
qltail(Z,A,B)

using Plots    

xx = -4:0.1:4
L = (λ -> try 
        real(ql(J-λ*I).L[1,1] )
    catch ContinuousSpectrumError 
        NaN
    end).(xx)

plot(xx,L)    


θθ = 2π*(0:0.0001:1)
    zz = exp.(im*θθ)
    plot(b./zz  .+ c.*zz)
θθ = 2π*(0:0.0001:0.5)
    zz = exp.(im*θθ)
    plot!(b./zz  .+ c.*zz)

b,c

c,a,b

plot(θθ, abs.(c./zz .+ a .+ b.*zz))


Z,A,B = c,-1.0001im,b

Bandw(J+A*I)
qltail(Z,A,B)


b,a,c,_ = toeptail(B)
X, τ = qltail(c,a,b)

T = promote_type(eltype(Z),eltype(A),eltype(B))
ñ1 = (A + sqrt(A^2-4B*Z))/2
ñ2 = (A - sqrt(A^2-4B*Z))/2
ñ = abs(ñ1) > abs(ñ2) ? ñ1 : ñ2
(n,σ) = (abs(ñ),conj(sign(ñ)))
if n^2 < abs2(B)
    throw(ContinuousSpectrumError())
end
n^2 < abs2(B)

ql(J-1.007im*I)
###

import InfiniteBandedMatrices: tailiterate, tailiterate!, reflector!, reflectorApply!

c,a,b = 0.6,2,0.5
T = Float64
X, τ = tailiterate(c,a,b)
        X[2,:] .= (zero(T), X[1,1], X[1,2])
        X[1,:] .= (c,a,b)

tailiterate(c,a,b)

function givenstail(Z::Real, A::Real, B::Real)
    s = (-A + sqrt(A^2-4B*Z))/(2Z)

    if s^2 > 1
        X, τ = givenstail(-Z,-A,-B)
        X[1,3] = -X[1,3]
        return -X, τ
    end

    c = -sqrt(1-s^2)
    γ¹ = Z*c
    γ⁰ = c*A + s*γ¹
    X = [Z A B;
         0 -γ¹ -γ⁰]
    QL = ql!(X)
    X, QL.τ[end]         
end

function givenstail(Z, A, B)
    ñ = (A + sign(real(A))*sqrt(A^2-4B*Z))/2
    (n,σ) = (abs(ñ),conj(sign(ñ)))
    e = sqrt(n^2 - abs2(B))
    d = σ*e*Z/n

    X = [Z A B;
         0 d e]
    QL = ql!(X)

    # two iterations to correct for sign
    X[2,:] .= (zero(T), X[1,1], X[1,2]);
    X[1,:] .= (Z,A,B);
    QL = ql!(X)

    X, QL.τ[end]         
end


Z,A,B = 2-0.2im,0,0.5+im
    givenstail(Z,A,B) .≈  tailiterate(Z,A,B)

ñ = (A - sqrt(A^2-4B*Z))/2
(n,σ) = (abs(ñ),conj(sign(ñ)))
e = sqrt(n^2 - abs2(B))
d = σ*e*Z/n

d,e = X[1,1:2]

X = [Z A B;
        0 d e]
QL = ql!(X)
X, QL.τ[end]         

tailiterate(c,a,b)

all(givenstail(c,a,b) .≈ tailiterate(c,a,b))


Z,A,B = -1+0.2im,-2+2im,-0.5+0.1im
givenstail(Z,A,B)


Z,A,B = -1,-2im,0.5
X, τ = tailiterate(Z,A,B)

d,e = X[1,1:2]

n = sqrt(e^2+abs2(B))
σ = conj(sign(e*A-B*d))
H = [σ*e -σ*B; -conj(B) -conj(e)]/n
H*[Z A B; 0 d e]

σ*e*Z/n - d


(A + sqrt(A^2-4B*Z))/2
ñ = (A - sqrt(A^2-4B*Z))/2
(n,σ) .== (abs(ñ),conj(sign(ñ)))

sqrt(n^2 - abs2(B))
e
e^2

(n,σ)
(abs(ñ),conj(sign(ñ)))
(n,σ)

σ
n

n/σ

σ|>abs
d,e

H2 = diagm(0 => [1/sign(e*A-B*d),1])

d,e

H2*H2'

x = [e,B]
    τ = reflector!(x)
    v = [x[2],1]
    H = I - τ*v*v' 
   
[e -B; -conj(B) -conj(e)]/n    
    
ql([Z A B;
    0 d e]).τ

H*[B,e]

H'H
e*Z/n -d

e^2

e^4 + 2*(abs2(B)+B*Z-A^2)*e^2 + (abs2(B)+B*Z)^2 - A^2*abs2(B)



s, t = -X[1,3]/norm(X[:,3]), -X[2,3]/norm(X[:,3])
H = [t -s; conj(s) conj(t)]
H'H
H*X



(-A + sqrt(abs2(A)-4B*conj(Z)))/(2conj(Z))


H

X

Z,A,B

X

J = SymTridiagonal(Vcat(Float64[], Fill(b,∞)), 
                    Vcat(Float64[], Fill(a,∞)),
                    Vcat(Float64[], Fill(c,∞)))


givenstail(-c,-a,-b)

tailiterate(c,a,b)

X


X

x = X[2:-1:1,3]
    τ = reflector!(x)
    v = [1; x[2]]
    H = I - τ*v*v' 
    H*X[2:-1:1,3]
x = X[2:-1:1,3]
s, t = -x[1]/sqrt(x[1]^2+x[2]^2), -x[2]/sqrt(x[1]^2+x[2]^2)
[s t;t -s] * x

s, t = x[2]/sqrt(x[1]^2+x[2]^2), x[1]/sqrt(x[1]^2+x[2]^2)
H = [t -s; s t]
H*X




v
τ

1/H[1,1]

X

function givenstail2(Z, A, B)
    # @assert a^2-4c*b ≥ 0
    s = (-A + sqrt(A^2-4B*Z))/(2Z)
    l⁰ = (A + sqrt(A^2-4B*Z))/2
    # if s∞^2 > 1
    #     s∞ = (t₀ + sqrt(t₀^2-4t₁^2))/(2t₁)
    #     l0 = (t₀ - sqrt(t₀^2-4t₁^2))/2
    # end
    c = -sqrt(1-s^2)
    γ¹ = Z*c
    γ⁰ = c*A + s*γ¹
    l¹ = B+Z  # = c*γ¹ - st₁
    l² = -Z*s
    c,s,l⁰,l¹,l²,γ¹,γ⁰
end

givenstail2(c,a,b)

v

s,t


H
 
[s t;t -s]

H*x

[s t;t -s]

x
x1 = X[2,3]
v1, v2 = x
X[2,2]
X
reflectorApply!(x, τ, X[2:-1:1,2])
A = X[2:-1:1,1]
x2 = X[2,1]

vAj = conj(τ)*v2'*a
A[1] = - vAj
A[2] = a -  v2*vAj
x2
function it(x1, x2)
    ν = sqrt(abs2(x1) + abs2(b))
    ξ1 = x1 + ν * sign(real(x1))
    v1 = -ν 
    v2 = b/ξ1
    τ = ξ1/ν

    a -  v2*conj(τ)*(x2 + v2'*a), c -  v2*conj(τ)*v2'*c
end

x2,x1 = X[2,2:3]
it(x1,x2)  .≈ (x1,x2)

# simplify
function it(x1, x2, c, a, b)
    ν = sqrt(abs2(x1) + abs2(b))
    ξ1 = x1 + ν * sign(real(x1))
    v2 = b/ξ1
    τ = ξ1/ν
    a -  v2*conj(τ)*(x2 + v2*a), c -  v2^2*conj(τ)*c
end

ν = sqrt(abs2(x1) + abs2(b))
ξ1 = x1 + ν * sign(real(x1))
v2 = b/ξ1
τ = ξ1/ν
a -  v2*conj(τ)*(x2 + v2'*a), c -  v2*conj(τ)*v2'*c


v = copy(x); reflector!(v); v


X

x1

x1
x2
A[1]
ql(X).factors

X
x1 = x[1]
ξ1 = x[1] 
normu = abs2(ξ1) + b^2
normu = sqrt(normu)
ν = normu * sign(real(ξ1))
ξ1 += ν



ξ1/ν


ql(X)

###


J = SymTridiagonal(Vcat([randn(5);0;0], Fill(0.0,∞)), Vcat([randn(4);0.5;0.5], Fill(0.5,∞)))
A = J + 3I
F = ql(A);
Q,L = F;
Q̃, L̃ = ql(BandedMatrix(A)[1:100,1:100]);
@test L̃[1:20,1:20] ≈ L[1:20,1:20]
@test Q̃.τ[1:20] ≈ Q.τ[1:20]
@test Q̃[1:20,1:20] ≈ Q[1:20,1:20]

x = Vcat(randn(10), Zeros(∞))
@test (Q*(Q'x))[1:100] ≈ x[1:100]
@test (Q'*(Q*x))[1:100] ≈ x[1:100]

A = J + 1.00001I
Q,L = ql(A);
Q̃, L̃ = ql(BandedMatrix(A)[1:10000,1:10000]);
@test L̃[1:20,1:20] ≈ L[1:20,1:20]
@test Q̃.τ[1:20] ≈ Q.τ[1:20]
@test Q̃[1:20,1:20] ≈ Q[1:20,1:20]


A = J + 3I
F = ql(A);
Q,L = F;

@test Q.factors === parent(L)


J = Tridiagonal(Vcat([randn(4);0.5;0.5], Fill(0.3,∞)), 
                Vcat([randn(5);0;0], Fill(0.0,∞)), 
                Vcat([randn(4);0.5;0.5], Fill(0.5,∞)))

@test BandedMatrix(J)[1:100,1:100] == J[1:100,1:100]
A = J - 2I
Q,L = ql(A)
Q̃, L̃ = ql(BandedMatrix(A)[1:1000,1:1000]);
@test L̃[1:20,1:20] ≈ L[1:20,1:20]
@test Q̃.τ[1:20] ≈ Q.τ[1:20]
@test Q̃[1:20,1:20] ≈ Q[1:20,1:20]


T = Tridiagonal(Vcat(ComplexF64[], Fill(1/2+0im,∞)), 
                Vcat(ComplexF64[], Fill(0.0im,∞)), 
                Vcat(ComplexF64[], Fill(2.0+0im,∞)))



scatter((z -> 1/(2z) + 2z).(exp.(im*(0:0.01:2π))))
ql(J-1.6im*I).L[1,1]

eigvals(Matrix(J[1:100,1:100]))

scatter(eigvals(J[1:100,1:100]+eps()*randn(100,100)))

using Plots

L*Q

Q, L = randn(

A = brand(10,10,1,1);
Q,L = ql(A);
L*Q


n = 500; kr = range(-2; length=n, stop=2); h = step(kr)

Q, R = qr(A);
    norm((Matrix(R)*Matrix(Q))[band(2)])



### Semiseparable
A = Tridiagonal((1/h^2)*ones(length(kr)-1).+0im, Vector(kr).^3 .*im .- (2/h^2) .+ 0.0im,(1/h^2)*ones(length(kr)-1).+0im)
λ = eigvals(Matrix(A))
    scatter(real.(λ), imag.(λ))

Q,R = qr(Matrix(A))    
R*Q
using Plots


Q, R = qr(Matrix(Tridiagonal(ones(length(kr)-1).+0im, Vector(kr).*im .+ 0.0im,ones(length(kr)-1).+0im)));
    (R*Q)[band(2)]

Q, L = ql(Tridiagonal(ones(length(kr)-1).+0im, Vector(kr) .+ 0.0im,ones(length(kr)-1).+0im))
L[1:100,1:100]*Q[1:100,1]

MulArray(L,Q)[1,1]

Q'

L'