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
        a = reverse(T.data.args[1])
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
        a = reverse(A.data.args[1])
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
        a = reverse(A.data.args[1])
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
    F∞ = QL(F∞.factors, Vcat(zero(ComplexF64),F∞.τ.args[2]))
    Q∞, L∞ = F∞
    @test Q∞[1:10,1:12] * L∞[1:12,1:10] ≈ T[1:10,1:10]

    n = 100_000; Q,L = ql(B[1:n,1:n]);
    @test Q.τ[3] ≈ F∞.τ[2]

    data = bandeddata(B).args[1]
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
    @test Q.τ[1:10] ≈ Vcat(F.τ, F∞.τ.args[2])[1:10]

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


function Toep_L11(T)
    l,u = bandwidths(T)
    @assert u == 2
    # shift by one
    H = _BandedMatrix(T.data, ∞, l+1, 1)
    Q1,L1 = ql(H)

    d = Q1[1:3,1]'T[1:1+l,1]
    ℓ = Q1.factors.data.args[2].args[1][2:end] # new L
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
    ℓ = F1.factors.data.args[2].args[1][2:end]
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
        T2 = _BandedMatrix(rightasymptotics(parent(L1).data).args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l, u)
        l1 = L1[1,1]
        

        A2 = [[D1 l1 zeros(1,10-size(D1,2)-1)]; T2[1:10-1,1:10]]
        @test Q1[1:13,1:10]'A[1:13,1:10] ≈ A2
        

        B2 = _BandedMatrix(T2.data, ∞, l+u-2, 2)
        D2, Q2, L2 = reduceband(B2)
        l2 = L2[1,1]
        T3 = _BandedMatrix(rightasymptotics(parent(L2).data).args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l+1, u-1)
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
        T2 = _BandedMatrix(rightasymptotics(parent(L1).data).args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l, u)
        l1 = L1[1,1]
        
        A2 = [[D1 l1 zeros(1,10-size(D1,2)-1)]; T2[1:10-1,1:10]]
        @test Q1[1:13,1:10]'A[1:13,1:10] ≈ A2
        
        B2 = _BandedMatrix(T2.data, ∞, l+u-2, 2)
        D2, Q2, L2 = reduceband(B2)
        l2 = L2[1,1]

        # peroidic tail
        T3 = _BandedMatrix(rightasymptotics(parent(L2).data).args[2], ∞, l+1, u-1)
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
        T2 = _BandedMatrix(rightasymptotics(parent(L1).data).args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l, u)
        l1 = L1[1,1]
        
        A2 = [[D1 l1 zeros(1,10-size(D1,2)-1)]; T2[1:10-1,1:10]]
        @test Q1[1:13,1:10]'A[1:13,1:10] ≈ A2
        
        B2 = _BandedMatrix(T2.data, ∞, l+u-2, 2)
        D2, Q2, L2 = reduceband(B2)
        l2 = L2[1,1]

        # peroidic tail
        T3 = _BandedMatrix(rightasymptotics(parent(L2).data).args[1][2:end] * Ones{ComplexF64}(1,∞), ∞, l+1, u-1)
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

_Lrightasymptotics(D::Vcat) = D.args[2]
_Lrightasymptotics(D::ApplyArray) = D.args[1][2:end] * Ones{ComplexF64}(1,∞)
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