using Revise, InfiniteBandedMatrices, BlockBandedMatrices, BlockArrays, BandedMatrices, LazyArrays, FillArrays, MatrixFactorizations, Plots
import MatrixFactorizations: reflectorApply!, QLPackedQ
import InfiniteBandedMatrices: blocktailiterate, _ql, qltail, rightasymptotics
import BandedMatrices: bandeddata,_BandedMatrix

function ℓ11(A,λ) 
    try 
        abs(ql(A-λ*I).L[1,1]) 
    catch DomainError 
        (-1) 
    end
end

function qlplot(A; x=range(-4,4; length=200), y=range(-4,4;length=200), kwds...)
    z = ℓ11.(Ref(A), x' .+ y.*im)
    contourf(x,y,z; kwds...)
end

###
# Non-normal
###

A = BandedMatrix(-1 => Fill(1/4,∞), 1 => Fill(1,∞))
qlplot(A; title="A")
qlplot(BandedMatrix(A'); title="A'", linewidth=0, nlevels=100)
θ = range(0,2π-0.5; length=1000)
a = z -> z + 0.25/z
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)



A = BandedMatrix(-10 => Fill(4.0,∞), 1 => Fill(1,∞))
ql(A+0im*I)




θ = range(0,2π-0.5; length=1000)
a = z -> 4z^10 + 1/z
θ = range(0,2π; length=1000)
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)

import InfiniteBandedMatrices: tail_de
a = reverse(A.data.applied.args[1]) .+ 0im

a = randn(10) .+ im*randn(10); a[1] += 10; a[end] = 1;
de = tail_de(a)

@which tail_de(a)

ql([transpose(a); 0 transpose(de)])

ql([transpose(a); 0 transpose(de2)])

de

contourf(x,y,angle.(a.(x' .+ y.*im)))

A'
Fun(a, Laurent())

A'


a(1.0exp(0.5im))

#### To be cleaned


function reduceband(A)
        l,u = bandwidths(A)
        H = _BandedMatrix(A.data, ∞, l+u-1, 1)
        Q1,L1 = ql(H)
        D = Q1[1:l+u+1,1:1]'A[1:l+u+1,1:u-1]
        D, Q1, L1
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
# Bull's head
A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
ℓ = λ -> try abs(ql(A-λ*I).L[1,1]) catch DomainError 
            (-1) end
x,y = range(-4,4; length=200),range(-4,4;length=200)
z = ℓ.(x' .+ y.*im)
contourf(x,y,z; nlevels=100, title="A", linewidth=0)
θ = range(0,2π; length=1000)
a = z -> 2im/z + z^2 + 7/10 * z^3
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)

Ac = BandedMatrix(A')

ℓ = λ -> try abs(qdL(Ac-conj(λ)*I)[1,1]) catch DomainError 
            (-1.0) end
x,y = range(-4,4; length=100),range(-4,4;length=100)
@time z = ℓ.(x' .+ y.*im)
contourf(x,y,z; nlevels=100, title="A'", linewidth=0.0)
θ = range(0,2π; length=1000)
a = z -> 2im/z + z^2 + 7/10 * z^3
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)

# Grcar
A = BandedMatrix(-3 => Fill(1,∞), -2 => Fill(1,∞), -1 => Fill(1,∞), 0 => Fill(1,∞), 1 => Fill(-1,∞))
ℓ = λ -> try abs(ql(A-λ*I).L[1,1]) catch DomainError 
        -1 end
x,y = range(-4,4; length=200),range(-4,4;length=200)
z = ℓ.(x' .+ y.*im)

θ = range(0,2π; length=1000)
a = z -> -1/z + 1 + z + z^2 + z^3
contourf(x,y,z; nlevels=50)
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)


# Triangle
A = BandedMatrix(-2 => Fill(1/4,∞), 1 => Fill(1,∞))
ℓ = λ -> try abs(ql(A-λ*I).L[1,1]) catch DomainError 
            (-1) end
x,y = range(-2,2; length=200),range(-2,2;length=200)
z = ℓ.(x' .+ y.*im)
θ = range(0,2π; length=1000)
a = z -> z^2/4 + 1/z

contourf(x,y,z; nlevels=50)
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)


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



A = BandedMatrix(2 => Fill(1/4,∞), -1 => Fill(1,∞), -2 => Fill(0.0, ∞))
@time Toep_L11(A-(0.1+0im)I)

T = A-(0.1+0im)I


ℓ = λ -> abs(Toep_L11(A-λ*I)) 
x,y = range(-2,2; length=51),range(-2,2;length=50)
z = ℓ.(x' .+ y.*im)
θ = range(0,2π; length=1000)
a = z -> z^2/4 + 1/z

contour(x,y,z; nlevels=50)
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)