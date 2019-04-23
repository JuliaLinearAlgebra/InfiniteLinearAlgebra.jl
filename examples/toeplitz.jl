using Revise, InfiniteBandedMatrices, BlockBandedMatrices, BlockArrays, BandedMatrices, LazyArrays, FillArrays, MatrixFactorizations, Plots
import MatrixFactorizations: reflectorApply!, QLPackedQ
import InfiniteBandedMatrices: blocktailiterate, _ql, qltail
import BandedMatrices: bandeddata,_BandedMatrix

# Bull's head
A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
ℓ = λ -> try abs(ql(A-λ*I).L[1,1]) catch DomainError 
            (-1) end
x,y = range(-4,4; length=200),range(-4,4;length=200)
z = ℓ.(x' .+ y.*im)
contourf(x,y,z; nlevels=100)
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