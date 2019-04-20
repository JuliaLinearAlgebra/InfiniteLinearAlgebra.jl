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


