####
# This file shows how the ∞-dimensional QL decomposition can be used
# for spectral theory of Toeplitz operators. We investigate the
# QL decomposition on each of the examples in Trefethen & Embree.
#


using InfiniteLinearAlgebra, BandedMatrices, PyPlot

###
# Basic routines for plotting
###

function ℓ11(A,λ; kwds...) 
    try 
        abs(ql(A-λ*I; kwds...).L[1,1]) 
    catch DomainError 
        -1.0
    end
end

function branch(k)
    function(λ)
        j = sortperm(λ)[end-k+1]
        λ[j], j
    end
end

qlplot(A::AbstractMatrix; kwds...) = qlplot(BandedMatrix(A); kwds...)
function qlplot(A::BandedMatrix; branch=findmax, x=range(-4,4; length=200), y=range(-4,4;length=200), kwds...)
    z = ℓ11.(Ref(A), x' .+ y.*im; branch=branch)
    contourf(x,y,z; kwds...)
end



toepcoeffs(A::BandedMatrix) = InfiniteLinearAlgebra.rightasymptotics(A.data).args[1]
toepcoeffs(A::Adjoint) = reverse(toepcoeffs(A'))

function symbolplot(A::BandedMatrix; kwds...)
    l,u = bandwidths(A)
    a = toepcoeffs(A)
    θ = range(0,2π; length=1000)
    i = map(t -> dot(exp.(im.*(u:-1:-l).*t),a),θ)
    plot(real.(i), imag.(i); kwds...)
end


###
# Trefethen & Embree example
###

A = BandedMatrix(1 => Fill(2im,∞), 2 => Fill(-1,∞), 3 => Fill(2,∞), -2 => Fill(-4,∞), -3 => Fill(-2im,∞))
clf(); qlplot(A; x=range(-10,7; length=100), y=range(-7,8;length=100)); symbolplot(A; color=:black); title("Trefethen & Embree")
clf(); qlplot(transpose(A); x=range(-10,7; length=200), y=range(-7,8;length=200)); symbolplot(A; color=:black); title("Trefethen & Embree, transpose")

###
# limaçon
###

A = BandedMatrix(-2 => Fill(1.0,∞), -1 => Fill(1.0,∞), 1 => Fill(eps(),∞))
qlplot(A; x=range(-2,3; length=100), y=range(-2.5,2.5;length=100)); symbolplot(A; color=:black); title("Limacon")
clf(); qlplot(A; branch=findsecond, x=range(-2,3; length=100), y=range(-2.5,2.5;length=100)); symbolplot(A; color=:black); title("Limacon, second branch")
clf(); qlplot(transpose(A); x=range(-2,3; length=100), y=range(-2.5,2.5;length=100)); symbolplot(A; color=:black); title("Limacon, transpose")


### 
# bull-head
###

A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
clf(); qlplot(A; x=range(-10,7; length=100), y=range(-7,8;length=100)); symbolplot(A; color=:black); title("Bull-head")
clf(); qlplot(transpose(A); x=range(-10,7; length=100), y=range(-7,8;length=100)); symbolplot(A; color=:black); title("Bull-head, transpose")

###
# Grcar
###

A = BandedMatrix(-3 => Fill(1,∞), -2 => Fill(1,∞), -1 => Fill(1,∞), 0 => Fill(1,∞), 1 => Fill(-1,∞))
clf(); qlplot(A; x=range(-4,5; length=100), y=range(-6,5;length=100)); symbolplot(A; color=:black); title("Grcar")
clf(); qlplot(A; branch=branch(2), x=range(-4,5; length=100), y=range(-6,5;length=100)); symbolplot(A; color=:black); title("Grcar, branch 2")
clf(); qlplot(A; branch=branch(3), x=range(-4,5; length=100), y=range(-6,5;length=100)); symbolplot(A; color=:black); title("Grcar, branch 3")
clf(); qlplot(transpose(A); x=range(-4,5; length=100), y=range(-6,5;length=100)); symbolplot(A; color=:black); title("Grcar, transpose")

###
# Triangle
###

A = BandedMatrix(-2 => Fill(1/4,∞), 1 => Fill(1,∞))
clf(); qlplot(A; x=range(-2,2; length=100), y=range(-2,2;length=100)); symbolplot(A; color=:black); title("Triangle")
clf(); qlplot(transpose(A); x=range(-2,2; length=100), y=range(-2,2;length=100)); symbolplot(A; color=:black); title("Triangle, transpose")
clf(); qlplot(transpose(A); branch=branch(2), x=range(-2,2; length=100), y=range(-2,2;length=100)); symbolplot(A; color=:black); title("Triangle, transpose, branch 2")

###
# Whale
###

A = BandedMatrix(-4 => Fill(im,∞), -3 => Fill(4,∞), -2 => Fill(3+im,∞), -1 => Fill(10,∞), 
                    1 => Fill(1,∞), 2 => Fill(im,∞), 3 => Fill(-(3+2im),∞), 4=>Fill(-1,∞))

clf(); qlplot(A; x=range(-15,20; length=100), y=range(-20,20;length=100)); symbolplot(A; color=:black); title("Whale")
clf(); qlplot(transpose(A); x=range(-15,20; length=100), y=range(-20,20;length=100)); symbolplot(A; color=:black); title("Whale, transpose")

###
# Butterfly
###

A = BandedMatrix(-2 => Fill(1,∞), -1 => Fill(-im,∞), 1 => Fill(im,∞), 2 => Fill(-1,∞))
clf(); qlplot(A; x=range(-3,3; length=100), y=range(-3,3;length=100)); symbolplot(A; color=:black); title("Butterfly")
clf(); qlplot(A; branch=branch(2), x=range(-3,3; length=100), y=range(-3,3;length=100)); symbolplot(A; color=:black); title("Butterfly, branch 2")
clf(); qlplot(transpose(A); x=range(-3,3; length=100), y=range(-3,3;length=100)); symbolplot(A; color=:black); title("Butterfly")
clf(); qlplot(transpose(A); branch=branch(2), x=range(-3,3; length=100), y=range(-3,3;length=100)); symbolplot(A; color=:black); title("Butterfly")
