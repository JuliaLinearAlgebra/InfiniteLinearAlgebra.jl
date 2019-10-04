using InfiniteLinearAlgebra, BandedMatrices, PyPlot


function ℓ11(A,λ; kwds...) 
    try 
        abs(ql(A-λ*I; kwds...).L[1,1]) 
    catch DomainError 
        -1.0
    end
end

function findsecond(λ) 
    j = sortperm(λ)[end-1]
    λ[j], j
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
clf();qlplot(A; x=range(-10,7; length=100), y=range(-7,8;length=100)); symbolplot(A; color=:black); title("Trefethen & Embree")
clf(); qlplot(transpose(A); x=range(-10,7; length=200), y=range(-7,8;length=200)); symbolplot(A; color=:black); title("Trefethen & Embree, transpose")

###
# limaçon
###

A = BandedMatrix(-2 => Fill(1.0,∞), -1 => Fill(1.0,∞), 1 => Fill(eps(),∞))
qlplot(A; x=range(-2,3; length=100), y=range(-2.5,2.5;length=100)); symbolplot(A; color=:black); title("Limacon")
clf(); qlplot(A; branch=findsecond, x=range(-2,3; length=100), y=range(-2.5,2.5;length=100)); symbolplot(A; color=:black); title("Limacon, second branch")
clf(); qlplot(transpose(A); x=range(-2,3; length=100), y=range(-2.5,2.5;length=100)); symbolplot(A; color=:black); title("Limacon, transpose")


ql(A+(0.5+0.000001im)*I; branch=findsecond)

### 
# bull-head
###

A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
clf();qlplot(A; x=range(-10,7; length=100), y=range(-7,8;length=100)); symbolplot(A; color=:black); title("Bull-head")
clf(); qlplot(transpose(A); x=range(-10,7; length=100), y=range(-7,8;length=100)); symbolplot(A; color=:black); title("Bull-head, transpose")


###
# Grcar
###
