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

function qlplot(A; branch=findmax, x=range(-4,4; length=200), y=range(-4,4;length=200), kwds...)
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
clf(); qlplot(BandedMatrix(transpose(A)); x=range(-10,7; length=200), y=range(-7,8;length=200)); symbolplot(A; color=:black); title("Trefethen & Embree, transpose")

ql(A+(2im)*I).L

symbolplot(A')
ql((A-(5.1)*I)[1:1000,1:1000]).L
BandedMatrix(view(A,:,3:∞))

ql(A)
p =plot(); symbolplot!(A)


###
# bull-head
###

A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
clf(); qlplot(A; x=range(-4,4; length=100), y=range(-4,4;length=100))
symbolplot(A; color=:black)

clf(); qlplot(BandedMatrix(transpose(A)); x=range(-4,4; length=100), y=range(-4,4;length=100))
symbolplot(A; color=:black)

At = BandedMatrix(transpose(A))
Q, L = ql(At)
Q[1:10,1:14]*L[1:14,1:10]
L
ql(At[1:N,1:N])

A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞)); At = A = B = BandedMatrix(transpose(A)); 
Q,L = ql(B)
Q[1:10,1:14] * L[1:14,1:10]


Q,L = ql(At);
Q[1:100,1:104] * L[1:104,1:100] - At[1:100,1:100] |> norm
A = B
import InfiniteLinearAlgebra: ql_pruneband
A = At
Q1,H1 = ql_pruneband(A)
Q1[1:10,1:15] * H1[1:15,1:10] - A[1:10,1:10] |> norm
Q2,H2 = ql_pruneband(H1)
Q2[1:10,1:15] * H2[1:15,1:10] - H1[1:10,1:10] |> norm
Q3,L = ql(H2)
Q3[1:10,1:15] * L[1:15,1:10] - H2[1:10,1:10] |> norm

Q2[1:10,1:15] * Q3[1:15,1:19] * L[1:19,1:10] - H1[1:10,1:10] |> norm
Q1[1:10,1:15] * Q2[1:15,1:19] * Q3[1:19,1:23] * L[1:23,1:10] - A[1:10,1:10] |> norm
Q,L = ql(At)
ql(At).Qs[2]

Q.Qs[1][1:10,1:15] - Q1[1:10,1:15]

Q.Qs[1][1:10,1:15] * Q.Qs[2][1:15,1:19] * Q.Qs[3][1:19,1:23] * L[1:23,1:10] - A[1:10,1:10] |> norm

e = zeros(ComplexF64,∞); e[1] = 1; lmul!(Q.Qs[1]', e); lmul!(Q.Qs[2]', e);  lmul!(Q.Qs[3]', e)
@which lmul!(Q', e)

e = zeros(ComplexF64,∞); e[1] = 1; lmul!((Q').Qs[end], e)


@which (Q')[1,1]
Q[1,1]



A = H1
A_hess = A[:,u:end]

B = BandedMatrix(A_hess, (bandwidth(A_hess,1)+bandwidth(A_hess,2),bandwidth(A_hess,2)))
l,u = bandwidths(B)
@assert u == 1
T = toeptail(B)
# Tail QL
F∞ = ql_hessenberg(T; kwds...)
Q∞, L∞ = F∞

Q∞[1:10,1:15] * L∞[1:15,1:10] - T[1:10,1:10] |> norm

data = bandeddata(B).args[1]
B̃ = _BandedMatrix(data, size(data,2), l,u)
B̃[end,end] = L∞[1,1]
B̃[end:end,end-l+1:end-1] = adjoint(Q∞)[1:1,1:l-1]*T[l:2(l-1),1:l-1]

m = size(data,2)-1
Q_tail = [Matrix(I,m,m) zeros(m,N-m); zeros(N-m,m)  Q∞[1:N-m,1:N-m]]
(Q_tail' * A_hess[1:N,1:N])[1:7,1:7] - B̃ 


# populate finite data and do ql!
F = ql(B̃)

# fill in data with L∞
B̃ = _BandedMatrix(B̃.data, size(data,2)+l, l,u)
B̃[size(data,2)+1:end,end-l+1:end] = adjoint(Q∞)[2:l+1,1:l+1] * T[l:2l,1:l]


# combine finite and infinite data
H = Hcat(B̃.data, rightasymptotics(F∞.factors.data))
QLHessenberg(_BandedMatrix(H, ∞, l, 1), Vcat( LowerHessenbergQ(F.Q).q, F∞.q))


@which ql_hessenberg(A_hess)
Q,L = ql_hessenberg(A_hess; kwds...)

Q[1:10,1:15] * L[1:15,1:10] - A_hess[1:10,1:10]

Q1[1:10,1:15] * H2[1:15,1:10] - H1[1:10,1:10] 
@which ql_pruneband(H1)


QN = ([F.Q zeros(6,N-6); zeros(N-6,6) Matrix(I,N-6,N-6)]' * [Matrix(I,5,5) zeros(5,N-5); zeros(N-5,5) Q∞[1:N-5,1:N-5]]')'



ql_pruneband(H1)

H2


(Q2*H2)[1:10,1:10] - H1[1:10,1:10]

using LazyArrays
ApplyArray(*,Q1,Q1')

Q = Q1[1:100,1:100]; Q'Q

H1


A

###
# Non-normal
###

A = BandedMatrix(-1 => Vcat(Float64[], Fill(1/4,∞)), 0 => Vcat([1im],Fill(0,∞)), 1 => Vcat(Float64[], Fill(1,∞)))
qlplot(A; title="A", linewidth=0, x=range(-2,2; length=100), y=range(-2,2;length=100))
symbolplot!(A; linewidth=2.0, linecolor=:blue, legend=false)
qlplot(BandedMatrix(A'); title="A', 1st", linewidth=0, nlevels=100, x=range(-2,2; length=100), y=range(-2,2;length=100))
symbolplot!(A; linewidth=2.0, linecolor=:blue, legend=false)
qlplot(BandedMatrix(A'); branch=findsecond, title="A'", linewidth=0, nlevels=100, x=range(-2,2; length=100), y=range(-2,2;length=100))


heatmap!(x,y,fill(-100,length(y),length(x)); legend=false, color=:grays, fillalpha=(z -> isnan(z) ? 0.0 : 1.0).(z))

Matrix((A)[1:1000,1:1000]) |> eigvals |> scatter

ql(A-2I)


θ = range(0,2π; length=1000)
a = z -> z + 0.25/z
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)





A = BandedMatrix(-10 => Fill(4.0,∞), 1 => Fill(1,∞))
ql(A+0im*I)




θ = range(0,2π-0.5; length=1000)
a = z -> 4z^10 + 1/z
θ = range(0,2π; length=1000)
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)

import InfiniteLinearAlgebra: tail_de
a = reverse(A.data.args[1]) .+ 0im

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
A = BandedMatrix(-3 => Fill(7/10,10), -2 => Fill(1,11), 1 => Fill(2im,9))

A = BandedMatrix(-3 => Fill(7/10,∞), -2 => Fill(1,∞), 1 => Fill(2im,∞))
qlplot(A;  title="largest", linewidth=0)


qlplot(A; branch=findsecond, title="second largest", linewidth=0)



sortp

ql(A-(1+2im)*I)

θ = range(0,2π; length=1000)
a = z -> 2im/z + z^2 + 7/10 * z^3
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)
qlplot(A; branch=2, title="branch=2", linewidth=0)




θ = range(0,2π; length=1000)
a = z -> 2im/z + z^2 + 7/10 * z^3
plot!(a.(exp.(im.*θ)); linewidth=2.0, linecolor=:blue, legend=false)


ql(A - (5+2im)*I; branch=1)

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
        ℓ = Q1.factors.data.args[2].args[1][2:end] # new L
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