
a = [-0.1,0.2,0.3]
A = BandedMatrix(-2 => Vcat([1], Fill(1,∞)),  0 => Vcat(a, Fill(0,∞)), 1 => Vcat([1/4], Fill(1/4,∞)))
ℓ = λ -> ql(A-λ*I).L[1,1]
x,y = range(-2,2; length=200),range(-2,2;length=200)
    z = abs.(ℓ.(x' .+ y.*im))
    contour(x,y,z; nlevels=50, title="z^2 + 1/(4z)")
     scatter!(eigvals(Matrix(A[1:100,1:100])))
     scatter!(eigvals(Matrix(A[1:100,1:100]')))

a = [-0.5,0.2,0.3]
A = BandedMatrix(-2 => Vcat([1], Fill(1,∞)),  0 => Vcat(a, Fill(0,∞)), 1 => Vcat([1/4], Fill(1/4,∞)))
ℓ = λ -> ql(A-λ*I).L[1,1]
x,y = range(-2,2; length=200),range(-2,2;length=200)
    z = abs.(ℓ.(x' .+ y.*im))
    contour(x,y,z; nlevels=50, title="z^2 + 1/(4z)")
     scatter!(eigvals(Matrix(A[1:100,1:100])); label="A finite section")
     scatter!(eigvals(Matrix(A[1:100,1:100]')); label="A' finite section")

a = [-0.5,0.2,0.5]
A = BandedMatrix(-2 => Vcat([1], Fill(1,∞)),  0 => Vcat(a, Fill(0,∞)), 1 => Vcat([1/4], Fill(1/4,∞)))
ℓ = λ -> ql(A-λ*I).L[1,1]
x,y = range(-2,2; length=200),range(-2,2;length=200)
    z = abs.(ℓ.(x' .+ y.*im))
    contour(x,y,z; nlevels=50, title="z^2 + 1/(4z)")
    scatter!(eigvals(Matrix(A[1:100,1:100]')); label="A' finite section")
    scatter!(eigvals(Matrix(A[1:100,1:100])); label="A finite section")

x = range(-2,2,length=1_000)
    plot(x, abs.(ℓ.(x.+eps()im)); label="abs(L[1,1])")

a = [-0.1,0.2,0.3]
A = BandedMatrix(-2 => Vcat([1], Fill(1,∞)),  0 => Vcat(a, Fill(0,∞)), 1 => Vcat([1/4], Fill(1/4,∞)))
Q,L = ql(A-(0.5+0.1im)*I)
Q[1:10,1:11]*L[1:11,1:10] - (A-(0.5+0.1im)*I)[1:10,1:10]

ℓ(0.5+0.1im)
ql((A-(0.5+0.1im)*I)[1:1000,1:1000]).L[1,1]

