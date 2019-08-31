const TriToeplitz{T} = Tridiagonal{T,Fill{T,1,Tuple{OneToInf{Int}}}}
const ConstRows{T} = ApplyMatrix{T,typeof(*),<:Tuple{<:AbstractVector,<:AbstractFill}}
const InfToeplitz{T} = BandedMatrix{T,<:ConstRows{T},OneToInf{Int}}
const PertToeplitz{T} = BandedMatrix{T,<:Hcat{T,<:Tuple{Matrix{T},<:ConstRows{T}}},OneToInf{Int}}

const SymTriPertToeplitz{T} = SymTridiagonal{T,Vcat{T,1,Tuple{Vector{T},Fill{T,1,Tuple{OneToInf{Int}}}}}}
const TriPertToeplitz{T} = Tridiagonal{T,Vcat{T,1,Tuple{Vector{T},Fill{T,1,Tuple{OneToInf{Int}}}}}}
const AdjTriPertToeplitz{T} = Adjoint{T,Tridiagonal{T,Vcat{T,1,Tuple{Vector{T},Fill{T,1,Tuple{OneToInf{Int}}}}}}}
const InfBandedMatrix{T,V<:AbstractMatrix{T}} = BandedMatrix{T,V,OneToInf{Int}}

# Construct InfToeplitz
function BandedMatrix{T}(kv::Tuple{Vararg{Pair{<:Integer,<:Fill{<:Any,1,Tuple{OneToInf{Int}}}}}},
                         mn::NTuple{2,Integer},
                         lu::NTuple{2,Integer}) where T
    m,n = mn
    @assert isinf(n)
    l,u = lu
    t = zeros(T, u+l+1)
    for (k,v) in kv
        p = length(v)
        t[u-k+1] = v.value
    end

    return _BandedMatrix(t * Ones{T}(1,∞), m, l, u)
end

function BandedMatrix{T}(kv::Tuple{Vararg{Pair{<:Integer,<:Vcat{<:Any,1,<:Tuple{<:AbstractVector,Fill{<:Any,1,Tuple{OneToInf{Int}}}}}}}},
                         mn::NTuple{2,Integer},
                         lu::NTuple{2,Integer}) where T
    m,n = mn
    @assert isinf(n)
    l,u = lu
    M = mapreduce(x -> length(x.second.args[1]) + max(0,x.first), max, kv) # number of data rows
    data = zeros(T, u+l+1, M)
    t = zeros(T, u+l+1)
    for (k,v) in kv
        a,b = v.args
        p = length(a)
        t[u-k+1] = b.value
        if k ≤ 0
            data[u-k+1,1:p] = a
            data[u-k+1,p+1:end] .= b.value
        else
            data[u-k+1,k+1:k+p] = a
            data[u-k+1,k+p+1:end] .= b.value
        end
    end

    return _BandedMatrix(Hcat(data, t * Ones{T}(1,∞)), m, l, u)
end


function BandedMatrix(Ac::Adjoint{T,<:InfToeplitz}) where T
    A = parent(Ac)
    l,u = bandwidths(A)
    a = A.data.args[1]
    _BandedMatrix(reverse(conj(a)) * Ones{T}(1,∞), ∞, u, l)
end

function BandedMatrix(Ac::Transpose{T,<:InfToeplitz}) where T
    A = parent(Ac)
    l,u = bandwidths(A)
    a = A.data.args[1]
    _BandedMatrix(reverse(a) * Ones{T}(1,∞), ∞, u, l)
end

function BandedMatrix(Ac::Adjoint{T,<:PertToeplitz}) where T
    A = parent(Ac)
    l,u = bandwidths(A)
    a,b = A.data.args
    Ac_fd = BandedMatrix(_BandedMatrix(Hcat(a, b[:,1:l+1]), size(a,2)+l, l, u)')
    _BandedMatrix(Hcat(Ac_fd.data, reverse(conj(b.args[1])) * Ones{T}(1,∞)), ∞, u, l)
end

function BandedMatrix(Ac::Transpose{T,<:PertToeplitz}) where T
    A = parent(Ac)
    l,u = bandwidths(A)
    a,b = A.data.args
    Ac_fd = BandedMatrix(transpose(_BandedMatrix(Hcat(a, b[:,1:l+1]), size(a,2)+l, l, u)))
    _BandedMatrix(Hcat(Ac_fd.data, reverse(b.args[1]) * Ones{T}(1,∞)), ∞, u, l)
end


for op in (:-, :+)
    @eval begin
        function $op(A::SymTriPertToeplitz{T}, λ::UniformScaling) where T 
            TV = promote_type(T,eltype(λ))
            dv = Vcat(convert.(AbstractVector{TV}, A.dv.args)...)
            ev = Vcat(convert.(AbstractVector{TV}, A.ev.args)...)
            SymTridiagonal(broadcast($op, dv, Ref(λ.λ)), ev)
        end
        function $op(λ::UniformScaling, A::SymTriPertToeplitz{V}) where V
            TV = promote_type(eltype(λ),V)
            SymTridiagonal(convert(AbstractVector{TV}, broadcast($op, Ref(λ.λ), A.dv)), 
                           convert(AbstractVector{TV}, broadcast($op, A.ev)))
        end
        function $op(A::SymTridiagonal{T,<:AbstractFill}, λ::UniformScaling) where T 
            TV = promote_type(T,eltype(λ))
            SymTridiagonal(convert(AbstractVector{TV}, broadcast($op, A.dv, Ref(λ.λ))),
                           convert(AbstractVector{TV}, A.ev))
        end

        function $op(A::TriPertToeplitz{T}, λ::UniformScaling) where T 
            TV = promote_type(T,eltype(λ))
            Tridiagonal(Vcat(convert.(AbstractVector{TV}, A.dl.args)...), 
                        Vcat(convert.(AbstractVector{TV}, broadcast($op, A.d, λ.λ).args)...), 
                        Vcat(convert.(AbstractVector{TV}, A.du.args)...))
        end
        function $op(λ::UniformScaling, A::TriPertToeplitz{V}) where V
            TV = promote_type(eltype(λ),V)
            Tridiagonal(Vcat(convert.(AbstractVector{TV}, broadcast($op, A.dl.args))...), 
                        Vcat(convert.(AbstractVector{TV}, broadcast($op, λ.λ, A.d).args)...), 
                        Vcat(convert.(AbstractVector{TV}, broadcast($op, A.du.args))...))
        end
        function $op(adjA::AdjTriPertToeplitz{T}, λ::UniformScaling) where T 
            A = parent(adjA)
            TV = promote_type(T,eltype(λ))
            Tridiagonal(Vcat(convert.(AbstractVector{TV}, A.du.args)...), 
                        Vcat(convert.(AbstractVector{TV}, broadcast($op, A.d, λ.λ).args)...), 
                        Vcat(convert.(AbstractVector{TV}, A.dl.args)...))
        end
        function $op(λ::UniformScaling, adjA::AdjTriPertToeplitz{V}) where V
            A = parent(adjA)
            TV = promote_type(eltype(λ),V)
            Tridiagonal(Vcat(convert.(AbstractVector{TV}, broadcast($op, A.du.args))...), 
                        Vcat(convert.(AbstractVector{TV}, broadcast($op, λ.λ, A.d).args)...), 
                        Vcat(convert.(AbstractVector{TV}, broadcast($op, A.dl.args))...))
        end

        function $op(λ::UniformScaling, A::InfToeplitz{V}) where V
            l,u = bandwidths(A)
            TV = promote_type(eltype(λ),V)
            a = convert(AbstractVector{TV}, $op.(A.data.args[1]))
            a[u+1] += λ.λ
            _BandedMatrix(a*Ones{TV}(1,∞), ∞, l, u)
        end

        function $op(A::InfToeplitz{T}, λ::UniformScaling) where T
            l,u = bandwidths(A)
            TV = promote_type(T,eltype(λ))
            a = AbstractVector{TV}(A.data.args[1])
            a[u+1] = $op(a[u+1], λ.λ)
            _BandedMatrix(a*Ones{TV}(1,∞), ∞, l, u)
        end

        function $op(λ::UniformScaling, A::PertToeplitz{V}) where V
            l,u = bandwidths(A)
            TV = promote_type(eltype(λ),V)
            a, t = convert.(AbstractVector{TV}, A.data.args)
            b = $op.(t.args[1])
            a[u+1,:] += λ.λ
            b[u+1] += λ.λ
            _BandedMatrix(Hcat(a, b*Ones{TV}(1,∞)), ∞, l, u)
        end

        function $op(A::PertToeplitz{T}, λ::UniformScaling) where T
            l,u = bandwidths(A)
            TV = promote_type(T,eltype(λ))
            ã, t = A.data.args
            a = AbstractArray{TV}(ã)
            b = AbstractVector{TV}(t.args[1])
            a[u+1,:] .= $op.(a[u+1,:],λ.λ)
            b[u+1] = $op(b[u+1], λ.λ)
            _BandedMatrix(Hcat(a, b*Ones{TV}(1,∞)), ∞, l, u)
        end
    end
end



####
# Conversions to BandedMatrix
####        

function BandedMatrix(A::PertToeplitz{T}, (l,u)::Tuple{Int,Int}) where T
    @assert A.u == u # Not implemented
    a, b = A.data.args
    t = b.args[1] # topelitz part
    t_pad = vcat(t,Zeros(l-A.l))
    data = Hcat([vcat(a,Zeros{T}(l-A.l,size(a,2))) repeat(t_pad,1,l)], t_pad * Ones{T}(1,∞))
    _BandedMatrix(data, ∞, l, u)
end

function BandedMatrix(A::SymTriPertToeplitz{T}, (l,u)::Tuple{Int,Int}) where T
    a,a∞ = A.dv.args
    b,b∞ = A.ev.args
    n = max(length(a), length(b)+1) + 1
    data = zeros(T, l+u+1, n)
    data[u,2:length(b)+1] .= b
    data[u,length(b)+2:end] .= b∞.value
    data[u+1,1:length(a)] .= a
    data[u+1,length(a)+1:end] .= a∞.value
    data[u+2,1:length(b)] .= b
    data[u+2,length(b)+1:end] .= b∞.value
    _BandedMatrix(Hcat(data, [Zeros{T}(u-1); b∞.value; a∞.value; b∞.value; Zeros{T}(l-1)] * Ones{T}(1,∞)), ∞, l, u)
end

function BandedMatrix(A::SymTridiagonal{T,Fill{T,1,Tuple{OneToInf{Int}}}}, (l,u)::Tuple{Int,Int}) where T
    a∞ = A.dv
    b∞ = A.ev
    n = 2
    data = zeros(T, l+u+1, n)
    data[u,2:end] .= b∞.value
    data[u+1,1:end] .= a∞.value
    data[u+2,1:end] .= b∞.value
    _BandedMatrix(Hcat(data, [Zeros{T}(u-1); b∞.value; a∞.value; b∞.value; Zeros{T}(l-1)] * Ones{T}(1,∞)), ∞, l, u)
end

function BandedMatrix(A::TriPertToeplitz{T}, (l,u)::Tuple{Int,Int}) where T
    a,a∞ = A.d.args
    b,b∞ = A.du.args
    c,c∞ = A.dl.args
    n = max(length(a), length(b)+1, length(c)-1) + 1
    data = zeros(T, l+u+1, n)
    data[u,2:length(b)+1] .= b
    data[u,length(b)+2:end] .= b∞.value
    data[u+1,1:length(a)] .= a
    data[u+1,length(a)+1:end] .= a∞.value
    data[u+2,1:length(c)] .= c
    data[u+2,length(c)+1:end] .= c∞.value
    _BandedMatrix(Hcat(data, [Zeros{T}(u-1); b∞.value; a∞.value; c∞.value; Zeros{T}(l-1)] * Ones{T}(1,∞)), ∞, l, u)
end

function BandedMatrix(A::Tridiagonal{T,Fill{T,1,Tuple{OneToInf{Int}}}}, (l,u)::Tuple{Int,Int}) where T
    a∞ = A.d
    b∞ = A.du
    c∞ = A.dl
    n = 2
    data = zeros(T, l+u+1, n)
    data[u,2:end] .= b∞.value
    data[u+1,1:end] .= a∞.value
    data[u+2,1:end] .= c∞.value
    _BandedMatrix(Hcat(data, [Zeros{T}(u-1); b∞.value; a∞.value; c∞.value; Zeros{T}(l-1)] * Ones{T}(1,∞)), ∞, l, u)
end

function InfToeplitz(A::Tridiagonal{T,Fill{T,1,Tuple{OneToInf{Int}}}}, (l,u)::Tuple{Int,Int}) where T
    a∞ = A.d
    b∞ = A.du
    c∞ = A.dl
    _BandedMatrix([Zeros{T}(u-1); b∞.value; a∞.value; c∞.value; Zeros{T}(l-1)] * Ones{T}(1,∞), ∞, l, u)
end

InfToeplitz(A::Tridiagonal{T,Fill{T,1,Tuple{OneToInf{Int}}}}) where T = InfToeplitz(A, bandwidths(A))