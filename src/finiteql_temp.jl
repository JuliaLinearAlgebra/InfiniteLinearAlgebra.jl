###
# Experimental adaptive finite section QL
###
mutable struct QLFiniteSectionFactor{T} <: LazyArrays.AbstractCachedMatrix{T}
    data::AbstractMatrix{T}
    M
    factor::Symbol
    datasize::Integer
    tol
    QLFiniteSectionFactor{T}(D::AbstractMatrix{T}, M, factor::Symbol, N::Integer, tol) where T = new{T}(D, M, factor, N, tol)
end

size(::QLFiniteSectionFactor) = (ℵ₀, ℵ₀)

mutable struct AdaptiveQLFiniteSection{T}
    Q::QLFiniteSectionFactor{T}
    L::LowerTriangular{T, QLFiniteSectionFactor{T}}
    tol
end

# Computes the initial data for the finite section based QL decomposition
function AdaptiveQLFiniteSection(A::AbstractMatrix{T}, tol = eps(float(T)), maxN = 10000) where T
    @assert size(A) == (ℵ₀, ℵ₀) # only makes sense for infinite matrices
    j = 50 # We initialize with a 50 × 50 block that is adaptively expanded
    Lerr = one(T)
    N = j
    checkinds = max(1,j-bandwidth(A,1)-bandwidth(A,2))
    @inbounds Ls = ql(A[checkinds:N,checkinds:N]).L[2:j-checkinds+1,2:j-checkinds+1]
    @inbounds while Lerr > tol
        # compute QL for small finite section and large finite section
        Ll = ql(A[checkinds:2N,checkinds:2N]).L[2:j-checkinds+1,2:j-checkinds+1]
        # compare bottom right sections and stop if desired level of convergence achieved
        Lerr = norm(Ll-Ls,2)
        if N == maxN
            error("Reached max. iterations in finite section QL without convergence to desired tolerance.")
        end
        Ls = Ll
        N = 2*N
    end
    F = ql(A[1:(N÷2),1:(N÷2)])
    return AdaptiveQLFiniteSection{float(T)}(QLFiniteSectionFactor{float(T)}(F.Q[1:50,1:50], A, :Q, 50, tol),LowerTriangular(QLFiniteSectionFactor{float(T)}(F.L[1:50,1:50], A, :L, 50, tol)),tol)
end

# Resize and filling functions for cached implementation
function resizedata!(K::QLFiniteSectionFactor, nm::Integer)
    νμ = K.datasize
    if nm > νμ
        olddata = copy(K.data)
        K.data = similar(K.data, nm, nm)
        K.data[axes(olddata)...] = olddata
        inds = νμ:nm
        cache_filldata!(K, inds)
        K.datasize = size(K.data,1)
    end
    K
end

function cache_filldata!(A::QLFiniteSectionFactor{T}, inds::UnitRange{Int}) where T
    j = maximum(inds)
    maxN = 1000*j
    Lerr = one(T)
    N = j
    checkinds = max(1,j-bandwidth(A.M,1)-bandwidth(A.M,2))
    @inbounds Ls = ql(A.M[checkinds:N,checkinds:N]).L[2:j-checkinds+1,2:j-checkinds+1]
    @inbounds while Lerr > A.tol
        # compute QL for small finite section and large finite section
        Ll = ql(A.M[checkinds:2N,checkinds:2N]).L[2:j-checkinds+1,2:j-checkinds+1]
        # compare bottom right sections and stop if desired level of convergence achieved
        Lerr = norm(Ll-Ls,2)
        if N == maxN
            error("Reached max. iterations in finite section QL without convergence to desired tolerance.")
        end
        Ls = Ll
        N = 2*N
    end
    if A.factor == :Q
        A.data = ql(A.M[1:(N÷2),1:(N÷2)]).Q[1:j,1:j]
    else
        A.data = ql(A.M[1:(N÷2),1:(N÷2)]).L[1:j,1:j]
    end
end

function getindex(K::QLFiniteSectionFactor, k::Integer, j::Integer)
    resizedata!(K, max(k,j))
    K.data[k, j]
end
function getindex(K::QLFiniteSectionFactor, k::Integer, jr::UnitRange{Int})
    resizedata!(K, max(k,maximum(jr)))
    K.data[k, jr]
end
function getindex(K::QLFiniteSectionFactor, kr::UnitRange{Int}, j::Integer)
    resizedata!(K, max(j,maximum(kr)))
    K.data[kr, j]
end
function getindex(K::QLFiniteSectionFactor, kr::UnitRange{Int}, jr::UnitRange{Int})
    resizedata!(K, max(maximum(jr),maximum(kr)))
    K.data[kr, jr]
end

function show(io::IO, mime::MIME{Symbol("text/plain")}, F::AdaptiveQLFiniteSection)
    summary(io, F); println(io)
    println(io, "Q factor:")
    show(io, mime, F.Q)
    println(io, "\nL factor:")
    show(io, mime, F.L)
end

*(L::LowerTriangular{T, QLFiniteSectionFactor{T}}, b::LayoutVector) where T = LazyArrays.ApplyArray(*, L, b)

Base.iterate(S::AdaptiveQLFiniteSection) = (S.Q, Val(:L))
Base.iterate(S::AdaptiveQLFiniteSection, ::Val{:L}) = (S.L, Val(:done))
Base.iterate(S::AdaptiveQLFiniteSection, ::Val{:done}) = nothing