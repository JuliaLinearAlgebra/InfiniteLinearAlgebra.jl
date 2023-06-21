
mutable struct AdaptiveCholeskyFactors{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} <: LayoutMatrix{T}
    data::CachedMatrix{T,DM,M}
    ncols::Int
end

size(U::AdaptiveCholeskyFactors) = size(U.data.array)
bandwidths(A::AdaptiveCholeskyFactors) = (0,bandwidth(A.data,2))

const SymmetricBandedLayouts = Union{SymTridiagonalLayout,SymmetricLayout{<:AbstractBandedLayout},DiagonalLayout}

function AdaptiveCholeskyFactors(::SymmetricBandedLayouts, S::AbstractMatrix{T}) where T
    A = parent(S)
    l,u = bandwidths(A)
    data = BandedMatrix{T}(undef,(0,0),(l,u)) # pad super
    AdaptiveCholeskyFactors(CachedArray(data,A), 0)
end
AdaptiveCholeskyFactors(A::AbstractMatrix{T}) where T = AdaptiveCholeskyFactors(MemoryLayout(A), A)
MemoryLayout(::Type{AdaptiveCholeskyFactors{T,DM,M}}) where {T,DM,M} = AdaptiveLayout{typeof(MemoryLayout(DM))}()


function partialcholesky!(F::AdaptiveCholeskyFactors{T,<:BandedMatrix}, n::Int) where T
    if n > F.ncols
        _,u = bandwidths(F.data.array)
        resizedata!(F.data,n+u,n+u);
        ncols = F.ncols
        kr = ncols+1:n
        factors = view(F.data.data,kr,kr)
        banded_chol!(factors, UpperTriangular)
        # multiply remaining columns
        kr2 = max(n-u+1,kr[1]):n
        U1 = UpperTriangular(view(F.data.data,kr2,kr2))
        B = view(F.data.data,kr2,n+1:n+u)
        ldiv!(U1',B)
        muladd!(-one(T),B',B,one(T),view(F.data.data,n+1:n+u,n+1:n+u))
        F.ncols = n
    end
    F
end

function getindex(F::AdaptiveCholeskyFactors, k::Int, j::Int)
    partialcholesky!(F, max(k,j))
    F.data.data[k,j]
end


adaptivecholesky(A) = Cholesky(AdaptiveCholeskyFactors(A), :U, 0)


ArrayLayouts._cholesky(::SymmetricBandedLayouts, ::NTuple{2,OneToInf{Int}}, A, ::CNoPivot) = adaptivecholesky(A)

function colsupport(F::AdaptiveCholeskyFactors, j)
    partialcholesky!(F, maximum(j)+bandwidth(F,2))
    colsupport(F.data.data, j)
end

function rowsupport(F::AdaptiveCholeskyFactors, j)
    partialcholesky!(F, maximum(j)+bandwidth(F,2))
    rowsupport(F.data.data, j)
end

colsupport(F::AdjOrTrans{<:Any,<:AdaptiveCholeskyFactors}, j) = rowsupport(parent(F), j)
rowsupport(F::AdjOrTrans{<:Any,<:AdaptiveCholeskyFactors}, j) = colsupport(parent(F), j)

function materialize!(M::MatLdivVec{<:TriangularLayout{'L','N',<:AdaptiveLayout},<:PaddedLayout})
    A,B = M.A,M.B
    T = eltype(M)
    COLGROWTH = 1000 # rate to grow columns
    tol = floatmin(real(T))

    require_one_based_indexing(B)
    mA, nA = size(A)
    mB = length(B)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    sB = B.datasize[1]
    l,_ = bandwidths(A)

    jr = 1:min(COLGROWTH,nA)

    P = parent(parent(A))

    @inbounds begin
        while first(jr) < nA
            j = first(jr)
            cs = colsupport(A, last(jr))
            cs_max = maximum(cs)
            kr = j:cs_max
            resizedata!(B, min(cs_max,mB))
            if (j > sB && maximum(abs,view(B.data,j:last(colsupport(P,j)))) ≤ tol)
                break
            end
            partialcholesky!(P, min(cs_max,nA))
            U_N = UpperTriangular(view(P.data.data, jr, jr))
            ldiv!(U_N', view(B.data, jr))
            jr1 = last(jr)-l+1:last(jr)
            jr2 = last(jr)+1:last(jr)+l
            muladd!(-one(T), view(P.data.data, jr1,jr2)', view(B.data,jr1), one(T), view(B.data,jr2))
            jr = last(jr)+1:min(last(jr)+COLGROWTH,nA)
        end
    end
    B
end

function ldiv!(R::UpperTriangular{<:Any,<:AdaptiveCholeskyFactors}, B::CachedVector{<:Any,<:Any,<:Zeros{<:Any,1}})
    n = B.datasize[1]
    partialcholesky!(parent(R), n)
    ldiv!(UpperTriangular(view(parent(R).data.data,oneto(n),oneto(n))), view(B.data,oneto(n)))
    B
end
