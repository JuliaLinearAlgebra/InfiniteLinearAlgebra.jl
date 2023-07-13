
mutable struct AdaptiveQRData{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}}
    data::CachedMatrix{T,DM,M}
    τ::Vector{T}
    ncols::Int
end

function AdaptiveQRData(::Union{SymmetricLayout{<:AbstractBandedLayout},AbstractBandedLayout}, A::AbstractMatrix{T}) where T
    l,u = bandwidths(A)
    FT = float(T)
    data = BandedMatrix{FT}(undef,(2l+u+1,0),(l,l+u)) # pad super
    AdaptiveQRData(CachedArray(data,A), Vector{FT}(), 0)
end

function AdaptiveQRData(::AbstractAlmostBandedLayout, A::AbstractMatrix{T}) where T
    l,u = almostbandwidths(A)
    r = almostbandedrank(A)
    data = AlmostBandedMatrix(Zeros{T}(2l+u+1,0),(l,l+u),r) # pad super

    AdaptiveQRData(CachedArray(data,A,(0,0)), Vector{T}(), 0)
end

function AdaptiveQRData(::AbstractBlockLayout, A::AbstractMatrix{T}) where T
    l,u = blockbandwidths(A)
    m,n = axes(A)
    data = BlockBandedMatrix{T}(undef,(m[Block.(1:2l+u+1)],n[Block.(1:0)]),(l,l+u)) # pad super
    AdaptiveQRData(CachedArray(data,A), Vector{T}(), 0)
end

AdaptiveQRData(A::AbstractMatrix{T}) where T = AdaptiveQRData(MemoryLayout(A), A)

function partialqr!(F::AdaptiveQRData{<:Any,<:BandedMatrix}, n::Int)
    if n > F.ncols
        l,u = bandwidths(F.data.data)
        resizedata!(F.data,n+l,n+u);
        resize!(F.τ,n);
        ñ = F.ncols
        τ = view(F.τ,ñ+1:n);
        if l ≤ 0
            zero!(τ)
        else
            factors = view(F.data.data,ñ+1:n+l,ñ+1:n+u);
            _banded_qr!(factors, τ, n-ñ)
        end
        F.ncols = n
    end
    F
end

function partialqr!(F::AdaptiveQRData{<:Any,<:AlmostBandedMatrix}, n::Int)
    if n > F.ncols
        l,u = almostbandwidths(F.data.data)
        resizedata!(F.data,n+l,n+l+u);
        resize!(F.τ,n);
        ñ = F.ncols
        τ = view(F.τ,ñ+1:n)
        if l ≤ 0
            zero!(τ)
        else
            factors = view(F.data.data,ñ+1:n+l,ñ+1:n+l+u)
            _almostbanded_qr!(factors, τ, n-ñ)
        end
        F.ncols = n
    end
    F
end

function partialqr!(F::AdaptiveQRData{<:Any,<:BlockSkylineMatrix}, N::Block{1})
    n = last(axes(F.data,2)[N])
    if n > F.ncols
        l,u = blockbandwidths(F.data.data)
        resizedata!(F.data,N+l,N+u);
        resize!(F.τ,n);
        ñ = F.ncols
        Ñ = ñ == 0 ? Block(0) : findblock(axes(F.data,2), ñ)
        τ = view(F.τ,ñ+1:n)
        if l ≤ 0
            zero!(τ)
        else
            factors = view(F.data.data,Ñ+1:N+l,Ñ+1:N+u);
            _blockbanded_qr!(factors, PseudoBlockVector(τ, (axes(factors,2)[Block(1):(N-Ñ)],)), N-Ñ)
        end
        F.ncols = n
    end
    F
end

partialqr!(F::AdaptiveQRData{<:Any,<:BlockSkylineMatrix}, n::Int) =
    partialqr!(F, findblock(axes(F.data,2), n))

struct AdaptiveQRFactors{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} <: LayoutMatrix{T}
    data::AdaptiveQRData{T,DM,M}
end

struct AdaptiveLayout{M} <: MemoryLayout end
MemoryLayout(::Type{AdaptiveQRFactors{T,DM,M}}) where {T,DM,M} = AdaptiveLayout{typeof(MemoryLayout(DM))}()
triangularlayout(::Type{Tri}, ::ML) where {Tri, ML<:AdaptiveLayout} = Tri{ML}()
transposelayout(A::AdaptiveLayout{ML}) where ML = AdaptiveLayout{typeof(transposelayout(ML()))}()

size(F::AdaptiveQRFactors) = size(F.data.data)
axes(F::AdaptiveQRFactors) = axes(F.data.data)
bandwidths(F::AdaptiveQRFactors) = bandwidths(F.data.data)

axes(A::UpperOrLowerTriangular{<:Any,<:AdaptiveQRFactors}) = axes(parent(A))

function colsupport(F::AdaptiveQRFactors, j)
    partialqr!(F.data, maximum(j))
    colsupport(F.data.data.data, j)
end

function rowsupport(F::AdaptiveQRFactors, j)
    partialqr!(F.data, maximum(j)+bandwidth(F,2))
    rowsupport(F.data.data.data, j)
end

function blockcolsupport(F::AdaptiveQRFactors, J)
    partialqr!(F.data, maximum(J))
    blockcolsupport(F.data.data.data, J)
end


function getindex(F::AdaptiveQRFactors, k::Int, j::Int)
    partialqr!(F.data, j)
    F.data.data[k,j]
end

colsupport(F::QRPackedQ{<:Any,<:AdaptiveQRFactors}, j) = 1:last(colsupport(F.factors, j))
rowsupport(F::QRPackedQ{<:Any,<:AdaptiveQRFactors}, j) = first(rowsupport(F.factors, j)):size(F,2)

blockcolsupport(F::QRPackedQ{<:Any,<:AdaptiveQRFactors}, j) = blockcolsupport(F.factors, j)


struct AdaptiveQRTau{T,DM<:AbstractMatrix{T},M<:AbstractMatrix{T}} <: LayoutVector{T}
    data::AdaptiveQRData{T,DM,M}
end

size(F::AdaptiveQRTau) = (size(F.data.data,1),)

function getindex(F::AdaptiveQRTau, j::Int)
    partialqr!(F.data, j)
    F.data.τ[j]
end

getR(Q::QR, ::NTuple{2,InfiniteCardinal{0}}) = UpperTriangular(Q.factors)


function adaptiveqr(A)
    data = AdaptiveQRData(A)
    QR(AdaptiveQRFactors(data), AdaptiveQRTau(data))
end

_qr(::AbstractBandedLayout, ::NTuple{2,OneToInf{Int}}, A) = adaptiveqr(A)
_qr(::AbstractAlmostBandedLayout, ::NTuple{2,OneToInf{Int}}, A) = adaptiveqr(A)
__qr(_, ::NTuple{2,InfiniteCardinal{0}}, A) = adaptiveqr(A)
_qr(::AbstractBlockBandedLayout, ::NTuple{2,InfiniteCardinal{0}}, A) = adaptiveqr(A)
_factorize(::AbstractBandedLayout, ::NTuple{2,OneToInf{Int}}, A) = qr(A)

partialqr!(F::QR, n) = partialqr!(F.factors, n)
partialqr!(F::AdaptiveQRFactors, n) = partialqr!(F.data, n)

#########
# getindex
#########

getindex(Q::QRPackedQ{<:Any,<:AdaptiveQRFactors,<:AdaptiveQRTau}, I::AbstractVector{Int}, J::AbstractVector{Int64}) =
    hcat((Q[:,j][I] for j in J)...)

#########
# lmul!
#########

_view_QRPackedQ(A, kr, jr) = QRPackedQ(view(A.factors.data.data.data,kr,jr), view(A.τ.data.τ,jr))
function materialize!(M::MatLmulVec{<:QRPackedQLayout{<:AdaptiveLayout},<:PaddedLayout})
    A,B = M.A,M.B
    sB = size(paddeddata(B),1)
    partialqr!(A.factors.data,sB)
    jr = oneto(sB)
    m = maximum(colsupport(A,jr))
    kr = oneto(m)
    resizedata!(B, m)
    b = paddeddata(B)
    lmul!(_view_QRPackedQ(A,kr,jr), b)
    B
end

function resizedata_chop!(v::CachedVector, tol)
    c = paddeddata(v)
    n = length(c)
    k_tol = n
    for k = n:-1:1
        if abs(c[k]) > tol
            v.datasize = (k_tol,)
            return v
        end
    end
    v.datasize = (0,)
    v
end

function resizedata_chop!(v::PseudoBlockVector, tol)
    c = paddeddata(v.blocks)
    n = length(c)
    k_tol = choplength(c, tol)
    ax = axes(v,1)
    K = findblock(ax,k_tol)
    n2 = last(ax[K])
    resize!(v.blocks.data, n2)
    zero!(view(v.blocks.data, n+1:n2))
    v.blocks.datasize = (n2,)
    v
end

_norm(x::Number) = abs(x)

function materialize!(M::MatLmulVec{<:AdjQRPackedQLayout{<:AdaptiveLayout},<:PaddedLayout}; tolerance=floatmin(real(eltype(M))))
    adjA,B = M.A,M.B
    COLGROWTH = 1000 # rate to grow columns

    require_one_based_indexing(B)
    A = parent(adjA)
    mA, nA = size(A.factors)
    mB = length(B)
    if mA != mB
        throw(DimensionMismatch("matrix A has dimensions ($mA,$nA) but B has dimensions ($mB, $nB)"))
    end
    Bdata = paddeddata(B)
    sB = size(Bdata,1)
    l,u = bandwidths(A.factors)
    if l == 0 # diagonal special case
        return B
    end

    jr = 1:min(COLGROWTH,nA)

    @inbounds begin
        while first(jr) < nA
            j = first(jr)
            cs = colsupport(A.factors, last(jr))
            cs_max = maximum(cs)
            kr = j:cs_max
            resizedata!(B, min(cs_max,mB))
            Bdata = paddeddata(B)
            if (j > sB && maximum(_norm,view(Bdata,j:last(colsupport(A.factors,j)))) ≤ tolerance)
                break
            end
            partialqr!(A.factors.data, min(cs_max,nA))
            Q_N = _view_QRPackedQ(A, kr, jr)
            lmul!(Q_N', view(Bdata, kr))
            jr = last(jr)+1:min(last(jr)+COLGROWTH,nA)
        end
    end
    resizedata_chop!(B, tolerance)
end

function resizedata!(B::PseudoBlockVector, M::Block{1})
    resizedata!(B.blocks, last(axes(B,1)[M]))
    B
end


function _view_QRPackedQ(A, KR::BlockRange, JR::BlockRange)
    jr = UnitRange{Int}(axes(A,2)[JR])
    QRPackedQ(view(A.factors.data.data.data,KR,JR), view(A.τ.data.τ,jr))
end

function materialize!(M::MatLmulVec{<:QRPackedQLayout{<:AdaptiveLayout{<:AbstractBlockBandedLayout}},<:PaddedLayout})
    A,B_in = M.A,M.B
    sB = length(paddeddata(B_in))
    ax1,ax2 = axes(A.factors.data.data)
    B = PseudoBlockVector(B_in, (ax2,))
    SB = findblock(ax2, sB)
    partialqr!(A.factors.data,SB)
    JR = Block(1):SB
    M = maximum(blockcolsupport(A.factors,JR))
    KR = Block(1):M
    resizedata!(B, M)
    b = paddeddata(B)
    lmul!(_view_QRPackedQ(A,KR,JR), b)
    B
end

function materialize!(M::MatLmulVec{<:AdjQRPackedQLayout{<:AdaptiveLayout{<:AbstractBlockBandedLayout}},<:PaddedLayout}; tolerance=1E-30)
    adjA,B_in = M.A,M.B
    A = parent(adjA)
    T = eltype(M)
    COLGROWTH = 300 # rate to grow columns
    ax1,ax2 = axes(A.factors.data.data)
    B = PseudoBlockVector(B_in, (ax1,))

    SB = findblock(ax1, length(paddeddata(B_in)))
    MA, NA = blocksize(A.factors.data.data.array)
    JR = Block(1):findblock(ax1,COLGROWTH)

    @inbounds begin
        while Int(first(JR)) < NA
            J = first(JR)
            J_last = last(JR)
            CS = blockcolsupport(A.factors.data.data.array, J_last)
            CS_max = maximum(CS)
            KR = J:CS_max
            resizedata!(B, CS_max)
            mx = maximum(abs,view(B,J:last(blockcolsupport(A.factors.data.data.array,J))))
            isnan(mx) && error("Not-a-number encounted")
            if J > SB && mx ≤ tolerance
                break
            end
            partialqr!(A.factors.data, CS_max)
            kr = first(ax1[KR[1]]):last(ax1[KR[end]])
            jr = first(ax2[JR[1]]):last(ax2[JR[end]])
            Q_N = QRPackedQ(view(A.factors.data.data.data,KR,JR), view(A.τ.data.τ,jr));
            lmul!(Q_N', view(B.blocks.data, kr))
            JR = last(JR)+1:findblock(ax1,last(jr)+COLGROWTH)
        end
    end
    resizedata_chop!(B, tolerance)
end


function _lmul_copymutable(A::Union{AbstractMatrix{T},AbstractQ{T}}, x::AbstractVector{S}; kwds...) where {T,S}
    TS = promote_op(matprod, T, S)
    lmul!(A, Base.copymutable(convert(AbstractVector{TS},x)); kwds...)
end

(*)(A::QRPackedQ{T,<:AdaptiveQRFactors}, x::AbstractVector; kwds...) where {T} = _lmul_copymutable(A, x; kwds...)
(*)(A::AdjointQtype{T,<:QRPackedQ{T,<:AdaptiveQRFactors}}, x::AbstractVector; kwds...) where {T} = _lmul_copymutable(A, x; kwds...)
(*)(A::QRPackedQ{T,<:AdaptiveQRFactors}, x::LayoutVector; kwds...) where {T} = _lmul_copymutable(A, x; kwds...)
(*)(A::AdjointQtype{T,<:QRPackedQ{T,<:AdaptiveQRFactors}}, x::LayoutVector; kwds...) where {T} = _lmul_copymutable(A, x; kwds...)

function ldiv!(R::UpperTriangular{<:Any,<:AdaptiveQRFactors}, B::CachedVector{<:Any,<:Any,<:Zeros{<:Any,1}})
    n = B.datasize[1]
    partialqr!(parent(R).data, n)
    materialize!(Ldiv(UpperTriangular(view(parent(R).data.data.data,oneto(n),oneto(n))), view(B.data,oneto(n))))
    B
end


function ldiv!(R::UpperTriangular{<:Any,<:AdaptiveQRFactors}, B::PseudoBlockArray)
    n = B.blocks.datasize[1]
    N = findblock(axes(R,1),n)
    partialqr!(parent(R).data, N)
    resizedata!(B,N)
    KR = Block(1):N
    materialize!(Ldiv(UpperTriangular(view(parent(R).data.data.data,KR,KR)), paddeddata(B)))
    B
end



ldiv!(dest::AbstractVector, F::QR{<:Any,<:AdaptiveQRFactors}, b::AbstractVector; kwds...) = ldiv!(F, copyto!(dest, b); kwds...)
ldiv!(F::QR{<:Any,<:AdaptiveQRFactors}, b::AbstractVector; kwds...) = ldiv!(F.R, lmul!(F.Q',b; kwds...))
ldiv!(F::QR{<:Any,<:AdaptiveQRFactors}, b::LayoutVector; kwds...) = ldiv!(F.R, lmul!(F.Q',b; kwds...))
\(F::QR{<:Any,<:AdaptiveQRFactors}, B::AbstractVector; kwds...) = ldiv!(F.R, *(F.Q', B; kwds...))
\(F::QR{<:Any,<:AdaptiveQRFactors}, B::LayoutVector; kwds...) = ldiv!(F.R, *(F.Q', B; kwds...))


factorize(A::BandedMatrix{<:Any,<:Any,<:OneToInf}) = qr(A)
qr(A::SymTridiagonal{T,<:AbstractFill{T,1,Tuple{OneToInf{Int}}}}) where T = adaptiveqr(A)

copy(M::Mul{<:QRPackedQLayout{<:AdaptiveLayout}}) = ApplyArray(*, M.A, M.B)
copy(M::Mul{<:Any,<:QRPackedQLayout{<:AdaptiveLayout}}) = ApplyArray(*, M.A, M.B)
