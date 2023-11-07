const OneToInfCumsum = RangeCumsum{Int,OneToInf{Int}}

BlockArrays.sortedunion(::AbstractVector{<:PosInfinity}, ::AbstractVector{<:PosInfinity}) = [∞]
function BlockArrays.sortedunion(::AbstractVector{<:PosInfinity}, b)
    @assert isinf(length(b))
    b
end

function BlockArrays.sortedunion(b, ::AbstractVector{<:PosInfinity})
    @assert isinf(length(b))
    b
end
BlockArrays.sortedunion(a::OneToInfCumsum, ::OneToInfCumsum) = a



function BlockArrays.sortedunion(a::Vcat{Int,1,<:Tuple{Union{Int,AbstractVector{Int}},<:AbstractRange}},
                                 b::Vcat{Int,1,<:Tuple{Union{Int,AbstractVector{Int}},<:AbstractRange}})
    @assert a == b # TODO: generailse? Not sure how to do so in a type stable fashion
    a
end

sizes_from_blocks(A::AbstractVector, ::Tuple{OneToInf{Int}}) = (map(length,A),)
length(::BlockedUnitRange{<:InfRanges}) = ℵ₀

const OneToInfBlocks = BlockedUnitRange{OneToInfCumsum}
const OneToBlocks = BlockedUnitRange{OneToCumsum}

axes(a::OneToInfBlocks) = (a,)
axes(a::OneToBlocks) = (a,)

LazyBandedMatrices.unitblocks(a::OneToInf) = blockedrange(Ones{Int}(length(a)))

BlockArrays.dimlength(start, ::Infinity) = ℵ₀

function copy(bc::Broadcasted{<:BroadcastStyle,<:Any,typeof(*),<:Tuple{Ones{T,1,Tuple{OneToInfBlocks}},AbstractArray{V,N}}}) where {N,T,V}
    a,b = bc.args
    @assert bc.axes == axes(b)
    convert(AbstractArray{promote_type(T,V),N}, b)
end

function copy(bc::Broadcasted{<:BroadcastStyle,<:Any,typeof(*),<:Tuple{AbstractArray{T,N},Ones{V,1,Tuple{OneToInfBlocks}}}}) where {N,T,V}
    a,b = bc.args
    @assert bc.axes == axes(a)
    convert(AbstractArray{promote_type(T,V),N}, a)
end

_block_interlace_axes(::Int, ax::Tuple{BlockedUnitRange{OneToInf{Int}}}...) = (blockedrange(Fill(length(ax), ∞)),)

_block_interlace_axes(nbc::Int, ax::NTuple{2,BlockedUnitRange{OneToInf{Int}}}...) =
    (blockedrange(Fill(length(ax) ÷ nbc, ∞)),blockedrange(Fill(mod1(length(ax),nbc), ∞)))


include("infblocktridiagonal.jl")


#######
# block broadcasted
######


BroadcastStyle(::Type{<:SubArray{T,N,Arr,<:NTuple{N,BlockSlice{BlockRange{1,Tuple{II}}}},false}}) where {T,N,Arr<:BlockArray,II<:InfRanges} =
    LazyArrayStyle{N}()

# TODO: generalise following
BroadcastStyle(::Type{<:BlockArray{T,N,<:AbstractArray{<:AbstractArray{T,N},N},<:NTuple{N,BlockedUnitRange{<:InfRanges}}}}) where {T,N} = LazyArrayStyle{N}()
BroadcastStyle(::Type{<:Transpose{T,<:BlockArray{T,N,<:AbstractArray{<:AbstractArray{T,N},N},<:NTuple{N,BlockedUnitRange{<:InfRanges}}}}}) where {T,N} = LazyArrayStyle{2}()
# BroadcastStyle(::Type{<:PseudoBlockArray{T,N,<:AbstractArray{T,N},<:NTuple{N,BlockedUnitRange{<:InfRanges}}}}) where {T,N} = LazyArrayStyle{N}()
BroadcastStyle(::Type{<:BlockArray{T,N,<:AbstractArray{<:AbstractArray{T,N},N},<:NTuple{N,BlockedUnitRange{<:RangeCumsum{Int,<:InfRanges}}}}}) where {T,N} = LazyArrayStyle{N}()
# BroadcastStyle(::Type{<:PseudoBlockArray{T,N,<:AbstractArray{T,N},<:NTuple{N,BlockedUnitRange{<:RangeCumsum{Int,<:InfRanges}}}}}) where {T,N} = LazyArrayStyle{N}()


###
# KronTrav
###

_krontrav_axes(A::OneToInf{Int}, B::OneToInf{Int}) = blockedrange(oneto(length(A)))


struct InfKronTravBandedBlockBandedLayout <: AbstractLazyBandedBlockBandedLayout end
MemoryLayout(::Type{<:KronTrav{<:Any,2,<:Any,NTuple{2,BlockedUnitRange{OneToInfCumsum}}}}) = InfKronTravBandedBlockBandedLayout()

sublayout(::InfKronTravBandedBlockBandedLayout, ::Type{<:NTuple{2,BlockSlice1}}) = BroadcastBandedLayout{typeof(*)}()
sublayout(::InfKronTravBandedBlockBandedLayout, ::Type{<:NTuple{2,BlockSlice{BlockRange{1,Tuple{OneTo{Int}}}}}}) = KronTravBandedBlockBandedLayout()

copy(M::Mul{InfKronTravBandedBlockBandedLayout, InfKronTravBandedBlockBandedLayout}) = KronTrav((krontravargs(M.A) .* krontravargs(M.B))...)

_broadcast_sub_arguments(::InfKronTravBandedBlockBandedLayout, M, V) = _broadcast_sub_arguments(KronTravBandedBlockBandedLayout(), M, V)
