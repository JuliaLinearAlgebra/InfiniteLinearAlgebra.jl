const OneToInfCumsum = InfiniteArrays.RangeCumsum{Int,OneToInf{Int}}
const OneToCumsum = InfiniteArrays.RangeCumsum{Int,OneTo{Int}}

BlockArrays.sortedunion(::AbstractVector{Infinity}, ::AbstractVector{Infinity}) = [∞]
function BlockArrays.sortedunion(::AbstractVector{Infinity}, b)
    @assert isinf(length(b))
    b
end

function BlockArrays.sortedunion(b, ::AbstractVector{Infinity})
    @assert isinf(length(b))
    b
end
BlockArrays.sortedunion(a::OneToInfCumsum, ::OneToInfCumsum) = a
BlockArrays.sortedunion(a::OneToCumsum, ::OneToCumsum) = a
function BlockArrays.sortedunion(a::RangeCumsum{<:Any,<:InfStepRange}, b::RangeCumsum{<:Any,<:InfStepRange})
    @assert a == b
    a
end


function BlockArrays.sortedunion(a::Vcat{Int,1,<:Tuple{<:AbstractVector{Int},InfStepRange{Int,Int}}},
                                 b::Vcat{Int,1,<:Tuple{<:AbstractVector{Int},InfStepRange{Int,Int}}})
    @assert a == b
    a
end

sizes_from_blocks(A::AbstractVector, ::Tuple{OneToInf{Int}}) = (map(length,A),)

const OneToInfBlocks = BlockedUnitRange{OneToInfCumsum}
const OneToBlocks = BlockedUnitRange{OneToCumsum}

axes(a::OneToInfBlocks) = (a,)
axes(a::OneToBlocks) = (a,)


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
BroadcastStyle(::Type{<:PseudoBlockArray{T,N,<:AbstractArray{T,N},<:NTuple{N,BlockedUnitRange{<:InfRanges}}}}) where {T,N} = LazyArrayStyle{N}()
BroadcastStyle(::Type{<:BlockArray{T,N,<:AbstractArray{<:AbstractArray{T,N},N},<:NTuple{N,BlockedUnitRange{<:RangeCumsum{Int,<:InfRanges}}}}}) where {T,N} = LazyArrayStyle{N}()
BroadcastStyle(::Type{<:PseudoBlockArray{T,N,<:AbstractArray{T,N},<:NTuple{N,BlockedUnitRange{<:RangeCumsum{Int,<:InfRanges}}}}}) where {T,N} = LazyArrayStyle{N}()


BlockArrays._length(::BlockedUnitRange, ::OneToInf) = ∞
BlockArrays._last(::BlockedUnitRange, ::OneToInf) = ∞

###
# KronTrav
###

_krontrav_axes(A::NTuple{N,OneToInf{Int}}, B::NTuple{N,OneToInf{Int}}) where N =
     @. blockedrange(OneTo(length(A)))
