const OneToInfCumsum = InfiniteArrays.RangeCumsum{Int,OneToInf{Int}}

BlockArrays.sortedunion(a::OneToInfCumsum, ::OneToInfCumsum) = a

function BlockArrays.sortedunion(a::Vcat{Int,1,<:Tuple{<:AbstractVector{Int},InfStepRange{Int,Int}}},
                                 b::Vcat{Int,1,<:Tuple{<:AbstractVector{Int},InfStepRange{Int,Int}}})
    @assert a == b
    a
end

sizes_from_blocks(A::AbstractVector, ::Tuple{OneToInf{Int}}) = (map(length,A),)

const OneToInfBlocks = BlockedUnitRange{OneToInfCumsum}

axes(a::OneToInfBlocks) = (a,)


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

_block_interlace_axes(::Int, ax::Tuple{OneToInf{Int}}...) = (blockedrange(Fill(length(ax), ∞)),)

_block_interlace_axes(nbc::Int, ax::NTuple{2,OneToInf{Int}}...) =
    (blockedrange(Fill(length(ax) ÷ nbc, ∞)),blockedrange(Fill(mod1(length(ax),nbc), ∞)))


include("infblocktridiagonal.jl")


#######
# block broadcasted
######




map(::typeof(length), A::BroadcastArray{OneTo{Int},1,Type{OneTo}}) = A.args[1]
map(::typeof(length), A::BroadcastArray{<:Fill,1,Type{Fill}}) = A.args[2]
map(::typeof(length), A::BroadcastArray{<:Zeros,1,Type{Zeros}}) = A.args[1]
map(::typeof(length), A::BroadcastArray{<:Vcat,1,Type{Vcat}}) = broadcast(+,map.(length,A.args)...)
broadcasted(::LazyArrayStyle{1}, ::typeof(length), A::BroadcastArray{OneTo{Int},1,Type{OneTo}}) =
    A.args[1]
broadcasted(::LazyArrayStyle{1}, ::typeof(length), A::BroadcastArray{<:Fill,1,Type{Fill}}) =
    A.args[2]

BlockArrays._length(::BlockedUnitRange, ::OneToInf) = ∞
BlockArrays._last(::BlockedUnitRange, ::OneToInf) = ∞

###
# KronTrav
###

_krontrav_axes(A::NTuple{N,OneToInf{Int}}, B::NTuple{N,OneToInf{Int}}) where N =
     @. blockedrange(OneTo(length(A)))
