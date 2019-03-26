const BlockTriPertToeplitz{T} = BlockMatrix{T,Tridiagonal{Matrix{T},Vcat{Matrix{T},1,Tuple{Vector{Matrix{T}},Fill{Matrix{T},1,Tuple{OneToInf{Int}}}}}},
                                        BlockSizes{2,Vcat{Int,1,Tuple{Int,Vcat{Int,1,Tuple{Vector{Int},InfStepRange{Int,Int}}}}}}}


function BlockTridiagonal(adjA::Adjoint{T,BlockTriPertToeplitz{T}}) where T
    A = parent(adjA)
    BlockTridiagonal(Matrix.(adjoint.(A.blocks.du)), 
                     Matrix.(adjoint.(A.blocks.d)), 
                     Matrix.(adjoint.(A.blocks.dl)))
end                                       

for op in (:-, :+)
    @eval begin
        function $op(A::BlockTriPertToeplitz{T}, λ::UniformScaling) where T 
            TV = promote_type(T,eltype(λ))
            BlockTridiagonal(Vcat(convert.(AbstractVector{Matrix{TV}}, A.blocks.dl.arrays)...), 
                             Vcat(convert.(AbstractVector{Matrix{TV}}, broadcast($op, A.blocks.d, Ref(λ)).arrays)...), 
                             Vcat(convert.(AbstractVector{Matrix{TV}}, A.blocks.du.arrays)...))
        end
        function $op(λ::UniformScaling, A::BlockTriPertToeplitz{V}) where V
            TV = promote_type(eltype(λ),V)
            BlockTridiagonal(Vcat(convert.(AbstractVector{Matrix{TV}}, broadcast($op, A.blocks.dl.arrays))...), 
                             Vcat(convert.(AbstractVector{Matrix{TV}}, broadcast($op, Ref(λ), A.blocks.d).arrays)...), 
                             Vcat(convert.(AbstractVector{Matrix{TV}}, broadcast($op, A.blocks.du.arrays))...))
        end
        $op(adjA::Adjoint{T,BlockTriPertToeplitz{T}}, λ::UniformScaling) where T = $op(BlockTridiagonal(adjA), λ)
        $op(λ::UniformScaling, adjA::Adjoint{T,BlockTriPertToeplitz{T}}) where T = $op(λ, BlockTridiagonal(adjA))
    end
end

*(a::AbstractVector, b::AbstractFill{<:Any,2,Tuple{OneTo{Int},OneToInf{Int}}}) = MulArray(a,b)


sizes_from_blocks(A::AbstractVector, ::Tuple{OneToInf{Int}}) = BlockSizes((Vcat(1, 1 .+ cumsum(length.(A))),))

function sizes_from_blocks(A::Tridiagonal, ::NTuple{2,OneToInf{Int}}) 
    sz = size.(A.d, 1), size.(A.d,2)
    BlockSizes(Vcat.(1,(c -> 1 .+ c).(cumsum.(sz))))
end

_find_block(cs::Number, i::Integer) = i ≤ cs ? 1 : 0
function _find_block(cs::Vcat, i::Integer)
    n = 0
    for a in cs.arrays
        i < first(a) && return n
        if i ≤ last(a)
            return _find_block(a, i) + n
        end
        n += length(a)
    end 
    return 0
end

print_matrix_row(io::IO,
        X::AbstractBlockVecOrMat, A::Vector,
        i::Integer, cols::AbstractVector{<:Infinity}, sep::AbstractString) = nothing

print_matrix_row(io::IO,
        X::Union{AbstractTriangular{<:Any,<:AbstractBlockMatrix},
                 Symmetric{<:Any,<:AbstractBlockMatrix},
                 Hermitian{<:Any,<:AbstractBlockMatrix}}, A::Vector,
        i::Integer, cols::AbstractVector{<:Infinity}, sep::AbstractString) = nothing        
                                        
function BlockSkylineSizes(A::BlockTriPertToeplitz, (l,u)::NTuple{2,Int})
    N = max(length(A.blocks.du.arrays[1])+1,length(A.blocks.d.arrays[1]),length(A.blocks.dl.arrays[1]))
    block_sizes = Vector{Int}(undef, N) # assume square
    block_starts = BandedMatrix{Int}(undef, (N+l,N),  (l,u))
    block_strides = Vector{Int}(undef, N)
    for J=1:N
        block_starts[max(1,J-u),J] = J == 1 ? 1 :
                            block_starts[max(1,J-1-u),J-1]+block_sizes[J-1]*block_strides[J-1]
                                
        for K=max(1,J-u)+1:J+l
            block_starts[K,J] = block_starts[K-1,J]+size(A[Block(K-1,J)],1)
        end
        block_strides[J] = block_starts[J+l,J] + size(A[Block(J+l,J)],1) - block_starts[max(1,J-u),J]
        block_sizes[J] = size(A[Block(J,J)],2)
    end

    block_stride∞ = 0
    for K=max(1,N+1-u):N+1+l
        block_stride∞ += size(A[Block(K,N+1)],1)
    end
    block_size∞ = size(A[Block(N+1,N+1)],2)

    bs∞ = fill(block_starts[max(1,N-u),N]+block_strides[N]*size(A[Block(N,N)],2):block_stride∞*block_size∞:∞, l+u+1)
    for k=2:l+u+1
        bs∞[k] = bs∞[k-1] .+ size(A[Block(N+1-u+k-1,N+1)],1)
    end

    BlockSkylineSizes(blocksizes(A),
                        _BandedMatrix(Hcat(block_starts.data, Vcat(adjoint.(bs∞)...)), ∞, l, u),
                        Vcat(block_strides, Fill(block_stride∞,∞)),
                        Fill(l,∞),Fill(u,∞))
end

function BlockBandedMatrix(A::BlockTriPertToeplitz{T}, (l,u)::NTuple{2,Int}) where T
    data = T[]                      
    append!(data,vec(A[Block.(1:1+l),Block(1)]))
    N = max(length(A.blocks.du.arrays[1])+1,length(A.blocks.d.arrays[1]),length(A.blocks.dl.arrays[1]))
    for J=2:N
        append!(data, vec(A[Block.(max(1,J-u):J+l),Block(J)]))
    end
    tl = mortar(Fill(vec(A[Block.(max(1,N+1-u):N+1+l),Block(N+1)]),∞))
    
    B = _BlockSkylineMatrix(Vcat(data,tl), BlockSkylineSizes(A, (l,u)))
end    

BlockBandedMatrix(A::BlockTriPertToeplitz) = BlockBandedMatrix(A, blockbandwidths(A))