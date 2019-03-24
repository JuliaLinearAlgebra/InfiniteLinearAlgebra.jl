const BlockTriPertToeplitz{T} = BlockMatrix{T,Tridiagonal{Matrix{T},Vcat{Matrix{T},1,Tuple{Vector{Matrix{T}},Fill{Matrix{T},1,Tuple{OneToInf{Int}}}}}},
                                        BlockSizes{2,Vcat{Int,1,Tuple{Int,Vcat{Int,1,Tuple{Vector{Int},InfStepRange{Int,Int}}}}}}}


function BlockSkylineSizes(A::BlockTriPertToeplitz)
    l,u = blockbandwidths(A)

    N = max(length(A.blocks.du.arrays[1])+1,length(A.blocks.d.arrays[1]),length(A.blocks.dl.arrays[1]))
    block_sizes = Vector{Int}(undef, N) # assume square
    block_starts = BandedMatrix{Int}(undef, (N+l,N),  (l,u))
    block_strides = Vector{Int}(undef, N)
    block_starts[1,1] = 1
    block_starts[2,1] = block_starts[1,1]+size(A[Block(1,1)],1)
    block_strides[1] = block_starts[2,1]+size(A[Block(2,1)],1)-1
    block_sizes[1] = size(A[Block(1,1)],2)
    for J=2:N
        block_starts[J-1,J] = block_starts[max(1,J-1-u),J-1]+block_sizes[J-1]*block_strides[J-1]
        for K=J-l+1:J+u
            block_starts[K,J] = block_starts[K-1,J]+size(A[Block(K-1,J)],1)
        end
        block_strides[J] = block_starts[J+u,J] + size(A[Block(J+u,J)],1) - block_starts[J-1,J]
        block_sizes[J] = size(A[Block(J,J)],2)
    end
    b = A.blocks.du.arrays[end][1]
    a = A.blocks.d.arrays[end][1]
    c = A.blocks.dl.arrays[end][1]
    block_stride∞ = size(b,1)+size(a,1)+size(c,1)
    block_size∞ = size(b,2)

    bs∞_b = block_starts[max(1,N-u),N]+block_strides[N]*size(A[Block(N,N)],2):block_stride∞*block_size∞:∞
    bs∞_a = bs∞_b .+ size(b,1)
    bs∞_c = bs∞_a .+ size(a,1)

    BlockSkylineSizes(blocksizes(A),
                        _BandedMatrix(Hcat(block_starts.data, Vcat(bs∞_b',bs∞_a',bs∞_c')), ∞, l, u),
                        Vcat(block_strides, Fill(block_stride∞,∞)),
                        Fill(l,∞),Fill(u,∞))
end

function BlockSkylineMatrix(A::BlockTriPertToeplitz{T}) where T
    data = T[]                      
    append!(data,vec(A[Block.(1:2),Block(1)]))
    N = max(length(A.blocks.du.arrays[1])+1,length(A.blocks.d.arrays[1]),length(A.blocks.dl.arrays[1]))
    for J=2:N
        append!(data, vec(A[Block.(J-1:J+1),Block(J)]))
    end
    tl = mortar(Fill(vec(Vcat(A.blocks.du.arrays[2][1],A.blocks.d.arrays[2][1],A.blocks.dl.arrays[2][1])),∞))
    
    B = _BlockSkylineMatrix(Vcat(data,tl), BlockSkylineSizes(A))
end    