sizes_from_blocks(A::AbstractVector, ::Tuple{OneToInf{Int}}) = (map(length,A),)


# for LazyLay in (:(BlockLayout{LazyLayout}), :(TriangularLayout{UPLO,UNIT,BlockLayout{LazyLayout}} where {UPLO,UNIT}))
#     @eval begin
#         combine_mul_styles(::$LazyLay) = LazyArrayApplyStyle()
#         mulapplystyle(::AbstractQLayout, ::$LazyLay) = LazyArrayApplyStyle()
#     end
# end

# BlockArrays.blockbroadcaststyle(::LazyArrayStyle{N}) where N = LazyArrayStyle{N}()


include("infblocktridiagonal.jl")

