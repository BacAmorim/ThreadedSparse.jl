module ThreadedSparse

using SparseArrays, ChunkSplitters

export maskedspmatmul!, initaccumulators

include("masked_sparse_matmul.jl")

end
