module ThreadedSparse

using SparseArrays, ChunkSplitters

include("masked_sparse_matmul.jl")

export maskedspmatmul!, initaccumulators

end
