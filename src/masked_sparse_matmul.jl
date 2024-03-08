using SparseArrays, ChunkSplitters

#=
Masked Sparse Accumulator as described in "Parallel Algorithms for Masked Sparse Matrix-Matrix Products", https://arxiv.org/abs/2111.09947

Is formed by two vectors of length = ncols(B): states and values
There are 3 possible states NOTALLOWED, ALLOWED, SET

- Initially, all of the states are NOTALLOWED

the following methods are defined:

- setallowed: changes a state NOTALLOWED -> ALLOWED

- insert (k, v):
    - if state[k] == ALLOWED: change state ALLOWED -> SET and values[k] = v
    - if state[k] == SET: accumulate values[k] += v

- remove (k):
    - if state[k] == SET: changes state SET -> NOTALLOWED and returns values[k]
    - if state[k] != SET: return nothing
=#
struct MaskedSparseAccumulator{Tv}
    values::Vector{Tv}
    states::Vector{Int}
end

const ALLOWED = 0
const SET = 1
const NOTALLOWED = 2

function MaskedSparseAccumulator(t::T, n) where T<:DataType
    values = zeros(t, n)
    states = fill(NOTALLOWED, n)
    return MaskedSparseAccumulator(values, states)
end

function init!(accum::MaskedSparseAccumulator{Tv}) where Tv
    fill!(accum.states, NOTALLOWED)
end

function setallowed!(accum::MaskedSparseAccumulator, i)
    accum.states[i] = ALLOWED
    return nothing
end

function isallowed(accum::MaskedSparseAccumulator, i)
    return accum.states[i] != NOTALLOWED
end

function insert!(accum::MaskedSparseAccumulator, i, value)
    if accum.states[i] == SET
        accum.values[i] += value
        
    elseif accum.states[i] == ALLOWED
        accum.states[i] = SET
        accum.values[i] = value
   
    end
    return nothing
end

function isset(accum::MaskedSparseAccumulator, i)
    return accum.states[i] == SET
end

function remove!(accum::MaskedSparseAccumulator, i)
    return accum.values[i]
end

function setnotallowed!(accum::MaskedSparseAccumulator, i)
    accum.states[i] = NOTALLOWED
end

#=
Modified version of MaskedSparseAccumulator: MaskedBitAccumulator.
The idea is that it is that the difference between the states ALLOWED and SET is only used in the methods insert! and isset/remove!

In insert!(accum, i, value) 
- if the state==ALLOWED, we just set  accum.values[i] = value;
- if state==SET we increment accum.values[i] += value
These two can be combined if we make sure that the values are all initially set to 0. If that is the case, we can always increment.

In isset(accum, i)/remove!(accum, i)
- if only retrive the value if the state is SET. If it is only ALLOWED, we do nothing. We can check instead if the stored value != 0

therefore we see that we can make the equivalence 
MaskedSparseAccumulator  <->   MaskedBitAccumulator

   state == NOTALLOWED   <->     state == NOTALLOWEDmod
   state == ALLOWED      <->     state == ALLOWEDmod, value == 0
   state == SET          <->     state == ALLOWEDmod, value != 0

Therefore, we only need two state values: ALLOWED and NOTALLOWED. Therefore, instead of using a standard vector to store the states, we can use a BitVector. We can therefore save some memory. 

=#
struct MaskedBitAccumulator{Tv}
    values::Vector{Tv}
    states::BitVector
end


function MaskedBitAccumulator(t::T, n) where T<:DataType
    values = zeros(t, n)
    states = BitVector(fill(false, n))
    return MaskedBitAccumulator(values, states)
end

function init!(accum::MaskedBitAccumulator{Tv}) where Tv
    fill!(accum.states, false)
end

function setallowed!(accum::MaskedBitAccumulator, i)
    accum.states[i] = true
    return nothing
end

function isallowed(accum::MaskedBitAccumulator, i)
    return accum.states[i] == true
end

function insert!(accum::MaskedBitAccumulator, i, value)
    accum.values[i] += value
    return nothing
end

function isset(accum::MaskedBitAccumulator, i)
    return !iszero(accum.values[i]) 
end

function remove!(accum::MaskedBitAccumulator{Tv}, i) where Tv
    val = accum.values[i]
    accum.values[i] = zero(Tv)
    return val
end

function setnotallowed!(accum::MaskedBitAccumulator, i)
    accum.states[i] = false
end

#=
Some thoughts:

- each thread gets a chunk os the columns of B.
- each thread gets its own accumulator. This accumulator is reused between each column of the same chunk
- we can avoid allocations by allocating one accumulator per thread at the begining.
- When performing masked C = M \odot A.B, the pattern of non-zeros of the matrix C depends both on the pattern of non-zero of the bare multiplication A.B which is then filtered by the mask M. 
- Usually only non-zeros that survive from this two-step process are stored. Meaning we have to find the strucutre of C. Two common approaches are: one-step and two-step
- If however, we fix the structure of the output matrix C to be the same as the mask M we do not need to find the structure of C.
=#
function initaccumulators(C, A, B, nchunks, accum_type)
    
    typeC = eltype(C)
    nrowsC = size(C, 1)
    
    if accum_type == :MSA

        return [MaskedSparseAccumulator(typeC, nrowsC) for i in 1:nchunks]
    
    elseif accum_type == :MBA
        
        return [MaskedBitAccumulator(typeC, nrowsC) for i in 1:nchunks]
        
    end

end

function maskedspmatmul!(C, A, B, accumulators)
    
    fill!(nonzeros(C), zero(eltype(C)))
    
    nrowsC, ncolsC = size(C)
    nrowsA, ncolsA = size(A)
    nrowsB, ncolsB = size(B)
    nrowsC == nrowsA && ncolsC == ncolsB && ncolsA == nrowsB || throw(DimensionMismatch())
    
    nchunks = length(accumulators)
    
    Threads.@threads for (colrange, ichunk) in chunks(1:ncolsC, nchunks)
        for j in colrange
            masked_SpMatColVec!(C, A, B, j, accumulators[ichunk])
        end
    end
    
    return C
    
end

function masked_SpMatColVec!(C, A, B, j, accum)
    
    rowsC = rowvals(C)
    valsC = nonzeros(C)
    rowsA = rowvals(A)
    valsA = nonzeros(A)
    rowsB = rowvals(B)
    valsB = nonzeros(B)
    
    #init!(accum)
    
    for n in nzrange(C, j)
        i = rowsC[n]
        setallowed!(accum, i)
    end
    
    
    for n in nzrange(B, j)
        
        k = rowsB[n]
        Bkj = valsB[n]
        
        for m in nzrange(A, k)
            
            i = rowsA[m]
            if isallowed(accum, i)
               
                Aik = valsA[m]
                cij = Aik*Bkj
                insert!(accum, i, cij)
            
            end
            
        end
        
    end
    
    for n in nzrange(C, j)
        i = rowsC[n]
        if isset(accum, i)
            valsC[n] = remove!(accum, i)
        end
        setnotallowed!(accum, i)
    end
    
    return nothing
end