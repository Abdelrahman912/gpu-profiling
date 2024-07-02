using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()


using CUDA
using BenchmarkTools
using NVTX


N = 16
A = CUDA.rand(1024,1024,N)
B = CUDA.rand(1024,1024,N)
C = CUDA.similar(A, N);

function rmse_kernel(C, A, B, Rmain, Rbatch)
    batch = blockIdx().x
    Ibatch = Rbatch[batch]
    
    # initialize the memory
    if threadIdx().x == 1
        C[batch] = 0
    end
    sync_threads()
    
    # grid-stride loop to process each batch in a block
    for i in threadIdx().x:blockDim().x:length(Rmain)
        Imain = Rmain[i]
        I = max(Imain, Ibatch)
        a = A[I]
        b = B[I]
        CUDA.@atomic C[batch] += (a-b)^2
    end    
    sync_threads()
    
    # finalize the computation
    if threadIdx().x == 1
        C[batch] = sqrt(C[batch] / length(Rmain))
    end
    return
end

function rmse(C, A, B)
    Rmain = ntuple(i->i == ndims(A) ? Base.OneTo(1) : axes(A)[i], ndims(A)) |> CartesianIndices
    Rbatch = ntuple(i->i != ndims(A) ? Base.OneTo(1) : axes(A)[i], ndims(A)) |> CartesianIndices
    @cuda threads=256 blocks=N rmse_kernel(C, A, B, Rmain, Rbatch)
    return
end

rmse(C,A,B)
C

#@benchmark CUDA.@sync rmse(C, A, B)

#command: ncu --mode=launch julia
#macro: @device_code_llvm debuginfo=:none fun()
#CUDA.registers(kernel)
# it's important to lower number of registers
# use Int32 to reduce the number of registers