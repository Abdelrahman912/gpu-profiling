using CUDA, NVTX, BenchmarkTools
A = CUDA.rand(1024, 1024)
B = CUDA.rand(1024, 1024)

A * B ; # Warm up the GPU

# Profile the code
# Note: it's better to profile the code twice for more accurate results.
CUDA.@profile begin
    NVTX.@range "mul! 1" CUDA.@sync A*B
    NVTX.@range "mul! 2" CUDA.@sync A*B
end

# command: nsys launch julia --project
