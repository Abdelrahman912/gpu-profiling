#ref : https://www.youtube.com/watch?v=jqBHgix42AI&t=1209s&ab_channel=JuliaHub
#ref : https://github.com/maleadt/cscs_gpu_course/blob/main/2-1-application_analysis_optimization.ipynb
# case study : calculate the rmse (root mean square error) of the model

using CUDA, NVTX, BenchmarkTools

N = 16
A = CUDA.rand(1024, 1024, N)
B = CUDA.rand(1024, 1024, N)

NVTX.@range function rmse(A::AbstractMatrix, B::AbstractMatrix)
    (sum((A .- B).^2) / length(A)) |> sqrt
end

NVTX.@range function doit()
    rmses = Vector{eltype(A)}(undef,N)
    for i in 1:N
        rmses[i] = rmse(A[:,:,i], B[:,:,i])
    end
    rmses
end

# normal benchmark to look at initial performance.
@benchmark doit()