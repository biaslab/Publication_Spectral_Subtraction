module Extensions
using RxInfer
using StatsFuns: logsumexp
include("nodes/sigmoid.jl")
include("rules/sigmoid/out.jl")
include("rules/sigmoid/in.jl")
include("rules/sigmoid/zeta.jl")
export Sigmoid
end
