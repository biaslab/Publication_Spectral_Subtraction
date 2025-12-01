# Main test file that includes all submodule tests
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Test

# Include submodule tests
@testset "HASoundProcessing" begin
    include("HASoundProcessing/runtests.jl")
end

@testset "VirtualHearingAid" begin
    include("VirtualHearingAid/runtests.jl")
end

# HADatasets doesn't have tests, so we skip it

