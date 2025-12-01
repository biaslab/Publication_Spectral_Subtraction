#!/usr/bin/env julia
"""
Script to convert VOICEBANK_DEMAND_resampled dataset to WFB-processed version.

Usage:
    julia scripts/convert_to_wfb.jl [--num-samples N]

Options:
    --num-samples N    Limit processing to first N samples (optional, for testing)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Experiments
const VirtualHearingAid = Experiments.VirtualHearingAid
const HADatasets = Experiments.HADatasets

# Paths
project_root = dirname(@__DIR__)
config_path = joinpath(project_root, "configurations", "BaselineHearingAid", "BaselineHearingAid.toml")
source_dir = joinpath(project_root, "databases", "VOICEBANK_DEMAND_resampled")

# Parse command line arguments
num_samples = nothing
if length(ARGS) > 0
    for arg in ARGS
        if startswith(arg, "--num-samples=")
            global num_samples = parse(Int, split(arg, "=")[2])
        elseif arg == "--num-samples" && length(ARGS) > 1
            global num_samples = parse(Int, ARGS[2])
        end 
    end
end

println("=" ^ 60)
println("VOICEBANK_DEMAND_resampled to WFB Conversion")
println("=" ^ 60)
println("\nConfiguration: $config_path")
println("Source directory: $source_dir")
if !isnothing(num_samples)
    println("Processing first $num_samples samples only")
end
println()

# Run conversion
try
    wfb_base_dir = convert_VOICEBANK_DEMAND_resampled_to_wfb(
        config_path, 
        source_dir; 
        num_samples=num_samples
    )
    
    println("\n" * "=" ^ 60)
    println("✓ Conversion completed successfully!")
    println("=" ^ 60)
    println("WFB dataset created at: $wfb_base_dir")
    
catch e
    println("\n" * "=" ^ 60)
    println("✗ Conversion failed!")
    println("=" ^ 60)
    println("Error: $e")
    rethrow(e)
end

