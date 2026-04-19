#!/usr/bin/env julia
"""
Stage 1: resample raw 48 kHz VOICEBANK_DEMAND to 16 kHz.

Usage:
    julia scripts/resample_voicebank_demand.jl [--target-fs 16000] [--force]
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Experiments

target_fs = 16_000
force = false
let i = 1
    while i <= length(ARGS)
        a = ARGS[i]
        if a == "--target-fs" && i < length(ARGS)
            global target_fs = parse(Int, ARGS[i + 1]); i += 2
        elseif startswith(a, "--target-fs=")
            global target_fs = parse(Int, split(a, "=")[2]); i += 1
        elseif a == "--force"
            global force = true; i += 1
        else
            i += 1
        end
    end
end

println("Stage 1 — Resample VOICEBANK_DEMAND → $(target_fs) Hz")
println("Raw:       ", Experiments.Databases.raw())
println("Resampled: ", Experiments.Databases.resampled())

Experiments.resample_VOICEBANK_DEMAND(; target_fs=target_fs, force=force)

println("✓ Stage 1 complete")
