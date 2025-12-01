"""
Performance optimization utilities for hearing aid processing.

This module contains optimized functions and utilities for improving
performance in batch processing operations.
"""

"""
    allocate_power_matrix(num_blocks::Int, nbands::Int)

Pre-allocate power matrix with proper type annotation for better performance.
"""
function allocate_power_matrix(num_blocks::Int, nbands::Int)
    return Matrix{Float64}(undef, num_blocks, nbands)
end

"""
    allocate_taps_history(num_blocks::Int, buffer_size::Int, nfft::Int)

Pre-allocate taps history with proper dimensions.
"""
function allocate_taps_history(num_blocks::Int, buffer_size::Int, nfft::Int)
    return Vector{Matrix{Float64}}(undef, num_blocks)
end
