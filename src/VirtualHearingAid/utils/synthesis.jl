"""
Synthesis utility functions for hearing aid processing.

This module contains utility functions for synthesizing audio output
from processed frequency domain data.
"""

"""
    synthesize_block(taps::Matrix{Float64}, synthesis_matrix::Matrix{Float64}, gains::Vector{Float64}, buffer_size::Int)

Synthesize a single block using the given taps, synthesis matrix, and gains.

# Arguments
- `taps`: Taps matrix for the current block
- `synthesis_matrix`: Synthesis matrix for frequency domain to time domain conversion
- `gains`: Gain vector for each frequency band
- `buffer_size`: Size of the output buffer

# Returns
- Vector of synthesized audio samples for the block
"""
@inline function synthesize_block(
    taps::Matrix{Float64},
    synthesis_matrix::Matrix{Float64},
    gains::Vector{Float64},
    buffer_size::Int,
)
    weights_temp = synthesis_matrix * gains
    n = length(weights_temp)
    weights = Vector{Float64}(undef, 2 * n)
    @inbounds for i = 1:n
        weights[i] = weights_temp[i]
    end
    weights[n+1] = 0.0
    @inbounds for i = 1:(n-1)
        weights[n+1+i] = weights_temp[n-i]
    end
    block = taps * weights
    return block[(end-buffer_size+1):end]
end

"""
    synthesize_blocks_batch(taps_history, synthesis_matrix, gains_matrix, buffer_size, gain_converter_type)

Type-stable batch synthesis function using multiple dispatch instead of Function types.

# Arguments
- `taps_history`: History of taps matrices
- `synthesis_matrix`: Synthesis matrix
- `gains_matrix`: Gain matrix or nothing
- `buffer_size`: Buffer size
- `gain_converter_type`: Symbol indicating converter type (:unity, :sem)

# Returns
- Concatenated output signal
"""
function synthesize_blocks_batch(
    taps_history::Vector{Matrix{Float64}},
    synthesis_matrix::Matrix{Float64},
    gains_matrix::Union{Matrix{Float64},Nothing},
    buffer_size::Int,
    gain_converter_type::Symbol,
)
    num_blocks = length(taps_history)
    total_samples = num_blocks * buffer_size
    output = Vector{Float64}(undef, total_samples)

    nbands = size(synthesis_matrix, 2)
    @inbounds for i = 1:num_blocks
        gains = _get_gains_for_block(Val(gain_converter_type), gains_matrix, i, nbands)
        block_output =
            synthesize_block(taps_history[i], synthesis_matrix, gains, buffer_size)
        start_idx = (i - 1) * buffer_size + 1
        copyto!(output, start_idx, block_output, 1, buffer_size)
    end

    return output
end

# Internal type-stable gain converter dispatcher using Val types for compile-time dispatch
@inline
function _get_gains_for_block(::Val{:unity}, ::Nothing, ::Int, nbands::Int)
    return ones(Float64, nbands)
end

@inline
function _get_gains_for_block(::Val{:unity}, gains::Matrix{Float64}, ::Int, nbands::Int)
    return ones(Float64, nbands)
end

@inline
function _get_gains_for_block(::Val{:sem}, gains::Matrix{Float64}, block_index::Int, ::Int)
    return collect(@view gains[block_index, :])
end

@inline
function _get_gains_for_block(::Val{:sem}, ::Nothing, ::Int, ::Int)
    throw(ArgumentError("SEM requires gains matrix"))
end


"""
    synthesize_offline(::BatchProcessingOffline, frontend::WFBFrontend, taps_history, buffer_size)

Generic synthesis using unity gains.
"""
function synthesize_offline(
    ::BatchProcessingOffline,
    frontend::WFBFrontend,
    taps_history::Vector{Matrix{Float64}},
    buffer_size::Int,
)
    synthesis_matrix = Frontends.WFB.get_synthesis_matrix(frontend)

    return synthesize_blocks_batch(
        taps_history,
        synthesis_matrix,
        nothing,
        buffer_size,
        :unity,
    )
end

"""
    synthesize_offline(::BatchProcessingOffline, ha::BaselineHearingAid, gains::Nothing, taps_history, buffer_size)

Synthesize output for Baseline hearing aid using unity gains.
"""
function synthesize_offline(
    ::BatchProcessingOffline,
    ha::BaselineHearingAid,
    gains::Nothing,
    taps_history::Vector{Matrix{Float64}},
    buffer_size::Int,
)
    frontend = get_frontend(ha)
    synthesis_matrix = Frontends.WFB.get_synthesis_matrix(frontend)

    return synthesize_blocks_batch(
        taps_history,
        synthesis_matrix,
        nothing,
        buffer_size,
        :unity,
    )
end

"""
    synthesize_offline(::BatchProcessingOffline, ha::SEMHearingAid, gains, taps_history, buffer_size)

Synthesize output for SEM hearing aid (gains already in linear scale).
"""
function synthesize_offline(
    ::BatchProcessingOffline,
    ha::SEMHearingAid,
    gains::Matrix{Float64},
    taps_history::Vector{Matrix{Float64}},
    buffer_size::Int,
)
    frontend = get_frontend(ha)
    synthesis_matrix = Frontends.WFB.get_synthesis_matrix(frontend)

    return synthesize_blocks_batch(taps_history, synthesis_matrix, gains, buffer_size, :sem)
end

"""
    synthesize_offline(::BatchProcessingOffline, ha::AbstractHearingAid, gains, taps_history, buffer_size)

Generic synthesis fallback for any hearing aid type.
"""
function synthesize_offline(
    ::BatchProcessingOffline,
    ha::AbstractHearingAid,
    gains::Union{Matrix{Float64},Nothing},
    taps_history::Vector{Matrix{Float64}},
    buffer_size::Int,
)
    frontend = get_frontend(ha)
    return synthesize_offline(BatchProcessingOffline(), frontend, taps_history, buffer_size)
end
