"""
Batch processing implementation for complete audio files.

This module provides batch processing capabilities for hearing aids, organized by hearing aid type.
Each hearing aid type has its own section with backend processing, frontend processing, and complete pipelines.
"""

# =============================================================================
# SHARED FRONTEND PROCESSING
# =============================================================================

"""
    process_frontend(::BatchProcessingOffline, ha::AbstractHearingAid, sound::SampleBuf)

Process the entire `sound` in blocks using only the frontend. Accumulates:
- powerdb_matrix: rows per block, columns per band (in dB)
- taps_history: vector of taps matrices (per block), each of size `buffer_size × nfft`

Returns `(powerdb_matrix::Matrix{Float64}, taps_history::Vector{Matrix{Float64}})`.
"""
function process_frontend(
    ::BatchProcessingOffline,
    ha::AbstractHearingAid,
    sound::SampleBuf,
)
    # Validate inputs
    validate_batch_processing_inputs(sound, ha)

    frontend = get_frontend(ha)
    block_length = get_params(frontend).buffer_size
    nbands = Frontends.get_nbands(frontend)

    num_blocks = cld(length(sound), block_length)
    powerdb_matrix = allocate_power_matrix(num_blocks, nbands)
    taps_history = allocate_taps_history(num_blocks, block_length, (nbands - 1) * 2)

    current_index = 1
    block_index = 1

    while current_index <= length(sound)
        remaining_samples = length(sound) - current_index + 1
        current_block_size = min(block_length, remaining_samples)
        # Use @view to avoid allocation, then convert to Array and SampleBuf for process_frontend
        block_view = @view sound[current_index:(current_index+current_block_size-1)]
        block = SampleBuf(collect(block_view), sound.samplerate)

        # Frontend processing only
        powerdb = Frontends.process_frontend(frontend, block)

        # Accumulate power in dB - use @view for row assignment
        powerdb_matrix[block_index, :] = powerdb

        # Snapshot current taps matrix for this block - avoid copy if possible
        taps = Frontends.WFB.get_taps(frontend)
        taps_history[block_index] = copy(taps)  # Copy needed for history

        current_index += current_block_size
        block_index += 1
    end

    return powerdb_matrix, taps_history
end

"""
    process_frontend(::BatchProcessingOnline, ha::AbstractHearingAid, sound::SampleBuf)

Process the entire `sound` in blocks using only the frontend, returning a
matrix of power in dB with shape (num_blocks × nbands). Taps history is not
retained for the online variant and will be returned as `nothing`.
"""
function process_frontend(::BatchProcessingOnline, ha::AbstractHearingAid, sound::SampleBuf)
    # Validate inputs
    validate_batch_processing_inputs(sound, ha)

    frontend = get_frontend(ha)
    block_length = get_params(frontend).buffer_size
    nbands = Frontends.get_nbands(frontend)

    num_blocks = cld(length(sound), block_length)
    powerdb_matrix = allocate_power_matrix(num_blocks, nbands)

    current_index = 1
    block_index = 1

    while current_index <= length(sound)
        remaining_samples = length(sound) - current_index + 1
        current_block_size = min(block_length, remaining_samples)
        # Use @view to avoid allocation, then convert to Array and SampleBuf for process_frontend
        block_view = @view sound[current_index:(current_index+current_block_size-1)]
        block = SampleBuf(collect(block_view), sound.samplerate)

        powerdb = Frontends.process_frontend(frontend, block)
        powerdb_matrix[block_index, :] = powerdb

        current_index += current_block_size
        block_index += 1
    end

    return powerdb_matrix, nothing
end

# =============================================================================
# GENERIC OFFLINE BATCH PROCESSING
# =============================================================================

"""
    process_backend_batch(ha::AbstractHearingAid, powerdb_matrix::AbstractMatrix{<:Real})

Process backend for entire batch. Returns (gains, results, ha_name).
For hearing aids without backends, returns (nothing, nothing, nothing).
ha_name is only needed when gains exist (for logging).
"""
function process_backend_batch(
    ha::AbstractHearingAid,
    powerdb_matrix::AbstractMatrix{<:Real},
)
    return nothing, nothing, nothing
end


"""
    process_backend_batch(ha::SEMHearingAid, powerdb_matrix::AbstractMatrix{<:Real})

Process SEM backend for entire batch. Returns (gains, results, ha_name).
"""
function process_backend_batch(ha::SEMHearingAid, powerdb_matrix::AbstractMatrix{<:Real})
    validate_backend_processing_inputs(powerdb_matrix, ha.backend)

    backend_start_time = time()
    results = run_backend(ha.backend, powerdb_matrix)
    gains = results.gains
    backend_time = time() - backend_start_time

    ha_name = getname(ha)
    log_backend_processing(ha_name, size(powerdb_matrix, 1), backend_time)

    return gains, results, ha_name
end

"""
    process_offline_batch(ha::AbstractHearingAid, sound::SampleBuf)

Generic offline batch processing pipeline.
Handles the common flow: frontend → backend → synthesis → return.
"""
function process_offline_batch(ha::AbstractHearingAid, sound::SampleBuf)
    powerdb_matrix, taps_history = process_frontend(BatchProcessingOffline(), ha, sound)
    buffer_size = get_params(get_frontend(ha)).buffer_size

    # Backend processing (dispatched per hearing aid type)
    gains, results, ha_name = process_backend_batch(ha, powerdb_matrix)

    # Validate synthesis inputs
    validate_synthesis_inputs(taps_history, gains, buffer_size)

    # Synthesis
    synthesis_start_time = time()
    outputs =
        synthesize_offline(BatchProcessingOffline(), ha, gains, taps_history, buffer_size)
    synthesis_time = time() - synthesis_start_time

    # Log synthesis metrics (if backend exists)
    if !isnothing(gains)
        log_synthesis_metrics(ha_name, size(powerdb_matrix, 1), synthesis_time)
    end

    output_signal = SampleBuf(outputs, sound.samplerate)
    return output_signal, results
end

# =============================================================================
# BASELINE HEARING AID
# =============================================================================

@inline
"""
    process(::BaselineBackend, powerdb::AbstractVector{<:Real})

Baseline backend processing - returns unity gains and no results.
"""
process(::BaselineBackend, powerdb::AbstractVector{<:Real}) =
    (ones(Float64, length(powerdb)), nothing) # unity gains and no results

"""
    compute_gains(backend::BaselineBackend, powerdb, nbands::Integer)

Baseline gain computation delegation.
"""
compute_gains(backend::BaselineBackend, powerdb) = process(backend, powerdb)

"""
    process(::BatchProcessingOnline, ha::AbstractHearingAid, sound::SampleBuf)

Online batch processing - processes audio in blocks sequentially.
"""
function process(::BatchProcessingOnline, ha::AbstractHearingAid, sound::SampleBuf)
    # Validate inputs
    validate_batch_processing_inputs(sound, ha)

    # Pre-allocate output - zeros() is well-optimized in Julia
    output = zeros(Float64, length(sound))
    block_length = get_params(get_frontend(ha)).buffer_size
    current_index = 1

    while current_index <= length(sound)
        remaining_samples = length(sound) - current_index + 1
        current_block_size = min(block_length, remaining_samples)
        # Use @view to avoid allocation, then convert to Array and SampleBuf for process_block
        block_view = @view sound[current_index:(current_index+current_block_size-1)]
        block = SampleBuf(collect(block_view), sound.samplerate)

        processed_block, _ = process_block(ha, block)
        # Use @view for assignment to avoid creating intermediate array
        @inbounds output[current_index:(current_index+current_block_size-1)] =
            @view processed_block[1:current_block_size]
        current_index += current_block_size
    end

    return SampleBuf(output, sound.samplerate), nothing
end

"""
    process(::BatchProcessingOffline, ha::BaselineHearingAid, sound::SampleBuf)

Offline batch processing for Baseline hearing aid.
"""
function process(::BatchProcessingOffline, ha::BaselineHearingAid, sound::SampleBuf)
    return process_offline_batch(ha, sound)
end




# =============================================================================
# SEM HEARING AID
# =============================================================================

@inline
"""
    run_backend(backend::SEMBackend, powerdb_matrix::AbstractMatrix{<:Real})

Run SEM backend processing for the entire power matrix.
Returns SEMResults containing wiener_gains and inference_results.
"""
function run_backend(backend::SEMBackend, powerdb_matrix::AbstractMatrix{<:Real})
    # Validate inputs
    validate_backend_processing_inputs(powerdb_matrix, backend)

    nframes, nbands = size(powerdb_matrix)
    wiener_gains = similar(powerdb_matrix, Float64)

    inference_results_list = []

    @inbounds for band_idx = 1:nbands
        inference_results, wiener_gain = HASoundProcessing.SEM.process_backend(
            backend,
            powerdb_matrix[:, band_idx],
            band_idx,
        )
        push!(inference_results_list, inference_results)
        wiener_gains[:, band_idx] = wiener_gain

    end

    sem_results = SEMResults(wiener_gains, inference_results_list)
    return sem_results
end

"""
    compute_gains(backend::SEMBackend, powerdb, nbands::Integer)

SEM gain computation delegation.
Returns: (gains, results) for compatibility with generic interface.
The gains are the Wiener gains actually used for synthesis.
"""
function compute_gains(backend::SEMBackend, powerdb)
    results = run_backend(backend, powerdb)
    return results.gains, results
end

"""
    denoise(ha::SEMHearingAid, sound::SampleBuf)

Denoise audio signal using SEM (Spectral Enhancement Model) hearing aid.

Processes the entire audio file through frontend and backend, returning both
the denoised output signal and inference results.

# Arguments
- `ha::SEMHearingAid`: The SEM hearing aid instance
- `sound::SampleBuf`: Input audio signal to denoise

# Returns
- `Tuple{SampleBuf, SEMResults}`: (output_signal, results)
  - `output_signal`: Denoised audio output
  - `results`: SEMResults containing wiener_gains and inference_results

# Note
This function performs the complete denoising pipeline: frontend processing,
backend spectral enhancement, and synthesis. Inference results are always computed
as part of the SEM algorithm and are always returned.
"""
function denoise(ha::SEMHearingAid, sound::SampleBuf)
    return process_offline_batch(ha, sound)
end
