using RxInfer: mean, precision

"""
Process a single block of power spectrum data through SEM.

# Arguments

  - `backend::SEMBackend{T}`: The SEM backend instance
  - `powerdb::Vector{T}`: Power spectrum data in dB
  - `idx::Int`: Frequency band index

# Returns

  - `Tuple`: Inference results and computed gains
"""
function process_backend(
    backend::SEMBackend{T},
    powerdb::Vector{T},
    idx::Int,
) where {T <: Real}
    # Validate band index
    nbands = backend.params.modules.nbands
    (1 <= idx <= nbands) || throw(
        ArgumentError(
            "Band index $idx is out of range. Backend has $nbands bands (valid indices: 1:$nbands)",
        ),
    )
    # Run inference
    results = infer_SEM_filtering(backend, powerdb, idx)
    gain_history = results.history[:w]

    gains = [h.p[1] for h in gain_history]

    wiener_gain =
        wiener_gain_spectral_floor(
            gains,
            get_gain_threshold_lin(backend),
        )
    return results, wiener_gain
end
