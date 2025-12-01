"""
HADatasets - Speech enhancement dataset preparation module for Voice Bank Demand dataset processing.
"""
module HADatasets

using SampledSignals: SampleBuf

include("misc.jl")
include("dataset_types.jl")
include("Metrics.jl")

export Metrics, Misc, DatasetTypes
export AudioFile
export evaluate_audio_metrics
export AbstractDataset, VOICEBANKDEMANDDataset
export download_data, resample_data, get_speech_dir, get_noise_dir, get_metadata, create_dataset

"""
    evaluate_audio_metrics(reference, processed)

Evaluate all metrics for a reference-processed audio pair.

Accepts file paths, `Vector{Float64}`, or `SampleBuf` inputs.
Returns a dictionary with PESQ, SIG, BAK, and OVRL scores.

# Type Stability
This function uses multiple dispatch for better type stability and performance.
"""
# Type-stable version for file paths
function evaluate_audio_metrics(reference::AbstractString, processed::AbstractString)
    if !isfile(reference)
        error("Reference file does not exist: $reference")
    end
    if !isfile(processed)
        error("Processed file does not exist: $processed")
    end
    
    ref_vec, ref_fs_any, _ = Misc.load_audio_file(reference)
    proc_vec, proc_fs_any, _ = Misc.load_audio_file(processed)
    
    if isnothing(ref_vec)
        error("Failed to load reference audio file: $reference")
    end
    if isnothing(proc_vec)
        error("Failed to load processed audio file: $processed")
    end
    
    ref_fs = Int(ref_fs_any)
    proc_fs = Int(proc_fs_any)
    
    if ref_fs != proc_fs
        error("Sample rate mismatch: reference=$ref_fs Hz, processed=$proc_fs Hz")
    end
    
    if ref_fs != 16000
        error("Sample rate must be 16000 Hz, got $ref_fs Hz")
    end
    
    if length(ref_vec) != length(proc_vec)
        error("Signal length mismatch: reference=$(length(ref_vec)) samples, processed=$(length(proc_vec)) samples")
    end
    
    return Metrics.evaluate_all(ref_vec, proc_vec, ref_fs)
end

# Type-stable version for Vector{Float64} with explicit sample rate
function evaluate_audio_metrics(reference::Vector{Float64}, processed::Vector{Float64}, fs::Int=16000)
    # Input validation
    if isempty(reference) || isempty(processed)
        error("Audio signals cannot be empty")
    end
    
    if fs <= 0
        error("Sample rate must be positive, got $fs Hz")
    end
    
    if fs != 16000
        error("Sample rate must be 16000 Hz, got $fs Hz")
    end
    
    if length(reference) != length(processed)
        error("Signal length mismatch: reference=$(length(reference)) samples, processed=$(length(processed)) samples")
    end
    
    return Metrics.evaluate_all(reference, processed, fs)
end

# Type-stable version for SampleBuf
function evaluate_audio_metrics(reference::SampleBuf, processed::SampleBuf)
    ref_vec, proc_vec, fs = Misc.prepare_audio_for_evaluation(reference, processed)
    return Metrics.evaluate_all(ref_vec, proc_vec, fs)
end

end
