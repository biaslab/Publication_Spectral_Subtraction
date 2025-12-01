"""
Miscellaneous utility functions for HADatasets.
"""
module Misc

using WAV
using Logging
using Statistics
using DSP
using Glob
using SampledSignals: SampleBuf

export load_audio_file, discover_audio_files, AudioFile
export samplebuf_to_vector, prepare_audio_for_evaluation

struct AudioFile
    path::String
    duration::Float64
    fs::Int
    category::String
end

"""
    load_audio_file(file_path)

Load audio file and extract metadata.
"""
function load_audio_file(file_path::String)
    try
        audio_data, fs_orig = wavread(file_path)
        mono = ndims(audio_data) == 2 && size(audio_data, 2) > 1 ? vec(audio_data[:, 1]) : vec(audio_data)
        duration = length(mono) / fs_orig
        return mono, fs_orig, duration
    catch e
        @error "Failed to load audio file: $file_path" exception=(e, catch_backtrace())
        return nothing, nothing, nothing
    end
end

"""
    discover_audio_files(dir_path, target_fs, min_duration)

Discover and validate audio files recursively.
"""
function discover_audio_files(dir_path::String, target_fs::Int, min_duration::Float64=1.0)
    audio_files = AudioFile[]
    wav_files = glob("*.wav", dir_path)
    
    for file_path in wav_files
        try
            if isfile(file_path)
                signal, fs, duration = load_audio_file(file_path)
                if signal !== nothing && duration >= min_duration
                    category = basename(dirname(file_path))
                    push!(audio_files, AudioFile(
                        file_path,
                        duration,
                        fs,
                        category
                    ))
                elseif signal !== nothing
                    @info "Skipping file shorter than minimum duration: $file_path (duration: $(round(duration, digits=2))s)"
                end
            end
        catch e
            @warn "Failed to process file: $file_path" exception=(e, catch_backtrace())
        end
    end
    
    return audio_files
end

"""
    samplebuf_to_vector(signal)

Convert SampleBuf to Vector{Float64} and sample rate.
"""
function samplebuf_to_vector(signal::SampleBuf)
    return vec(signal.data), Int(signal.samplerate)
end

"""
    prepare_audio_for_evaluation(reference, processed)

Prepare audio for evaluation (convert SampleBuf if needed).
Returns (reference_vector, processed_vector, sample_rate).

# Type Stability
This function uses multiple dispatch for type stability.
"""
# Type-stable version for Vector{Float64}
function prepare_audio_for_evaluation(reference::Vector{Float64}, processed::Vector{Float64}, fs::Int=16000)
    # Input validation
    if isempty(reference) || isempty(processed)
        error("Audio signals cannot be empty")
    end
    
    if fs <= 0
        error("Sample rate must be positive, got $fs Hz")
    end
    
    if length(reference) != length(processed)
        error("Signal length mismatch: reference=$(length(reference)) samples, processed=$(length(processed)) samples")
    end
    
    return reference, processed, fs
end

# Type-stable version for SampleBuf
function prepare_audio_for_evaluation(reference::SampleBuf, processed::SampleBuf)
    ref_vec, ref_fs = samplebuf_to_vector(reference)
    proc_vec, proc_fs = samplebuf_to_vector(processed)
    
    # Input validation
    if isempty(ref_vec) || isempty(proc_vec)
        error("Audio signals cannot be empty")
    end
    
    if ref_fs <= 0 || proc_fs <= 0
        error("Sample rate must be positive: reference=$ref_fs Hz, processed=$proc_fs Hz")
    end
    
    if ref_fs != proc_fs
        error("Sample rate mismatch: reference=$ref_fs Hz, processed=$proc_fs Hz")
    end
    
    if length(ref_vec) != length(proc_vec)
        error("Signal length mismatch: reference=$(length(ref_vec)) samples, processed=$(length(proc_vec)) samples")
    end
    
    return ref_vec, proc_vec, ref_fs
end

# Fallback for mixed types (less type-stable but more flexible)
function prepare_audio_for_evaluation(reference, processed)
    ref_vec, ref_fs = if reference isa SampleBuf
        samplebuf_to_vector(reference)
    elseif reference isa Vector{Float64}
        (reference, 16000)
    else
        error("Unsupported reference type: $(typeof(reference)). Expected Vector{Float64} or SampleBuf")
    end
    
    proc_vec, proc_fs = if processed isa SampleBuf
        samplebuf_to_vector(processed)
    elseif processed isa Vector{Float64}
        (processed, 16000)
    else
        error("Unsupported processed type: $(typeof(processed)). Expected Vector{Float64} or SampleBuf")
    end
    
    # Input validation
    if isempty(ref_vec) || isempty(proc_vec)
        error("Audio signals cannot be empty")
    end
    
    if ref_fs <= 0 || proc_fs <= 0
        error("Sample rate must be positive: reference=$ref_fs Hz, processed=$proc_fs Hz")
    end
    
    if ref_fs != proc_fs
        error("Sample rate mismatch: reference=$ref_fs Hz, processed=$proc_fs Hz")
    end
    
    if length(ref_vec) != length(proc_vec)
        error("Signal length mismatch: reference=$(length(ref_vec)) samples, processed=$(length(proc_vec)) samples")
    end
    
    return ref_vec, proc_vec, ref_fs
end

end
