"""
Dataset type abstraction for different speech enhancement dataset sources.

Provides a unified interface for managing dataset sources (VOICEBANK_DEMAND, etc.)
with operations like downloading, resampling, and accessing dataset-specific paths and metadata.

This module handles dataset source management, while `Dataset` module handles
the actual mixing and preparation of training data from these sources.
"""
module DatasetTypes

using Logging
using FileIO
using Glob
using DSP
using WAV
using CSV
using DataFrames

include("misc.jl")

export AbstractDataset, VOICEBANKDEMANDDataset
export download_data, resample_data, get_speech_dir, get_noise_dir, get_metadata
export create_dataset

"""
    AbstractDataset

Abstract type for all dataset implementations.
"""
abstract type AbstractDataset end

"""
    VOICEBANKDEMANDDataset

VOICEBANK_DEMAND dataset implementation.

# Fields
- `base_dir::String`: Base directory for VOICEBANK_DEMAND dataset
- `resampled_dir::String`: Directory for resampled data
- `target_fs::Int`: Target sample rate (default: 16000)
- `min_duration::Float64`: Minimum audio duration in seconds (default: 1.0)
"""
struct VOICEBANKDEMANDDataset <: AbstractDataset
    base_dir::String
    resampled_dir::String
    target_fs::Int
    min_duration::Float64
end

"""
    VOICEBANKDEMANDDataset(base_dir; target_fs, min_duration)

Create VOICEBANKDEMANDDataset instance.

# Arguments
- `base_dir::String`: Base directory for VOICEBANK_DEMAND dataset (default: "data/VOICEBANK_DEMAND")
- `target_fs::Int`: Target sample rate (default: 16000)
- `min_duration::Float64`: Minimum audio duration in seconds (default: 1.0)
"""
function VOICEBANKDEMANDDataset(base_dir::String="data/VOICEBANK_DEMAND"; target_fs::Int=16000, min_duration::Float64=1.0)
    resampled_dir = joinpath(dirname(base_dir), "$(basename(base_dir))_resampled")
    return VOICEBANKDEMANDDataset(base_dir, resampled_dir, target_fs, min_duration)
end

"""
    download_data(dataset::AbstractDataset)

Download dataset data. Must be implemented by each dataset type.
"""
function download_data(dataset::AbstractDataset)
    error("download_data not implemented for $(typeof(dataset))")
end

"""
    download_data(dataset::VOICEBANKDEMANDDataset)

Download VOICEBANK_DEMAND dataset. Manual download required.
"""
function download_data(dataset::VOICEBANKDEMANDDataset)
    @info "VOICEBANK_DEMAND dataset download must be performed manually."
    @info "Download from official source and extract to: $(dataset.base_dir)"
    return nothing
end

"""
    resample_data(dataset::AbstractDataset; target_fs, min_duration)

Resample dataset to target sample rate. Must be implemented by each dataset type.
"""
function resample_data(dataset::AbstractDataset; target_fs=nothing, min_duration=nothing)
    error("resample_data not implemented for $(typeof(dataset))")
end

"""
    resample_data(dataset::VOICEBANKDEMANDDataset; target_fs, min_duration)

Resample VOICEBANK_DEMAND dataset to target sample rate.
"""
function resample_data(dataset::VOICEBANKDEMANDDataset; target_fs::Union{Int, Nothing}=nothing, min_duration::Union{Float64, Nothing}=nothing)
    target_fs = isnothing(target_fs) ? dataset.target_fs : target_fs
    min_duration = isnothing(min_duration) ? dataset.min_duration : min_duration
    
    @info "Starting VOICEBANK_DEMAND dataset resampling..."
    @info "Input directory: $(dataset.base_dir)"
    @info "Output directory: $(dataset.resampled_dir)"
    @info "Target sample rate: $(target_fs)Hz"
    @info "Minimum duration: $(min_duration)s"
    
    if !isdir(dataset.base_dir)
        error("Base directory not found: $(dataset.base_dir)")
    end
    
    clean_output_dir = joinpath(dataset.resampled_dir, "clean_testset_wav")
    noisy_output_dir = joinpath(dataset.resampled_dir, "noisy_testset_wav")
    log_output_dir = joinpath(dataset.resampled_dir, "logfiles")
    
    mkpath(clean_output_dir)
    mkpath(noisy_output_dir)
    mkpath(log_output_dir)
    
    log_file = joinpath(dataset.base_dir, "logfiles", "log_testset.txt")
    snr_info = parse_VOICEBANK_DEMAND_log(log_file)
    @info "Parsed SNR information for $(length(snr_info)) files"
    
    clean_input_dir = joinpath(dataset.base_dir, "data", "clean_testset_wav")
    clean_files = process_VOICEBANK_DEMAND_directory(clean_input_dir, clean_output_dir, 
                                             target_fs, min_duration, "clean")
    
    noisy_input_dir = joinpath(dataset.base_dir, "data", "noisy_testset_wav")
    noisy_files = process_VOICEBANK_DEMAND_directory(noisy_input_dir, noisy_output_dir, 
                                             target_fs, min_duration, "noisy")
    
    create_resampled_log(clean_files, noisy_files, snr_info, log_output_dir)
    
    total_files = length(clean_files) + length(noisy_files)
    @info "Resampling complete! Total files processed: $total_files"
    @info "Clean files: $(length(clean_files))"
    @info "Noisy files: $(length(noisy_files))"
    @info "Output saved to: $(dataset.resampled_dir)"
    
    return Dict("clean" => clean_files, "noisy" => noisy_files)
end

"""
    get_speech_dir(dataset::AbstractDataset)

Get directory containing speech files. Must be implemented by each dataset type.
"""
function get_speech_dir(dataset::AbstractDataset)
    error("get_speech_dir not implemented for $(typeof(dataset))")
end

"""
    get_speech_dir(dataset::VOICEBANKDEMANDDataset)

Get VOICEBANK_DEMAND clean speech directory.
"""
function get_speech_dir(dataset::VOICEBANKDEMANDDataset)
    return joinpath(dataset.resampled_dir, "clean_testset_wav")
end

"""
    get_noise_dir(dataset::AbstractDataset)

Get directory containing noise files. Must be implemented by each dataset type.
"""
function get_noise_dir(dataset::AbstractDataset)
    error("get_noise_dir not implemented for $(typeof(dataset))")
end

"""
    get_noise_dir(dataset::VOICEBANKDEMANDDataset)

Get VOICEBANK_DEMAND noisy speech directory (contains mixed signals).
"""
function get_noise_dir(dataset::VOICEBANKDEMANDDataset)
    return joinpath(dataset.resampled_dir, "noisy_testset_wav")
end

"""
    get_metadata(dataset::AbstractDataset)

Get dataset metadata. Optional, returns nothing by default.
"""
function get_metadata(dataset::AbstractDataset)
    return nothing
end

"""
    get_metadata(dataset::VOICEBANKDEMANDDataset)

Get VOICEBANK_DEMAND metadata from log file.
"""
function get_metadata(dataset::VOICEBANKDEMANDDataset)
    log_file = joinpath(dataset.resampled_dir, "logfiles", "log_testset_resampled.txt")
    if isfile(log_file)
        return parse_VOICEBANK_DEMAND_log(log_file)
    else
        log_file = joinpath(dataset.base_dir, "logfiles", "log_testset.txt")
        if isfile(log_file)
            return parse_VOICEBANK_DEMAND_log(log_file)
        end
    end
    return nothing
end

"""
    create_dataset(dataset_type::Symbol, base_dir; kwargs...)

Create a dataset instance by type name.

# Arguments
- `dataset_type::Symbol`: Dataset type (`:VOICEBANK_DEMAND`)
- `base_dir::String`: Base directory for the dataset
- `kwargs...`: Additional arguments passed to dataset constructor
"""
function create_dataset(dataset_type::Symbol, base_dir::String; kwargs...)
    if dataset_type == :VOICEBANK_DEMAND
        return VOICEBANKDEMANDDataset(base_dir; kwargs...)
    else
        error("Unknown dataset type: $dataset_type. Available: :VOICEBANK_DEMAND")
    end
end

# Internal helper functions

"""
    preprocess_directory(input_dir, output_dir, target_fs, min_duration)

Preprocess audio files in a directory recursively.
"""
function preprocess_directory(input_dir::String, output_dir::String, target_fs::Int, 
                              min_duration::Float64)
    audio_files = Misc.AudioFile[]
    wav_files = glob("*.wav", input_dir)
    
    @info "Found $(length(wav_files)) WAV files in $input_dir"
    
    for (i, file_path) in enumerate(wav_files)
        try
            if isfile(file_path)
                rel_path = relpath(file_path, input_dir)
                output_path = joinpath(output_dir, rel_path)
                
                mkpath(dirname(output_path))
                
                if isfile(output_path)
                    @info "Skipping existing file ($i/$(length(wav_files))): $(basename(file_path))"
                    continue
                end
                
                @info "Processing file ($i/$(length(wav_files))): $(basename(file_path))"
                
                signal, fs, duration = Misc.load_audio_file(file_path)
                
                if signal !== nothing && duration >= min_duration
                    if fs != target_fs
                        ratio = target_fs / fs
                        signal = resample(signal, ratio)
                        fs = target_fs
                    end
                    
                    wavwrite(signal, output_path, Fs=fs)
                    
                    push!(audio_files, Misc.AudioFile(
                        output_path,
                        duration,
                        fs,
                        basename(dirname(file_path))
                    ))
                    
                    @info "Saved resampled file: $(basename(output_path)) (duration: $(round(duration, digits=2))s, fs: $(fs)Hz)"
                    
                elseif signal !== nothing
                    @info "Skipping file shorter than minimum duration: $(basename(file_path)) (duration: $(round(duration, digits=2))s)"
                else
                    @warn "Failed to load audio file: $(basename(file_path))"
                end
            end
        catch e
            @warn "Failed to process file: $(basename(file_path))" exception=(e, catch_backtrace())
        end
    end
    
    return audio_files
end

"""
    parse_VOICEBANK_DEMAND_log(log_file)

Parse VOICEBANK_DEMAND log file to get SNR information.
"""
function parse_VOICEBANK_DEMAND_log(log_file::String)
    snr_info = Dict{String, Tuple{String, Float64}}()
    
    try
        lines = readlines(log_file)
        for line in lines
            if !isempty(strip(line)) && !startswith(strip(line), "#")
                parts = split(strip(line))
                if length(parts) >= 3
                    filename = parts[1]
                    noise_type = parts[2]
                    snr_value = parse(Float64, parts[3])
                    snr_info[filename] = (noise_type, snr_value)
                end
            end
        end
    catch e
        @warn "Failed to parse log file: $log_file" exception=(e, catch_backtrace())
    end
    
    return snr_info
end

"""
    process_VOICEBANK_DEMAND_directory(input_dir, output_dir, target_fs, min_duration, category)

Process audio files in a VOICEBANK_DEMAND directory.
"""
function process_VOICEBANK_DEMAND_directory(
    input_dir::String, 
    output_dir::String, 
    target_fs::Int, 
    min_duration::Float64,
    category::String
)
    audio_files = Misc.AudioFile[]
    wav_files = glob("*.wav", input_dir)
    
    @info "Found $(length(wav_files)) WAV files in $input_dir"
    
    for (i, file_path) in enumerate(wav_files)
        try
            if isfile(file_path)
                filename = basename(file_path)
                output_path = joinpath(output_dir, filename)
                
                if isfile(output_path)
                    @info "Skipping existing file ($i/$(length(wav_files))): $filename"
                    continue
                end
                
                @info "Processing file ($i/$(length(wav_files))): $filename"
                
                signal, fs, duration = Misc.load_audio_file(file_path)
                
                if signal !== nothing && duration >= min_duration
                    if fs != target_fs
                        ratio = target_fs / fs
                        if ndims(signal) == 2
                            signal = mean(signal, dims=2)[:, 1]
                        end
                        signal = resample(signal, ratio)
                        fs = target_fs
                    end
                    
                    wavwrite(signal, output_path, Fs=fs)
                    
                    push!(audio_files, Misc.AudioFile(
                        output_path,
                        duration,
                        fs,
                        category
                    ))
                    
                    @info "Saved resampled file: $filename (duration: $(round(duration, digits=2))s, fs: $(fs)Hz)"
                    
                elseif signal !== nothing
                    @info "Skipping file shorter than minimum duration: $filename (duration: $(round(duration, digits=2))s)"
                else
                    @warn "Failed to load audio file: $filename"
                end
            end
        catch e
            @warn "Failed to process file: $(basename(file_path))" exception=(e, catch_backtrace())
        end
    end
    
    return audio_files
end

"""
    create_resampled_log(clean_files, noisy_files, snr_info, log_output_dir)

Create resampled log file with updated information.
"""
function create_resampled_log(
    clean_files::Vector{Misc.AudioFile}, 
    noisy_files::Vector{Misc.AudioFile}, 
    snr_info::Dict{String, Tuple{String, Float64}}, 
    log_output_dir::String
)
    log_output_file = joinpath(log_output_dir, "log_testset_resampled.txt")
    
    try
        clean_filenames = Set([basename(f.path) for f in clean_files])
        noisy_filenames = Set([basename(f.path) for f in noisy_files])
        common_filenames = intersect(clean_filenames, noisy_filenames)
        
        open(log_output_file, "w") do io
            println(io, "# VOICEBANK_DEMAND Test Set - Resampled")
            println(io, "# Format: filename noise_type snr_value")
            println(io, "# Generated by HADatasets.DatasetTypes")
            println(io, "")
            
            for filename in sort(collect(common_filenames))
                if haskey(snr_info, filename)
                    noise_type, snr_value = snr_info[filename]
                    println(io, "$filename $noise_type $snr_value")
                end
            end
        end
        
        @info "Created resampled log file: $log_output_file"
        @info "Processed $(length(common_filenames)) file pairs"
        
    catch e
        @warn "Failed to create resampled log file" exception=(e, catch_backtrace())
    end
end

end

