module Experiments

# Load all submodules first
include("HASoundProcessing/HASoundProcessing.jl")
include("HADatasets/HADatasets.jl")
include("VirtualHearingAid/VirtualHearingAid.jl")

using CSV
using DataFrames
using TOML
using Dates
using Logging
using .HADatasets
using SampledSignals: SampleBuf
using Statistics
using WAV
using .VirtualHearingAid

# Re-export key functions from submodules for convenience
export HADatasets, VirtualHearingAid, HASoundProcessing

const HearingAidType = String

"""
    load_audio_file(file_path::String)::SampleBuf

Load audio file using HADatasets and convert to SampleBuf.
"""
function load_audio_file(file_path::String)::SampleBuf
    mono, fs, duration = HADatasets.Misc.load_audio_file(file_path)
    if isnothing(mono) || isnothing(fs)
        throw(ArgumentError("Failed to load audio file: $file_path"))
    end
    return SampleBuf(mono, fs)
end

"""
    load_config(config_path::String)::Dict{String, Any}

Load a configuration file from a path.
"""
function load_config(config_path::String)::Dict{String, Any}
    if !isfile(config_path)
        throw(ArgumentError("Configuration file does not exist: $config_path"))
    end
    return TOML.parsefile(config_path)
end

"""
    create_hearing_aid_from_config(config::Dict{String, Any})

Create a hearing aid instance from a configuration dictionary.
"""
function create_hearing_aid_from_config(config::Dict{String, Any})
    # Extract hearing aid type from config
    ha_type_str = config["parameters"]["hearingaid"]["type"]
    
    # Map string to type
    ha_type = if ha_type_str == "BaselineHearingAid"
        VirtualHearingAid.BaselineHearingAid
    elseif ha_type_str == "SEMHearingAid"
        VirtualHearingAid.SEMHearingAid
    else
        throw(ArgumentError("Unknown hearing aid type: $ha_type_str"))
    end
    
    return VirtualHearingAid.from_config(ha_type, config)
end

"""
    create_hearing_aid_from_config_file(config_path::String)

Create a hearing aid instance from a configuration file path.
"""
function create_hearing_aid_from_config_file(config_path::String)
    config = load_config(config_path)
    return create_hearing_aid_from_config(config)
end

end # module
