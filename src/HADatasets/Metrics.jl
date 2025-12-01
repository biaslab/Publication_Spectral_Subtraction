"""
Metrics - Speech enhancement evaluation module integrating PESQ and DNSMOS.
"""
module Metrics

using Logging
using PyCall

export get_metrics, pesq_score, dnsmos_score, evaluate_all

const pesq_module = Ref{Any}(nothing)
const dnsmos_module = Ref{Any}(nothing)

function __init__()
    try
        global pesq_module
        global dnsmos_module
        
        pesq_module[] = pyimport("pesq")
        
        try
            dnsmos_module[] = pyimport("dnsmos_wrapper")
        catch e
            error("Failed to import dnsmos_wrapper: $e")
        end
        
        @info "Python modules loaded successfully: pesq, dnsmos"
    catch e
        error("Failed to load Python modules. To enable all metrics, install: pip install pesq dnsmos. Error: $e")
    end
end

"""
    pesq_score(reference, denoised, fs; mode)

Calculate PESQ (Perceptual Evaluation of Speech Quality) score.

Requires 16 kHz sample rate. Mode can be "wb" (wideband) or "nb" (narrowband).

# Performance
This function is marked @inline for better performance in hot loops.
"""
@inline function pesq_score(reference::Vector{Float64}, denoised::Vector{Float64}, fs::Int; 
                   mode::String="wb")::Float64
    # Input validation
    if isempty(reference) || isempty(denoised)
        error("Signals cannot be empty")
    end
    
    if length(reference) != length(denoised)
        error("Signal length mismatch: reference=$(length(reference)) samples, denoised=$(length(denoised)) samples")
    end
    
    if fs != 16000
        error("PESQ only supports 16000 Hz sample rate, got $fs Hz")
    end
        
    if pesq_module[] === nothing
        error("PESQ module not available. Please install pesq: pip install pesq")
    end
    
    try
        score = pesq_module[].pesq(fs, reference, denoised, mode)
        return Float64(score)
    catch e
        @error "PESQ calculation failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    DNSMOSResults

Type alias for DNSMOS evaluation results as NamedTuple.
"""
const DNSMOSResults = NamedTuple{(:SIG, :BAK, :OVRL), Tuple{Float64, Float64, Float64}}

"""
    dnsmos_score(processed, fs)

Calculate DNSMOS (Deep Noise Suppression Mean Opinion Score) using Microsoft's DNSMOS P.835.

Returns a NamedTuple with SIG, BAK, and OVRL scores. Requires 16 kHz sample rate.

# Performance
Uses NamedTuple instead of Dict for better type stability and performance.
"""
function dnsmos_score(processed::Vector{Float64}, fs::Int)::DNSMOSResults
    # Input validation
    if isempty(processed)
        error("Processed signal cannot be empty")
    end
    
    if fs != 16000
        error("DNSMOS only supports 16000 Hz sample rate, got $fs Hz")
    end
    
    if dnsmos_module[] === nothing
        error("DNSMOS module not available. Please install dnsmos: pip install dnsmos")
    end
    
    try
        scores = dnsmos_module[].dnsmos(processed, fs)
        # Return as NamedTuple for better performance
        return (SIG=Float64(scores["SIG"]), 
                BAK=Float64(scores["BAK"]), 
                OVRL=Float64(scores["OVRL"]))
    catch e
        @error "DNSMOS calculation failed" exception=(e, catch_backtrace())
        rethrow(e)
    end
end

"""
    MetricResults

Type alias for metric evaluation results as NamedTuple.
More efficient than Dict for small fixed-size results.
"""
const MetricResults = NamedTuple{(:PESQ, :SIG, :BAK, :OVRL), Tuple{Float64, Float64, Float64, Float64}}

"""
    evaluate_all(reference, denoised, fs)

Evaluate all metrics for a reference-processed pair.

Returns a NamedTuple with PESQ, SIG, BAK, and OVRL scores. Requires 16 kHz sample rate.

# Performance
Uses NamedTuple instead of Dict for better type stability and performance.
"""
function evaluate_all(reference::Vector{Float64}, denoised::Vector{Float64}, fs::Int)::MetricResults
    # Input validation
    if isempty(reference) || isempty(denoised)
        error("Signals cannot be empty")
    end
    
    if length(reference) != length(denoised)
        error("Signal length mismatch: reference=$(length(reference)) samples, denoised=$(length(denoised)) samples")
    end
    
    # Calculate metrics
    pesq_val = pesq_score(reference, denoised, fs)
    
    dnsmos_scores = dnsmos_score(denoised, fs)
    
    # Return as NamedTuple for better performance
    return (PESQ=pesq_val, SIG=dnsmos_scores.SIG, 
            BAK=dnsmos_scores.BAK, OVRL=dnsmos_scores.OVRL)
end

"""
    get_metrics()

Get dictionary of available metric functions.
"""
function get_metrics()
    return Dict{String, Function}(
        "pesq" => pesq_score,
        "dnsmos" => dnsmos_score
    )
end

end
