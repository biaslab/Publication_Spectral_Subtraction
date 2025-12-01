using RxInfer
using SampledSignals

"""
Result structures for hearing aid processing algorithms.

This module contains data structures used to store results from various
hearing aid processing algorithms, including SEM and Baseline.

"""

"""
    BaselineResults

Results structure for Baseline hearing aid processing.

# Fields
- `gains`: Gain values applied to the signal
- `processing_info`: Additional processing information
"""
struct BaselineResults
    gains::AbstractMatrix{<:Real}
    processing_info::Dict{String,Any}
end

"""
    SEMResults

Complete results structure for SEM (Spectral Enhancement Model) processing.

# Fields
- `wiener_gains`: Wiener gain values applied to the WFB (matrix: nframes × nbands)
    Rows represent time frames, columns represent frequency bands.
    These are the gains actually used for synthesis (with spectral floor applied).
- `inference_results`: Optional inference engine results for validation/debugging
    Can be nothing to save memory, or a collection of inference results per frame/band.
    If stored, raw inference gains can be extracted from `inference_results.history[:w]`.
"""
struct SEMResults
    gains::AbstractMatrix{<:Real}
    inference_results::Union{Nothing,Any}
end

"""
    SEMHearingAidResults

Complete results structure for SEM hearing aid processing.

# Fields
- `output_signal`: Processed audio output
- `result`: SEM algorithm results
- `processing_metrics`: Performance and timing metrics

# Constructors
    SEMHearingAidResults(output_signal, result, gains, config_name)

Create SEM hearing aid results from processing output and SEMResults.
"""
struct SEMHearingAidResults
    output_signal::SampleBuf
    result::SEMResults
    processing_metrics::Dict{String,Any}

    SEMHearingAidResults(
        output_signal::SampleBuf,
        result::SEMResults,
        processing_metrics::Dict{String,Any},
    ) = new(output_signal, result, processing_metrics)

    function SEMHearingAidResults(
        output_signal::SampleBuf,
        result::SEMResults,
        gains::Matrix{Float64},
        config_name::String,
    )
        # result is already a SEMResults structure
        return new(
            output_signal,
            result,
            Dict("processing_time" => 0.0, "config_name" => config_name),
        )
    end
end

"""
    BaselineHearingAidResults

Complete results structure for Baseline hearing aid processing.

# Fields
- `output_signal`: Processed audio output
- `baseline_results`: Baseline algorithm results
- `processing_metrics`: Performance and timing metrics

# Constructors
    BaselineHearingAidResults(output_signal, result, gains, config_name)

Create Baseline hearing aid results from processing output.
"""
struct BaselineHearingAidResults
    output_signal::SampleBuf
    result::BaselineResults
    processing_metrics::Dict{String,Any}

    BaselineHearingAidResults(
        output_signal::SampleBuf,
        result::BaselineResults,
        processing_metrics::Dict{String,Any},
    ) = new(output_signal, result, processing_metrics)
    function BaselineHearingAidResults(
        output_signal::SampleBuf,
        result,
        gains::Union{Matrix{Float64},Nothing},
        config_name::String,
    )
        baseline_results = BaselineResults(gains, Dict("config_name" => config_name))
        return new(
            output_signal,
            baseline_results,
            Dict("processing_time" => 0.0, "config_name" => config_name),
        )
    end
end


