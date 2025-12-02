"""
SEM (Spectral Enhancement Model) backend implementation.
"""

module SEM

using RxInfer
using ..HASoundProcessing:
    AbstractSPMBackend,
    AbstractSPMParameters,
    AbstractSPMFilter,
    AbstractSPMStates,
    AbstractSPMFilterParameters,
    AbstractInferenceParameters

include("../extensions/extensions.jl")
using .Extensions

include("types.jl")
include("states.jl")
include("model.jl")
include("inference.jl")
include("processing.jl")

export SEMBackend, SEMParameters, SEMStates
export SourceState, SourceTransitionState, BLIState, SourceParameters
export GainState, GainParameters, VADState, VADParameters
export ModuleParameters, InferenceParameters
export process_backend, get_gain, get_speech, get_noise, get_vad
export infer_SEM_filtering, wiener_gain_spectral_floor
export SEM_filtering_model
export update_state!, update_transition!
export get_source_mean, get_source_precision, get_transition_precision
export get_gain_auxiliary, get_vad_auxiliary
export get_vad_threshold_dB, get_gain_threshold_dB,
    get_gain_threshold_lin, get_gain_slope_dB
end # module
