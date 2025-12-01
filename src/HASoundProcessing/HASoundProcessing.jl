
module HASoundProcessing


using RxInfer
# Include base types first
include("types.jl")

# Export abstract types
export AbstractSPMBackend,
    AbstractSPMParameters, AbstractSPMFilter, AbstractSPMStates,
    AbstractSPMFilterParameters,
    AbstractInferenceParameters

# Include general utilities
include("backends/utils.jl")
using .Utils

# Extensions (needed for SEM)
include("backends/extensions/extensions.jl")
using .Extensions

# Backends
include("backends/sem/sem.jl")
using .SEM

# Re-export SEM types
export SEMParameters, SEMBackend, SEMStates
export ModuleParameters, InferenceParameters
export SourceParameters, GainParameters, VADParameters
export process_backend
export SEM_filtering_model

# Re-export shared utility functions
export tau2ff,
    calculate_fc, calculate_λ, λ2process_var, λ2observation_var, Γ_hyperparameters_shaperate

# Re-export SEM constants
export SEM_SOURCE_FIELDS, SEM_STATE_FIELDS

end
