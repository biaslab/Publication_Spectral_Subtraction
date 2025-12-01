module VirtualHearingAid

using SampledSignals
using DataStructures: CircularBuffer
using DSP
using LinearAlgebra
using Base: atan
using FFTW
using ..HASoundProcessing
using TOML

# Import specific types from HASoundProcessing
import ..HASoundProcessing:
    SEMBackend,
    SEMParameters,
    SourceParameters
import ..HASoundProcessing.SEM: GainParameters, VADParameters

# Core types and traits
include("types.jl")
include("traits.jl")

# Include getters early to define the get_params pattern
include("getters.jl")

include("frontends/frontends.jl")
import .Frontends
import .Frontends:
    AbstractFrontend,
    AbstractFrontendParameters,
    WFBFrontend,
    WFBFrontendParameters,
    synthesize

# Re-export only the stable frontend API surface
export AbstractFrontend, AbstractFrontendParameters
export get_params

# Include backend definitions
include("backends/abstract_backend.jl")

# Export backends
export BaselineBackend,
    AbstractSPMBackend, SEMBackend, SEMParameters

# Include hearing aid implementations
include("hearing_aids/abstract_hearing_aid.jl")
include("hearing_aids/baseline/type.jl")
include("hearing_aids/baseline/config.jl")
include("hearing_aids/baseline/processing.jl")
include("hearing_aids/sem/type.jl")
include("hearing_aids/sem/config.jl")
include("hearing_aids/sem/processing.jl")

"""
    processing_strategy(::Type{BaselineHearingAid})

Default processing strategy for Baseline hearing aids.
"""
processing_strategy(::Type{BaselineHearingAid}) = BatchProcessingOnline()

"""
    processing_strategy(::Type{SEMHearingAid})

Default processing strategy for SEM hearing aids.
"""
processing_strategy(::Type{SEMHearingAid}) = BatchProcessingOffline()

@inline
get_backend(ha::BaselineHearingAid) = ha.backend
@inline
get_backend(ha::SEMHearingAid) = ha.backend

include("utils/config.jl")
include("utils/validation.jl")
include("utils/results.jl")
include("utils/synthesis.jl")
include("utils/performance.jl")
include("utils/logging.jl")

include("processing/batch.jl")
include("processing/stream.jl")

# Export public API
export

    # Core types
    AbstractHearingAid,
    AbstractFrontend,
    AbstractFrontendParameters,
    AbstractSPMBackend,

    # Backends
    BaselineBackend,
    # Frontends 
    synthesize,

    # Hearing Aid types
    BaselineHearingAid,
    SEMHearingAid,

    # Main processing functions
    process,
    process_block,
    denoise,

    # Configuration
    from_config,
    get_hearing_aids,

    # Traits
    backend_type,
    processing_strategy,
    get_processing_strategy,
    set_processing_strategy!,
    ProcessingStrategy,
    StreamProcessing,
    BatchProcessingOnline,
    BatchProcessingOffline,

    # Getters/setters
    get_frontend,
    get_backend,
    get_params,
    getname,

    # Offline helpers
    synthesize_offline,

    # Result structures
    SEMResults,
    BaselineResults,
    SEMHearingAidResults,
    BaselineHearingAidResults

end # module
