module WFB

using SampledSignals
using DataStructures: CircularBuffer
using DSP
using LinearAlgebra
using Base: atan
using FFTW

# Add these constants at the top of the file
const EDGE_BAND_ADJUSTMENT_DB = 3.0103
const SCALE_FACTOR_DB = 20.0
const POWER_SCALE_FACTOR = 10.0

# Include types and implementations
include("../abstract_frontend.jl")
include("wfb_types.jl")
include("wfb_processing.jl")

# Export public API
export
    # Types
    WFBFrontendParameters,
    WFBFrontend,

    # Processing functions
    process_frontend,

    # Helper functions
    calculate_window,
    calculate_input_calibration_db,
    calculate_center_frequencies,
    calculate_synthesis_matrix,
    db2lin,
    synthesize,

    # Getter methods
    get_weights,
    get_sample_buffer,
    get_taps,
    get_temp,
    get_window,
    get_calibration_db,
    get_synthesis_matrix,
    get_nbands,
    get_nfft,
    get_fs,
    get_params

end # module 
