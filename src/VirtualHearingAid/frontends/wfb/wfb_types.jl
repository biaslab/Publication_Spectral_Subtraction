"""
Parameters for Warped-frequency Filter Bank frontend.
"""
struct WFBFrontendParameters <: AbstractFrontendParameters
    nbands::Int          # Number of frequency bands
    nfft::Int            # FFT size
    fs::Float64          # Sampling frequency (Hz)
    spl_reference_db::Float64    # Reference SPL in dB
    spl_power_estimate_lower_bound_db::Float64  # Lower bound for power estimation in dB
    apcoefficient::Float64       # Coefficient for power estimation
    buffer_size::Int     # Size of processing buffer
end

"""
Warped-frequency Filter Bank frontend implementation.
"""
mutable struct WFBFrontend <: AbstractFrontend
    params::WFBFrontendParameters
    weights::Vector{Float64}
    sample_buffer::CircularBuffer{Float64}
    taps::Matrix{Float64}
    temp::Vector{Float64}
    window::Vector{Float64}
    calibration_db::Matrix{Float64}
    synthesis_matrix::Matrix{Float64}
end

"""
Constructor for WFBFrontend.
"""
function WFBFrontend(params::WFBFrontendParameters)
    nbands = params.nbands
    nfft = params.nfft
    window = calculate_window(nfft)
    WFBFrontend(
        params,
        vcat(zeros(nbands - 1), 1, zeros(nbands - 2)),
        CircularBuffer{Float64}(params.buffer_size),
        zeros(params.buffer_size, nfft),
        zeros(nfft),
        window,
        calculate_input_calibration_db(window, nbands, params.spl_reference_db),
        calculate_synthesis_matrix(nbands, nfft),
    )
end

get_weights(front_end::WFBFrontend) = front_end.weights
get_sample_buffer(front_end::WFBFrontend) = front_end.sample_buffer
get_taps(front_end::WFBFrontend) = front_end.taps
get_temp(front_end::WFBFrontend) = front_end.temp
get_window(front_end::WFBFrontend) = front_end.window
get_calibration_db(front_end::WFBFrontend) = front_end.calibration_db
get_synthesis_matrix(front_end::WFBFrontend) = front_end.synthesis_matrix

# Setters
set_weights!(front_end::WFBFrontend, weights::Vector{Float64}) = front_end.weights = weights
set_sample_buffer!(front_end::WFBFrontend, buffer::CircularBuffer{Float64}) =
    front_end.sample_buffer = buffer
set_taps!(front_end::WFBFrontend, taps::Matrix{Float64}) = front_end.taps = taps
set_temp!(front_end::WFBFrontend, temp::Vector{Float64}) = front_end.temp = temp
set_window!(front_end::WFBFrontend, window::Vector{Float64}) = front_end.window = window
set_calibration_db!(front_end::WFBFrontend, calibration_db::Matrix{Float64}) =
    front_end.calibration_db = calibration_db
set_synthesis_matrix!(front_end::WFBFrontend, synthesis_matrix::Matrix{Float64}) =
    front_end.synthesis_matrix = synthesis_matrix
# Getters for params
get_nbands(front_end::WFBFrontend) = front_end.params.nbands
get_nfft(front_end::WFBFrontend) = front_end.params.nfft
get_fs(front_end::WFBFrontend) = front_end.params.fs
get_spl_reference_db(front_end::WFBFrontend) = front_end.params.spl_reference_db
get_spl_power_estimate_lower_bound_db(front_end::WFBFrontend) =
    front_end.params.spl_power_estimate_lower_bound_db
get_apcoefficient(front_end::WFBFrontend) = front_end.params.apcoefficient
get_params(front_end::WFBFrontend) = front_end.params
get_buffer_size(front_end::WFBFrontend) = front_end.params.buffer_size
