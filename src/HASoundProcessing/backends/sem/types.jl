# ============================================================================
# SEM Types and Data Structures
# ============================================================================

# Import shared utility functions
using ..HASoundProcessing.Utils: tau2ff, calculate_fc, λ2process_var

# ============================================================================
# Basic State Types
# ============================================================================

"""
    SourceState{T<:AbstractFloat} <: AbstractSPMStates

State container for source states (Basic Leaky Integrator).

# Fields

  - `mean::Vector{T}`: Mean values per frequency band
  - `precision::Vector{T}`: Precision values per frequency band
"""
struct SourceState{T <: AbstractFloat} <: AbstractSPMStates
    mean::Vector{T}
    precision::Vector{T}

    function SourceState{T}(
        nbands::Integer,
        prior_mean::Vector{T},
        prior_precision::Vector{T},
    ) where {T <: AbstractFloat}
        nbands > 0 || throw(ArgumentError("Number of bands must be positive"))
        length(prior_mean) == nbands ||
            throw(ArgumentError("Prior mean must have exactly nbands elements"))
        length(prior_precision) == nbands ||
            throw(ArgumentError("Prior precision must have exactly nbands elements"))
        all(prior_precision .> zero(T)) ||
            throw(ArgumentError("All prior precision values must be strictly positive"))
        new{T}(prior_mean, prior_precision)
    end
end

"""
    GainState{T<:AbstractFloat} <: AbstractSPMStates

State container for gain variable with Bernoulli distribution.

# Fields

  - `p::Vector{T}`: Probability of gain (Bernoulli success probability)
  - `q::Vector{T}`: Probability of no gain (Bernoulli failure probability, q = 1 - p)
  - `auxiliary::Vector{T}`: Auxiliary parameters for variational inference
  - `wiener_gain_spectral_floor::Vector{T}`: Wiener gain with spectral floor applied per frequency band
"""
struct GainState{T <: AbstractFloat} <: AbstractSPMStates
    p::Vector{T}
    q::Vector{T}
    auxiliary::Vector{T}
    wiener_gain_spectral_floor::Vector{T}
    function GainState{T}(
        nbands::Integer,
        prior::Vector{T},
        auxiliary::Vector{T},
        wiener_gain_spectral_floor::Vector{T},
    ) where {T <: AbstractFloat}
        nbands > 0 || throw(ArgumentError("Number of bands must be positive"))
        length(prior) == 2 || throw(ArgumentError("Prior must have exactly 2 elements"))
        # Validate that p + q = 1
        isapprox(prior[1] + prior[2], one(T), atol = eps(T)) ||
            throw(
                ArgumentError(
                    "Prior probabilities must sum to 1, got $(prior[1] + prior[2])",
                ),
            )
        length(auxiliary) == nbands ||
            throw(ArgumentError("Auxiliary must have exactly nbands elements"))
        # Validate that all auxiliary values are positive
        all(auxiliary .> zero(T)) ||
            throw(ArgumentError("All auxiliary values must be greater than 0"))
        length(wiener_gain_spectral_floor) == nbands ||
            throw(
                ArgumentError(
                    "Wiener gain spectral floor must have exactly nbands elements",
                ),
            )
        # prior[1] is the probability of "no gain" state (q, failure probability)
        # prior[2] is the probability of "gain" state (p, success probability)
        p = fill(prior[2], nbands)
        q = fill(prior[1], nbands)
        new{T}(p, q, auxiliary, wiener_gain_spectral_floor)
    end
end

"""
    VADState{T<:AbstractFloat} <: AbstractSPMStates

State container for vad variable with Bernoulli distribution.

# Fields

  - `p::Vector{T}`: Probability of vad on (Bernoulli success probability)
  - `q::Vector{T}`: Probability of vad off (Bernoulli failure probability, q = 1 - p)
  - `auxiliary::Vector{T}`: Auxiliary parameters for variational inference
"""
struct VADState{T <: AbstractFloat} <: AbstractSPMStates
    p::Vector{T}
    q::Vector{T}
    auxiliary::Vector{T}

    function VADState{T}(
        nbands::Integer,
        prior::Vector{T},
        auxiliary::Vector{T},
    ) where {T <: AbstractFloat}
        nbands > 0 || throw(ArgumentError("Number of bands must be positive"))
        length(prior) == 2 || throw(ArgumentError("Prior must have exactly 2 elements"))
        # Validate that p + q = 1
        isapprox(prior[1] + prior[2], one(T), atol = eps(T)) ||
            throw(
                ArgumentError(
                    "Prior probabilities must sum to 1, got $(prior[1] + prior[2])",
                ),
            )
        length(auxiliary) == nbands ||
            throw(ArgumentError("Auxiliary must have exactly nbands elements"))
        # Validate that all auxiliary values are positive
        all(auxiliary .> zero(T)) ||
            throw(ArgumentError("All auxiliary values must be greater than 0"))
        # prior[1] is the probability of "vad off" state (q, failure probability)
        # prior[2] is the probability of "vad on" state (p, success probability)
        p = fill(prior[2], nbands)
        q = fill(prior[1], nbands)
        new{T}(p, q, auxiliary)
    end
end

"""
    SourceTransitionState{T<:AbstractFloat} <: AbstractSPMStates

Transition parameters for source state evolution.

# Fields

  - `precision::Vector{T}`: Transition precision parameters per frequency band
"""
struct SourceTransitionState{T <: AbstractFloat} <: AbstractSPMStates
    precision::Vector{T}

    function SourceTransitionState{T}(nbands::Integer, λ::T) where {T <: AbstractFloat}
        nbands > 0 || throw(ArgumentError("Number of bands must be positive"))
        process_var = λ2process_var(λ)
        precision_val = T(1.0) / process_var
        new{T}(fill(precision_val, nbands))
    end
end

"""
    SourceParameters{T<:AbstractFloat} <: AbstractSPMFilterParameters

Parameters for a source (speech or noise).

# Fields

  - `τ90::T`: Time constant at 90% response (ms)
  - `sampling_frequency::T`: Sampling frequency (Hz)
  - `λ::T`: Forgetting factor (computed from τ90)
  - `fc::T`: Cutoff frequency (computed from τ90)
"""
struct SourceParameters{T <: AbstractFloat} <: AbstractSPMFilterParameters
    τ90::T
    sampling_frequency::T
    λ::T
    fc::T

    function SourceParameters{T}(τ90::T, sampling_frequency::T) where {T <: AbstractFloat}
        τ90 > 0 || throw(ArgumentError("Time constant must be positive"))
        sampling_frequency > 0 ||
            throw(ArgumentError("Sampling frequency must be positive"))
        λ = tau2ff(τ90, sampling_frequency)
        fc = calculate_fc(τ90)
        new{T}(τ90, sampling_frequency, λ, fc)
    end
end

"""
    GainParameters{T<:AbstractFloat} <: AbstractSPMFilterParameters

Parameters for the gain state.

# Fields

  - `slope_dB::T`: Slope parameter for sigmoid function (in dB)
  - `threshold_dB::T`: Threshold parameter for sigmoid function (in dB)
  - `threshold_lin::T`: Linear threshold computed from threshold_dB (10^(-threshold_dB / 20.0))
"""
struct GainParameters{T <: AbstractFloat} <: AbstractSPMFilterParameters
    slope_dB::T
    threshold_dB::T
    threshold_lin::T

    function GainParameters{T}(slope_dB::T, threshold_dB::T) where {T <: AbstractFloat}
        gmin_suppression_linear = 10^(-threshold_dB / 20)
        gmin_residual_suppression_linear = 1 - gmin_suppression_linear
        gmin_residual_suppression_db = round(-20 * log10(gmin_residual_suppression_linear))
        safe_guard_linear_attenuation = 10^(-gmin_residual_suppression_db / 20)
        threshold_lin = 10^(20 * log10(1 - safe_guard_linear_attenuation + eps()) / 20)
        new{T}(slope_dB, gmin_residual_suppression_db, threshold_lin)
    end
end

"""
    VADParameters{T<:AbstractFloat} <: AbstractSPMFilterParameters

Parameters for the vad state.

# Fields

  - `slope_dB::T`: Slope parameter for sigmoid function (in dB)
  - `threshold_dB::T`: Threshold parameter for sigmoid function (in dB)
"""
struct VADParameters{T <: AbstractFloat} <: AbstractSPMFilterParameters
    slope_dB::T
    threshold_dB::T

    function VADParameters{T}(slope_dB::T, threshold_dB::T) where {T <: AbstractFloat}
        new{T}(slope_dB, threshold_dB)
    end
end

struct InferenceParameters <: AbstractInferenceParameters
    iterations::Int
    autostart::Bool
    free_energy::Bool

    function InferenceParameters(iterations::Int = 1, autostart::Bool = true,
        free_energy::Bool = true)
        iterations > 0 || throw(ArgumentError("Number of iterations must be positive"))
        new(iterations, autostart, free_energy)
    end

end

"""
    ModuleParameters{T<:AbstractFloat} <: AbstractSPMParameters

Module parameters for SEM algorithm.

# Fields

  - `speech::SourceParameters{T}`: Speech source parameters
  - `noise::SourceParameters{T}`: Noise source parameters
  - `ξ_smooth::SourceParameters{T}`: ξ_smooth source parameters
  - `gain::GainParameters{T}`: Gain parameters
  - `vad::VADParameters{T}`: VAD parameters
  - `sampling_frequency::T`: Sampling frequency (Hz)
  - `nbands::Int`: Number of frequency bands
"""
struct ModuleParameters{T <: AbstractFloat} <: AbstractSPMParameters
    speech::SourceParameters{T}
    noise::SourceParameters{T}
    ξ_smooth::SourceParameters{T}
    gain::GainParameters{T}
    vad::VADParameters{T}
    sampling_frequency::T
    nbands::Int

    function ModuleParameters{T}(
        speech::SourceParameters{T},
        noise::SourceParameters{T},
        ξ_smooth::SourceParameters{T},
        gain::GainParameters{T},
        vad::VADParameters{T},
        sampling_frequency::T,
        nbands::Int,
    ) where {T <: AbstractFloat}
        sampling_frequency > 0 ||
            throw(ArgumentError("Sampling frequency must be positive"))
        nbands > 0 || throw(ArgumentError("Number of bands must be positive"))
        new{T}(speech, noise, ξ_smooth, gain, vad, sampling_frequency, nbands)
    end
end

"""
    SEMParameters{T<:AbstractFloat} <: AbstractSPMParameters

Common parameters for the SEM algorithm.

# Fields

  - `inference::InferenceParameters`: Global inference configuration (iterations, autostart, free_energy)
  - `modules::ModuleParameters{T}`: Per-module parameters for speech, noise, ξ_smooth, gain, vad, plus shared sampling_frequency and nbands
"""
struct SEMParameters{T <: AbstractFloat} <: AbstractSPMParameters

    inference::InferenceParameters
    modules::ModuleParameters{T}

    function SEMParameters{T}(
        inference::InferenceParameters,
        modules::ModuleParameters{T},
    ) where {T <: AbstractFloat}
        new{T}(inference, modules)
    end
end

# ============================================================================
# State Container Types
# ============================================================================

"""
    BLIState{T<:AbstractFloat} <: AbstractSPMStates

Unified source structure for speech and noise sources (Basic Leaky Integrator).

# Fields

  - `state::SourceState{T}`: Current state (mean and precision)
  - `transition::SourceTransitionState{T}`: Transition parameters

# Note

Parameters are stored in `SEMParameters.modules` and accessed from the backend when needed.
This avoids duplication and follows the principle: States = mutable runtime data, Parameters = immutable configuration.
"""
struct BLIState{T <: AbstractFloat} <: AbstractSPMStates
    state::SourceState{T}
    transition::SourceTransitionState{T}

    function BLIState{T}(
        params::SourceParameters{T},
        nbands::Integer,
        prior_mean::T = zero(T),
        prior_precision::T = T(1.0),
    ) where {T <: AbstractFloat}
        nbands > 0 || throw(ArgumentError("Number of bands must be positive"))
        state =
            SourceState{T}(nbands, fill(prior_mean, nbands), fill(prior_precision, nbands))
        transition = SourceTransitionState{T}(nbands, params.λ)
        new{T}(state, transition)
    end
end


"""
    SEMStates{T<:AbstractFloat} <: AbstractSPMStates

Container for all SEM algorithm states.

# Fields

  - `speech::BLIState{T}`: Speech source state
  - `noise::BLIState{T}`: Noise source state
  - `ξ_smooth::BLIState{T}`: Smoothed SNR state
  - `gain::GainState{T}`: Gain state
  - `vad::VADState{T}`: vad state
"""
struct SEMStates{T <: AbstractFloat} <: AbstractSPMStates
    speech::BLIState{T}
    noise::BLIState{T}
    ξ_smooth::BLIState{T}
    gain::GainState{T}
    vad::VADState{T}

    function SEMStates{T}(
        params::SEMParameters{T},
        speech_prior_mean::T = zero(T),
        speech_prior_precision::T = T(1.0),
        noise_prior_mean::T = zero(T),
        noise_prior_precision::T = T(1.0),
        ξ_smooth_prior_mean::T = zero(T),
        ξ_smooth_prior_precision::T = T(1.0),
        gain_prior::Vector{T} = [T(0.5), T(0.5)],
        gain_auxiliary::Union{Vector{T}, Nothing} = nothing,
        gain_wiener_floor::Union{Vector{T}, Nothing} = nothing,
        vad_prior::Vector{T} = [T(0.5), T(0.5)],
        vad_auxiliary::Union{Vector{T}, Nothing} = nothing,
    ) where {T <: AbstractFloat}
        nbands = params.modules.nbands
        nbands > 0 || throw(ArgumentError("Number of bands must be positive"))
        # Use provided values or defaults
        gain_auxiliary = isnothing(gain_auxiliary) ? fill(T(1.0), nbands) : gain_auxiliary
        gain_wiener_floor =
            isnothing(gain_wiener_floor) ? fill(T(0.0), nbands) : gain_wiener_floor
        vad_auxiliary = isnothing(vad_auxiliary) ? fill(T(1.0), nbands) : vad_auxiliary
        speech = BLIState{T}(
            params.modules.speech,
            nbands,
            speech_prior_mean,
            speech_prior_precision,
        )
        noise = BLIState{T}(
            params.modules.noise,
            nbands,
            noise_prior_mean,
            noise_prior_precision,
        )
        ξ_smooth = BLIState{T}(
            params.modules.ξ_smooth,
            nbands,
            ξ_smooth_prior_mean,
            ξ_smooth_prior_precision,
        )
        # Initialize gain and vad states using default or provided parameters
        gain = GainState{T}(nbands, gain_prior, gain_auxiliary, gain_wiener_floor)
        vad = VADState{T}(nbands, vad_prior, vad_auxiliary)
        new{T}(speech, noise, ξ_smooth, gain, vad)
    end
end

"""
    SEMBackend{T<:AbstractFloat} <: AbstractSPMBackend

SEM backend implementation.

# Fields

  - `states::SEMStates{T}`: All SEM states
  - `params::SEMParameters{T}`: SEM parameters
"""
struct SEMBackend{T <: AbstractFloat} <: AbstractSPMBackend
    states::SEMStates{T}
    params::SEMParameters{T}
end

# ============================================================================
# Constructors
# ============================================================================

"""
    SEMBackend(params::SEMParameters{T}, speech_prior_mean::T, speech_prior_precision::T, noise_prior_mean::T, noise_prior_precision::T, ξ_smooth_prior_mean::T, ξ_smooth_prior_precision::T) -> SEMBackend{T}

Create a SEM backend with specified parameters and priors.

# Arguments

  - `params::SEMParameters{T}`: SEM parameters
  - `speech_prior_mean::T`: Prior mean for speech source (default: `zero(T)`)
  - `speech_prior_precision::T`: Prior precision for speech source (default: `T(1.0)`)
  - `noise_prior_mean::T`: Prior mean for noise source (default: `zero(T)`)
  - `noise_prior_precision::T`: Prior precision for noise source (default: `T(1.0)`)
  - `ξ_smooth_prior_mean::T`: Prior mean for ξ_smooth source (default: `zero(T)`)
  - `ξ_smooth_prior_precision::T`: Prior precision for ξ_smooth source (default: `T(1.0)`)

# Returns

  - `SEMBackend{T}`: Initialized SEM backend
"""
function SEMBackend(
    params::SEMParameters{T},
    speech_prior_mean::T = zero(T),
    speech_prior_precision::T = T(1.0),
    noise_prior_mean::T = zero(T),
    noise_prior_precision::T = T(1.0),
    ξ_smooth_prior_mean::T = zero(T),
    ξ_smooth_prior_precision::T = T(1.0),
) where {T <: AbstractFloat}
    states = SEMStates{T}(
        params,
        speech_prior_mean,
        speech_prior_precision,
        noise_prior_mean,
        noise_prior_precision,
        ξ_smooth_prior_mean,
        ξ_smooth_prior_precision,
    )
    return SEMBackend{T}(states, params)
end

# ============================================================================
# Getter Functions
# ============================================================================

"""
    get_speech(backend::SEMBackend{T}) -> BLIState{T}

Get the speech source from the backend.
"""
get_speech(backend::SEMBackend{T}) where {T} = backend.states.speech

"""
    get_noise(backend::SEMBackend{T}) -> BLIState{T}

Get the noise source from the backend.
"""
get_noise(backend::SEMBackend{T}) where {T} = backend.states.noise

"""
    get_vad(backend::SEMBackend{T}) -> Vector{T}

Get the vad values from the backend.
"""
get_vad(backend::SEMBackend{T}) where {T} = backend.states.vad.p

"""
    get_gain(backend::SEMBackend{T}) -> Vector{T}

Get the Wiener gain with spectral floor applied from the backend.
This is the processed gain value used by other modules.
"""
get_gain(backend::SEMBackend{T}) where {T} =
    backend.states.gain.wiener_gain_spectral_floor
