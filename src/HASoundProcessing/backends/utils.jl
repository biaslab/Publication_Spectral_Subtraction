"""
    General utilities for all backends in HASoundProcessing.

This module provides shared functionality that can be used across different
backend implementations, ensuring consistency in mathematical operations.
"""

module Utils

"""
    tau2ff(τ::T, fs::T)::T where {T<:AbstractFloat}

Convert time constant to forgetting factor using the correct formula.

# Arguments

  - `τ`: Time constant in milliseconds
  - `fs`: Sampling frequency in Hz

# Returns

  - Forgetting factor value

# Performance

  - Type stable for all AbstractFloat types
  - Inlined for optimal performance
  - Uses fused multiply-add operations where possible

# Notes

This is the correct implementation used in the backends and should be
used consistently across all backends for proper time constant conversion.
"""
function tau2ff(τ::T, fs::T)::T where {T <: AbstractFloat}
    1 - exp(-one(T) / ((τ / 2.3) * fs + eps()))
end

"""
    calculate_fc(τ::T)::T where {T<:AbstractFloat}

Calculate cutoff frequency from time constant.

# Arguments

  - `τ`: Time constant in milliseconds

# Returns

  - Cutoff frequency in Hz
"""
function calculate_fc(τ::T)::T where {T <: AbstractFloat}
    one(T) / (2π * (τ / 2.3 / 1000))
end

"""
    calculate_λ(λs::T, λn::T, θ::T)::T where {T<:AbstractFloat}

Calculate adaptive lambda value.

# Arguments

  - `λs`: Signal lambda
  - `λn`: Noise lambda
  - `θ`: Adaptation parameter

# Returns

  - Adaptive lambda value
"""
function calculate_λ(λs::T, λn::T, θ::T)::T where {T <: AbstractFloat}
    λs * θ + λn * (one(T) - θ)
end

"""
    λ2process_var(λ::T) where {T<:AbstractFloat}

Convert forgetting factor to process variance.

# Arguments

  - `λ::T`: Forgetting factor

# Returns

  - `T`: Process variance
"""
function λ2process_var(λ::T) where {T <: AbstractFloat}
    λ^2 / (one(T) - λ)
end

"""
    λ2observation_var(λ::T) where {T<:AbstractFloat}

Convert forgetting factor to observation variance.

# Arguments

  - `λ::T`: Forgetting factor

# Returns

  - `T`: Observation variance
"""
function λ2observation_var(λ::T) where {T <: AbstractFloat}
    return T(1) / λ
end

"""
    Γ_hyperparameters_shaperate(E_τ; strength=1.0)

Calculate Gamma distribution hyperparameters from expectation and strength.

# Arguments

  - `E_τ`: Expected value
  - `strength`: Concentration parameter (default: 1.0)

# Returns

  - `Tuple{T, T}`: Shape and rate parameters
"""
function Γ_hyperparameters_shaperate(E_τ; strength = 1.0)
    # For Gamma(α, β) where E[X] = α/β
    E_α = strength
    E_β = E_α / E_τ
    return (E_α, E_β)
end




# SEM specific constants
const SEM_SOURCE_FIELDS = (:speech, :noise, :ξ_smooth)
const SEM_STATE_FIELDS = (:gain, :vad)

# Export all functions
export tau2ff,
    calculate_fc, calculate_λ, λ2process_var, λ2observation_var, Γ_hyperparameters_shaperate

# Export constants
export SEM_SOURCE_FIELDS, SEM_STATE_FIELDS

end # module Utils
