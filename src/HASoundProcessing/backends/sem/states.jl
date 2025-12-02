"""
State management functions for SEM backend.
"""

"""
    update_state!(state::SourceState{T}, mean::T, precision::T, band::Int) where {T<:AbstractFloat}

Update the state of a Basic Leaky Integrator for a specific frequency band.
"""
function update_state!(
    state::SourceState{T},
    mean::T,
    precision::T,
    band::Int,
) where {T <: AbstractFloat}
    state.mean[band] = mean
    state.precision[band] = precision
end

"""
    update_transition!(transition::SourceTransitionState{T}, precision::T, band::Int) where {T<:AbstractFloat}

Update the transition parameters for a specific frequency band.
"""
function update_transition!(
    transition::SourceTransitionState{T},
    precision::T,
    band::Int,
) where {T <: AbstractFloat}
    transition.precision[band] = precision
end

"""
    update_state!(source::BLIState{T}, mean::T, precision::T, band::Int) where {T<:AbstractFloat}

Update a source's state for a specific frequency band.
"""
function update_state!(
    source::BLIState{T},
    mean::T,
    precision::T,
    band::Int,
) where {T <: AbstractFloat}
    update_state!(source.state, mean, precision, band)
end

"""
    update_transition!(source::BLIState{T}, precision::T, band::Int) where {T<:AbstractFloat}

Update a source's transition parameters for a specific frequency band.
"""
function update_transition!(
    source::BLIState{T},
    precision::T,
    band::Int,
) where {T <: AbstractFloat}
    update_transition!(source.transition, precision, band)
end

"""
    update_state!(backend::SEMBackend{T}, ::Val{S}, mean::T, precision::T, band::Int) where {T<:AbstractFloat, S}

Update a source's state in the backend for a specific frequency band using Val-based dispatch.
"""
function update_state!(
    backend::SEMBackend{T},
    ::Val{S},
    mean::T,
    precision::T,
    band::Int,
) where {T <: AbstractFloat, S}
    source = getproperty(backend.states, S)
    update_state!(source, mean, precision, band)
end

"""
    update_transition!(backend::SEMBackend{T}, ::Val{S}, precision::T, band::Int) where {T<:AbstractFloat, S}

Update a source's transition parameters in the backend for a specific frequency band using Val-based dispatch.

Supports:

  - `:speech` - updates speech transition precision
  - `:noise` - updates noise transition precision
  - `:ξ` - updates ξ transition precision
  - `:τs` - alias for `:speech`
  - `:τn` - alias for `:noise`
  - `:τξ` - alias for `:ξ`
"""
function update_transition!(
    backend::SEMBackend{T},
    ::Val{S},
    precision::T,
    band::Int,
) where {T <: AbstractFloat, S}
    # Map transition aliases to actual state names
    if S === :τs
        update_transition!(backend.states.speech, precision, band)
    elseif S === :τn
        update_transition!(backend.states.noise, precision, band)
    elseif S === :τξ
        update_transition!(backend.states.ξ, precision, band)
    elseif S === :speech || S === :noise || S === :ξ
        source = getproperty(backend.states, S)
        update_transition!(source, precision, band)
    else
        throw(
            ArgumentError(
                "Unknown transition state: $S. Use :speech, :noise, :ξ, :τs, :τn, or :τξ",
            ),
        )
    end
end

"""
    update_bernoulli_state!(state, value::T, band::Int) where {T<:AbstractFloat}

Update a Bernoulli state (p and q) for a specific frequency band.
Used internally by update_state! for gain, vad, and switch states.
"""
function update_bernoulli_state!(
    state,
    value::T,
    band::Int,
) where {T <: AbstractFloat}
    state.p[band] = value
    state.q[band] = T(1.0) - value
end

"""
    update_state!(backend::SEMBackend{T}, ::Val{S}, value::T, band::Int) where {T<:AbstractFloat, S}

Update a state for a specific frequency band using Val-based dispatch.

Supports:

  - Bernoulli states: `:switch`, `:gain`, `:vad` - updates p and q from a single probability value
  - Transition states: `:τs` (speech), `:τn` (noise) - updates transition precision
  - Source states: `:speech`, `:noise` - requires mean and precision (use separate method)
"""
function update_state!(
    backend::SEMBackend{T},
    ::Val{S},
    value::T,
    band::Int,
) where {T <: AbstractFloat, S}
    # Bernoulli states (switch is an alias for vad, gain, vad)
    if S === :switch
        # switch is an alias for vad (they represent the same state)
        update_bernoulli_state!(backend.states.vad, value, band)
    elseif S === :gain || S === :vad
        state = getproperty(backend.states, S)
        update_bernoulli_state!(state, value, band)
        # Transition states (τs, τn, τξ)
    elseif S === :τs
        update_transition!(backend.states.speech, value, band)
    elseif S === :τn
        update_transition!(backend.states.noise, value, band)
    elseif S === :τξ
        update_transition!(backend.states.ξ, value, band)
    else
        throw(
            ArgumentError("Unknown state: $S. Use :switch, :gain, :vad, :τs, :τn, or :τξ"),
        )
    end
end

function get_state_mean(state::SourceState{T}, band::Int) where {T <: AbstractFloat}
    return state.mean[band]
end

function get_state_precision(
    state::SourceState{T},
    band::Int,
) where {T <: AbstractFloat}
    return state.precision[band]
end

function get_source_mean(source::BLIState{T}, band::Int) where {T <: AbstractFloat}
    return get_state_mean(source.state, band)
end

function get_source_precision(
    source::BLIState{T},
    band::Int,
) where {T <: AbstractFloat}
    return get_state_precision(source.state, band)
end


function get_transition_precision(
    source::BLIState{T},
    band::Int,
) where {T <: AbstractFloat}
    return source.transition.precision[band]
end

function get_transition_precision(
    transition::SourceTransitionState{T},
    band::Int,
) where {T <: AbstractFloat}
    return transition.precision[band]
end


function get_gain_auxiliary(
    backend::SEMBackend{T},
    band::Int,
) where {T <: AbstractFloat}
    return backend.states.gain.auxiliary[band]
end


function get_vad_auxiliary(
    backend::SEMBackend{T},
    band::Int,
) where {T <: AbstractFloat}
    return backend.states.vad.auxiliary[band]
end


function get_vad_threshold_dB(backend::SEMBackend{T}) where {T <: AbstractFloat}
    return backend.params.modules.vad.threshold_dB
end

function get_gain_threshold_dB(backend::SEMBackend{T}) where {T <: AbstractFloat}
    return backend.params.modules.gain.threshold_dB
end


function get_gain_threshold_lin(backend::SEMBackend{T}) where {T <: AbstractFloat}
    return backend.params.modules.gain.threshold_lin
end


function get_gain_slope_dB(backend::SEMBackend{T}) where {T <: AbstractFloat}
    return backend.params.modules.gain.slope_dB
end
