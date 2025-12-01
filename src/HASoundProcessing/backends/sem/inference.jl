"""
    infer_SEM_filtering(backend::SEMBackend{T}, data::Vector{T}, band::Int) where {T<:AbstractFloat}

Perform variational inference for the SEM filtering model.

# Arguments

  - `backend::SEMBackend{T}`: The SEM backend instance
  - `data::Vector{T}`: Input data for inference
  - `band::Int`: Frequency band index

# Returns

  - Inference engine results
"""
function infer_SEM_filtering(
    backend::SEMBackend{T},
    data::Vector{T},
    band::Int,
) where {T <: AbstractFloat}
    # Validate band index
    nbands = backend.params.modules.nbands
    (1 <= band <= nbands) || throw(
        ArgumentError(
            "Band index $band is out of range. Backend has $nbands bands (valid indices: 1:$nbands)",
        ),
    )

    speech_mean = get_source_mean(backend.states.speech, band)
    speech_precision = get_source_precision(backend.states.speech, band)
    noise_mean = get_source_mean(backend.states.noise, band)
    noise_precision = get_source_precision(backend.states.noise, band)

    ξ_mean = get_source_mean(backend.states.ξ_smooth, band)
    ξ_precision = get_source_precision(backend.states.ξ_smooth, band)

    τξ = get_transition_precision(backend.states.ξ_smooth, band)
    τs = get_transition_precision(backend.states.speech, band)
    τn = get_transition_precision(backend.states.noise, band)

    η = get_vad_slope_dB(backend)
    κ = get_vad_threshold_dB(backend)
    θ = get_gain_threshold_dB(backend)


    initialization = @initialization begin
        q(s) = NormalMeanPrecision(speech_mean, speech_precision)
        q(n) = NormalMeanPrecision(noise_mean, noise_precision)
        q(ξ) = NormalMeanPrecision(0.0, 1.0)
        q(ξ_smooth) = NormalMeanPrecision(ξ_mean, ξ_precision)
        q(ζ_switch) = PointMass(get_vad_auxiliary(backend, band))
        q(ζ_gain) = PointMass(get_gain_auxiliary(backend, band))
        q(π_switch) =
            Categorical([backend.states.vad.p[band], backend.states.vad.q[band]])
        q(w) = Categorical([backend.states.gain.p[band], backend.states.gain.q[band]])

    end

    constraints = @constraints begin
        q(s, n, ξ_smooth, ξ, ζ_switch, ζ_gain, π_switch, w) =
            q(s)q(n)q(ξ_smooth)q(ξ)q(ζ_switch)q(ζ_gain)q(π_switch)q(w)

        for q in random_w_model
            q(x_prior, x) = q(x_prior, x)
        end

    end

    autoupdates = @autoupdates begin
        μs_prior, τs_prior = mean_precision(q(s))
        μn_prior, τn_prior = mean_precision(q(n))
        μξ_mean, τξ_mean = mean_precision(q(ξ_smooth))
    end

    rxengine = infer(
        model = SEM_filtering_model(τs = τs, τn = τn, η = η, κ = κ, θ = θ, τξ = τξ),
        data = (y = data,),
        initialization = initialization,
        keephistory = length(data),
        constraints = constraints,
        autoupdates = autoupdates,
        historyvars = (
            s = KeepLast(),
            n = KeepLast(),
            ξ = KeepLast(),
            ξ_smooth = KeepLast(),
            π_switch = KeepLast(),
            w = KeepLast(),
        ),
        autostart = backend.params.inference.autostart,
        free_energy = backend.params.inference.free_energy,
        iterations = backend.params.inference.iterations,
    )

    return rxengine

end

"""
    wiener_gain_spectral_floor(w_gain, gmin)

Compute Wiener gain with spectral floor constraint.

# Arguments

  - `gains`: Vector of gain values
  - `gmin_lin`: Minimum gain in linear scale

# Returns

  - `T` or `Vector{T}`: Gain value(s) with spectral floor applied (scalar if input is scalar)
"""
function wiener_gain_spectral_floor(
    gains::Vector{T},
    gmin_lin::T,
) where {T <: AbstractFloat}
    return max.(gains, gmin_lin)    # return a vector of the same length as gains with the spectral floor applied
end
