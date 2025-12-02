"""
    from_config(::Type{SEMHearingAid}, config::AbstractDict)

Create a SEMHearingAid from a configuration dictionary.

Uses the generic `_from_config_impl` helper from `utils/config.jl`.

# Arguments
- `config::AbstractDict`: Configuration dictionary containing hearing aid parameters

# Returns
- `SEMHearingAid`: A configured SEM hearing aid instance

# Example
```julia
config = TOML.parsefile("config.toml")
ha = from_config(SEMHearingAid, config)
```
"""
function from_config(::Type{SEMHearingAid}, config::AbstractDict)
    return _from_config_impl(SEMHearingAid, config)
end

"""
    init_backend(::Type{SEMBackend}, config::AbstractDict)

Initialize SEM backend from configuration.

Creates a SEMBackend with parameters extracted from the configuration dictionary.
SEM requires speech, noise, and ξ (xnr) parameters; context and VAI fields are ignored
for backward compatibility.

# Arguments
- `config::AbstractDict`: Configuration dictionary containing backend parameters

# Returns
- `SEMBackend`: A configured SEM backend instance

# Throws
- `ArgumentError`: If required configuration fields are missing or invalid

# Note
The configuration may include optional fields (context, vai) for backward compatibility
with old BPNR configs, but these are not used by SEM.
"""
function init_backend(::Type{SEMBackend}, config::AbstractDict)
    validate_sem_config(config)

    T = Float64

    buffer_size_s = config["parameters"]["frontend"]["buffer_size_s"]
    fs_algorithm = 1.0 / (buffer_size_s * 1000)
    nbands = config["parameters"]["frontend"]["nbands"]

    iterations = config["parameters"]["backend"]["inference"]["iterations"]
    autostart = config["parameters"]["backend"]["inference"]["autostart"]
    free_energy = config["parameters"]["backend"]["inference"]["free_energy"]

    tc_config = config["parameters"]["backend"]["filters"]["time_constants90"]

    function get_first_value(value)
        if value isa Number
            return T(value)
        elseif value isa Vector
            return T(value[1])  # Use first value for SEM (single source parameter)
        else
            throw(ArgumentError("Time constant must be either a number or an array"))
        end
    end

    τs = get_first_value(tc_config["s"])
    τn = get_first_value(tc_config["n"])
    τξ = get_first_value(tc_config["xnr"])

    priors_config = config["parameters"]["backend"]["priors"]
    speech_mean = T(priors_config["speech"]["mean"])
    speech_precision = T(priors_config["speech"]["precision"])
    noise_mean = T(priors_config["noise"]["mean"])
    noise_precision = T(priors_config["noise"]["precision"])
    # ξ does not require prior mean/precision as it only has transition state

    # Gain slope is no longer used, use default value
    gain_slope = T(1.0)
    gain_threshold =
        haskey(config["parameters"]["backend"], "gain") &&
        haskey(config["parameters"]["backend"]["gain"], "threshold") ?
        T(config["parameters"]["backend"]["gain"]["threshold"]) : T(12.0)
    # VAD slope is no longer used (η removed), use default value
    switch_slope = T(1.0)
    switch_threshold =
        haskey(config["parameters"]["backend"], "switch") &&
        haskey(config["parameters"]["backend"]["switch"], "threshold") ?
        T(config["parameters"]["backend"]["switch"]["threshold"]) : T(0.0)

    speech_params = SourceParameters{T}(τs, fs_algorithm)
    noise_params = SourceParameters{T}(τn, fs_algorithm)
    ξ_params = SourceParameters{T}(τξ, fs_algorithm)

    # Note: gain_threshold should be in dB, and GainParameters will convert it
    # Use gain.threshold (defaults to 12.0 dB if not specified)
    gain_params = HASoundProcessing.SEM.GainParameters{T}(gain_slope, gain_threshold)
    # VAD parameters (switch is now called VAD in the new API)
    vad_params = HASoundProcessing.SEM.VADParameters{T}(switch_slope, switch_threshold)

    # Create module parameters
    module_params = HASoundProcessing.SEM.ModuleParameters{T}(
        speech_params,
        noise_params,
        ξ_params,
        gain_params,
        vad_params,
        fs_algorithm,
        nbands,
    )

    # Create inference parameters
    inference_params =
        HASoundProcessing.SEM.InferenceParameters(iterations, autostart, free_energy)

    # Create SEM parameters with new structure
    params = SEMParameters{T}(inference_params, module_params)

    backend = SEMBackend(
        params,
        speech_mean,
        speech_precision,
        noise_mean,
        noise_precision,
    )
    return backend
end
