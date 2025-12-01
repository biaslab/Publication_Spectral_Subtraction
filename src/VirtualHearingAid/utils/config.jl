using Dates

"""
    _from_config_impl(::Type{T}, config::AbstractDict) where {T <: AbstractHearingAid}

Internal generic implementation for creating a hearing aid from configuration.

This function handles the common pattern used by all hearing aid types:
1. Validate configuration structure
2. Initialize frontend via `init_wfb_frontend(config)`
3. Initialize backend via `init_backend(backend_type(T), config)`
4. Determine processing strategy (from config or type default)
5. Create hearing aid instance
6. Set processing strategy override (for runtime flexibility)

This is an internal helper function. Specific hearing aid types should implement
`from_config` which calls this helper. This eliminates code duplication across
all hearing aid implementations.

# Arguments
- `T`: The hearing aid type (must be a concrete subtype of `AbstractHearingAid`)
- `config::AbstractDict`: Configuration dictionary

# Returns
- `T`: An initialized hearing aid instance

# Throws
- `ArgumentError`: If configuration validation fails

# See also
- `from_config`: Public API for creating hearing aids from configuration
"""
function _from_config_impl(::Type{T}, config::AbstractDict) where {T<:AbstractHearingAid}
    # Validate config structure
    validate_config(T, config)

    # Initialize components
    frontend = init_wfb_frontend(config)
    backend = init_backend(backend_type(T), config)

    # Determine processing strategy (from config or default)
    strategy =
        haskey(config, "parameters") &&
        haskey(config["parameters"], "hearingaid") &&
        haskey(config["parameters"]["hearingaid"], "processing_strategy") ?
        processing_strategy_from_config(config) : processing_strategy(T)

    # Create and return hearing aid
    ha = T(frontend, backend, strategy)
    set_processing_strategy!(ha, strategy)
    return ha
end

"""
    processing_strategy_from_config(config::AbstractDict)

Extract processing strategy from configuration dictionary.

Parses the `processing_strategy` field from the configuration and returns
the corresponding `ProcessingStrategy` instance.

# Arguments
- `config::AbstractDict`: Configuration dictionary with `parameters.hearingaid.processing_strategy`

# Returns
- `ProcessingStrategy`: The processing strategy instance
  - `StreamProcessing()` for "streaming" or "stream"
  - `BatchProcessingOnline()` for "batchprocessingonline", "online", or "batch_online"
  - `BatchProcessingOffline()` for "batchprocessingoffline", "offline", or "batch_offline"

# Example
```julia
config = Dict("parameters" => Dict("hearingaid" => Dict("processing_strategy" => "offline")))
strategy = processing_strategy_from_config(config)  # Returns BatchProcessingOffline()
```
"""
function processing_strategy_from_config(config::AbstractDict)
    processing_type = config["parameters"]["hearingaid"]["processing_strategy"]
    strat = lowercase(String(processing_type))
    if strat == "streaming" || strat == "stream"
        return StreamProcessing()
    elseif strat == "batchprocessingonline" || strat == "online" || strat == "batch_online"
        return BatchProcessingOnline()
    elseif strat == "batchprocessingoffline" ||
           strat == "offline" ||
           strat == "batch_offline"
        return BatchProcessingOffline()
    end
end

"""
    init_wfb_frontend(config::AbstractDict)

Shared initializer for the WFB (Warped Filter Bank) frontend from configuration.

This function is used by all hearing aid types to initialize the frontend,
eliminating code duplication.

# Arguments
- `config::AbstractDict`: Configuration dictionary containing frontend parameters

# Returns
- `WFBFrontend`: An initialized WFB frontend instance

# Throws
- `ArgumentError`: If required frontend configuration fields are missing

# Required Configuration Fields
- `parameters.frontend.nbands`: Number of frequency bands
- `parameters.frontend.fs`: Sampling frequency (Hz)
- `parameters.frontend.buffer_size_s`: Buffer size in seconds
- `parameters.frontend.spl_reference_db`: SPL reference level in dB
- `parameters.frontend.spl_power_estimate_lower_bound_db`: Lower bound for power estimation
- `parameters.frontend.apcoefficient`: All-pass coefficient

# Example
```julia
config = TOML.parsefile("config.toml")
frontend = init_wfb_frontend(config)
```
"""
function init_wfb_frontend(config::AbstractDict)
    validate_frontend_config(config)

    cfg = config["parameters"]["frontend"]
    buffer_size_s = cfg["buffer_size_s"]
    nbands = cfg["nbands"]
    nfft = (nbands - 1) * 2
    samplerate = cfg["fs"]

    params_frontend = WFBFrontendParameters(
        nbands,
        nfft,
        samplerate,
        cfg["spl_reference_db"],
        cfg["spl_power_estimate_lower_bound_db"],
        cfg["apcoefficient"],
        buffer_size_s * samplerate,
    )

    return WFBFrontend(params_frontend)
end

"""
Get a list of available hearing aid types.

Returns:
- Vector of hearing aid types
"""
function get_hearing_aids()
    aids = [BaselineHearingAid, SEMHearingAid]
    return aids
end

"""
Prepare a configuration dictionary from a TOML configuration.
Formats the TOML config to match the expected structure for from_config.

# Arguments
- `config_dict::AbstractDict`: The parsed TOML configuration

# Returns
- A dictionary in the format expected by from_config
"""
function _prepare_config_from_toml(config_dict::AbstractDict)
    # Extract the hearing aid type
    ha_type_str = config_dict["parameters"]["hearingaid"]["type"]

    # Basic hearing aid configuration is always required
    processed_config = Dict{String,Any}(
        "parameters" => Dict(
            "hearingaid" => Dict(
                "type" => ha_type_str,
                "name" => config_dict["parameters"]["hearingaid"]["name"],
                "processing_strategy" =>
                    config_dict["parameters"]["hearingaid"]["processing_strategy"],
            ),
            "frontend" => config_dict["parameters"]["frontend"],
        ),
    )

    # For SEM, we need to add the backend parameters
    if ha_type_str == "SEMHearingAid"
        # Make sure the backend section exists
        haskey(config_dict["parameters"], "backend") || throw(
            ArgumentError(
                "Missing 'parameters.backend' section required for $(ha_type_str) hearing aid",
            ),
        )

        # Add backend configuration
        processed_config["parameters"]["backend"] = config_dict["parameters"]["backend"]
    end

    # Add metadata if present
    if haskey(config_dict, "metadata")
        processed_config["metadata"] = config_dict["metadata"]
    end

    return processed_config
end

"""
Get default parameters for a WFB (Weighted Filter Bank) frontend in dictionary form.

# Returns
- A dictionary containing all default parameters for a WFB frontend
"""
function default_wfb_frontend_parameters()
    Dict(
        "type" => "WFBFrontend",
        "name" => "WFB",
        "nbands" => 17,
        "fs" => 16000.0,
        "spl_reference_db" => 100.0,
        "spl_power_estimate_lower_bound_db" => 30.0,
        "apcoefficient" => 0.5,
        "buffer_size_s" => 0.0015,
    )
end


"""
Get default parameters for a Baseline hearing aid in dictionary form.

# Returns
- A dictionary containing all default parameters for a Baseline hearing aid
"""
function default_baseline_parameters()
    Dict(
        "parameters" => Dict(
            "hearingaid" => Dict(
                "type" => "BaselineHearingAid",
                "name" => "Baseline Hearing Aid",
                "processing_strategy" => "batch_online",
            ),
            "frontend" => Dict(
                "type" => "WFBFrontend",
                "name" => "WFB",
                "nbands" => 17,
                "fs" => 16000.0,
                "spl_reference_db" => 100.0,
                "spl_power_estimate_lower_bound_db" => 30.0,
                "apcoefficient" => 0.5,
                "buffer_size_s" => 0.0015,
            ),
            "backend" => Dict(
                "general" => Dict("type" => "BaselineBackend", "name" => "Baseline"),
            ),
        ),
        "metadata" => Dict(
            "author" => "VirtualHearingAid",
            "date" => string(today()),
            "description" => "Default Baseline Hearing Aid configuration",
            "name" => "Baseline",
        ),
    )
end

"""
Create a default configuration for a specific hearing aid type.

# Arguments
- `ha_type::Type{T}`: The type of hearing aid to create configuration for
- `kwargs...`: Optional parameters to override defaults

# Returns
- A dictionary containing the default configuration
"""
function create_default_config(ha_type::Type{T}; kwargs...) where {T<:AbstractHearingAid}
    # Create base config with common frontend parameters
    config = Dict(
        "parameters" => Dict(
            "frontend" => Dict(
                "name" => "WFB",
                "type" => "WFBFrontend",
                "nbands" => 17,
                "fs" => 16000.0,
                "spl_reference_db" => 100.0,
                "spl_power_estimate_lower_bound_db" => 30.0,
                "apcoefficient" => 0.5,
                "buffer_size_s" => 0.0015,
            ),
        ),
    )

    # Add type-specific defaults
    if ha_type == BaselineHearingAid
        config["parameters"]["hearingaid"] = Dict(
            "name" => "Baseline Hearing Aid",
            "type" => "BaselineHearingAid",
            "processing_strategy" => "streaming",
        )
        config["parameters"]["backend"] =
            Dict("general" => Dict("name" => "Baseline", "type" => "BaselineBackend"))
        config["metadata"] = Dict(
            "author" => "Marco Hidalgo-Araya",
            "date" => "2025-04-03",
            "description" => "Baseline Hearing Aid configuration",
            "name" => "Baseline",
        )
    elseif ha_type == SEMHearingAid
        config["parameters"]["hearingaid"] = Dict(
            "name" => "SEM Hearing Aid",
            "type" => "SEMHearingAid",
            "processing_strategy" => "BatchProcessingOffline",
        )
        config["parameters"]["backend"] = Dict(
            "general" => Dict("name" => "SEM", "type" => "SEMBackend"),
            "inference" =>
                Dict("autostart" => true, "free_energy" => false, "iterations" => 2),
            "filters" => Dict("time_constants90" => Dict(
                "s" => 5.0,     # Speech time constant (ms)
                "n" => 10000.0, # Noise time constant (ms)
            )),
            "priors" => Dict(
                "speech" => Dict("mean" => 80.0, "precision" => 1.0),
                "noise" => Dict("mean" => 70.0, "precision" => 1.0),
            ),
            "gain" => Dict(
                "slope" => 1.0,
                "threshold" => 12.0, # (GMIN)
            ),
            "switch" => Dict("slope" => 8.0, "threshold" => 4.0),
        )
        config["metadata"] = Dict(
            "author" => "VirtualHearingAid",
            "date" => "2025-01-27",
            "description" => "SEM Hearing Aid configuration",
            "name" => "SEM",
        )
    else
        throw(ArgumentError("Unknown hearing aid type: $ha_type"))
    end

    # Apply any overrides from kwargs
    for (k, v) in kwargs
        # Convert dot notation to nested dict
        keys = split(string(k), ".")
        d = config
        for key in keys[1:(end-1)]
            d = d[key]
        end
        d[keys[end]] = v
    end

    # Validate the configuration
    validate_config(ha_type, config)

    return config
end
