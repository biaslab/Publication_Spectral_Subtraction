"""
Validate configuration based on hearing aid type.
"""
function validate_config(::Type{T}, config::AbstractDict) where {T<:AbstractHearingAid}
    validate_schema(config)

    haskey(config["parameters"], "hearingaid") ||
        throw(ArgumentError("Missing 'hearing_aid' section"))

    if backend_type(T) == BaselineBackend
        validate_frontend_config(config)
    elseif backend_type(T) == SEMBackend
        validate_frontend_config(config)
        validate_sem_config(config)
    else
        throw(ArgumentError("Unknown backend type for hearing aid type $T"))
    end

    return true
end

"""
Type-specific validation for Baseline.
"""
function _validate_config(::BaselineBackend, config::AbstractDict)
    validate_frontend_config(config)
end

"""
Validate frontend configuration.
"""
function validate_frontend_config(config::AbstractDict)
    required_fields = [
        "buffer_size_s",
        "nbands",
        "fs",
        "spl_reference_db",
        "spl_power_estimate_lower_bound_db",
        "apcoefficient",
    ]

    for field in required_fields
        haskey(config["parameters"]["frontend"], field) ||
            throw(ArgumentError("Missing required field '$field' in hearing_aid config"))
    end
end


"""
Validate filter configuration.
"""
function validate_filter_config(config::AbstractDict)
    haskey(config["parameters"]["backend"], "filters") ||
        throw(ArgumentError("Missing 'filters' in backend parameters"))

    required_filters = ["speech", "noise", "context", "switch"]
    for filter_name in required_filters
        haskey(config["parameters"]["backend"]["filters"], filter_name) ||
            throw(ArgumentError("Missing required filter '$filter_name' in filters"))

        filter_config = config["parameters"]["backend"]["filters"][filter_name]
        haskey(filter_config, "tau") ||
            throw(ArgumentError("Missing 'tau' for filter '$filter_name'"))
        haskey(filter_config["tau"], "s") ||
            throw(ArgumentError("Missing 's' parameter in tau for filter '$filter_name'"))
        haskey(filter_config["tau"], "n") ||
            throw(ArgumentError("Missing 'n' parameter in tau for filter '$filter_name'"))
    end
end

"""
Validate time constant parameters to ensure they are either single values or arrays of size nbands.
"""
function validate_time_constants(
    tc_config::AbstractDict,
    nbands::Integer,
    required_fields::Vector{String},
)
    for field in required_fields
        haskey(tc_config, field) ||
            throw(ArgumentError("Missing required field '$field' in time_constants90"))

        value = tc_config[field]

        if value isa Number
            continue
        elseif value isa Vector
            if length(value) != nbands
                throw(
                    ArgumentError(
                        "Time constant '$field' array must have exactly $nbands elements (got $(length(value)))",
                    ),
                )
            end

            for (i, elem) in enumerate(value)
                if !(elem isa Number)
                    throw(
                        ArgumentError(
                            "Time constant '$field' array element $i must be a number (got $(typeof(elem)))",
                        ),
                    )
                end
            end
        else
            throw(
                ArgumentError(
                    "Time constant '$field' must be either a number or an array of numbers",
                ),
            )
        end
    end
end

"""
Validate SEM backend configuration.

SEM (Spectral Enhancement Model) only requires speech and noise parameters.
Context and VAI fields in config are ignored for backward compatibility but not required.
"""
function validate_sem_config(config::AbstractDict)
    haskey(config, "parameters") || throw(ArgumentError("Missing 'parameters' section"))
    haskey(config["parameters"], "backend") ||
        throw(ArgumentError("Missing 'backend' in parameters"))

    haskey(config["parameters"]["backend"], "general") ||
        throw(ArgumentError("Missing 'general' in backend parameters"))

    required_general_fields = ["name", "type"]
    for field in required_general_fields
        haskey(config["parameters"]["backend"]["general"], field) || throw(
            ArgumentError("Missing required field '$field' in backend general parameters"),
        )
    end

    haskey(config["parameters"]["backend"], "inference") ||
        throw(ArgumentError("Missing 'inference' in backend parameters"))

    required_inference_fields = ["autostart", "free_energy", "iterations"]
    for field in required_inference_fields
        haskey(config["parameters"]["backend"]["inference"], field) || throw(
            ArgumentError(
                "Missing required field '$field' in backend inference parameters",
            ),
        )
    end

    # Validate inference parameter types
    inference_config = config["parameters"]["backend"]["inference"]
    if !(inference_config["autostart"] isa Bool)
        throw(
            ArgumentError(
                "autostart must be a Bool, got $(typeof(inference_config["autostart"]))",
            ),
        )
    end
    if !(inference_config["free_energy"] isa Bool)
        throw(
            ArgumentError(
                "free_energy must be a Bool, got $(typeof(inference_config["free_energy"]))",
            ),
        )
    end
    if !(inference_config["iterations"] isa Integer)
        throw(
            ArgumentError(
                "iterations must be an Integer, got $(typeof(inference_config["iterations"]))",
            ),
        )
    end

    haskey(config["parameters"]["backend"], "filters") ||
        throw(ArgumentError("Missing 'filters' in backend parameters"))

    haskey(config["parameters"]["backend"]["filters"], "time_constants90") ||
        throw(ArgumentError("Missing 'time_constants90' in filters"))

    nbands = config["parameters"]["frontend"]["nbands"]
    tc_config = config["parameters"]["backend"]["filters"]["time_constants90"]

    # SEM requires speech (s), noise (n), and ξ (xnr) time constants
    required_tc_fields = ["s", "n", "xnr"]
    validate_time_constants(tc_config, nbands, required_tc_fields)

    # Validate time constant values: must be floats, > 0, and s < n
    function validate_tc_value(value, name)
        if value isa Number
            if !(value isa AbstractFloat)
                throw(
                    ArgumentError(
                        "Time constant '$name' must be a Float, got $(typeof(value))",
                    ),
                )
            end
            if value <= 0
                throw(
                    ArgumentError(
                        "Time constant '$name'=$value: the time constant needs to be greater than zero",
                    ),
                )
            end
            return value
        elseif value isa Vector
            for (i, elem) in enumerate(value)
                if !(elem isa AbstractFloat)
                    throw(
                        ArgumentError(
                            "Time constant '$name'[$i] must be a Float, got $(typeof(elem))",
                        ),
                    )
                end
                if elem <= 0
                    throw(
                        ArgumentError(
                            "Time constant '$name'[$i]=$elem: the time constant needs to be greater than zero",
                        ),
                    )
                end
            end
            return value
        else
            throw(
                ArgumentError(
                    "Time constant '$name' must be either a Float or an array of Floats",
                ),
            )
        end
    end

    s_value = validate_tc_value(tc_config["s"], "s")
    n_value = validate_tc_value(tc_config["n"], "n")

    # Validate s < n (s tracks speech envelope, must be faster than noise envelope)
    function compare_tc_values(s_val, n_val, s_name, n_name)
        if s_val isa Number && n_val isa Number
            if s_val >= n_val
                throw(
                    ArgumentError(
                        "s=$s_val >= n=$n_val: s tracks the speech envelope so it must be faster (smaller) than the noise envelope that tracks the background noise",
                    ),
                )
            end
        elseif s_val isa Vector && n_val isa Vector
            for i = 1:length(s_val)
                if s_val[i] >= n_val[i]
                    throw(
                        ArgumentError(
                            "s[$i]=$(s_val[i]) >= n[$i]=$(n_val[i]): s tracks the speech envelope so it must be faster (smaller) than the noise envelope that tracks the background noise",
                        ),
                    )
                end
            end
        elseif s_val isa Number && n_val isa Vector
            for i = 1:length(n_val)
                if s_val >= n_val[i]
                    throw(
                        ArgumentError(
                            "s=$s_val >= n[$i]=$(n_val[i]): s tracks the speech envelope so it must be faster (smaller) than the noise envelope that tracks the background noise",
                        ),
                    )
                end
            end
        elseif s_val isa Vector && n_val isa Number
            for i = 1:length(s_val)
                if s_val[i] >= n_val
                    throw(
                        ArgumentError(
                            "s[$i]=$(s_val[i]) >= n=$n_val: s tracks the speech envelope so it must be faster (smaller) than the noise envelope that tracks the background noise",
                        ),
                    )
                end
            end
        end
    end

    compare_tc_values(s_value, n_value, "s", "n")

    # Validate xnr time constant
    xnr_value = validate_tc_value(tc_config["xnr"], "xnr")

    haskey(config["parameters"]["backend"], "priors") ||
        throw(ArgumentError("Missing 'priors' in backend parameters"))

    # SEM requires speech and noise priors (ξ does not need priors as it only has transition state)
    required_priors = ["speech", "noise"]
    for prior_name in required_priors
        haskey(config["parameters"]["backend"]["priors"], prior_name) ||
            throw(ArgumentError("Missing required prior '$prior_name' in priors"))

        prior_config = config["parameters"]["backend"]["priors"][prior_name]
        haskey(prior_config, "mean") ||
            throw(ArgumentError("Missing 'mean' for prior '$prior_name'"))
        haskey(prior_config, "precision") ||
            throw(ArgumentError("Missing 'precision' for prior '$prior_name'"))

        # Validate mean is a Float
        if !(prior_config["mean"] isa AbstractFloat)
            throw(
                ArgumentError(
                    "Prior '$prior_name' mean must be a Float, got $(typeof(prior_config["mean"]))",
                ),
            )
        end

        # Validate precision is a Float and > 0
        if !(prior_config["precision"] isa AbstractFloat)
            throw(
                ArgumentError(
                    "Prior '$prior_name' precision must be a Float, got $(typeof(prior_config["precision"]))",
                ),
            )
        end
        if prior_config["precision"] <= 0
            throw(
                ArgumentError(
                    "Prior '$prior_name' precision=$(prior_config["precision"]) must be greater than zero (precision is for Gaussian parametrization and cannot be negative)",
                ),
            )
        end
    end

    # Mandatory: gain and switch parameters (for VAD)
    haskey(config["parameters"]["backend"], "gain") ||
        throw(ArgumentError("Missing 'gain' in backend parameters"))

    gain_config = config["parameters"]["backend"]["gain"]
    haskey(gain_config, "threshold") ||
        throw(ArgumentError("Missing 'threshold' in gain parameters"))

    # Validate gain threshold is a Float
    if !(gain_config["threshold"] isa AbstractFloat)
        throw(
            ArgumentError(
                "Gain threshold must be a Float, got $(typeof(gain_config["threshold"]))",
            ),
        )
    end

    haskey(config["parameters"]["backend"], "switch") ||
        throw(ArgumentError("Missing 'switch' in backend parameters"))

    switch_config = config["parameters"]["backend"]["switch"]
    haskey(switch_config, "threshold") ||
        throw(ArgumentError("Missing 'threshold' in switch parameters"))

    # Validate switch threshold is a Float
    if !(switch_config["threshold"] isa AbstractFloat)
        throw(
            ArgumentError(
                "Switch threshold must be a Float, got $(typeof(switch_config["threshold"]))",
            ),
        )
    end
end

"""
Validate configuration schema against expected structure.
"""
function validate_schema(config::AbstractDict)
    haskey(config, "parameters") || throw(ArgumentError("Missing 'parameters' section"))
    haskey(config["parameters"], "hearingaid") ||
        throw(ArgumentError("Missing 'hearingaid' section"))
    haskey(config["parameters"], "frontend") ||
        throw(ArgumentError("Missing 'frontend' section"))

    ha_config = config["parameters"]["hearingaid"]
    haskey(ha_config, "type") ||
        throw(ArgumentError("Missing 'type' in hearingaid section"))
    haskey(ha_config, "name") ||
        throw(ArgumentError("Missing 'name' in hearingaid section"))

    fe_config = config["parameters"]["frontend"]
    required_frontend_fields = [
        "type",
        "name",
        "nbands",
        "fs",
        "spl_reference_db",
        "spl_power_estimate_lower_bound_db",
        "apcoefficient",
        "buffer_size_s",
    ]
    for field in required_frontend_fields
        haskey(fe_config, field) ||
            throw(ArgumentError("Missing required field '$field' in frontend section"))
    end

    return true
end

"""
    validate_batch_processing_inputs(sound::SampleBuf, ha::AbstractHearingAid)

Validate inputs for batch processing operations.

# Arguments
- `sound`: Input audio signal
- `ha`: Hearing aid instance

# Throws
- `ArgumentError`: If inputs are invalid
"""
function validate_batch_processing_inputs(sound::SampleBuf, ha::AbstractHearingAid)
    if isempty(sound)
        throw(ArgumentError("Input sound cannot be empty"))
    end

    if length(sound) < get_params(get_frontend(ha)).buffer_size
        throw(
            ArgumentError(
                "Input sound length ($(length(sound))) must be at least buffer_size ($(get_params(get_frontend(ha)).buffer_size))",
            ),
        )
    end

    if isnothing(get_frontend(ha))
        throw(ArgumentError("Hearing aid must have a valid frontend"))
    end

    expected_samplerate = get_params(get_frontend(ha)).fs
    if sound.samplerate != expected_samplerate
        throw(
            ArgumentError(
                "Sound samplerate ($(sound.samplerate)) must match frontend samplerate ($(expected_samplerate))",
            ),
        )
    end
end

"""
    validate_synthesis_inputs(taps_history::Vector{Matrix{Float64}}, gains::Union{Matrix{Float64}, Nothing}, buffer_size::Int)

Validate inputs for synthesis operations.

# Arguments
- `taps_history`: History of taps matrices
- `gains`: Gain matrix or nothing
- `buffer_size`: Expected buffer size

# Throws
- `ArgumentError`: If inputs are invalid
"""
function validate_synthesis_inputs(
    taps_history::Vector{Matrix{Float64}},
    gains::Union{Matrix{Float64},Nothing},
    buffer_size::Int,
)
    if isempty(taps_history)
        throw(ArgumentError("Taps history cannot be empty"))
    end

    if buffer_size <= 0
        throw(ArgumentError("Buffer size must be positive, got $buffer_size"))
    end

    if !isnothing(gains)
        if size(gains, 1) != length(taps_history)
            throw(
                ArgumentError(
                    "Gains matrix rows ($(size(gains, 1))) must match number of blocks ($(length(taps_history)))",
                ),
            )
        end

        # Validate all gains are between 0 and 1
        if any(x -> x < 0 || x > 1, gains)
            min_gain = minimum(gains)
            max_gain = maximum(gains)
            throw(
                ArgumentError(
                    "All gains must be between 0 and 1, got range [$min_gain, $max_gain]",
                ),
            )
        end
    end
end

"""
    validate_backend_processing_inputs(powerdb_matrix::AbstractMatrix{<:Real}, backend)

Validate inputs for backend processing operations.

# Arguments
- `powerdb_matrix`: Power matrix in dB
- `backend`: Backend instance

# Throws
- `ArgumentError`: If inputs are invalid
"""
function validate_backend_processing_inputs(powerdb_matrix::AbstractMatrix{<:Real}, backend)
    if isempty(powerdb_matrix)
        throw(ArgumentError("Power matrix cannot be empty"))
    end

    if any(isnan, powerdb_matrix) || any(isinf, powerdb_matrix)
        throw(ArgumentError("Power matrix contains NaN or Inf values"))
    end

    if isnothing(backend)
        throw(ArgumentError("Backend cannot be nothing"))
    end
end
