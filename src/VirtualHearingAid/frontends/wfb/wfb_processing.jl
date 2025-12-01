"""
    process_frontend(frontend::WFBFrontend, block::SampleBuf{Float64, 1})

Process an audio block through the WFB frontend.
Returns the power spectrum in decibels.
"""
function process_frontend(frontend::WFBFrontend, block::SampleBuf)
    # Get parameters
    params = WFB.get_params(frontend)
    for sample in block.data
        push!(frontend.sample_buffer, sample)
    end

    allpass_filter!(frontend)
    power = compute_power(frontend)
    return convert_to_db(frontend, power)
end

# Helpers
function calculate_window(nfft)
    window_temp = hanning(nfft)
    window = window_temp ./ maximum(window_temp)
    return window
end

function calculate_input_calibration_db(
    window::Vector{Float64},
    nbands::Int64,
    spl_reference_db::Float64,
)
    central_band_power = 0.25 * sum(window)^2
    input_calibration_db_offset = spl_reference_db - 10 * log10(central_band_power)
    input_calibration_db = fill(input_calibration_db_offset, 1, nbands)
    input_calibration_db[[1, nbands]] .-= EDGE_BAND_ADJUSTMENT_DB
    return input_calibration_db
end

function calculate_center_frequencies(
    apcoefficient::AbstractFloat,
    nfft::Int64,
    fs::Float64,
)
    f = (0:(nfft/2))' * (2 * π / nfft)
    freq =
        atan.(
            (1 - apcoefficient^2) * sin.(f),
            (1 + apcoefficient^2) * cos.(f) .+ 2 * apcoefficient,
        ) * (fs / 2) / π
    return vec(freq)
end

calculate_center_frequencies(frontend::WFBFrontend) = calculate_center_frequencies(
    get_apcoefficient(frontend),
    get_nfft(frontend),
    get_fs(frontend),
)

function calculate_synthesis_matrix(nbands::Int64, nfft::Int64)
    nhalf = Int64(nfft / 2)
    synthesis_matrix = real(DSP.fft(Matrix{Float64}(I, nfft, nfft), 1)) / nfft
    dmatrix = Matrix{Float64}(I, nbands, nbands)
    extender_matrix = [
        dmatrix
        zeros(nbands - 2, 1) reverse(dmatrix[1:(nbands-2), 1:(nbands-2)], dims = 2) zeros(nbands - 2, 1)
    ]
    synthesis_matrix = synthesis_matrix * extender_matrix
    synthesis_matrix = synthesis_matrix[1:nhalf, :]
    synthesis_matrix =
        reverse(Matrix{Float64}(I, nhalf, nhalf), dims = 2) * synthesis_matrix
    synthesis_window = hanning(nfft - 1)
    synthesis_matrix = Diagonal(synthesis_window[1:nhalf]) * synthesis_matrix
    return synthesis_matrix
end

function db2lin(gains::Vector{Float64})
    return 10.0 .^ (gains / SCALE_FACTOR_DB)
end

function update_weights!(frontend::WFBFrontend, gains::Vector{Float64})
    weights_temp = get_synthesis_matrix(frontend) * gains
    weights = vcat(weights_temp, 0, weights_temp[(end-1):-1:1])
    WFB.set_weights!(frontend, weights)
end



"""
    allpass_filter!(frontend::WFBFrontend)

Apply allpass filtering to the samples in the frontend's buffer.
This warps the frequency scale to approximate human hearing sensitivity.
"""
function allpass_filter!(frontend::WFBFrontend)
    buffer_size, nfft = size(get_taps(frontend))
    ap_coeff = get_apcoefficient(frontend)

    for n = 1:buffer_size
        # Set first tap directly from sample buffer
        frontend.taps[n, 1] = get_sample_buffer(frontend)[n]

        # Process remaining taps
        for tap = 2:nfft
            prev_tap = tap - 1
            # Apply allpass filter equation
            frontend.taps[n, tap] =
                get_temp(frontend)[tap] - ap_coeff * get_taps(frontend)[n, prev_tap]
            frontend.temp[tap] =
                ap_coeff * get_taps(frontend)[n, tap] + get_taps(frontend)[n, prev_tap]
        end
    end
end

function compute_power(frontend::WFBFrontend)
    nbands = get_nbands(frontend)

    # Apply window to taps
    windowed_taps = get_window(frontend) .* get_taps(frontend)[end, :]

    # Transform to frequency domain
    fft_taps = DSP.fft(windowed_taps')[1:nbands]

    # Calculate power spectrum
    power = real(fft_taps .* conj(fft_taps))

    # Double power for all but DC and Nyquist components
    power[2:(nbands-1)] = 2 * power[2:(nbands-1)]

    return power
end

function convert_to_db(frontend::WFBFrontend, power::Vector{Float64})
    power_db = 10 .* log10.(power) .+ vec(get_calibration_db(frontend))
    return power_db
end

function synthesize(frontend::WFBFrontend)
    synthesized =
        last(get_taps(frontend) * get_weights(frontend), get_params(frontend).buffer_size)

    return SampleBuf(synthesized, get_params(frontend).fs)
end
