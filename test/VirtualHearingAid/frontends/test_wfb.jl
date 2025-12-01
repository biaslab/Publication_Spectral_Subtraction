@testitem "WFB Frontend" begin

    using Test, VirtualHearingAid
    using Experiments.VirtualHearingAid.Frontends.WFB
    using Experiments.VirtualHearingAid.Frontends.WFB:
        WFBFrontend,
        WFBFrontendParameters,
        process_frontend,
        allpass_filter!,
        compute_power,
        convert_to_db,
        calculate_window,
        calculate_input_calibration_db,
        calculate_center_frequencies,
        calculate_synthesis_matrix,
        db2lin,
        update_weights!,
        get_params,
        get_weights,
        get_taps,
        get_window,
        get_calibration_db,
        get_synthesis_matrix,
        get_nbands,
        get_nfft,
        get_fs,
        get_spl_reference_db,
        get_spl_power_estimate_lower_bound_db,
        get_apcoefficient,
        get_buffer_size,
        get_temp,
        get_sample_buffer,
        set_taps!
    using SampledSignals
    using DataStructures: CircularBuffer
    using DSP
    using LinearAlgebra

    # Define constants at module level
    const EDGE_BAND_ADJUSTMENT_DB = 3.0103
    const SCALE_FACTOR_DB = 20.0
    # Default test parameters
    nbands = 17
    nfft = 32
    fs = 44100.0
    spl_reference_db = 100.0
    spl_power_estimate_lower_bound_db = -30.0
    apcoefficient = 0.9
    buffer_size = 441

    # Create default frontend for tests
    function create_test_frontend()
        params = WFBFrontendParameters(
            nbands,
            nfft,
            fs,
            spl_reference_db,
            spl_power_estimate_lower_bound_db,
            apcoefficient,
            buffer_size,
        )
        return WFBFrontend(params)
    end

    @testset "Constructor" begin
        frontend = create_test_frontend()

        # Check that the frontend is correctly initialized
        @test frontend isa WFBFrontend
        @test get_params(frontend) isa WFBFrontendParameters
        @test get_nbands(frontend) == nbands
        @test get_nfft(frontend) == nfft
        @test get_fs(frontend) == fs
        @test get_apcoefficient(frontend) == apcoefficient
        @test get_buffer_size(frontend) == buffer_size

        # Check that the internal structures are correctly initialized
        # The constructor actually creates weights of length nbands*2-1
        @test length(get_weights(frontend)) == nfft
        @test size(get_taps(frontend)) == (buffer_size, nfft)
        @test length(get_window(frontend)) == nfft
        @test size(get_synthesis_matrix(frontend)) == (div(nfft, 2), nbands)
        @test size(get_calibration_db(frontend)) == (1, nbands)
    end

    @testset "calculate_window" begin
        window = calculate_window(nfft)

        # The window should be normalized to a maximum of 1.0
        @test maximum(window) ≈ 1.0

        # Should be a hanning window of the specified length
        @test length(window) == nfft

        # The window should be symmetric
        @test window[1:div(nfft, 2)] ≈ window[end:-1:(div(nfft, 2)+1)]
    end

    @testset "calculate_input_calibration_db" begin
        window = calculate_window(nfft)
        calibration = calculate_input_calibration_db(window, nbands, spl_reference_db)

        # Check dimensions
        @test size(calibration) == (1, nbands)

        # Edge bands should have less power than central bands due to adjustment
        @test calibration[1, 1] ≈ calibration[1, 2] - EDGE_BAND_ADJUSTMENT_DB atol = 1e-4
        @test calibration[1, nbands] ≈ calibration[1, nbands-1] - EDGE_BAND_ADJUSTMENT_DB atol =
            1e-4

        # Central bands should all have the same value
        @test all(x -> x ≈ calibration[1, 2], calibration[1, 2:(nbands-1)])
    end

    @testset "calculate_center_frequencies" begin
        frontend = create_test_frontend()

        # Get center frequencies
        center_freqs = calculate_center_frequencies(frontend)

        # Should have correct number of frequencies
        @test length(center_freqs) == div(nfft, 2) + 1

        # Frequencies should be in ascending order
        @test issorted(center_freqs)
    end

    @testset "calculate_synthesis_matrix" begin
        synthesis_matrix = calculate_synthesis_matrix(nbands, nfft)

        # Check dimensions
        @test size(synthesis_matrix) == (div(nfft, 2), nbands)

        # The synthesis matrix should be real
        @test all(x -> isreal(x), synthesis_matrix)
    end

    @testset "db2lin" begin
        # Test conversion from dB to linear scale
        db_values = [0.0, -6.0, -12.0, -18.0]
        expected_lin = [1.0, 0.5011872336272722, 0.251188643150958, 0.12589254117941673]
        lin_values = db2lin(db_values)

        # Check that the conversion is correct
        @test all(lin_values .≈ expected_lin)
    end

    @testset "update_weights!" begin
        frontend = create_test_frontend()

        # Create some test gains
        gains = zeros(nbands)
        gains[div(nbands, 2)] = 1.0  # Set middle band to 1.0

        # Update weights
        update_weights!(frontend, gains)

        # Check that weights were updated
        @test !all(get_weights(frontend) .== 0)
    end

    @testset "allpass_filter!" begin
        frontend = create_test_frontend()

        # Fill the sample buffer with a simple sine wave
        t = 0:(1/fs):(buffer_size/fs-1/fs)
        sine_wave = sin.(2π * 440 * t)
        for sample in sine_wave
            push!(frontend.sample_buffer, sample)
        end

        # Apply allpass filter
        allpass_filter!(frontend)

        # Check that taps were updated
        @test !all(get_taps(frontend) .== 0)
    end

    @testset "compute_power" begin
        frontend = create_test_frontend()

        # Fill the sample buffer with a simple sine wave
        t = 0:(1/fs):(buffer_size/fs-1/fs)
        sine_wave = sin.(2π * 440 * t)
        for sample in sine_wave
            push!(frontend.sample_buffer, sample)
        end

        # Apply allpass filter
        allpass_filter!(frontend)

        # Compute power
        power = compute_power(frontend)

        # Check that power is computed correctly
        @test length(power) == div(nfft, 2) + 1
        @test all(power .>= 0)  # Power should be non-negative
    end

    @testset "convert_to_db" begin
        frontend = create_test_frontend()

        # Create some test power values
        power = ones(div(nfft, 2) + 1)

        # Convert to dB
        power_db = convert_to_db(frontend, power)

        # Check that conversion is correct
        @test length(power_db) == div(nfft, 2) + 1
        @test all(power_db .>= get_spl_power_estimate_lower_bound_db(frontend))
    end

    @testset "process_frontend" begin
        frontend = create_test_frontend()

        # Create a test audio block
        t = 0:(1/fs):(buffer_size/fs-1/fs)
        sine_wave = sin.(2π * 440 * t)
        block = SampleBuf(sine_wave, fs)

        # Process the block
        power_db = process_frontend(frontend, block)

        # Check that processing is correct
        @test length(power_db) == nbands
        @test all(power_db .>= get_spl_power_estimate_lower_bound_db(frontend))
    end

    @testset "synthesize" begin
        frontend = create_test_frontend()

        # Fill the sample buffer with a simple sine wave
        t = 0:(1/fs):(buffer_size/fs-1/fs)
        sine_wave = sin.(2π * 440 * t)
        for sample in sine_wave
            push!(frontend.sample_buffer, sample)
        end

        # Apply allpass filter
        allpass_filter!(frontend)

        # Create some test gains
        gains = zeros(nbands)
        gains[div(nbands, 2)] = 1.0  # Set middle band to 1.0

        # Synthesize
        update_weights!(frontend, gains)
        synthesized = synthesize(frontend)

        # Check that synthesis is correct
        @test length(synthesized) == buffer_size
    end
end
