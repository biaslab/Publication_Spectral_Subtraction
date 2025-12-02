@testitem "SEM Hearing Aid Configuration Tests" begin
    using VirtualHearingAid
    using SampledSignals
    using TOML
    using HASoundProcessing
    using DataStructures
    using Experiments.VirtualHearingAid.Frontends.WFB:
        process_frontend, synthesize, update_weights!, get_nbands
    import VirtualHearingAid:
        from_config,
        process,
        get_frontend,
        getname,
        SEMHearingAid,
        SEMBackend,
        SEMResults,
        WFBFrontend

    # Test constructor
    @testset "Constructor" begin
        # Create from example configuration
        config = TOML.parsefile(joinpath(@__DIR__, "example_SEM_config.toml"))
        ha = from_config(SEMHearingAid, config)

        @test ha isa SEMHearingAid
        @test get_frontend(ha) isa WFBFrontend
        @test getname(ha) == "SEMHearingAid"
        @test ha.backend isa SEMBackend
    end

    # Test processing
    @testset "Processing" begin

        config = TOML.parsefile(joinpath(@__DIR__, "example_SEM_config.toml"))
        ha = from_config(SEMHearingAid, config)

        # Create test input
        t = 0:(1/16000):(4410/16000-1/16000)
        sine_wave = sin.(2π .* 1000.0 .* t)
        test_input = SampleBuf(sine_wave, 16000)

        # Process
        result, results = process(ha, test_input)

        @test result isa SampleBuf
        @test abs(length(result) - length(test_input)) <= 10  # Allow small differences due to block processing
        @test results isa SEMResults

        # Test that Wiener gains matrix has correct dimensions
        nbands = get_nbands(get_frontend(ha))
        num_blocks = cld(length(test_input), get_params(get_frontend(ha)).buffer_size)
        @test size(results.gains) == (num_blocks, nbands)
        threshold_dB_ = config["parameters"]["backend"]["gain"]["threshold"]
        threshold_lin = 10^(-threshold_dB_ / 20.0)
        @test all(results.gains .>= threshold_lin) &&
              all(results.gains .<= 1.0)  # Wiener gains should be between SPECTRAL FLOOR and 1

        # Test frontend weights
        weights = VirtualHearingAid.Frontends.WFB.get_weights(ha.frontend)
        @test length(weights) == 32  # nfft = (nbands - 1) * 2 = 32
    end

    # Test configuration structure
    @testset "Configuration Structure" begin
        config = TOML.parsefile(joinpath(@__DIR__, "example_SEM_config.toml"))
        ha = from_config(SEMHearingAid, config)

        @test ha isa SEMHearingAid
        @test ha.backend isa SEMBackend

        @test ha.backend.params isa Experiments.HASoundProcessing.SEMParameters
        @test ha.backend.params.modules.speech isa HASoundProcessing.SourceParameters
        @test ha.backend.params.modules.noise isa HASoundProcessing.SourceParameters
    end

    # Test comprehensive structure matching config file
    @testset "Structure matches config file" begin
        config = TOML.parsefile(joinpath(@__DIR__, "example_SEM_config.toml"))
        ha = from_config(SEMHearingAid, config)

        # Test frontend structure (ha.frontend.*)
        @testset "Frontend structure" begin
            @test hasfield(typeof(ha.frontend), :calibration_db)
            @test hasfield(typeof(ha.frontend), :params)
            @test hasfield(typeof(ha.frontend), :sample_buffer)
            @test hasfield(typeof(ha.frontend), :synthesis_matrix)
            @test hasfield(typeof(ha.frontend), :taps)
            @test hasfield(typeof(ha.frontend), :temp)
            @test hasfield(typeof(ha.frontend), :weights)
            @test hasfield(typeof(ha.frontend), :window)

            # Verify frontend fields are initialized
            @test ha.frontend.calibration_db isa Matrix{Float64}
            @test ha.frontend.params isa
                  VirtualHearingAid.Frontends.WFB.WFBFrontendParameters
            @test ha.frontend.sample_buffer isa DataStructures.CircularBuffer{Float64}
            @test ha.frontend.synthesis_matrix isa Matrix{Float64}
            @test ha.frontend.taps isa Matrix{Float64}
            @test ha.frontend.temp isa Vector{Float64}
            @test ha.frontend.weights isa Vector{Float64}
            @test ha.frontend.window isa Vector{Float64}

            # Verify frontend params match config
            @test ha.frontend.params.nbands == config["parameters"]["frontend"]["nbands"]
            @test ha.frontend.params.fs == config["parameters"]["frontend"]["fs"]
            @test ha.frontend.params.spl_reference_db ==
                  config["parameters"]["frontend"]["spl_reference_db"]
            @test ha.frontend.params.apcoefficient ==
                  config["parameters"]["frontend"]["apcoefficient"]

        end

        # Test backend.params structure (ha.backend.params.*)
        @testset "Backend params structure" begin
            # Test inference parameters
            @test hasfield(typeof(ha.backend.params), :inference)
            @test hasfield(typeof(ha.backend.params.inference), :autostart)
            @test hasfield(typeof(ha.backend.params.inference), :free_energy)
            @test hasfield(typeof(ha.backend.params.inference), :iterations)

            # Verify inference params match config
            @test ha.backend.params.inference.autostart ==
                  config["parameters"]["backend"]["inference"]["autostart"]
            @test ha.backend.params.inference.free_energy ==
                  config["parameters"]["backend"]["inference"]["free_energy"]
            @test ha.backend.params.inference.iterations ==
                  config["parameters"]["backend"]["inference"]["iterations"]

            # Test modules structure
            @test hasfield(typeof(ha.backend.params), :modules)
            @test hasfield(typeof(ha.backend.params.modules), :gain)
            @test hasfield(typeof(ha.backend.params.modules), :nbands)
            @test hasfield(typeof(ha.backend.params.modules), :noise)
            @test hasfield(typeof(ha.backend.params.modules), :sampling_frequency)
            @test hasfield(typeof(ha.backend.params.modules), :speech)
            @test hasfield(typeof(ha.backend.params.modules), :vad)
            @test hasfield(typeof(ha.backend.params.modules), :ξ)
            @test ha.backend.params.modules.nbands ==
                  config["parameters"]["frontend"]["nbands"]

            # Test gain module structure
            @test hasfield(typeof(ha.backend.params.modules.gain), :slope_dB)
            @test hasfield(typeof(ha.backend.params.modules.gain), :threshold_dB)
            @test hasfield(typeof(ha.backend.params.modules.gain), :threshold_lin)
            # slope_dB is no longer read from config, uses default value of 1.0
            @test ha.backend.params.modules.gain.slope_dB == 1.0

            threshold_dB_ = config["parameters"]["backend"]["gain"]["threshold"]
            threshold_lin = 10^(-threshold_dB_ / 20.0)
            threshold_dB = round(-20 * log10(1 - threshold_lin))
            @test ha.backend.params.modules.gain.threshold_dB === threshold_dB

            # Test speech module structure
            @test hasfield(typeof(ha.backend.params.modules.speech), :fc)
            @test hasfield(typeof(ha.backend.params.modules.speech), :sampling_frequency)
            @test hasfield(typeof(ha.backend.params.modules.speech), :λ)
            @test hasfield(typeof(ha.backend.params.modules.speech), :τ90)
            expected_τs =
                config["parameters"]["backend"]["filters"]["time_constants90"]["s"]

            @test ha.backend.params.modules.speech.τ90 ≈ expected_τs

            # Test noise module structure
            @test hasfield(typeof(ha.backend.params.modules.noise), :fc)
            @test hasfield(typeof(ha.backend.params.modules.noise), :sampling_frequency)
            @test hasfield(typeof(ha.backend.params.modules.noise), :λ)
            @test hasfield(typeof(ha.backend.params.modules.noise), :τ90)
            expected_τn =
                config["parameters"]["backend"]["filters"]["time_constants90"]["n"]
            @test ha.backend.params.modules.noise.τ90 ≈ expected_τn

            # Test vad module structure
            @test hasfield(typeof(ha.backend.params.modules.vad), :slope_dB)
            @test hasfield(typeof(ha.backend.params.modules.vad), :threshold_dB)
            # slope_dB is no longer read from config, uses default value of 1.0
            @test ha.backend.params.modules.vad.slope_dB == 1.0
            @test ha.backend.params.modules.vad.threshold_dB ==
                  config["parameters"]["backend"]["switch"]["threshold"]

            # Verify sampling_frequency matches expected value (1 / (buffer_size_s * 1000))
            buffer_size_s = config["parameters"]["frontend"]["buffer_size_s"]
            expected_fs = 1.0 / (buffer_size_s * 1000)
            @test ha.backend.params.modules.sampling_frequency ≈ expected_fs
            @test ha.backend.params.modules.speech.sampling_frequency ≈ expected_fs
            @test ha.backend.params.modules.noise.sampling_frequency ≈ expected_fs

            # Test ξ module structure
            @test hasfield(typeof(ha.backend.params.modules.ξ), :fc)
            @test hasfield(typeof(ha.backend.params.modules.ξ), :sampling_frequency)
            @test hasfield(typeof(ha.backend.params.modules.ξ), :λ)
            @test hasfield(typeof(ha.backend.params.modules.ξ), :τ90)
            expected_τxnr =
                config["parameters"]["backend"]["filters"]["time_constants90"]["xnr"]
            @test ha.backend.params.modules.ξ.τ90 ≈ expected_τxnr
            @test ha.backend.params.modules.ξ.sampling_frequency ≈ expected_fs
        end
    end
end
