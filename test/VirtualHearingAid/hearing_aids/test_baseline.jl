@testitem "Baseline Hearing Aid Tests" begin
    using VirtualHearingAid
    using SampledSignals
    using TOML
    using DataStructures
    import VirtualHearingAid:
        from_config,
        process,
        get_frontend,
        getname,
        default_baseline_parameters,
        WFBFrontend

    # Test constructor
    @testset "Constructor" begin
        # Create from default parameters
        config = default_baseline_parameters()
        ha = from_config(BaselineHearingAid, config)

        @test ha isa BaselineHearingAid
        @test get_frontend(ha) isa WFBFrontend
        @test getname(ha) == "BaselineHearingAid"
        @test ha.backend isa BaselineBackend
    end

    # Test processing
    @testset "Processing" begin
        # Create from default parameters
        config = default_baseline_parameters()
        ha = from_config(BaselineHearingAid, config)

        # Create test input
        t = 0:(1/16000):(4410/16000-1/16000)
        sine_wave = sin.(2π .* 1000.0 .* t)
        test_input = SampleBuf(sine_wave, 16000)

        # Process
        result, results = process(ha, test_input)

        @test result isa SampleBuf
        @test length(result) == length(test_input)
        @test results === nothing  # Baseline should return nothing for results

        # Test frontend weights
        weights = VirtualHearingAid.Frontends.WFB.get_weights(ha.frontend)
        @test length(weights) == 32  # nfft = (nbands - 1) * 2 = 32
        @test all(isapprox.(weights[1:15], 0.0, atol=1e-14))  # First 15 bands should be zero
        @test isapprox(weights[16], 1.0, atol=1e-14)  # Middle band should be 1
        @test all(isapprox.(weights[17:end], 0.0, atol=1e-14))  # Remaining bands should be zero
    end

    # Test configuration structure
    @testset "Configuration Structure" begin
        config = TOML.parsefile(joinpath(@__DIR__, "example_Baseline_config.toml"))
        ha = from_config(BaselineHearingAid, config)

        @test ha isa BaselineHearingAid
        @test ha.backend isa BaselineBackend
        # BaselineBackend has no params (empty struct)
    end

    # Test comprehensive structure matching config file
    @testset "Structure matches config file" begin
        config = TOML.parsefile(joinpath(@__DIR__, "example_Baseline_config.toml"))
        ha = from_config(BaselineHearingAid, config)

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
            @test ha.frontend.params isa VirtualHearingAid.Frontends.WFB.WFBFrontendParameters
            @test ha.frontend.sample_buffer isa DataStructures.CircularBuffer{Float64}
            @test ha.frontend.synthesis_matrix isa Matrix{Float64}
            @test ha.frontend.taps isa Matrix{Float64}
            @test ha.frontend.temp isa Vector{Float64}
            @test ha.frontend.weights isa Vector{Float64}
            @test ha.frontend.window isa Vector{Float64}

            # Verify frontend params match config
            @test ha.frontend.params.nbands == config["parameters"]["frontend"]["nbands"]
            @test ha.frontend.params.fs == config["parameters"]["frontend"]["fs"]
            @test ha.frontend.params.spl_reference_db == config["parameters"]["frontend"]["spl_reference_db"]
            @test ha.frontend.params.apcoefficient == config["parameters"]["frontend"]["apcoefficient"]
        end
    end
end
