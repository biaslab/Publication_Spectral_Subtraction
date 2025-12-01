@testitem "Utils Module Tests" begin
    # Move imports inside the testitem block
    using Experiments.HASoundProcessing.Utils:
        tau2ff,
        calculate_fc,
        calculate_λ,
        λ2process_var,
        λ2observation_var,
        Γ_hyperparameters_shaperate,
        FILTER_FIELDS,
        STATE_FIELDS

    @testset "tau2ff - Time Constant to Forgetting Factor" begin
        @testset "Basic Functionality" begin
            # Test with typical values
            @test isapprox(tau2ff(5.0, 10.0), 0.0450, atol = 1e-3)
            @test isapprox(tau2ff(10.0, 100.0), 0.0023, atol = 1e-3)
            @test isapprox(tau2ff(100.0, 1000.0), 0.00023, atol = 1e-3)
        end

        @testset "Type Stability" begin
            # Test with different numeric types
            @test tau2ff(Float32(5.0), Float32(10.0)) isa Float32
            @test tau2ff(Float64(5.0), Float64(10.0)) isa Float64
            @test tau2ff(5.0, 10.0) isa Float64
        end

        @testset "Edge Cases" begin
            # Test with very small time constants
            @test tau2ff(0.001, 1000.0) > 0
            @test tau2ff(0.001, 1000.0) < 1.0

            # Test with very large time constants
            @test tau2ff(10000.0, 1.0) > 0
            @test tau2ff(10000.0, 1.0) < 1.0

            # Test with very high sampling frequencies
            @test tau2ff(1.0, 100000.0) > 0
            @test tau2ff(1.0, 100000.0) < 1.0

            # Test with very low sampling frequencies
            @test tau2ff(1.0, 0.1) > 0
            @test tau2ff(1.0, 0.1) < 1.0
        end

        @testset "Mathematical Properties" begin
            # Forgetting factor should always be between 0 and 1
            for τ in [0.1, 1.0, 10.0, 100.0, 1000.0]
                for fs in [0.1, 1.0, 10.0, 100.0, 1000.0]
                    λ = tau2ff(τ, fs)
                    @test λ > 0
                    @test λ <= 1.0  # Allow equality for edge cases
                end
            end

            # Larger time constants should give smaller forgetting factors
            @test tau2ff(1.0, 10.0) > tau2ff(10.0, 10.0)
            @test tau2ff(10.0, 10.0) > tau2ff(100.0, 10.0)
        end
    end

    @testset "calculate_fc - Cutoff Frequency Calculation" begin
        @testset "Basic Functionality" begin
            # Test with typical values
            @test isapprox(calculate_fc(5.0), 73.2112, atol = 1e-3)
            @test isapprox(calculate_fc(10.0), 36.6056, atol = 1e-3)
            @test isapprox(calculate_fc(100.0), 3.6606, atol = 1e-3)
        end

        @testset "Type Stability" begin
            # Test with different numeric types
            @test calculate_fc(Float32(5.0)) isa Float32
            @test calculate_fc(Float64(5.0)) isa Float64
            @test calculate_fc(5.0) isa Float64
        end

        @testset "Edge Cases" begin
            # Test with very small time constants
            @test calculate_fc(0.001) > 0
            @test calculate_fc(0.001) > 1000.0  # Should be very high frequency

            # Test with very large time constants
            @test calculate_fc(10000.0) > 0
            @test calculate_fc(10000.0) < 1.0  # Should be very low frequency
        end

        @testset "Mathematical Properties" begin
            # Cutoff frequency should always be positive
            for τ in [0.1, 1.0, 10.0, 100.0, 1000.0]
                fc = calculate_fc(τ)
                @test fc > 0
            end

            # Larger time constants should give smaller cutoff frequencies
            @test calculate_fc(1.0) > calculate_fc(10.0)
            @test calculate_fc(10.0) > calculate_fc(100.0)
        end
    end

    @testset "calculate_λ - Adaptive Lambda Calculation" begin
        @testset "Basic Functionality" begin
            # Test with typical values
            @test isapprox(calculate_λ(0.5, 0.1, 0.7), 0.38, atol = 1e-3)
            @test isapprox(calculate_λ(0.8, 0.2, 0.5), 0.5, atol = 1e-3)
            @test isapprox(calculate_λ(0.3, 0.9, 0.1), 0.84, atol = 1e-3)
        end

        @testset "Type Stability" begin
            # Test with different numeric types
            @test calculate_λ(Float32(0.5), Float32(0.1), Float32(0.7)) isa Float32
            @test calculate_λ(Float64(0.5), Float64(0.1), Float64(0.7)) isa Float64
            @test calculate_λ(0.5, 0.1, 0.7) isa Float64
        end

        @testset "Boundary Cases" begin
            # Test with θ = 0 (should return λn)
            @test calculate_λ(0.5, 0.1, 0.0) == 0.1
            @test calculate_λ(0.8, 0.2, 0.0) == 0.2

            # Test with θ = 1 (should return λs)
            @test calculate_λ(0.5, 0.1, 1.0) == 0.5
            @test calculate_λ(0.8, 0.2, 1.0) == 0.8

            # Test with θ = 0.5 (should return average)
            @test calculate_λ(0.5, 0.1, 0.5) == 0.3
            @test calculate_λ(0.8, 0.2, 0.5) == 0.5
        end

        @testset "Mathematical Properties" begin
            # Result should be between λn and λs when 0 ≤ θ ≤ 1
            λs, λn = 0.8, 0.2
            for θ in [0.0, 0.25, 0.5, 0.75, 1.0]
                λ = calculate_λ(λs, λn, θ)
                @test λ >= min(λs, λn)
                @test λ <= max(λs, λn)
            end

            # Linear interpolation property
            λs, λn = 0.6, 0.2
            θ1, θ2 = 0.25, 0.75
            λ1 = calculate_λ(λs, λn, θ1)
            λ2 = calculate_λ(λs, λn, θ2)
            @test λ2 > λ1  # Higher θ should give higher λ
        end
    end

    @testset "λ2process_var - Forgetting Factor to Process Variance" begin
        @testset "Basic Functionality" begin
            # Test with typical values
            @test isapprox(λ2process_var(0.1), 0.0111, atol = 1e-3)
            @test isapprox(λ2process_var(0.5), 0.5, atol = 1e-3)
            @test isapprox(λ2process_var(0.9), 8.1, atol = 1e-3)
        end

        @testset "Type Stability" begin
            # Test with different numeric types
            @test λ2process_var(Float32(0.5)) isa Float32
            @test λ2process_var(Float64(0.5)) isa Float64
            @test λ2process_var(0.5) isa Float64
        end

        @testset "Edge Cases" begin
            # Test with very small λ
            @test λ2process_var(0.001) > 0
            @test λ2process_var(0.001) < 0.001

            # Test with λ close to 1
            @test λ2process_var(0.999) > 0
            @test λ2process_var(0.999) > 100.0  # Should be very large
        end

        @testset "Mathematical Properties" begin
            # Process variance should always be positive for 0 < λ < 1
            for λ in [0.1, 0.3, 0.5, 0.7, 0.9]
                var = λ2process_var(λ)
                @test var > 0
            end

            # Larger λ should give larger process variance
            @test λ2process_var(0.3) < λ2process_var(0.5)
            @test λ2process_var(0.5) < λ2process_var(0.7)
            @test λ2process_var(0.7) < λ2process_var(0.9)
        end

        @testset "Error Cases" begin
            # Should handle λ = 0 (though this might not be realistic)
            @test λ2process_var(0.0) == 0.0

            # Should handle λ = 1 (though this might cause issues)
            @test λ2process_var(1.0) == Inf
        end
    end

    @testset "λ2observation_var - Forgetting Factor to Observation Variance" begin
        @testset "Basic Functionality" begin
            # Test with typical values
            @test isapprox(λ2observation_var(0.1), 10.0, atol = 1e-3)
            @test isapprox(λ2observation_var(0.5), 2.0, atol = 1e-3)
            @test isapprox(λ2observation_var(0.9), 1.1111, atol = 1e-3)
        end

        @testset "Type Stability" begin
            # Test with different numeric types
            @test λ2observation_var(Float32(0.5)) isa Float32
            @test λ2observation_var(Float64(0.5)) isa Float64
            @test λ2observation_var(0.5) isa Float64
        end

        @testset "Edge Cases" begin
            # Test with very small λ
            @test λ2observation_var(0.001) > 0
            @test λ2observation_var(0.001) > 100.0  # Should be very large

            # Test with λ close to 1
            @test λ2observation_var(0.999) > 0
            @test λ2observation_var(0.999) < 2.0  # Should be close to 1
        end

        @testset "Mathematical Properties" begin
            # Observation variance should always be positive for λ > 0
            for λ in [0.1, 0.3, 0.5, 0.7, 0.9]
                var = λ2observation_var(λ)
                @test var > 0
            end

            # Smaller λ should give larger observation variance
            @test λ2observation_var(0.1) > λ2observation_var(0.3)
            @test λ2observation_var(0.3) > λ2observation_var(0.5)
            @test λ2observation_var(0.5) > λ2observation_var(0.7)
        end

        @testset "Error Cases" begin
            # Should handle λ = 0 (though this might not be realistic)
            @test λ2observation_var(0.0) == Inf
        end
    end

    @testset "Γ_hyperparameters_shaperate - Gamma Distribution Hyperparameters" begin
        @testset "Basic Functionality" begin
            # Test with typical values
            shape, rate = Γ_hyperparameters_shaperate(1.0)
            @test shape == 1.0
            @test rate == 1.0

            shape, rate = Γ_hyperparameters_shaperate(2.0)
            @test shape == 1.0
            @test rate == 0.5

            shape, rate = Γ_hyperparameters_shaperate(0.5)
            @test shape == 1.0
            @test rate == 2.0
        end

        @testset "Custom Strength Parameter" begin
            # Test with different strength values
            shape, rate = Γ_hyperparameters_shaperate(1.0, strength = 2.0)
            @test shape == 2.0
            @test rate == 2.0

            shape, rate = Γ_hyperparameters_shaperate(2.0, strength = 0.5)
            @test shape == 0.5
            @test rate == 0.25
        end

        @testset "Type Stability" begin
            # Test with different numeric types
            shape, rate = Γ_hyperparameters_shaperate(Float32(1.0))
            @test shape isa Float64  # Default type is Float64
            @test rate isa Float64

            shape, rate = Γ_hyperparameters_shaperate(Float64(1.0))
            @test shape isa Float64
            @test rate isa Float64
        end

        @testset "Mathematical Properties" begin
            # For Gamma(α, β), E[X] = α/β
            for E_τ in [0.5, 1.0, 2.0, 5.0]
                for strength in [0.5, 1.0, 2.0]
                    shape, rate = Γ_hyperparameters_shaperate(E_τ, strength = strength)
                    expected_mean = shape / rate
                    @test isapprox(expected_mean, E_τ, atol = 1e-10)
                end
            end

            # Shape and rate should always be positive
            for E_τ in [0.1, 1.0, 10.0]
                shape, rate = Γ_hyperparameters_shaperate(E_τ)
                @test shape > 0
                @test rate > 0
            end
        end

        @testset "Edge Cases" begin
            # Test with very small expected value
            shape, rate = Γ_hyperparameters_shaperate(0.001)
            @test shape > 0
            @test rate > 0

            # Test with very large expected value
            shape, rate = Γ_hyperparameters_shaperate(1000.0)
            @test shape > 0
            @test rate > 0
        end
    end


    @testset "Integration Tests" begin
        @testset "Consistency Between Functions" begin
            # Test that tau2ff and calculate_fc are consistent
            τ = 5.0
            fs = 10.0
            λ = tau2ff(τ, fs)
            fc = calculate_fc(τ)

            # Both should produce reasonable values
            @test λ > 0 && λ < 1.0
            @test fc > 0

            # Test that λ2process_var and λ2observation_var are consistent
            process_var = λ2process_var(λ)
            observation_var = λ2observation_var(λ)

            @test process_var > 0
            @test observation_var > 0
        end

        @testset "Real-world Parameter Ranges" begin
            # Test with realistic hearing aid parameters
            τ_speech = 5.0    # ms
            τ_noise = 10000.0 # ms
            fs = 0.667        # 1/ms (600 Hz)

            λ_speech = tau2ff(τ_speech, fs)
            λ_noise = tau2ff(τ_noise, fs)

            @test λ_speech > λ_noise  # Speech should have higher forgetting factor
            @test λ_speech > 0.01     # Should be reasonable value
            @test λ_noise < 0.001     # Should be very small

            # Test adaptive lambda calculation
            θ = 0.7  # 70% speech, 30% noise
            λ_adaptive = calculate_λ(λ_speech, λ_noise, θ)

            @test λ_adaptive > λ_noise
            @test λ_adaptive < λ_speech
        end
    end
end
