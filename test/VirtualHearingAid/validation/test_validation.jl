@testitem "Configuration Validation Tests" begin
    using VirtualHearingAid
    using TOML
    import VirtualHearingAid:
        validate_config, BaselineHearingAid, SEMHearingAid


    test_dir = @__DIR__

    @testset "Baseline Config Validation" begin
        # Test correct config
        @testset "Correct Baseline Config" begin
            config =
                TOML.parsefile(joinpath(test_dir, "example_Baseline_config_correct.toml"))

            @test validate_config(BaselineHearingAid, config) == true
        end

        # Test missing sections
        @testset "Missing Sections" begin
            # Missing parameters
            config = Dict()
            @test_throws ArgumentError validate_config(BaselineHearingAid, config)

            # Missing hearingaid
            config = Dict("parameters" => Dict("frontend" => Dict()))
            @test_throws ArgumentError validate_config(BaselineHearingAid, config)

            # Missing frontend
            config = Dict("parameters" => Dict("hearingaid" => Dict()))
            @test_throws ArgumentError validate_config(BaselineHearingAid, config)
        end

        # Test missing frontend fields
        @testset "Missing Frontend Fields" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_Baseline_config_correct.toml"))

            for field in [
                "buffer_size_s",
                "nbands",
                "fs",
                "spl_reference_db",
                "spl_power_estimate_lower_bound_db",
                "apcoefficient",
            ]

                config = deepcopy(correct_config)
                delete!(config["parameters"]["frontend"], field)
                @test_throws ArgumentError validate_config(BaselineHearingAid, config)
            end
        end

        # Test missing hearingaid fields
        @testset "Missing HearingAid Fields" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_Baseline_config_correct.toml"))

            for field in ["name", "type"]
                config = deepcopy(correct_config)
                delete!(config["parameters"]["hearingaid"], field)
                @test_throws ArgumentError validate_config(BaselineHearingAid, config)
            end
        end
    end

    @testset "SEM Config Validation" begin
        # Test correct config
        @testset "Correct SEM Config" begin
            config = TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))
            @test validate_config(SEMHearingAid, config) == true
        end

        # Test missing backend
        @testset "Missing Backend" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            config = deepcopy(correct_config)
            delete!(config["parameters"], "backend")
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing backend.general
        @testset "Missing Backend General" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            config = deepcopy(correct_config)
            delete!(config["parameters"]["backend"], "general")
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing backend.general fields
        @testset "Missing Backend General Fields" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            for field in ["name", "type"]
                config = deepcopy(correct_config)
                delete!(config["parameters"]["backend"]["general"], field)
                @test_throws ArgumentError validate_config(SEMHearingAid, config)
            end
        end

        # Test missing inference
        @testset "Missing Inference" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            config = deepcopy(correct_config)
            delete!(config["parameters"]["backend"], "inference")
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing inference fields
        @testset "Missing Inference Fields" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            for field in ["autostart", "free_energy", "iterations"]
                config = deepcopy(correct_config)
                delete!(config["parameters"]["backend"]["inference"], field)
                @test_throws ArgumentError validate_config(SEMHearingAid, config)
            end
        end

        # Test wrong inference types
        @testset "Wrong Inference Types" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            # autostart not a Bool
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["inference"]["autostart"] = 1
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # free_energy not a Bool
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["inference"]["free_energy"] = "true"
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # iterations not an Integer
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["inference"]["iterations"] = 2.5
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing filters
        @testset "Missing Filters" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            config = deepcopy(correct_config)
            delete!(config["parameters"]["backend"], "filters")
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing time_constants90
        @testset "Missing Time Constants" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            config = deepcopy(correct_config)
            delete!(config["parameters"]["backend"]["filters"], "time_constants90")
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing s or n in time_constants90
        @testset "Missing s or n in Time Constants" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            for param in ["s", "n"]
                config = deepcopy(correct_config)
                delete!(
                    config["parameters"]["backend"]["filters"]["time_constants90"],
                    param,
                )

                @test_throws ArgumentError validate_config(SEMHearingAid, config)
            end
        end

        # Test time constants not floats
        @testset "Time Constants Not Floats" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            # s as integer
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["filters"]["time_constants90"]["s"] = 5
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # n as integer
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["filters"]["time_constants90"]["n"] = 10000
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test time constants <= 0
        @testset "Time Constants <= 0" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            # s = 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["filters"]["time_constants90"]["s"] = 0.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # s < 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["filters"]["time_constants90"]["s"] = -5.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # n = 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["filters"]["time_constants90"]["n"] = 0.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # n < 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["filters"]["time_constants90"]["n"] = -10000.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test s >= n (should fail)
        @testset "s >= n (Invalid Relationship)" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            # s = n
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["filters"]["time_constants90"]["s"] = 10000.0
            config["parameters"]["backend"]["filters"]["time_constants90"]["n"] = 10000.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # s > n
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["filters"]["time_constants90"]["s"] = 20000.0
            config["parameters"]["backend"]["filters"]["time_constants90"]["n"] = 10000.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing priors
        @testset "Missing Priors" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            config = deepcopy(correct_config)
            delete!(config["parameters"]["backend"], "priors")
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing prior types
        @testset "Missing Prior Types" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            for prior_name in ["speech", "noise"]
                config = deepcopy(correct_config)
                delete!(config["parameters"]["backend"]["priors"], prior_name)
                @test_throws ArgumentError validate_config(SEMHearingAid, config)
            end
        end

        # Test missing prior fields
        @testset "Missing Prior Fields" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            for prior_name in ["speech", "noise"]
                for field in ["mean", "precision"]
                    config = deepcopy(correct_config)
                    delete!(config["parameters"]["backend"]["priors"][prior_name], field)
                    @test_throws ArgumentError validate_config(SEMHearingAid, config)
                end
            end
        end

        # Test prior mean not a float
        @testset "Prior Mean Not Float" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            config = deepcopy(correct_config)
            config["parameters"]["backend"]["priors"]["speech"]["mean"] = 80
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test prior precision not a float
        @testset "Prior Precision Not Float" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            config = deepcopy(correct_config)
            config["parameters"]["backend"]["priors"]["speech"]["precision"] = 1
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test prior precision <= 0
        @testset "Prior Precision <= 0" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            # precision = 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["priors"]["speech"]["precision"] = 0.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # precision < 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["priors"]["speech"]["precision"] = -1.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing gain
        @testset "Missing Gain" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            config = deepcopy(correct_config)
            delete!(config["parameters"]["backend"], "gain")
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing gain fields
        @testset "Missing Gain Fields" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            for field in ["slope", "threshold"]
                config = deepcopy(correct_config)
                delete!(config["parameters"]["backend"]["gain"], field)
                @test_throws ArgumentError validate_config(SEMHearingAid, config)
            end
        end

        # Test gain slope not a float
        @testset "Gain Slope Not Float" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            config = deepcopy(correct_config)
            config["parameters"]["backend"]["gain"]["slope"] = 1
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test gain slope <= 0
        @testset "Gain Slope <= 0" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            # slope = 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["gain"]["slope"] = 0.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # slope < 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["gain"]["slope"] = -1.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test gain threshold not a float
        @testset "Gain Threshold Not Float" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            config = deepcopy(correct_config)
            config["parameters"]["backend"]["gain"]["threshold"] = 12
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing switch
        @testset "Missing Switch" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))

            config = deepcopy(correct_config)
            delete!(config["parameters"]["backend"], "switch")
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test missing switch fields
        @testset "Missing Switch Fields" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            for field in ["slope", "threshold"]
                config = deepcopy(correct_config)
                delete!(config["parameters"]["backend"]["switch"], field)
                @test_throws ArgumentError validate_config(SEMHearingAid, config)
            end
        end

        # Test switch slope not a float
        @testset "Switch Slope Not Float" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            config = deepcopy(correct_config)
            config["parameters"]["backend"]["switch"]["slope"] = 8
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test switch slope <= 0
        @testset "Switch Slope <= 0" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            # slope = 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["switch"]["slope"] = 0.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)

            # slope < 0
            config = deepcopy(correct_config)
            config["parameters"]["backend"]["switch"]["slope"] = -8.0
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test switch threshold not a float
        @testset "Switch Threshold Not Float" begin
            correct_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_correct.toml"))


            config = deepcopy(correct_config)
            config["parameters"]["backend"]["switch"]["threshold"] = 4
            @test_throws ArgumentError validate_config(SEMHearingAid, config)
        end

        # Test incorrect config file (comprehensive errors)
        @testset "Incorrect SEM Config File" begin
            incorrect_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_incorrect.toml"))


            # This config file has multiple errors, validation should catch at least one
            @test_throws ArgumentError validate_config(SEMHearingAid, incorrect_config)
        end

        # Test negative values config file
        @testset "SEM Config with Negative Values" begin
            negative_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_negative.toml"))


            # This config file has negative values which should be caught
            @test_throws ArgumentError validate_config(SEMHearingAid, negative_config)
        end

        # Test zero values config file
        @testset "SEM Config with Zero Values" begin
            zero_config = TOML.parsefile(joinpath(test_dir, "example_SEM_config_zero.toml"))

            # This config file has zero values which should be caught
            @test_throws ArgumentError validate_config(SEMHearingAid, zero_config)
        end

        # Test invalid relationship config file (s > n)
        @testset "SEM Config with Invalid Relationship (s > n)" begin
            invalid_rel_config = TOML.parsefile(
                joinpath(test_dir, "example_SEM_config_invalid_relationship.toml"),
            )

            # This config file has s > n which should be caught
            @test_throws ArgumentError validate_config(SEMHearingAid, invalid_rel_config)
        end

        # Test invalid relationship config file (s == n)
        @testset "SEM Config with Invalid Relationship (s == n)" begin
            s_equal_n_config =
                TOML.parsefile(joinpath(test_dir, "example_SEM_config_s_equal_n.toml"))

            # This config file has s == n which should be caught
            @test_throws ArgumentError validate_config(SEMHearingAid, s_equal_n_config)
        end
    end

    @testset "Incorrect Config Files" begin
        # Test incorrect Baseline config file
        @testset "Incorrect Baseline Config File" begin
            incorrect_config =
                TOML.parsefile(joinpath(test_dir, "example_Baseline_config_incorrect.toml"))


            # This config file has multiple errors, validation should catch at least one
            @test_throws ArgumentError validate_config(BaselineHearingAid, incorrect_config)
        end

    end
end
