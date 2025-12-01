@testitem "rules:Sigmoid:in" begin
    using ReactiveMP, BayesBase, Random, ExponentialFamily, Distributions
    using StatsFuns: logistic
    using Experiments.Experiments.HASoundProcessing.SEM.Extensions: Sigmoid

    import ReactiveMP: @test_rules

    @testset "Mean Field: (q_out::Categorical, q_ζ::PointMass) - Float64" begin
        @test_rules [check_type_promotion = false, atol = [Float64 => 1e-5]] Sigmoid(
            :in,
            Marginalisation,
        ) [
            (
                input = (q_out = Categorical([0.5, 0.5]), q_ζ = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(0.0, 0.2310585786300049),
            ),
            (
                input = (q_out = Categorical([1.0, 0.0]), q_ζ = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(
                    0.5,
                    0.2310585786300049,
                ),
            ),
            (
                input = (q_out = Categorical([0.0, 1.0]), q_ζ = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(
                    -0.5,
                    0.2310585786300049,
                ),
            ),
        ]
    end

    @testset "Mean Field: (q_out::PointMass, q_ζ::PointMass) - Float64" begin
        @test_rules [check_type_promotion = false, atol = [Float64 => 1e-5]] Sigmoid(
            :in,
            Marginalisation,
        ) [
            (
                input = (q_out = PointMass(0.5), q_ζ = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(0.0, 0.2310585786300049),
            ),
            (
                input = (q_out = PointMass(1.0), q_ζ = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(
                    0.5,
                    0.2310585786300049,
                ),
            ),
            (
                input = (q_out = PointMass(0.0), q_ζ = PointMass(1.0)),
                output = NormalWeightedMeanPrecision(
                    -0.5,
                    0.2310585786300049,
                ),
            ),
        ]
    end
end
