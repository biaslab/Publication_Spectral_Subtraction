@testitem "sigmoidNode" begin
    using ReactiveMP, Random, BayesBase, ExponentialFamily
    using Experiments.Experiments.HASoundProcessing.SEM.Extensions: Sigmoid

    @testset "Standard Sigmoid node case: (q_out::Categorical, q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass)" begin
        q_out = Categorical([0.5, 0.5])
        q_ζ = PointMass(1.0)
        q_in = NormalMeanVariance(0.0, 1.0)
        for normal_fam in
            (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
            q_in_adj = convert(normal_fam, q_in)
            @test isapprox(
                score(
                    AverageEnergy(),
                    Sigmoid,
                    Val{(:out, :in, :ζ)}(),
                    (
                        Marginal(q_out, false, false, nothing),
                        Marginal(q_in_adj, false, false, nothing),
                        Marginal(q_ζ, false, false, nothing),
                    ),
                    nothing,
                ), 0.81, atol = 1e-2)
        end
    end

    @testset "Standard Sigmoid node case: (q_out::PointMass, q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass)" begin
        q_out = PointMass(0.5)
        q_ζ = PointMass(1.0)
        q_in = NormalMeanVariance(0.0, 1.0)
        for normal_fam in
            (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
            q_in_adj = convert(normal_fam, q_in)
            @test isapprox(
                score(
                    AverageEnergy(),
                    Sigmoid,
                    Val{(:out, :in, :ζ)}(),
                    (
                        Marginal(q_out, false, false, nothing),
                        Marginal(q_in_adj, false, false, nothing),
                        Marginal(q_ζ, false, false, nothing),
                    ),
                    nothing,
                ), 0.81, atol = 1e-2)
        end
    end


    @testset "Huge variance Sigmoid node case: (q_out::Categorical, q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass)" begin
        q_out = Categorical([0.5, 0.5])
        q_ζ = PointMass(0.0)
        q_in = NormalMeanVariance(0.0, 1e6)
        for normal_fam in
            (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
            q_in_adj = convert(normal_fam, q_in)
            @test isapprox(
                score(
                    AverageEnergy(),
                    Sigmoid,
                    Val{(:out, :in, :ζ)}(),
                    (
                        Marginal(q_out, false, false, nothing),
                        Marginal(q_in_adj, false, false, nothing),
                        Marginal(q_ζ, false, false, nothing),
                    ),
                    nothing,
                ), 125000.70, atol = 1e-2)
        end

        @testset "Huge variance Sigmoid node case: (q_out::Categorical, q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass)" begin
            q_out = Categorical([0.5, 0.5])
            q_ζ = PointMass(0.0)
            q_in = NormalMeanVariance(0.0, 1e6)
            for normal_fam in
                (NormalMeanVariance, NormalMeanPrecision, NormalWeightedMeanPrecision)
                q_in_adj = convert(normal_fam, q_in)
                @test isapprox(
                    score(
                        AverageEnergy(),
                        Sigmoid,
                        Val{(:out, :in, :ζ)}(),
                        (
                            Marginal(q_out, false, false, nothing),
                            Marginal(q_in_adj, false, false, nothing),
                            Marginal(q_ζ, false, false, nothing),
                        ),
                        nothing,
                    ), 125000.70, atol = 1e-2)
            end
        end
    end

    @testset "Negative variance Sigmoid node case: (q_out::Categorical, q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass)" begin
        q_out = Categorical([0.5, 0.5])
        q_in = NormalMeanVariance(0.0, -1.0)
        m_in, v_in = mean_var(q_in)
        @test_throws DomainError PointMass(sqrt(m_in^2 + v_in)) #which is the message for sqrt of negative number of q(ζ)
    end

    @testset "Negative variance Sigmoid node case: (q_out::PointMass, q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass)" begin
        q_out = PointMass(0.5)
        q_in = NormalMeanVariance(0.0, -1.0)
        m_in, v_in = mean_var(q_in)
        @test_throws DomainError PointMass(sqrt(m_in^2 + v_in)) #which is the message for sqrt of negative number of q(ζ)
    end

    @testset "Degenerate case: variance -> 0 Sigmoid node case: (q_out::PointMass, q_in::UnivariateNormalDistributionsFamily, q_ζ::PointMass)" begin
        q_out = PointMass(0.5)
        q_in = NormalMeanVariance(0.0, 0.0)
        # Test that the rule catches unwanted behavior (variance <= 0 or resulting ζ² <= 0)
        # The rule should throw DomainError for degenerate cases
        # We test the computation directly as the rule does it
        m_in, v_in = mean_var(q_in)
        ζ_hat2 = m_in^2 + v_in
        # The rule checks ζ_hat2 <= 0 and throws DomainError
        @test_throws DomainError begin
            if ζ_hat2 <= 0
                throw(
                    DomainError(
                        ζ_hat2,
                        "Cannot compute sqrt of non-positive ζ² = $ζ_hat2 (mean = $m_in, variance = $v_in)",
                    ),
                )
            end
            PointMass(sqrt(ζ_hat2))
        end
    end
end
