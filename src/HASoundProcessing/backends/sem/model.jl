# ============================================================================
# Spectral Enhancement Model (SEM) Equation 13 in the paper 
# "A Probabilistic Generative Model for Spectral Speech Enhancement"
# ============================================================================

@model function sigmoid_model(switch, ξ, η, θ, ζ)
    switch ~ Sigmoid(η * (ξ - θ), ζ)
end

@model function random_w_model(x, μx_prior, τx_prior, τx)
    x_prior ~ Normal(mean = μx_prior, precision = τx_prior)
    x ~ Normal(mean = x_prior, precision = τx)
end


@model function SEM_filtering_model(
    y, μs_prior, τs_prior, τs, μn_prior, τn_prior, τn, η, κ, θ, τξ, μξ_mean, τξ_mean)

    s ~ random_w_model(μx_prior = μs_prior, τx_prior = τs_prior, τx = τs)
    n ~ random_w_model(μx_prior = μn_prior, τx_prior = τn_prior, τx = τn)

    ξ := s - n
    ξ_smooth ~ random_w_model(μx_prior = μξ_mean, τx_prior = τξ_mean, τx = τξ)
    ξ ~ Normal(mean = ξ_smooth, precision = 1.0)

    ζ_switch ~ Uninformative()
    π_switch ~ sigmoid_model(ξ = ξ_smooth, η = η, θ = κ, ζ = ζ_switch)

    ζ_gain ~ Uninformative()
    w ~ sigmoid_model(ξ = ξ, η = η, θ = θ, ζ = ζ_gain)
    w ~ Categorical([0.5, 0.5])

    y ~ NormalMixture(switch = π_switch, m = (s, n), p = (1.0, 1.0))
end
