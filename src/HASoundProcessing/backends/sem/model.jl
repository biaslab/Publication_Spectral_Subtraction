# ============================================================================
# Spectral Enhancement Model (SEM) Equation 13 in the paper 
# "A Probabilistic Generative Model for Spectral Speech Enhancement"
# ============================================================================

@model function sigmoid_model(switch, ξ, θ, ζ)
    switch ~ Sigmoid(ξ - θ, ζ)
end

@model function random_w_model(x, μx_prior, τx_prior, τx)
    x_prior ~ Normal(mean = μx_prior, precision = τx_prior)
    x ~ Normal(mean = x_prior, precision = τx)
end


@model function SEM_filtering_model(
    y, μs_prior, τs_prior, τs, μn_prior, τn_prior, τn, κ, θ, τξ)

    s ~ random_w_model(μx_prior = μs_prior, τx_prior = τs_prior, τx = τs)
    n ~ random_w_model(μx_prior = μn_prior, τx_prior = τn_prior, τx = τn)

    ξ ~ Normal(mean = s - n, precision = τξ)

    ζ_switch ~ Uninformative()
    π_switch ~ sigmoid_model(ξ = ξ, θ = κ, ζ = ζ_switch)

    ζ_gain ~ Uninformative()
    w ~ sigmoid_model(ξ = ξ, θ = θ, ζ = ζ_gain)
    w ~ Categorical([0.5, 0.5])

    y ~ NormalMixture(switch = π_switch, m = (s, n), p = (1.0, 1.0))
end
