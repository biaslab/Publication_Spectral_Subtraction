using Distributions: pdf
using StatsFuns: logistic
@rule Sigmoid(:in, Marginalisation) (q_out::Categorical, q_ζ::PointMass) = begin
    m_out = pdf(q_out, 1)
    ζ_hat = mean(q_ζ)

    λ = 0.5 * (logistic(ζ_hat) - 0.5) / ζ_hat
    w = 2 * λ
    ξ = m_out - 0.5
    return NormalWeightedMeanPrecision(ξ, w)
end

@rule Sigmoid(:in, Marginalisation) (q_out::PointMass, q_ζ::PointMass) = begin
    m_out = mean(q_out)
    ζ_hat = mean(q_ζ)

    λ = 0.5 * (logistic(ζ_hat) - 0.5) / ζ_hat
    w = 2 * λ
    ξ = m_out - 0.5
    return NormalWeightedMeanPrecision(ξ, w)
end
