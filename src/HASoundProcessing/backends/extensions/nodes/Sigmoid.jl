using StatsFuns: logistic, softplus
using Distributions: pdf
export Sigmoid

struct Sigmoid end

@node Sigmoid Stochastic [out, in, ζ]

@average_energy Sigmoid (
    q_out::Categorical,
    q_in::UnivariateNormalDistributionsFamily,
    q_ζ::PointMass,
) = begin
    m_out = pdf(q_out, 1)
    m_in, v_in = mean_var(q_in)

    ζ_hat = mean(q_ζ)

    # Guard against invalid or infinite values
    if !isfinite(m_in) || !isfinite(v_in) || !isfinite(ζ_hat)
        msg = "Non-finite values in Sigmoid average energy: m_in = $m_in, v_in = $v_in, ζ_hat = $ζ_hat"

        error(msg)
    end

    if abs(ζ_hat) < 1e-8
        λ = 0.125  # limit as ζ -> 0
    else
        λ = 0.5 * (logistic(ζ_hat) - 0.5) / ζ_hat
    end

    # Check intermediate calculations for overflow
    if !isfinite(λ)
        msg = "Non-finite λ in Sigmoid average energy: λ = $λ, ζ_hat = $ζ_hat"
        error(msg)
    end

    # Check for potential overflow in intermediate terms
    m_in_sq = m_in^2
    ζ_hat_sq = ζ_hat^2
    if !isfinite(m_in_sq) || !isfinite(ζ_hat_sq)
        msg = "Overflow in intermediate calculations: m_in^2 = $m_in_sq, ζ_hat^2 = $ζ_hat_sq, m_in = $m_in, ζ_hat = $ζ_hat"
        error(msg)
    end

    term1 = m_in * m_out
    term2 = softplus(-ζ_hat)
    term3 = 0.5 * (m_in + ζ_hat)
    term4 = λ * (m_in_sq + v_in - ζ_hat_sq)

    # Check each term
    if !isfinite(term1) || !isfinite(term2) || !isfinite(term3) || !isfinite(term4)
        msg = "Non-finite intermediate terms: term1 = $term1, term2 = $term2, term3 = $term3, term4 = $term4"

        error(msg)
    end

    U = -(term1 - term2 - term3 - term4)

    # Guard against infinite or NaN result
    if !isfinite(U)
        msg = "Non-finite average energy in Sigmoid: U = $U, m_in = $m_in, v_in = $v_in, ζ_hat = $ζ_hat, λ = $λ, terms = ($term1, $term2, $term3, $term4)"

        error(msg)
    end

    return U
end

@average_energy Sigmoid (
    q_out::PointMass,
    q_in::UnivariateNormalDistributionsFamily,
    q_ζ::PointMass,
) = begin
    m_out = mean(q_out)
    m_in, v_in = mean_var(q_in)
    ζ_hat = mean(q_ζ)

    # Guard against invalid or infinite values
    if !isfinite(m_in) || !isfinite(v_in) || !isfinite(ζ_hat) || !isfinite(m_out)
        msg = "Non-finite values in Sigmoid average energy: m_out = $m_out, m_in = $m_in, v_in = $v_in, ζ_hat = $ζ_hat"
        error(msg)
    end

    if abs(ζ_hat) < 1e-8
        λ = 0.125  # limit as ζ -> 0, to avoid division by 0 and numerical instability
    else
        λ = 0.5 * (logistic(ζ_hat) - 0.5) / ζ_hat
    end

    # Check intermediate calculations for overflow
    if !isfinite(λ)
        msg = "Non-finite λ in Sigmoid average energy: λ = $λ, ζ_hat = $ζ_hat"
        error(msg)
    end

    # Check for potential overflow in intermediate terms
    m_in_sq = m_in^2
    ζ_hat_sq = ζ_hat^2
    if !isfinite(m_in_sq) || !isfinite(ζ_hat_sq)
        msg = "Overflow in intermediate calculations: m_in^2 = $m_in_sq, ζ_hat^2 = $ζ_hat_sq, m_in = $m_in, ζ_hat = $ζ_hat"
        error(msg)
    end

    term1 = m_in * m_out
    term2 = softplus(-ζ_hat)
    term3 = 0.5 * (m_in + ζ_hat)
    term4 = λ * (m_in_sq + v_in - ζ_hat_sq)

    # Check each term
    if !isfinite(term1) || !isfinite(term2) || !isfinite(term3) || !isfinite(term4)
        msg = "Non-finite intermediate terms: term1 = $term1, term2 = $term2, term3 = $term3, term4 = $term4"
        error(msg)
    end

    U = -(term1 - term2 - term3 - term4)

    # Guard against infinite or NaN result
    if !isfinite(U)
        msg = "Non-finite average energy in Sigmoid: U = $U, m_out = $m_out, m_in = $m_in, v_in = $v_in, ζ_hat = $ζ_hat, λ = $λ, terms = ($term1, $term2, $term3, $term4)"
        error(msg)
    end
    return U
end
