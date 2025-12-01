@rule Sigmoid(:ζ, Marginalisation) (
    q_out::Categorical,
    q_in::UnivariateNormalDistributionsFamily,
) = begin
    m_in, v_in = mean_var(q_in)

    # 1) Basic sanity: finite mean/variance
    if !isfinite(m_in) || !isfinite(v_in)
        error("""
        Degenerate or invalid Gaussian in Sigmoid(:ζ, Marginalisation):
        mean = $m_in, variance = $v_in. This should not happen.
        """)
    end

    # 2) Variance must be strictly positive in this model
    if v_in < 0.0
        throw(
            DomainError(
                v_in,
                "Non-positive variance in Sigmoid(:ζ, Marginalisation): variance = $v_in, mean = $m_in",
            ),
        )
    end

    ζ_hat2 = m_in^2 + v_in

    # 3) ζ² must be finite and positive
    if !isfinite(ζ_hat2)
        error("""
        Invalid ζ² in Sigmoid(:ζ, Marginalisation):
        ζ² = $ζ_hat2 from mean = $m_in, variance = $v_in.
        """)
    elseif ζ_hat2 <= 0
        throw(
            DomainError(
                ζ_hat2,
                "Cannot compute sqrt of non-positive ζ² = $ζ_hat2 (mean = $m_in, variance = $v_in)",
            ),
        )
    end

    return PointMass(sqrt(ζ_hat2))
end
@rule Sigmoid(:ζ, Marginalisation) (
    q_out::PointMass,
    q_in::UnivariateNormalDistributionsFamily,
) = begin
    m_in, v_in = mean_var(q_in)

    # 1) Basic sanity: finite mean/variance
    if !isfinite(m_in) || !isfinite(v_in)
        error("""
        Degenerate or invalid Gaussian in Sigmoid(:ζ, Marginalisation):
        mean = $m_in, variance = $v_in. This should not happen.
        """)
    end

    # 2) Variance must be strictly positive in this model
    if v_in < 0.0
        throw(
            DomainError(
                v_in,
                "Non-positive variance in Sigmoid(:ζ, Marginalisation): variance = $v_in, mean = $m_in",
            ),
        )
    end

    ζ_hat2 = m_in^2 + v_in

    # 3) ζ² must be finite and positive
    if !isfinite(ζ_hat2)
        error("""
        Invalid ζ² in Sigmoid(:ζ, Marginalisation):
        ζ² = $ζ_hat2 from mean = $m_in, variance = $v_in.
        """)
    elseif ζ_hat2 <= 0
        throw(
            DomainError(
                ζ_hat2,
                "Cannot compute sqrt of non-positive ζ² = $ζ_hat2 (mean = $m_in, variance = $v_in)",
            ),
        )
    end

    return PointMass(sqrt(ζ_hat2))
end
