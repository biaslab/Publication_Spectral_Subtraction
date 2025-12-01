using StatsFuns: logistic
@rule Sigmoid(:out, Marginalisation) (
    q_in::UnivariateNormalDistributionsFamily,
    q_ζ::Any,
) = begin
    m_in = mean(q_in)
    p = logistic(m_in)
    probs = clamp.([p, 1 - p], tiny, 1 - tiny)
    probs ./= sum(probs)
    return Categorical(probs)
end
