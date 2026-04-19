#!/usr/bin/env julia
"""
Process a single audio file through the SEM pipeline and plot posterior
trajectories for one frequency band.

Usage (from the codebase root):
    julia --project ../../FIGURES/data/plot_parameter_evolution.jl

Output:
    FIGURES/parameter_evolution.pdf
"""

using Pkg
# Activate the codebase project
const CODEBASE = joinpath(@__DIR__, "..", "..", "external",
    "Marco-2025-A-Probabilistic-Generative-Model-for-Spectral-Speech-Enhancement_Codebase")
Pkg.activate(CODEBASE)

using Experiments
const VHA = Experiments.VirtualHearingAid

# Load SEM rules
Experiments.HASoundProcessing.SEM.eval(
    :(include($(joinpath(pkgdir(Experiments), "src", "HASoundProcessing", "backends", "sem", "rules.jl")))),
)

import Experiments.VirtualHearingAid: from_config, process
using Experiments.VirtualHearingAid.Frontends.WFB: get_nbands
using RxInfer: mean, precision
using TOML, Plots
gr()

# ── Configuration ───────────────────────────────────────────────────────────
const BAND       = 5
const FRAME_DT_S = 32 / 16000  # 32 samples at 16 kHz = 2 ms per frame
const SYSTEM     = "SEM"       # θ=0, β=0 (canonical MMSE Wiener)

# Paths
const CONFIG_PATH = joinpath(CODEBASE, "configurations", "$(SYSTEM)HearingAid", "$(SYSTEM)HearingAid.toml")
const AUDIO_PATH  = joinpath(CODEBASE, "databases", "VOICEBANK_DEMAND_resampled",
                             "noisy_testset_wav", "p257_003.wav")
const OUT_PATH    = joinpath(@__DIR__, "..", "parameter_evolution.pdf")

# WFB center frequency
function wfb_center_frequency(k, N, alpha, fs)
    ω = π * k / N
    num = (1 - alpha^2) * sin(ω)
    den = (1 + alpha^2) * cos(ω) + 2alpha
    ω_orig = atan(num, den)
    ω_orig < 0 && (ω_orig += π)
    return ω_orig / (2π) * fs
end
const FC = round(Int, wfb_center_frequency(BAND, 17, 0.5, 16000.0))

# ── Process single file ────────────────────────────────────────────────────
println("Loading config: $SYSTEM")
config = TOML.parsefile(CONFIG_PATH)
ha = from_config(config)

println("Processing: p257_003.wav (7.5 dB SNR, bus noise)")
noisy_audio = Experiments.load_audio_file(AUDIO_PATH)
output, results = process(ha, noisy_audio)

# ── Extract posteriors for BAND ─────────────────────────────────────────────
eng = results.inference_results[BAND]
nframes = size(results.gains, 1)

s_hist  = eng.history[:s]
n_hist  = eng.history[:n]
xi_hist = eng.history[:ξ]
pi_hist = eng.history[:π_switch]
w_hist  = eng.history[:w]

time_ms = [(t - 1) * FRAME_DT_S * 1000 for t in 1:nframes]

m_s  = [mean(s_hist[t])  for t in 1:nframes]
m_n  = [mean(n_hist[t])  for t in 1:nframes]
m_xi = [mean(xi_hist[t]) for t in 1:nframes]
σ_s  = [sqrt(1.0 / precision(s_hist[t]))  for t in 1:nframes]
σ_n  = [sqrt(1.0 / precision(n_hist[t]))  for t in 1:nframes]
σ_xi = [sqrt(1.0 / precision(xi_hist[t])) for t in 1:nframes]
m_pi = [pi_hist[t].p[1] for t in 1:nframes]  # p[1] = σ(ξ-κ) = speech-presence probability
m_w  = [w_hist[t].p[1] for t in 1:nframes]   # p[1] = σ(ξ-θ) = posterior spectral gain

println("Extracted $nframes frames for band $BAND (fc ≈ $FC Hz)")

# ── Plot ────────────────────────────────────────────────────────────────────
common = Dict(
    :linewidth  => 0.8,
    :legend     => :topright,
    :legendfontsize => 6,
    :tickfontsize   => 6,
    :guidefontsize  => 7,
    :grid       => true,
    :gridalpha  => 0.2,
)

# (a) Speech and noise power
p1 = plot(time_ms, m_s; ribbon=σ_s, fillalpha=0.25, color=:blue,
    label="E[sₘ] ± σ", ylabel="dB SPL",
    title="p257_003, bus noise, 7.5 dB SNR, band $BAND (fc ≈ $FC Hz)",
    titlefontsize=8, xformatter=_->"", common...)
plot!(p1, time_ms, m_n; ribbon=σ_n, fillalpha=0.25, color=:red,
    label="E[nₘ] ± σ")

# (b) Log-SNR
p2 = plot(time_ms, m_xi; ribbon=σ_xi, fillalpha=0.2, color=:green4,
    label="E[ξₘ] ± σ", ylabel="ξ̂ₘ (dB)",
    xformatter=_->"", common...)
hline!(p2, [0.0]; color=:gray, ls=:dash, lw=0.5, label="0 dB")

# (c) VAD probability
p3 = plot(time_ms, m_pi; color=:orange, label="E[πₘ]",
    ylabel="E[πₘ]", ylims=(0, 1.05),
    xformatter=_->"", common...)
hline!(p3, [0.5]; color=:gray, ls=:dash, lw=0.5, label="σ(κ)=0.5")

# (d) Spectral gain
p4 = plot(time_ms, m_w; color=:purple, label="E[w̃ₘ]",
    ylabel="Gain", xlabel="Time (ms)", ylims=(0, 1.05), common...)
hline!(p4, [0.5]; color=:gray, ls=:dash, lw=0.5, label="σ(0)=0.5")

fig = plot(p1, p2, p3, p4; layout=(4, 1), size=(800, 700), margin=3Plots.mm)
savefig(fig, OUT_PATH)
println("Saved to $OUT_PATH")
