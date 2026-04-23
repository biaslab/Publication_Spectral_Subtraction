#!/usr/bin/env julia
"""
plot_parameter_evolution.jl — regenerate the paper's parameter-evolution
figures from the public repo.

Runs the SEM pipeline (`SEMHearingAid.toml`) on a fixed list of representative
utterances from VoiceBank+DEMAND and, for each (utterance, band) pair,
plots the posterior trajectories that populate the paper's
`\\label{fig:parameter-evolution}` figure:

  (a) latent log speech power \$s\$ (blue) and log noise power \$n\$ (red),
      each with a ±1 σ ribbon (dB SPL),
  (b) log-SNR \$\\xi\$ with a ±1 σ ribbon,
  (c) spectral filter coefficient \$\\mathbb{E}[\\tilde w_m]\$.

Output files are written under `figures/`:

    figures/parameter_evolution_<noise>_<snr>dB_band<K>.{pdf,png,jpeg}

The paper currently \\includegraphics the 7.5 dB / band 13 case, but the
script emits the full (utterance × band) grid so you can pick the best
illustrative combination.

Usage:
    julia --project=. scripts/plot_parameter_evolution.jl

Prerequisites:
  - `databases/VOICEBANK_DEMAND_resampled/noisy_testset_wav/` present
    (either from `run_paper_results.jl` stage 1, or symlinked from another
    checkout).
  - Python DNSMOS / pesq / pysepm need NOT be installed; this script skips
    metrics entirely and only runs inference + plotting.
"""

using Pkg
const PROJECT_DIR = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_DIR)

using Experiments
const VHA = Experiments.VirtualHearingAid

# ── The v1 backward-message override on the NormalMixture switch ────────────
# Duplicated verbatim from scripts/run_evaluation.jl so both entry points use
# the same uniform-Categorical backward message that the paper's results rely
# on.
using RxInfer
@rule NormalMixture{N}(:switch, Marginalisation) (q_out::Any, q_m::ManyOf{N,Any}, q_p::ManyOf{N,Any}) where {N} = begin
    return Categorical(0.5, 0.5)
end

import Experiments.VirtualHearingAid: process
import Experiments: create_hearing_aid_from_config
using RxInfer: mean, precision
using TOML, Plots
gr()

# ── Configuration ───────────────────────────────────────────────────────────

# One WFB block at 16 kHz is 32 samples = 2 ms, matching the SEM runtime.
const FRAME_DT_S = 32 / 16000
const SYSTEM     = "SEM"

const CONFIG_PATH = joinpath(PROJECT_DIR, "configurations", "$(SYSTEM)HearingAid", "$(SYSTEM)HearingAid.toml")
const NOISY_DIR   = joinpath(PROJECT_DIR, "databases", "VOICEBANK_DEMAND_resampled", "noisy_testset_wav")
const OUT_DIR     = joinpath(PROJECT_DIR, "figures")

# (filename, SNR-in-filename, human-readable SNR label)
#
# The paper figures exactly one case (p257_003, bus, 7.5 dB, band 13). Add
# more entries below to regenerate the full (utterance × band) grid the author
# used to pick the illustrative case.
const AUDIO_CASES = [
    ("p257_003.wav", "7p5dB", "7.5 dB SNR"),
    # ("p257_001.wav", "17p5dB", "17.5 dB SNR"),
    # ("p257_002.wav", "12p5dB", "12.5 dB SNR"),
    # ("p257_004.wav",  "2p5dB",  "2.5 dB SNR"),
]
const NOISE_LABEL = "bus"

# Bands to plot. The paper uses band 13 (fc ≈ 3.6 kHz at α=0.5); add
# [3, 5, 8, 11, 15] to also emit a low/mid/high grid.
const BANDS = [13]

# ── WFB center frequency (α-warped), for figure titles only ─────────────────

function wfb_center_frequency(k::Integer, N::Integer, alpha::Real, fs::Real)
    ω = π * k / N
    num = (1 - alpha^2) * sin(ω)
    den = (1 + alpha^2) * cos(ω) + 2alpha
    ω_orig = atan(num, den)
    ω_orig < 0 && (ω_orig += π)
    return ω_orig / (2π) * fs
end

# ── Per-band plotting ───────────────────────────────────────────────────────

function plot_band(results, band::Integer, stem::AbstractString, title_prefix::AbstractString)
    eng = results.inference_results[band]
    nframes = size(results.gains, 1)

    s_hist  = eng.history[:s]
    n_hist  = eng.history[:n]
    xi_hist = eng.history[:ξ]
    w_hist  = eng.history[:w]

    time_s = [(t - 1) * FRAME_DT_S for t in 1:nframes]

    m_s  = [mean(s_hist[t])  for t in 1:nframes]
    m_n  = [mean(n_hist[t])  for t in 1:nframes]
    m_xi = [mean(xi_hist[t]) for t in 1:nframes]

    σ_s  = [sqrt(1.0 / precision(s_hist[t]))  for t in 1:nframes]
    σ_n  = [sqrt(1.0 / precision(n_hist[t]))  for t in 1:nframes]
    σ_xi = [sqrt(1.0 / precision(xi_hist[t])) for t in 1:nframes]

    m_w = [w_hist[t].p[1] for t in 1:nframes]

    fc = round(Int, wfb_center_frequency(band, 17, 0.5, 16000.0))

    common = Dict(
        :linewidth      => 0.8,
        :legend         => :topright,
        :legendfontsize => 6,
        :tickfontsize   => 6,
        :guidefontsize  => 7,
        :grid           => true,
        :gridalpha      => 0.2,
    )

    p1 = plot(time_s, m_s; ribbon=σ_s, fillalpha=0.45, color=:blue,
        label="s", ylabel="dB SPL",
        title="$(title_prefix), band $band (fc ≈ $fc Hz)",
        titlefontsize=8, xformatter=_->"", common...)
    plot!(p1, time_s, m_n; ribbon=σ_n, fillalpha=0.45, color=:red, label="n")

    p2 = plot(time_s, m_xi; ribbon=σ_xi, fillalpha=0.45, color=:green4,
        label="ξ", ylabel="dB SPL", xformatter=_->"", common...)
    hline!(p2, [0.0]; color=:gray, ls=:dash, lw=0.5, label="0 dB")

    p3 = plot(time_s, m_w; color=:purple, label="E[w̃ₘ]",
        ylabel="Gain", xlabel="Time (s)", ylims=(0, 1.05), common...)
    hline!(p3, [0.5]; color=:gray, ls=:dash, lw=0.5, label="σ(0)=0.5")

    fig = plot(p1, p2, p3; layout=(3, 1), size=(800, 560), margin=3Plots.mm)

    mkpath(OUT_DIR)
    out_pdf  = joinpath(OUT_DIR, "$(stem).pdf")
    out_png  = joinpath(OUT_DIR, "$(stem).png")
    out_jpeg = joinpath(OUT_DIR, "$(stem).jpeg")

    savefig(fig, out_pdf)
    savefig(fig, out_png)
    # macOS ships `sips`, which is the cheapest pdf→jpeg converter available
    # without pulling in ImageMagick. Silently fall through on non-macOS.
    try
        run(`sips -s format jpeg $out_pdf --out $out_jpeg`)
    catch e
        @warn "sips conversion failed for $stem; pdf and png are still available" exception=e
    end

    return (; stem, nframes,
            σ_s_mean  = sum(σ_s)  / length(σ_s),
            σ_n_mean  = sum(σ_n)  / length(σ_n),
            σ_xi_mean = sum(σ_xi) / length(σ_xi))
end

# ── Driver ──────────────────────────────────────────────────────────────────

function main()
    isfile(CONFIG_PATH) || error("Expected SEM config at $CONFIG_PATH.")
    isdir(NOISY_DIR)    || error("Expected VoiceBank+DEMAND noisy testset at $NOISY_DIR.\nDid you run `run_paper_results.jl` Stage 1 (resampling)?")

    println("Loading config: $CONFIG_PATH")
    config = TOML.parsefile(CONFIG_PATH)

    stats = []
    for (audio_file, snr_stem, snr_label) in AUDIO_CASES
        audio_path = joinpath(NOISY_DIR, audio_file)
        isfile(audio_path) || (@warn "Missing audio, skipping" path=audio_path; continue)

        println("\nProcessing $audio_file ($snr_label)")
        ha = create_hearing_aid_from_config(config)
        noisy_audio = Experiments.load_audio_file(audio_path)
        _, results = process(ha, noisy_audio)

        utt_stem = splitext(audio_file)[1]
        title_prefix = "$(utt_stem), $(NOISE_LABEL) noise, $(snr_label)"

        for band in BANDS
            stem = "parameter_evolution_$(NOISE_LABEL)_$(snr_stem)_band$(band)"
            r = plot_band(results, band, stem, title_prefix)
            push!(stats, (snr_label = snr_label, band = band, r...))
            println("  → band $band saved: $(r.stem)  " *
                    "(σ̄ₛ=$(round(r.σ_s_mean; digits=3))  " *
                    "σ̄ₙ=$(round(r.σ_n_mean; digits=3))  " *
                    "σ̄ξ=$(round(r.σ_xi_mean; digits=3)))")
        end
    end

    println("\n=== Summary (σ means in dB) ===")
    for s in stats
        println("  $(s.snr_label)  band $(lpad(s.band, 2))  ",
                "σ̄ₛ=$(round(s.σ_s_mean; digits=3))  ",
                "σ̄ₙ=$(round(s.σ_n_mean; digits=3))  ",
                "σ̄ξ=$(round(s.σ_xi_mean; digits=3))")
    end
    println("\nAll figures saved under $OUT_DIR/")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
