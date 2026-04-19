#!/usr/bin/env julia
# Run SEM inference on selected noisy files and plot the per-frame posterior
# trace for the chosen WFB band as JPEGs. Produces individual per-case plots
# and a side-by-side high-SNR vs low-SNR comparison (same environment) to
# visualise speech-masking and gain flicker.

using Pkg
const CODEBASE = joinpath(@__DIR__, "..", "..", "external",
    "Marco-2025-A-Probabilistic-Generative-Model-for-Spectral-Speech-Enhancement_Codebase")
Pkg.activate(CODEBASE)

using Experiments
const VHA = Experiments.VirtualHearingAid

# Override the NormalMixture→switch backward message with a uniform
# Categorical([0.5, 0.5]); without this the VAD posterior is biased and the
# traces don't match the paper results.
Experiments.HASoundProcessing.SEM.eval(
    :(include($(joinpath(pkgdir(Experiments), "src", "HASoundProcessing", "backends", "sem", "rules.jl")))),
)

using TOML, Plots
import Experiments.VirtualHearingAid: from_config, process, get_frontend
using Experiments.VirtualHearingAid.Frontends.WFB: get_nbands
using RxInfer: mean, precision
using SampledSignals: samplerate, SampleBuf
const WFBmod = Experiments.VirtualHearingAid.Frontends.WFB
gr()

"""
    wfb_per_band_logpow(audio, ref_frontend, band)

Pass `audio` through a fresh WFB front-end whose parameters match `ref_frontend`,
returning the per-frame log-power for the requested `band` (in dB SPL units the
SEM observes). No BLI smoothing, no inference --- just deterministic spectral
analysis.
"""
function wfb_per_band_logpow(audio, ref_frontend, band::Int)
    p = WFBmod.get_params(ref_frontend)
    # Use the SAME dB floor as the SEM so the reference and the inferred
    # posterior live on identical dynamic range and scale.
    fe = WFBmod.WFBFrontend(p)
    bsz = p.buffer_size
    samples = vec(audio)
    nframes = div(length(samples), bsz)
    out = zeros(Float64, nframes)
    for k in 1:nframes
        blk = SampleBuf(Vector{Float64}(samples[(k-1)*bsz+1 : k*bsz]), Float64(p.fs))
        db_per_band = WFBmod.process_frontend(fe, blk)
        out[k] = db_per_band[band]
    end
    return out
end

# --- Global config ---
const BAND = 13
const Z    = 1.0

# (filename, noise label, SNR dB, output file tag)
const CASES_FULL = [
    # Stationary noise (bus engine, office HVAC)
    ("p257_004.wav", "bus",    2.5,  "bus_2p5dB"),       # stationary, VERY LOW SNR
    ("p257_003.wav", "bus",    7.5,  "bus_7p5dB"),       # stationary, moderate SNR (reference)
    ("p257_001.wav", "bus",    17.5, "bus_17p5dB"),      # stationary, HIGH SNR - θ damage max
    ("p257_017.wav", "office", 2.5,  "office_2p5dB"),    # stationary, low SNR
    ("p257_014.wav", "office", 17.5, "office_17p5dB"),   # stationary, high SNR
    # Non-stationary noise (cafe babble, living TV, psquare crowd)
    ("p257_009.wav", "cafe",   2.5,  "cafe_2p5dB"),      # non-stationary babble, low SNR
    ("p257_008.wav", "cafe",   7.5,  "cafe_7p5dB"),      # non-stationary babble, moderate
    ("p257_006.wav", "cafe",   17.5, "cafe_17p5dB"),     # non-stationary babble, high SNR
    ("p257_013.wav", "living", 2.5,  "living_2p5dB"),    # TV + speech, low SNR
    ("p257_022.wav", "psquare", 2.5, "psquare_2p5dB"),   # crowd, low SNR
]
# Set ONLY_TAG=<tag> in the env to regenerate a single case quickly.
const _ONLY = get(ENV, "ONLY_TAG", "")
const CASES = isempty(_ONLY) ? CASES_FULL :
    filter(c -> c[4] == _ONLY, CASES_FULL)

# Side-by-side comparison: same environment, high vs low SNR.
const SBS_LEFT  = "office_17p5dB"   # high SNR  → speech clearly above threshold
const SBS_RIGHT = "office_2p5dB"    # low SNR   → speech pulled down near threshold
const SBS_TAG   = "office_highVlow_SNR"

const RESULTS_CSV = joinpath(CODEBASE, "results", "VOICEBANK_DEMAND",
    "run_20260417_215956_factorial_full", "results.csv")

function fetch_scores(filename::String, system::String)
    isfile(RESULTS_CSV) || return nothing
    for line in eachline(RESULTS_CSV)
        fields = split(line, ',')
        length(fields) >= 15 || continue
        if fields[1] == filename && fields[4] == system
            try
                return (
                    PESQ = parse(Float64, fields[5]),
                    CSIG = parse(Float64, fields[13]),
                    CBAK = parse(Float64, fields[14]),
                    COVL = parse(Float64, fields[15]),
                )
            catch
                return nothing
            end
        end
    end
    return nothing
end

fmt_score(x) = x === nothing ? "—" : string(round(x, digits=2))
score_line(label, s) = s === nothing ? "$label: n/a" :
    "$label: PESQ=$(fmt_score(s.PESQ))  CSIG=$(fmt_score(s.CSIG))  CBAK=$(fmt_score(s.CBAK))  COVL=$(fmt_score(s.COVL))"

function wfb_center_frequency(k::Int, N::Int, alpha::Float64, fs::Float64)
    omega_warped = π * k / N
    num = (1 - alpha^2) * sin(omega_warped)
    den = (1 + alpha^2) * cos(omega_warped) + 2 * alpha
    omega_orig = atan(num, den)
    omega_orig < 0 && (omega_orig += π)
    return omega_orig / (2π) * fs
end

# --- Build the SEM pipeline once (shared across cases) ---
config_path = joinpath(CODEBASE, "configurations", "SEM_litHearingAid", "SEM_litHearingAid.toml")
isfile(config_path) || error("Config not found: $config_path")
config  = TOML.parsefile(config_path)
nbands  = get_nbands(get_frontend(from_config(config)))
θ       = Float64(config["parameters"]["backend"]["gain"]["threshold_dB"])
1 <= BAND <= nbands || error("BAND $BAND out of range 1:$nbands")

fc = round(Int, wfb_center_frequency(BAND, nbands, 0.5, 16000.0))
println("Band $BAND center frequency: $fc Hz; θ = $θ dB")

# Run inference on one file and return the extracted traces + scores + title.
function collect_traces(filename::String, noise_label::String, snr_db::Float64, tag::String)
    ha         = from_config(config)
    noisy_path = joinpath(CODEBASE, "databases", "VOICEBANK_DEMAND_resampled",
                         "noisy_testset_wav", filename)
    isfile(noisy_path) || error("Noisy file not found: $noisy_path")

    println("\n[$tag] $filename — $noise_label @ $(snr_db) dB SNR")
    noisy_audio = Experiments.load_audio_file(noisy_path)
    _, results  = process(ha, noisy_audio)

    eng     = results.inference_results[BAND]
    s_hist  = eng.history[:s]
    n_hist  = eng.history[:n]
    xi_hist = eng.history[:ξ]
    w_hist  = eng.history[:w]
    nframes = length(s_hist)
    frame_dt_ms = (length(noisy_audio) / samplerate(noisy_audio) * 1000.0) / nframes

    time_s = [(t - 1) * frame_dt_ms / 1000.0 for t in 1:nframes]
    return (
        filename    = filename,
        noise_label = noise_label,
        snr_db      = snr_db,
        tag         = tag,
        time_s      = time_s,
        m_s         = [mean(s_hist[t])  for t in 1:nframes],
        m_n         = [mean(n_hist[t])  for t in 1:nframes],
        m_xi        = [mean(xi_hist[t]) for t in 1:nframes],
        σ_s         = [sqrt(1.0 / precision(s_hist[t]))  for t in 1:nframes],
        σ_n         = [sqrt(1.0 / precision(n_hist[t]))  for t in 1:nframes],
        σ_xi        = [sqrt(1.0 / precision(xi_hist[t])) for t in 1:nframes],
        m_w         = [w_hist[t].p[1]  for t in 1:nframes],
        s_aida2     = fetch_scores(filename, "SEM_lit"),
        s_theta     = fetch_scores(filename, "AIDA2_thetaWFB"),
        s_unp       = fetch_scores(filename, "Unprocessed"),
    )
end

# Spectral floor value used in the manuscript (matches SEM_lit config).
const BETA = 0.25

function build_case_column(d; common, title=nothing, show_ylabels=true)
    fname_stem = replace(d.filename, ".wav" => "")
    ttl = title !== nothing ? title :
          "$fname_stem, $(d.noise_label) noise, $(d.snr_db) dB SNR, band $BAND (fc ≈ $fc Hz)"

    yl_s  = show_ylabels ? "dB SPL" : ""
    yl_xi = show_ylabels ? "dB SPL" : ""
    yl_w  = show_ylabels ? "Gain"   : ""

    p1 = plot(d.time_s, d.m_s; ribbon=Z .* d.σ_s, fillalpha=0.25, color=:blue,
        label="s", ylabel=yl_s, title=ttl, titlefontsize=12,
        xformatter=_->"", common...)
    plot!(p1, d.time_s, d.m_n; ribbon=Z .* d.σ_n, fillalpha=0.25, color=:red,
        label="n")

    p2 = plot(d.time_s, d.m_xi; ribbon=Z .* d.σ_xi, fillalpha=0.2, color=:green4,
        label="ξ", ylabel=yl_xi, xformatter=_->"", common...)
    plot!(p2, d.time_s, d.m_xi .- θ; color=:darkorange, linewidth=1.2,
        label="ξ − θ")

    # Row 3: expected gain w̃ (unfloored) and floored gain max(w̃, β).
    m_w_floored = max.(d.m_w, BETA)
    p4 = plot(d.time_s, d.m_w; color=:purple, linewidth=0.9,
        label="expected w̃ (no floor)",
        ylabel=yl_w, xlabel="Time (s)", ylims=(0, 1.05), common...)
    plot!(p4, d.time_s, m_w_floored;
        fillrange=d.m_w, fillcolor=:teal, fillalpha=0.35,
        color=:teal, linewidth=1.6,
        label="expected w̃ with spectral floor")
    hline!(p4, [BETA]; color=:gray, ls=:dash, lw=0.8,
        label="spectral floor = 0.25")

    return (p1, p2, p4)
end

function save_single(d)
    common = (linewidth=1.2, legend=:topright, legendfontsize=10,
              tickfontsize=10, guidefontsize=11, grid=true, gridalpha=0.2)
    println("  ", score_line("AIDA-2     ", d.s_aida2))
    println("  ", score_line("θ-only     ", d.s_theta))
    println("  ", score_line("Unprocessed", d.s_unp))
    (p1, p2, p4) = build_case_column(d; common=common)
    fig = plot(p1, p2, p4; layout=(3, 1), size=(900, 680), margin=4Plots.mm,
               top_margin=6Plots.mm)

    out_jpeg = joinpath(@__DIR__, "..", "parameter_evolution_$(d.tag).jpeg")
    png_tmp  = joinpath(@__DIR__, "..", "parameter_evolution_$(d.tag).png")
    savefig(fig, png_tmp)
    run(`sips -s format jpeg $png_tmp --out $out_jpeg`)
    rm(png_tmp; force=true)
    println("  Saved to $out_jpeg")
end

function save_side_by_side(d_left, d_right, tag)
    common = (linewidth=1.1, legend=:topright, legendfontsize=9,
              tickfontsize=9, guidefontsize=10, grid=true, gridalpha=0.2)

    (p1L, p2L, p4L) = build_case_column(d_left;  common=common, show_ylabels=true)
    (p1R, p2R, p4R) = build_case_column(d_right; common=common, show_ylabels=false)

    fig = plot(p1L, p1R, p2L, p2R, p4L, p4R;
        layout=(3, 2), size=(1500, 720), margin=3Plots.mm, top_margin=7Plots.mm)

    out_jpeg = joinpath(@__DIR__, "..", "parameter_evolution_$(tag).jpeg")
    png_tmp  = joinpath(@__DIR__, "..", "parameter_evolution_$(tag).png")
    savefig(fig, png_tmp)
    run(`sips -s format jpeg $png_tmp --out $out_jpeg`)
    rm(png_tmp; force=true)
    println("\n[SIDE BY SIDE] Saved to $out_jpeg")
end

# Run all cases, collect traces, save individual plots.
traces = Dict{String,Any}()
for (filename, noise_label, snr_db, tag) in CASES
    d = collect_traces(filename, noise_label, snr_db, tag)
    traces[tag] = d
    save_single(d)
end

# Build the side-by-side high-vs-low SNR comparison.
if haskey(traces, SBS_LEFT) && haskey(traces, SBS_RIGHT)
    save_side_by_side(traces[SBS_LEFT], traces[SBS_RIGHT], SBS_TAG)
end
