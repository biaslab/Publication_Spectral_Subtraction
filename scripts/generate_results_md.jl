#!/usr/bin/env julia
"""
generate_results_md.jl — produce `RESULTS.md` at the repo root.

`RESULTS.md` is a one-file snapshot of the paper's headline results, rendered
in GitHub-flavoured markdown so reviewers can read it directly in the
browser without compiling LaTeX.

Contents:
  • the parameter-evolution figure (embedded from `figures/`),
  • Table 3 — overall comparison (PESQ / CSIG / CBAK / COVL, unprocessed vs SEM),
  • Table 4 — WFB ablation (SEM WFB vs uFB, mean ± std over 824 files),
  • Table 5 — per-environment × per-SNR ablation
              (5 environments × 4 SNRs × 3 systems, one sub-table per metric),
  • Table 6 — per-environment improvement summary
              (Δ_U→W vs Δ_u→W, averaged across the four SNRs).

Reads the same CSVs as `scripts/generate_latex_tables.jl` (i.e. the most recent
`run_*/table/results.csv` under each of baseline_noise, SEMHearingAid,
SEMHearingAid_uFB) and writes `RESULTS.md` at the repo root. The figure link
points at `figures/parameter_evolution_bus_7p5dB_band13.png`; regenerate it
with `scripts/plot_parameter_evolution.jl` if missing.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Printf
using Statistics

const PROJECT_DIR = abspath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(PROJECT_DIR, "results", "VOICEBANK_DEMAND")
const OUT_PATH    = joinpath(PROJECT_DIR, "RESULTS.md")
const FIG_REL     = "figures/parameter_evolution_bus_7p5dB_band13.png"

const SYSTEMS = [
    (dir = "baseline_noise",      label = "Unprocessed"),
    (dir = "SEMHearingAid_uFB",   label = "SEM (uFB)"),
    (dir = "SEMHearingAid",       label = "SEM (WFB)"),
]
const ENVIRONMENTS = ["bus", "cafe", "living", "psquare", "office"]
const SNRS         = [2.5, 7.5, 12.5, 17.5]
const METRICS      = ["PESQ", "CSIG", "CBAK", "COVL"]

# ----------------------------------------------------------- input loading ---

function latest_results_path(system_dir::AbstractString)
    parent = joinpath(RESULTS_DIR, system_dir)
    isdir(parent) || error("Expected $parent — did `run_paper_results.jl` run?")
    runs = [joinpath(parent, r) for r in readdir(parent) if startswith(r, "run_")]
    isempty(runs) && error("No run_* directory inside $parent")
    sort!(runs; by = mtime, rev = true)
    csv = joinpath(runs[1], "table", "results.csv")
    isfile(csv) || error("Missing results.csv inside $(runs[1])")
    return csv
end

function load_runs()
    dfs = Dict{String,DataFrame}()
    for sys in SYSTEMS
        df = CSV.read(latest_results_path(sys.dir), DataFrame)
        df.noise_type = lowercase.(string.(df.noise_type))
        df.snr_db = Float64.(df.snr_db)
        dfs[sys.label] = df
    end
    return dfs
end

# --------------------------------------------------------- stats helpers -----

function cell_stats(df::DataFrame, env::AbstractString, snr::Real, metric::AbstractString)
    mask = (df.noise_type .== env) .& (df.snr_db .== snr)
    col = Symbol(metric)
    values = collect(skipmissing(df[mask, col]))
    isempty(values) && return (NaN, NaN)
    σ = length(values) ≥ 2 ? std(values) : 0.0
    return (mean(values), σ)
end

function overall_stats(df::DataFrame, metric::AbstractString)
    values = collect(skipmissing(df[!, Symbol(metric)]))
    isempty(values) && return (NaN, NaN)
    σ = length(values) ≥ 2 ? std(values) : 0.0
    return (mean(values), σ)
end

function env_avg(df::DataFrame, env::AbstractString, metric::AbstractString)
    # Average the per-SNR means for this environment. Matches the paper's
    # per-environment delta definition.
    vals = [cell_stats(df, env, snr, metric)[1] for snr in SNRS]
    vals = filter(isfinite, vals)
    isempty(vals) && return NaN
    return mean(vals)
end

fmt2(x::Real) = isfinite(x) ? @sprintf("%.2f", x) : "—"
fmt_meanstd(μ::Real, σ::Real) = isfinite(μ) && isfinite(σ) ? "$(fmt2(μ)) ± $(fmt2(σ))" : "—"
fmt_delta(x::Real) = isfinite(x) ? (x ≥ 0 ? "+" : "−") * @sprintf("%.2f", abs(x)) : "—"

# ------------------------------------------------------- table emitters ------

function table3(dfs)
    # One-row-per-system comparison, mean ± std.
    io = IOBuffer()
    println(io, "### Table 3 — Overall comparison")
    println(io)
    println(io, "Sample mean ± one sample standard deviation over the 824-file test set.")
    println(io)
    println(io, "| System | PESQ | CSIG | CBAK | COVL |")
    println(io, "|---|---|---|---|---|")
    for sys in SYSTEMS
        stats = [overall_stats(dfs[sys.label], m) for m in METRICS]
        cells = [fmt_meanstd(μ, σ) for (μ, σ) in stats]
        println(io, "| ", sys.label, " | ", cells[1], " | ", cells[2], " | ", cells[3], " | ", cells[4], " |")
    end
    return String(take!(io))
end

function table4(dfs)
    # WFB ablation — SEM (WFB) vs SEM (uFB), plus Δ.
    io = IOBuffer()
    println(io, "### Table 4 — WFB ablation")
    println(io)
    println(io, "SEM with the warped filter bank (α = 0.5) vs. the ablated uniform filter bank (α = 0), all other parameters unchanged. Sample mean ± one sample standard deviation over the 824-file test set. **Δ** is the per-metric difference WFB − uFB.")
    println(io)
    println(io, "| System | PESQ | CSIG | CBAK | COVL |")
    println(io, "|---|---|---|---|---|")

    u_stats = [overall_stats(dfs["SEM (uFB)"], m) for m in METRICS]
    w_stats = [overall_stats(dfs["SEM (WFB)"], m) for m in METRICS]
    u_cells = [fmt_meanstd(μ, σ) for (μ, σ) in u_stats]
    w_cells = [fmt_meanstd(μ, σ) for (μ, σ) in w_stats]
    deltas  = [fmt_delta(w_stats[i][1] - u_stats[i][1]) for i in eachindex(METRICS)]

    println(io, "| SEM (uFB, α = 0)   | ", u_cells[1], " | ", u_cells[2], " | ", u_cells[3], " | ", u_cells[4], " |")
    println(io, "| SEM (WFB, α = 0.5) | ", w_cells[1], " | ", w_cells[2], " | ", w_cells[3], " | ", w_cells[4], " |")
    println(io, "| **Δ (WFB − uFB)**  | ", deltas[1], " | ", deltas[2], " | ", deltas[3], " | ", deltas[4], " |")
    return String(take!(io))
end

function table5(dfs)
    # Per-environment × per-SNR quadrants. One sub-table per metric.
    io = IOBuffer()
    println(io, "### Table 5 — Per-environment × per-SNR ablation (mean ± std)")
    println(io)
    println(io, "For each of PESQ / CSIG / CBAK / COVL, the mean ± one sample standard deviation over the files assigned to each (environment, SNR) cell, for the three systems.")

    for metric in METRICS
        println(io)
        println(io, "#### ", metric)
        println(io)
        println(io, "| Env | System | 2.5 dB | 7.5 dB | 12.5 dB | 17.5 dB |")
        println(io, "|---|---|---|---|---|---|")
        for env in ENVIRONMENTS
            first_row = true
            for sys in SYSTEMS
                cells = [fmt_meanstd(cell_stats(dfs[sys.label], env, snr, metric)...) for snr in SNRS]
                env_cell = first_row ? uppercase(env) : ""
                first_row = false
                println(io, "| ", env_cell, " | ", sys.label, " | ",
                        cells[1], " | ", cells[2], " | ", cells[3], " | ", cells[4], " |")
            end
        end
    end
    return String(take!(io))
end

function table6(dfs)
    io = IOBuffer()
    println(io, "### Table 6 — Per-environment improvement (averaged across SNRs)")
    println(io)
    println(io, "**Δ_{U→W}** = SEM (WFB) − Unprocessed; **Δ_{u→W}** = SEM (WFB) − SEM (uFB) (WFB ablation). Each value is the mean over the four input SNRs.")
    println(io)
    println(io, "| Env | Δ_{U→W} PESQ | Δ_{U→W} CSIG | Δ_{U→W} CBAK | Δ_{U→W} COVL | Δ_{u→W} PESQ | Δ_{u→W} CSIG | Δ_{u→W} CBAK | Δ_{u→W} COVL |")
    println(io, "|---|---|---|---|---|---|---|---|---|")
    for env in ENVIRONMENTS
        dU = [env_avg(dfs["SEM (WFB)"], env, m) - env_avg(dfs["Unprocessed"], env, m) for m in METRICS]
        du = [env_avg(dfs["SEM (WFB)"], env, m) - env_avg(dfs["SEM (uFB)"], env, m) for m in METRICS]
        println(io, "| ", uppercase(env), " | ",
                fmt_delta(dU[1]), " | ", fmt_delta(dU[2]), " | ", fmt_delta(dU[3]), " | ", fmt_delta(dU[4]), " | ",
                fmt_delta(du[1]), " | ", fmt_delta(du[2]), " | ", fmt_delta(du[3]), " | ", fmt_delta(du[4]), " |")
    end
    return String(take!(io))
end

# --------------------------------------------------------- driver ------------

function main()
    dfs = load_runs()

    header = """
# Paper Results — VoiceBank+DEMAND

Auto-generated by `scripts/generate_results_md.jl` from the latest evaluation runs under `results/VOICEBANK_DEMAND/`. Regenerate with:

```bash
julia --project=. scripts/generate_results_md.jl
```

This is the reviewer-facing companion to the paper's Results section. Every value below comes from the same runs that populate the LaTeX tables under `tables/` (the `generate_latex_tables.jl` script emits both).
"""

    fig_block = """
## Parameter-evolution figure

Posterior trajectories during SEM inference for a representative utterance (`p257_003`, bus noise, 7.5 dB input SNR), frequency band 13 (centre frequency ≈ 3.6 kHz). From top to bottom: (a) latent log speech power *s* and log noise power *n* in dB SPL, each with a ±1σ ribbon; (b) log-SNR *ξ* with a ±1σ ribbon; (c) spectral filter coefficient 𝔼[*w̃ₘ*].

Regenerate with:

```bash
julia --project=. scripts/plot_parameter_evolution.jl
```

![Parameter evolution for p257_003, bus noise, 7.5 dB SNR, band 13]($(FIG_REL))
"""

    sections = [
        header,
        fig_block,
        "## Tables\n\n" * table3(dfs),
        table4(dfs),
        table5(dfs),
        table6(dfs),
    ]

    contents = join(sections, "\n\n")
    open(OUT_PATH, "w") do io
        write(io, contents)
    end

    println("Wrote ", relpath(OUT_PATH, PROJECT_DIR))
    println("Open with: open ", OUT_PATH)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
