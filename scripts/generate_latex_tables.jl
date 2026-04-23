#!/usr/bin/env julia
"""
generate_latex_tables.jl — populate the paper's LaTeX tables directly from the
most recent evaluation runs.

Reads per-file scores from the latest run of each of three systems:
  • baseline_noise         → "Unprocessed"   (clean vs. noisy, no processing)
  • SEMHearingAid_uFB      → "SEM (uFB)"     (uniform filter bank, alpha = 0.0)
  • SEMHearingAid          → "SEM (WFB)"     (warped filter bank, alpha = 0.5)

and emits three LaTeX files under `tables/`:

  tables/tab_comparison_with_params.tex
      One-row-per-system comparison with PESQ / CSIG / CBAK / COVL means and,
      for the SEM row, a second line with the ±std spread. Populates the
      `tab:comparison_with_params` table in the paper. Only our runs are
      written; published-baseline rows (Wiener, MAMBA-SENet, DSEGAN, CDiffuSE,
      WaveCRN, MOSE) remain quoted from their original publications and are
      therefore left to the paper author to insert by hand.

  tables/tab_metrics_quadrants.tex
      Full ablation with 4 sub-tables (PESQ, CSIG, CBAK, COVL), 5 environments
      (BUS, CAFE, LIVING, PSQUARE, OFFICE), 4 input SNRs (2.5, 7.5, 12.5, 17.5
      dB), 3 systems. Each cell is `mean ± std` over the files assigned to
      that (environment, SNR). Populates `tab:metrics-quadrants`.

  tables/tab_per_env_delta.tex
      Per-environment improvement table. For each of the 5 environments,
      reports two contrasts, averaged across the four input SNRs:
        Δ_{U→W} = SEM (WFB) − Unprocessed  (total perceptual benefit)
        Δ_{u→W} = SEM (WFB) − SEM (uFB)    (WFB ablation only)
      for PESQ, CSIG, CBAK, COVL. Populates `tab:per-env-delta`.

Usage:
    julia --project=. scripts/generate_latex_tables.jl

The script also prints each table to stdout so the output can be copy-pasted
straight into the paper if the `tables/` files are inconvenient.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Printf
using Statistics

const PROJECT_DIR = abspath(joinpath(@__DIR__, ".."))
const RESULTS_DIR = joinpath(PROJECT_DIR, "databases", "..", "results", "VOICEBANK_DEMAND")
const TABLES_DIR  = joinpath(PROJECT_DIR, "tables")

# Canonical ordering — matches the paper's tables.
const SYSTEMS = [
    (dir = "baseline_noise",      label = "Unprocessed"),
    (dir = "SEMHearingAid_uFB",   label = "SEM (uFB)"),
    (dir = "SEMHearingAid",       label = "SEM (WFB)"),
]
const ENVIRONMENTS = ["bus", "cafe", "living", "psquare", "office"]
const SNRS         = [2.5, 7.5, 12.5, 17.5]
const METRICS      = ["PESQ", "CSIG", "CBAK", "COVL"]

# Param counts shown in the comparison table. Keep in sync with the paper
# (currently Table `tab:comparison_with_params`).
const PARAM_COUNTS = Dict(
    "Unprocessed" => "N/A",
    "SEM (uFB)"   => raw"$\sim$85",
    "SEM (WFB)"   => raw"$\sim$85",
)

# ---------------------------------------------------------------- input loading

"""Latest `run_*/table/results.csv` directory for `system_dir` under `RESULTS_DIR`."""
function latest_results_path(system_dir::AbstractString)
    parent = joinpath(RESULTS_DIR, system_dir)
    isdir(parent) || error("Expected $parent — did `run_paper_results.jl` run?")
    runs = filter(x -> startswith(x, "run_"), readdir(parent))
    isempty(runs) && error("No run_* directory inside $parent")
    # Sort lexicographically; `run_DD_MM_YYYY_HH_MM` sorts by day-of-month first,
    # which is wrong across months — but within a single month it is fine.
    # For safety, pick the directory with the newest mtime.
    paths = [joinpath(parent, r) for r in runs]
    sort!(paths; by = mtime, rev = true)
    path = paths[1]
    csv = joinpath(path, "table", "results.csv")
    isfile(csv) || error("Missing results.csv inside $path")
    return csv
end

function load_runs()
    dfs = Dict{String,DataFrame}()
    for sys in SYSTEMS
        csv = latest_results_path(sys.dir)
        df = CSV.read(csv, DataFrame)
        # Normalise environment and SNR columns.
        df.noise_type = lowercase.(string.(df.noise_type))
        df.snr_db = Float64.(df.snr_db)
        dfs[sys.label] = df
    end
    return dfs
end

# ---------------------------------------------------------- summary statistics

"""(mean, std) of `metric` over the (environment, SNR) cell of `df`.

Returns `(NaN, NaN)` if the cell is empty; the caller emits an em-dash in
that case.
"""
function cell_stats(df::DataFrame, env::AbstractString, snr::Real, metric::AbstractString)
    mask = (df.noise_type .== env) .& (df.snr_db .== snr)
    col = Symbol(metric)
    values = skipmissing(df[mask, col])
    vals_vec = collect(values)
    if isempty(vals_vec)
        return (NaN, NaN)
    end
    μ = mean(vals_vec)
    σ = length(vals_vec) ≥ 2 ? std(vals_vec) : 0.0
    return (μ, σ)
end

"""Global mean (and, for the SEM rows, std) of `metric` across all files."""
function overall_stats(df::DataFrame, metric::AbstractString)
    col = Symbol(metric)
    values = collect(skipmissing(df[!, col]))
    isempty(values) && return (NaN, NaN)
    σ = length(values) ≥ 2 ? std(values) : 0.0
    return (mean(values), σ)
end

# --------------------------------------------------------------------- format

fmt2(x::Real) = isfinite(x) ? @sprintf("%.2f", x) : "--"

"""`mean ± std` cell content for the ablation table, e.g. `\$2.00\\,{\\pm}\\,0.56\$`."""
function cell_tex(μ::Real, σ::Real)
    if isfinite(μ) && isfinite(σ)
        return string("\$", fmt2(μ), raw"\,{\pm}\,", fmt2(σ), "\$")
    else
        return "--"
    end
end

# ----------------------------------------------------------- table 1 (comparison)

function write_comparison_table(dfs::Dict{String,DataFrame})
    io = IOBuffer()

    println(io, raw"% Auto-generated by scripts/generate_latex_tables.jl — do not hand-edit.")
    println(io, raw"% Regenerate with: julia --project=. scripts/generate_latex_tables.jl")
    println(io, raw"\begin{table}[t]")
    println(io, raw"\centering")
    println(io, raw"\caption{VoiceBank+DEMAND results. Best scores are bold.")
    println(io, raw"T--F = time--frequency domain, T = time domain.")
    println(io, raw"The Wiener gain is taken from the results reported in WaveCRN~\cite{hsieh_wavecrn_2020}.}")
    println(io, raw"\label{tab:comparison_with_params}")
    println(io, raw"\begin{tabular}{l@{\hskip 4pt}|l@{\hskip 3pt}|r@{\hskip 4pt}r@{\hskip 4pt}r@{\hskip 4pt}r@{\hskip 4pt}r}")
    println(io, raw"\hline")
    println(io, raw"System & Dom. & PESQ & \new{CSIG} & \new{CBAK} & \new{COVL} & Params \\")
    println(io, raw"\hline")

    # Our runs: Unprocessed, SEM (WFB). SEM (uFB) goes into the ablation table,
    # not this comparison, per the paper's current structure.
    reported_here = ["Unprocessed", "SEM (WFB)"]
    for label in reported_here
        df = dfs[label]
        means = [overall_stats(df, m)[1] for m in METRICS]
        stds  = [overall_stats(df, m)[2] for m in METRICS]

        display_label = label == "SEM (WFB)" ? raw"\textbf{SEM (ours)}" : label
        wrap_new(x) = label == "SEM (WFB)" ? string(raw"\new{", x, "}") : x

        @printf(io, "%-18s & T--F & %s & %s & %s & %s & %s \\\\\n",
                display_label,
                wrap_new(fmt2(means[1])),
                wrap_new(fmt2(means[2])),
                wrap_new(fmt2(means[3])),
                wrap_new(fmt2(means[4])),
                PARAM_COUNTS[label])

        if label == "SEM (WFB)"
            @printf(io, "                   &      & \\new{{\\small \$\\pm\\,%s\$}} & \\new{{\\small \$\\pm\\,%s\$}} & \\new{{\\small \$\\pm\\,%s\$}} & \\new{{\\small \$\\pm\\,%s\$}} &  \\\\\n",
                    fmt2(stds[1]), fmt2(stds[2]), fmt2(stds[3]), fmt2(stds[4]))
        end
    end

    println(io, raw"% --- Published-baseline rows (quoted from cited publications; this script does not populate them).")
    println(io, raw"% Wiener~\cite{hsieh_wavecrn_2020}            & T--F & 2.22 & 3.23 & 2.68 & 2.67 & --- \\")
    println(io, raw"% MAMBA-SENet~\cite{kim_mambabased_2025}      & T    & \textbf{3.62} & \textbf{4.79} & \textbf{4.01} & \textbf{4.34} & 0.99 M \\")
    println(io, raw"% DSEGAN~\cite{pascual_segan_2017}            & T    & 2.39 & 3.46 & 3.11 & 2.90 & 43.2 k \\")
    println(io, raw"% CDiffuSE~\cite{lu_conditional_2022}         & T    & 2.52 & 3.72 & 2.91 & 3.10 & --- \\")
    println(io, raw"% WaveCRN~\cite{hsieh_wavecrn_2020}           & T    & 2.64 & 3.94 & 3.37 & 3.29 & 4.65 M \\")
    println(io, raw"% MOSE~\cite{chen_metricoriented_2023}        & T    & 2.54 & 3.72 & 2.93 & 3.06 & --- \\")
    println(io, raw"\hline")
    println(io, raw"\end{tabular}")
    println(io, raw"\end{table}")

    return String(take!(io))
end

# ----------------------------------------------------- table 5 (ablation quadrants)

function write_metric_subtable(io::IO, metric::AbstractString, dfs::Dict{String,DataFrame})
    println(io, raw"\subfloat[", metric, raw"]{%")
    println(io, raw"\begin{minipage}{0.5\textwidth}\centering\scriptsize")
    println(io, raw"\begin{tabular}{l|cccc}")
    println(io, raw"\hline")

    for env in ENVIRONMENTS
        println(io, raw"\multicolumn{5}{c}{", uppercase(env), raw"} \\ \hline")
        println(io, raw"System & 2.5 dB & 7.5 dB & 12.5 dB & 17.5 dB \\ \hline")
        for sys in SYSTEMS
            cells = [cell_tex(cell_stats(dfs[sys.label], env, snr, metric)...) for snr in SNRS]
            @printf(io, "%-12s & %s & %s & %s & %s \\\\\n",
                    sys.label, cells[1], cells[2], cells[3], cells[4])
        end
        println(io, raw"\hline")
    end

    println(io, raw"\end{tabular}")
    println(io, raw"\end{minipage}}")
end

function write_ablation_table(dfs::Dict{String,DataFrame})
    io = IOBuffer()

    println(io, raw"% Auto-generated by scripts/generate_latex_tables.jl — do not hand-edit.")
    println(io, raw"% Regenerate with: julia --project=. scripts/generate_latex_tables.jl")
    println(io, raw"\begin{table*}[ht!]")
    println(io, raw"\centering")
    println(io, raw"\caption{\new{Objective scores by noise environment and input SNR, formatted as mean $\pm$ one sample standard deviation.}}")
    println(io, raw"\label{tab:metrics-quadrants}")
    println(io, "")
    println(io, raw"\new{")
    for (i, metric) in enumerate(METRICS)
        write_metric_subtable(io, metric, dfs)
        # Layout: PESQ | CSIG on row 1, CBAK | COVL on row 2 (matches the paper).
        if i == 1 || i == 3
            println(io, raw"\hfill")
        end
    end
    println(io, raw"}")
    println(io, "")
    println(io, raw"\end{table*}")

    return String(take!(io))
end

# --------------------------------------------------- per-environment delta table

function write_per_env_delta_table(dfs::Dict{String,DataFrame})
    io = IOBuffer()

    function snr_avg(df::DataFrame, env::AbstractString, metric::AbstractString)
        vals = [cell_stats(df, env, snr, metric)[1] for snr in SNRS]
        vals = filter(isfinite, vals)
        isempty(vals) && return NaN
        return mean(vals)
    end

    deltas_U = Dict{String,Dict{String,Float64}}()
    deltas_u = Dict{String,Dict{String,Float64}}()
    for env in ENVIRONMENTS
        deltas_U[env] = Dict()
        deltas_u[env] = Dict()
        for m in METRICS
            wfb = snr_avg(dfs["SEM (WFB)"], env, m)
            ufb = snr_avg(dfs["SEM (uFB)"], env, m)
            unp = snr_avg(dfs["Unprocessed"], env, m)
            deltas_U[env][m] = wfb - unp
            deltas_u[env][m] = wfb - ufb
        end
    end

    # Identify the best-positive improvement per column for bolding.
    best_U = Dict(m => argmax(env -> deltas_U[env][m], ENVIRONMENTS) for m in METRICS)
    best_u = Dict(m => argmax(env -> deltas_u[env][m], ENVIRONMENTS) for m in METRICS)

    function fmt_delta(x::Real, is_best::Bool)
        sign = x ≥ 0 ? "+" : "-"
        body = @sprintf("%s%.2f", sign, abs(x))
        inner = is_best ? "\\mathbf{" * body * "}" : body
        return "\$" * inner * "\$"
    end

    println(io, raw"% Auto-generated by scripts/generate_latex_tables.jl — do not hand-edit.")
    println(io, raw"% Regenerate with: julia --project=. scripts/generate_latex_tables.jl")
    println(io, raw"\begin{table}[h]")
    println(io, raw"\centering")
    println(io, raw"\caption{Per-environment average improvement, averaged across the four input SNRs. $\Delta_\text{U$\rightarrow$W}$ = SEM (WFB) $-$ Unprocessed; $\Delta_\text{u$\rightarrow$W}$ = SEM (WFB) $-$ SEM (uFB) (WFB ablation). Best positive improvement per column shown in bold.}")
    println(io, raw"\label{tab:per-env-delta}")
    println(io, raw"\small")
    println(io, raw"\setlength{\tabcolsep}{3pt}")
    println(io, raw"\begin{tabular}{l|cccc|cccc}")
    println(io, raw"\hline")
    println(io, raw" & \multicolumn{4}{c|}{$\Delta_\text{U$\rightarrow$W}$ (Unprocessed $\rightarrow$ WFB)} & \multicolumn{4}{c}{$\Delta_\text{u$\rightarrow$W}$ (uFB $\rightarrow$ WFB, ablation)}" * " \\\\")
    println(io, raw"Env & PESQ & CSIG & CBAK & COVL & PESQ & CSIG & CBAK & COVL" * " \\\\")
    println(io, raw"\hline")

    for env in ENVIRONMENTS
        dU = [fmt_delta(deltas_U[env][m], best_U[m] == env && deltas_U[env][m] > 0) for m in METRICS]
        du = [fmt_delta(deltas_u[env][m], best_u[m] == env && deltas_u[env][m] > 0) for m in METRICS]
        @printf(io, "\\textsc{%s} & %s & %s & %s & %s & %s & %s & %s & %s \\\\\n",
                env, dU[1], dU[2], dU[3], dU[4], du[1], du[2], du[3], du[4])
    end

    println(io, raw"\hline")
    println(io, raw"\end{tabular}")
    println(io, raw"\end{table}")

    return String(take!(io))
end

# --------------------------------------------------------------------- driver

function main()
    mkpath(TABLES_DIR)
    dfs = load_runs()

    println("="^72)
    println("Latest runs used:")
    for sys in SYSTEMS
        println("  ", rpad(sys.label, 16), " ← ", relpath(latest_results_path(sys.dir), PROJECT_DIR))
    end
    println("="^72)

    outputs = Dict(
        "tab_comparison_with_params.tex" => write_comparison_table(dfs),
        "tab_metrics_quadrants.tex"      => write_ablation_table(dfs),
        "tab_per_env_delta.tex"          => write_per_env_delta_table(dfs),
    )

    for (fname, contents) in outputs
        path = joinpath(TABLES_DIR, fname)
        open(path, "w") do io
            write(io, contents)
        end
        println("\nWrote ", relpath(path, PROJECT_DIR), " ($(length(contents)) bytes)")
        println("-"^72)
        print(contents)
    end

    println("\n", "="^72)
    println("All three tables written under $TABLES_DIR/.")
    println("Drop `\\input{tables/<filename>}` into the paper body to use them.")
    println("="^72)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
