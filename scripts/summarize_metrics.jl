#!/usr/bin/env julia
"""
Aggregate per-file PESQ / CSIG / CBAK / COVL scores into the breakdowns reported
in the paper's appendix:

  - overall summary               (one row per system)
  - by SNR                        (one row per (system, snr_db))
  - by environment × SNR          (one row per (system, noise_type, snr_db))
  - per-metric pivot tables       (rows = noise_type, cols = SNR), one CSV per
                                   metric and per system

Reads the per-file CSV produced by `run_evaluation.jl` (default
`results/test_metrics.csv`) and writes the summary CSVs next to it.

Usage:
    julia scripts/summarize_metrics.jl
    julia scripts/summarize_metrics.jl --in results/test_metrics.csv
    julia scripts/summarize_metrics.jl --in results/test_metrics.csv --out-dir results/
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV
using DataFrames
using Statistics
using Printf

const BASE_METRICS = (:PESQ, :SIG, :BAK, :OVRL)
const COMPOSITE_METRICS = (:CSIG, :CBAK, :COVL)

nanmean(x) = (v = collect(skipmissing(x)); v = v[.!isnan.(v)]; isempty(v) ? NaN : mean(v))

"""
    summarize_metrics(df) -> NamedTuple

Aggregate a per-file metrics DataFrame into the four breakdowns used by the
paper. The input must have columns: `system, noise, snr_db, PESQ, SIG, BAK, OVRL`.

Returns a NamedTuple with fields `overall`, `by_snr`, `by_env_snr`, `pivots`,
where `pivots` is a `Dict{Tuple{String,Symbol},DataFrame}` keyed by
`(system, metric)`.
"""
function summarize_metrics(df::DataFrame)
    # Detect which metrics are present (support runs with or without composite)
    METRICS = if :CSIG in propertynames(df) && any(!ismissing, df.CSIG)
        (BASE_METRICS..., COMPOSITE_METRICS...)
    else
        BASE_METRICS
    end

    available = filter(m -> m in propertynames(df), collect(METRICS))

    overall = combine(
        groupby(df, :system),
        (m => nanmean => m for m in available)...,
        nrow => :n_files,
    )

    by_snr = sort(
        combine(
            groupby(df, [:system, :snr_db]),
            (m => nanmean => m for m in available)...,
            nrow => :n_files,
        ),
        [:system, :snr_db],
    )

    by_env_snr = sort(
        combine(
            groupby(df, [:system, :noise, :snr_db]),
            (m => nanmean => m for m in available)...,
            nrow => :n_files,
        ),
        [:system, :noise, :snr_db],
    )

    # Per-metric pivot: rows = noise_type, cols = snr_db, one table per (system, metric)
    pivots = Dict{Tuple{String,Symbol},DataFrame}()
    for sys in unique(df.system)
        sub = df[df.system .== sys, :]
        for metric in available
            agg = combine(
                groupby(sub, [:noise, :snr_db]),
                metric => nanmean => metric,
            )
            piv = unstack(agg, :noise, :snr_db, metric)
            sort!(piv, :noise)
            pivots[(sys, metric)] = piv
        end
    end

    return (; overall, by_snr, by_env_snr, pivots)
end

"""
    write_summaries(s, out_dir; prefix="test_metrics")

Persist all four breakdowns from `summarize_metrics` under `out_dir`.
File names:

  - `<prefix>_summary.csv`
  - `<prefix>_by_snr.csv`
  - `<prefix>_by_env_snr.csv`
  - `<prefix>_pivot_<system>_<metric>.csv`        (one per system × metric)
"""
function write_summaries(s, out_dir::AbstractString; prefix::AbstractString="test_metrics")
    mkpath(out_dir)
    CSV.write(joinpath(out_dir, "$(prefix)_summary.csv"), s.overall)
    CSV.write(joinpath(out_dir, "$(prefix)_by_snr.csv"), s.by_snr)
    CSV.write(joinpath(out_dir, "$(prefix)_by_env_snr.csv"), s.by_env_snr)
    for ((sys, metric), piv) in s.pivots
        CSV.write(joinpath(out_dir, "$(prefix)_pivot_$(sys)_$(metric).csv"), piv)
    end
end

"""
    print_overall(overall::DataFrame)

Pretty-print the overall summary table.
"""
function print_overall(overall::DataFrame)
    println("\n" * "="^64)
    println("Overall summary (mean over the test set)")
    println("="^64)
    @printf("%-12s | %6s | %6s | %6s | %6s | %5s\n",
            "System", "PESQ", "SIG", "BAK", "OVRL", "N")
    println("-"^64)
    for r in eachrow(overall)
        @printf("%-12s | %6.3f | %6.3f | %6.3f | %6.3f | %5d\n",
                r.system, r.PESQ, r.SIG, r.BAK, r.OVRL, r.n_files)
    end
end

"""
    print_by_snr(by_snr::DataFrame)

Pretty-print the by-SNR breakdown.
"""
function print_by_snr(by_snr::DataFrame)
    println("\n" * "="^64)
    println("By SNR")
    println("="^64)
    @printf("%-12s | %6s | %6s | %6s | %6s | %6s | %5s\n",
            "System", "SNR", "PESQ", "SIG", "BAK", "OVRL", "N")
    println("-"^64)
    for r in eachrow(by_snr)
        @printf("%-12s | %6.1f | %6.3f | %6.3f | %6.3f | %6.3f | %5d\n",
                r.system, r.snr_db, r.PESQ, r.SIG, r.BAK, r.OVRL, r.n_files)
    end
end

# ── CLI ──────────────────────────────────────────────────────────────────────
function main(args)
    in_path = joinpath(@__DIR__, "..", "results", "test_metrics.csv")
    out_dir = nothing
    let i = 1
        while i <= length(args)
            a = args[i]
            if a == "--in" && i < length(args)
                in_path = args[i + 1]; i += 2
            elseif startswith(a, "--in=")
                in_path = split(a, "=")[2]; i += 1
            elseif a == "--out-dir" && i < length(args)
                out_dir = args[i + 1]; i += 2
            elseif startswith(a, "--out-dir=")
                out_dir = split(a, "=")[2]; i += 1
            else
                i += 1
            end
        end
    end
    isnothing(out_dir) && (out_dir = dirname(in_path))

    isfile(in_path) || error("Per-file metrics CSV not found: $in_path " *
                             "(run `make eval` first)")

    df = CSV.read(in_path, DataFrame)
    s = summarize_metrics(df)
    print_overall(s.overall)
    print_by_snr(s.by_snr)
    write_summaries(s, out_dir)
    println("\nWrote breakdowns under $out_dir")
end

# Only run main when executed as a script (not when included from another script)
if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
