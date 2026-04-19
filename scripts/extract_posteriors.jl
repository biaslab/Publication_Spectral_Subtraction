#!/usr/bin/env julia
"""
Extract per-frame posterior variances from SEM inference for uncertainty analysis.

Usage:
    julia --project scripts/extract_posteriors.jl [--num-samples N] [--system SEM]

Outputs:
    results/uncertainty/posterior_variances_<system>.csv
    results/uncertainty/timeseries_<file>_snr<snr>.csv  (representative files)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Experiments
const VHA = Experiments.VirtualHearingAid

# Load SEM paper rules
Experiments.HASoundProcessing.SEM.eval(
    :(include($(joinpath(pkgdir(Experiments), "src", "HASoundProcessing", "backends", "sem", "rules.jl")))),
)

using CSV, DataFrames, Statistics, TOML, Printf
using SampledSignals: SampleBuf
import Experiments.VirtualHearingAid: from_config, process, get_frontend, get_backend
using Experiments.VirtualHearingAid.Frontends.WFB: get_nbands
using RxInfer: mean, precision

# ── Parse args ───────────────────────────────────────────────────────────────

num_samples = 824
system_name = "SEM"
for i in 1:length(ARGS)-1
    if ARGS[i] == "--num-samples"
        global num_samples = parse(Int, ARGS[i+1])
    elseif ARGS[i] == "--system"
        global system_name = ARGS[i+1]
    end
end

# ── Setup ────────────────────────────────────────────────────────────────────

base_dir = joinpath(@__DIR__, "..")
config_path = joinpath(base_dir, "configurations", "$(system_name)HearingAid", "$(system_name)HearingAid.toml")
isfile(config_path) || error("Config not found: $config_path")

config = TOML.parsefile(config_path)
ha = from_config(config)
nbands = get_nbands(get_frontend(ha))

# Database
db_dir = joinpath(base_dir, "databases", "VOICEBANK_DEMAND_resampled")
noisy_dir = joinpath(db_dir, "noisy_testset_wav")
log_path = joinpath(db_dir, "logfiles", "log_testset.txt")
isfile(log_path) || error("Missing $log_path")

metadata = []
for line in readlines(log_path)
    parts = split(strip(line))
    length(parts) >= 3 || continue
    push!(metadata, (filename=String(parts[1]), noise_type=String(parts[2]),
                     snr_db=parse(Float64, parts[3])))
end
metadata = first(metadata, num_samples)
println("Processing $(length(metadata)) files with system $system_name (nbands=$nbands)")

out_dir = joinpath(base_dir, "results", "uncertainty")
mkpath(out_dir)

# Pick one representative per SNR
representative_files = Dict{Float64,String}()
for m in metadata
    haskey(representative_files, m.snr_db) || (representative_files[m.snr_db] = m.filename)
end
println("Representative files: ", representative_files)

# ── Process ──────────────────────────────────────────────────────────────────

summary_rows = NamedTuple[]

for (idx, meta) in enumerate(metadata)
    fname = meta.filename
    fname_wav = endswith(fname, ".wav") ? fname : fname * ".wav"
    noisy_path = joinpath(noisy_dir, fname_wav)
    isfile(noisy_path) || continue

    # Process through the full HA pipeline
    noisy_audio = Experiments.load_audio_file(noisy_path)
    output, results = process(ha, noisy_audio)

    # results.inference_results is a list of per-band RxInfer engines
    inference_list = results.inference_results
    nframes = size(results.gains, 1)

    # Extract per-band average variances
    avg_var_s = zeros(nbands)
    avg_var_n = zeros(nbands)
    avg_var_xi = zeros(nbands)

    is_repr = get(representative_files, meta.snr_db, "") == fname

    for band in 1:nbands
        eng = inference_list[band]
        s_hist = eng.history[:s]
        n_hist = eng.history[:n]
        xi_hist = eng.history[:ξ]

        for t in 1:nframes
            avg_var_s[band] += 1.0 / precision(s_hist[t])
            avg_var_n[band] += 1.0 / precision(n_hist[t])
            avg_var_xi[band] += 1.0 / precision(xi_hist[t])
        end
        avg_var_s[band] /= nframes
        avg_var_n[band] /= nframes
        avg_var_xi[band] /= nframes
    end

    push!(summary_rows, (
        file=fname,
        noise=meta.noise_type,
        snr_db=meta.snr_db,
        system=system_name,
        nframes=nframes,
        avg_var_s=Statistics.mean(avg_var_s),
        avg_var_n=Statistics.mean(avg_var_n),
        avg_var_xi=Statistics.mean(avg_var_xi),
    ))

    # Save representative time series
    if is_repr
        ts_rows = NamedTuple[]
        for band in 1:nbands
            eng = inference_list[band]
            s_hist = eng.history[:s]
            n_hist = eng.history[:n]
            xi_hist = eng.history[:ξ]
            pi_hist = eng.history[:π_switch]
            w_hist = eng.history[:w]
            for t in 1:nframes
                push!(ts_rows, (
                    frame=t, band=band,
                    mean_s=mean(s_hist[t]),
                    var_s=1.0 / precision(s_hist[t]),
                    mean_n=mean(n_hist[t]),
                    var_n=1.0 / precision(n_hist[t]),
                    mean_xi=mean(xi_hist[t]),
                    var_xi=1.0 / precision(xi_hist[t]),
                    gain=results.gains[t, band],
                    pi_prob=mean(pi_hist[t]),
                    w_prob=mean(w_hist[t]),
                ))
            end
        end
        ts_path = joinpath(out_dir, "timeseries_$(replace(fname, ".wav" => ""))_snr$(meta.snr_db).csv")
        CSV.write(ts_path, DataFrame(ts_rows))
        println("  [$idx/$(length(metadata))] $fname (SNR=$(meta.snr_db)) — representative → $ts_path")
    elseif idx % 50 == 0 || idx == length(metadata)
        println("  [$idx/$(length(metadata))] $(fname)")
    end
end

# ── Save summary ─────────────────────────────────────────────────────────────

summary_df = DataFrame(summary_rows)
summary_path = joinpath(out_dir, "posterior_variances_$(system_name).csv")
CSV.write(summary_path, summary_df)
println("\nSummary: $summary_path ($(nrow(summary_df)) files)")

# ── Calibration check ────────────────────────────────────────────────────────

println("\n=== Calibration Check ===")

# Load PESQ from existing results
ablation_dirs = filter(d -> occursin("full_ablation", d),
    readdir(joinpath(base_dir, "results", "VOICEBANK_DEMAND"); join=true))

if !isempty(ablation_dirs)
    rcsv = joinpath(first(ablation_dirs), "results.csv")
    if isfile(rcsv)
        df_m = CSV.read(rcsv, DataFrame)
        # In full_ablation: "SEM" = canonical, "SEM_lit" = proposed
        sys_key = system_name == "SEM_lit" ? "SEM_lit" : system_name
        df_sys = filter(r -> r.system == sys_key, df_m)

        joined = innerjoin(summary_df, select(df_sys, :file, :PESQ, :BAK), on=:file)
        if nrow(joined) > 0
            r_pesq = cor(joined.avg_var_xi, joined.PESQ)
            r_bak = cor(joined.avg_var_xi, joined.BAK)
            println("  corr(avg_var_xi, PESQ): $(round(r_pesq, digits=3))")
            println("  corr(avg_var_xi, BAK):  $(round(r_bak, digits=3))")

            qs = quantile(joined.avg_var_xi, [0.25, 0.5, 0.75])
            println("\n  ξ Variance Quartile Analysis:")
            for (label, lo, hi) in [
                ("Q1 (most confident)", -Inf, qs[1]),
                ("Q2", qs[1], qs[2]),
                ("Q3", qs[2], qs[3]),
                ("Q4 (least confident)", qs[3], Inf)]
                mask = (joined.avg_var_xi .> lo) .& (joined.avg_var_xi .<= hi)
                n = sum(mask)
                p = round(Statistics.mean(joined.PESQ[mask]), digits=3)
                b = round(Statistics.mean(joined.BAK[mask]), digits=3)
                println("    $label (n=$n): PESQ=$p, BAK=$b")
            end
        end
    end
end

println("\nDone.")
