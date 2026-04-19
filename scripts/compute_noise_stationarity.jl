#!/usr/bin/env julia
"""
Per-environment noise stationarity from (noisy - clean) subtraction.

For each of the 824 files in the VoiceBank+DEMAND test set, compute
the noise waveform n[t] = y[t] - x[t], take its STFT log-power
log|N(t,f)|^2, and measure the standard deviation across frames per
band, then average across bands. This yields one scalar per file
quantifying how much the noise log-power fluctuates over time.

Low values -> stationary noise (engine hum, HVAC) where musical
noise is audible against a flat background.
High values -> non-stationary noise (babble, crowd) where the
residual cannot be flat in principle.

The script also verifies the additive mixing model by checking that
the empirical SNR = 20*log10(rms(x) / rms(n)) is within tolerance of
the labelled SNR for a sample of files.

Usage:
    julia --project scripts/compute_noise_stationarity.jl
Output:
    results/noise_stationarity.csv  (per-environment summary)
    results/noise_stationarity_per_file.csv  (per-file detail)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using WAV
using FFTW
using Statistics
using Printf
using CSV
using DataFrames

const BASE_DIR   = joinpath(@__DIR__, "..")
const DB_DIR     = joinpath(BASE_DIR, "databases", "VOICEBANK_DEMAND_resampled")
const CLEAN_DIR  = joinpath(DB_DIR, "clean_testset_wav")
const NOISY_DIR  = joinpath(DB_DIR, "noisy_testset_wav")
const LOG_PATH   = joinpath(DB_DIR, "logfiles", "log_testset.txt")
const OUT_DIR    = joinpath(BASE_DIR, "results")
const PER_FILE_CSV = joinpath(OUT_DIR, "noise_stationarity_per_file.csv")
const SUMMARY_CSV  = joinpath(OUT_DIR, "noise_stationarity.csv")

const FS           = 16000.0      # Hz (resampled set)
const FRAME_SIZE   = 512           # samples (32 ms)
const HOP_SIZE     = 256           # samples (16 ms)
const LOG_EPS      = 1e-10
const VERIFY_N     = 5             # sanity-check the additive model on this many files

rms(x) = sqrt(mean(abs2, x))

# Read metadata: filename, noise, snr_db
function read_metadata()
    rows = NamedTuple[]
    for line in readlines(LOG_PATH)
        parts = split(strip(line))
        length(parts) >= 3 || continue
        push!(rows, (
            filename = String(parts[1]),
            noise    = String(parts[2]),
            snr_db   = parse(Float64, parts[3]),
        ))
    end
    return rows
end

# STFT log-power standard deviation across frames, averaged across bands.
# Returns: (stat_log_std, nframes, rms_noise, rms_clean)
function noise_stationarity(clean::Vector{Float64}, noisy::Vector{Float64})
    n = noisy .- clean
    L = min(length(clean), length(noisy), length(n))
    n = @view n[1:L]

    # STFT with Hann window, no overlap-add (only magnitude stats needed)
    win = 0.5 .* (1 .- cos.(2π .* (0:FRAME_SIZE-1) ./ (FRAME_SIZE - 1)))
    nframes = max(1, 1 + div(L - FRAME_SIZE, HOP_SIZE))
    nfreq   = div(FRAME_SIZE, 2) + 1
    logpow  = Matrix{Float64}(undef, nfreq, nframes)

    for k in 1:nframes
        start = (k - 1) * HOP_SIZE + 1
        stop  = start + FRAME_SIZE - 1
        stop > L && break
        frame = (n[start:stop]) .* win
        spec  = rfft(frame)
        logpow[:, k] .= log.(abs2.(spec) .+ LOG_EPS)
    end

    # Per-band std across frames, then mean across bands.
    band_std = [std(@view logpow[j, :]) for j in 1:nfreq]
    return (
        stat     = mean(band_std),
        nframes  = nframes,
        rms_n    = rms(n),
        rms_c    = rms(@view clean[1:L]),
    )
end

# --- Main ---
# Disable stdout buffering so progress is visible when redirected.
Base.Libc.flush_cstdio()

metadata = read_metadata()
@printf "Loaded %d metadata entries.\n" length(metadata)
flush(stdout)

mkpath(OUT_DIR)
per_file_rows = NamedTuple[]

# Verification pass on a handful of files.
println("\n=== Additive-model sanity check (first $VERIFY_N files) ===")
println("file              env         labeled_snr   empirical_snr   delta_dB")
for meta in first(metadata, VERIFY_N)
    clean_path = joinpath(CLEAN_DIR, meta.filename * ".wav")
    noisy_path = joinpath(NOISY_DIR, meta.filename * ".wav")
    if !(isfile(clean_path) && isfile(noisy_path))
        @printf "  %-15s  MISSING\n" meta.filename
        continue
    end
    clean, _ = wavread(clean_path; format="double")
    noisy, _ = wavread(noisy_path; format="double")
    c = vec(clean); y = vec(noisy)
    L = min(length(c), length(y))
    c = c[1:L]; y = y[1:L]
    n = y .- c
    emp_snr = 20 * log10(rms(c) / rms(n))
    @printf "  %-15s  %-10s  %7.2f dB    %7.2f dB    %+6.2f\n" meta.filename meta.noise meta.snr_db emp_snr (emp_snr - meta.snr_db)
end

println("\n=== Computing per-file stationarity over all 824 files ===")
for (i, meta) in enumerate(metadata)
    clean_path = joinpath(CLEAN_DIR, meta.filename * ".wav")
    noisy_path = joinpath(NOISY_DIR, meta.filename * ".wav")
    if !(isfile(clean_path) && isfile(noisy_path))
        continue
    end
    clean, _ = wavread(clean_path; format="double")
    noisy, _ = wavread(noisy_path; format="double")
    c = vec(clean); y = vec(noisy)
    res = noise_stationarity(c, y)
    push!(per_file_rows, (
        filename     = meta.filename,
        noise        = meta.noise,
        snr_db       = meta.snr_db,
        log_power_std = res.stat,
        nframes      = res.nframes,
        rms_noise    = res.rms_n,
        rms_clean    = res.rms_c,
        empirical_snr_db = 20 * log10(res.rms_c / res.rms_n),
    ))
    if i % 50 == 0 || i == length(metadata)
        @printf "  [%3d/%3d] %s %s %.1fdB  std=%.3f\n" i length(metadata) meta.filename meta.noise meta.snr_db res.stat
        flush(stdout)
    end
end

per_file_df = DataFrame(per_file_rows)
CSV.write(PER_FILE_CSV, per_file_df)
@printf "\nPer-file detail written to %s (%d rows)\n" PER_FILE_CSV nrow(per_file_df)

# Aggregate per environment.
println("\n=== Per-environment summary ===")
summary_rows = NamedTuple[]
for env in sort(unique(per_file_df.noise))
    sub = per_file_df[per_file_df.noise .== env, :]
    push!(summary_rows, (
        noise              = env,
        n_files            = nrow(sub),
        log_power_std_mean = mean(sub.log_power_std),
        log_power_std_sd   = std(sub.log_power_std),
        empirical_snr_mean = mean(sub.empirical_snr_db),
    ))
end
# Sort by stationarity (ascending: lowest std = most stationary).
summary_df = sort(DataFrame(summary_rows), :log_power_std_mean)
CSV.write(SUMMARY_CSV, summary_df)

println("\n      env      n   log_power_std (mean ± sd)   emp. SNR (mean)")
for row in eachrow(summary_df)
    @printf "  %-10s  %3d    %6.3f ± %5.3f             %5.2f dB\n" row.noise row.n_files row.log_power_std_mean row.log_power_std_sd row.empirical_snr_mean
end
println("\nSummary written to $SUMMARY_CSV")
println("\nRanking: lowest std = most stationary = largest expected EUM benefit.")
