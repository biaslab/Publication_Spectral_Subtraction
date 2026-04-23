#!/usr/bin/env julia
"""
run_paper_results.jl — one-command reproduction of the paper tables.

This script orchestrates the full pipeline a reviewer needs to regenerate
every aggregate table reported in the paper, including the WFB ablation and
the CSIG/CBAK/COVL composite metrics added during the revision.

Prerequisites (steps the reviewer does once):
  1. Clone the repository with `git clone --recursive …` so that the
     `python_modules/DNSMOS/` submodule is initialized.
  2. Instantiate the Julia project: `julia --project=. -e 'using Pkg; Pkg.instantiate()'`
  3. Install the Python dependencies: `python install_python_deps.py`
  4. Download the VoiceBank+DEMAND testset into `databases/VOICEBANK_DEMAND/`
     (see README §"Step 1: Download and Resample VOICEBANK_DEMAND Dataset").

Then run:

    julia --project=. scripts/run_paper_results.jl

The script is idempotent: already-resampled and already-WFB-processed data is
reused, partial evaluation runs can be resumed through their checkpoints, and
re-invoking the script after a clean run regenerates only the README tables.

Stages (each in its own Julia process so the Julia package precompile cache
is amortized across stages and a crash in one stage does not take down the
orchestrator):
  1. Resample VoiceBank+DEMAND to 16 kHz (skipped if already done).
  2. Generate the WFB-processed reference dataset (skipped if already done).
  3. Evaluate the baselines and the two SEM configurations with
     `--composite --checkpoint-interval 50`, yielding both DNSMOS and
     CSIG/CBAK/COVL columns in every per-file and summary CSV.
  4. Regenerate the README benchmark tables.
"""

const PROJECT_DIR = abspath(joinpath(@__DIR__, ".."))

# The four configurations needed to fill the paper tables:
#   baseline_clean  — upper bound (clean vs clean).
#   baseline_noise  — lower bound (unprocessed noisy mixture).
#   SEMHearingAid   — paper algorithm (warped FB, apcoefficient = 0.5).
#   SEMHearingAid_uFB — WFB ablation (uniform FB, apcoefficient = 0.0).
const CONFIGS = [
    "configurations/baseline_clean/baseline_clean.toml",
    "configurations/baseline_noise/baseline_noise.toml",
    "configurations/SEMHearingAid/SEMHearingAid.toml",
    "configurations/SEMHearingAid_uFB/SEMHearingAid_uFB.toml",
]

# One checkpoint every 50 files keeps the run restartable on long sweeps
# without flooding the table directory with intermediate CSVs.
const CHECKPOINT_INTERVAL = 50

function banner(title::AbstractString)
    println()
    println("=" ^ 72)
    println("  ", title)
    println("=" ^ 72)
end

function has_audio(path::AbstractString)
    isdir(path) || return false
    any(f -> endswith(lowercase(f), ".wav"), readdir(path))
end

function ensure_resampled()
    noisy = joinpath(PROJECT_DIR, "databases", "VOICEBANK_DEMAND_resampled", "noisy_testset_wav")
    clean = joinpath(PROJECT_DIR, "databases", "VOICEBANK_DEMAND_resampled", "clean_testset_wav")
    if has_audio(noisy) && has_audio(clean)
        @info "Stage 1/4: resampled VoiceBank+DEMAND already present — skipping."
        return
    end
    banner("Stage 1/4 — Resampling VoiceBank+DEMAND to 16 kHz")
    raw_root = joinpath(PROJECT_DIR, "databases", "VOICEBANK_DEMAND")
    if !isdir(joinpath(raw_root, "data"))
        error("Raw VoiceBank+DEMAND not found at $raw_root/data/. Download and extract it per README §'Step 1: Download and Resample VOICEBANK_DEMAND Dataset' before re-running this script.")
    end
    cmd = `julia --project=$PROJECT_DIR -e "using Experiments; using Experiments.HADatasets; HADatasets.resample_data(HADatasets.VOICEBANKDEMANDDataset(\"$raw_root\"))"`
    run(cmd)
end

function ensure_wfb()
    noisy = joinpath(PROJECT_DIR, "databases", "VOICEBANK_DEMAND_resampled_wfb", "noisy_testset_wav")
    clean = joinpath(PROJECT_DIR, "databases", "VOICEBANK_DEMAND_resampled_wfb", "clean_testset_wav")
    if has_audio(noisy) && has_audio(clean)
        @info "Stage 2/4: WFB-processed dataset already present — skipping."
        return
    end
    banner("Stage 2/4 — Generating WFB-processed reference dataset")
    run(`julia --project=$PROJECT_DIR $(joinpath(PROJECT_DIR, "scripts", "convert_to_wfb.jl"))`)
end

function run_all_evaluations()
    banner("Stage 3/4 — Running evaluations (--composite, checkpoint every $CHECKPOINT_INTERVAL)")
    script = joinpath(PROJECT_DIR, "scripts", "run_evaluation.jl")
    for cfg in CONFIGS
        cfg_path = joinpath(PROJECT_DIR, cfg)
        isfile(cfg_path) || error("Configuration not found: $cfg_path")
        banner("  Evaluating: $cfg")
        run(`julia --project=$PROJECT_DIR $script $cfg_path --composite --checkpoint-interval $CHECKPOINT_INTERVAL`)
    end
end

function update_tables()
    banner("Stage 4/4 — Updating README benchmark tables")
    run(`julia --project=$PROJECT_DIR $(joinpath(PROJECT_DIR, "scripts", "update_readme_benchmark.jl"))`)
end

function main()
    banner("SEM paper-results reproduction pipeline")
    println("Project root: $PROJECT_DIR")
    println("Configurations: $(join(CONFIGS, ", "))")
    println("Checkpoint interval: $CHECKPOINT_INTERVAL file(s)")

    ensure_resampled()
    ensure_wfb()
    run_all_evaluations()
    update_tables()

    banner("Done")
    println("""
    Per-file and summary CSVs (all seven metrics) are under:
      $(joinpath(PROJECT_DIR, "results", "VOICEBANK_DEMAND"))

    Aggregated benchmark tables have been written into README.md; open the
    `Benchmark Results` section to inspect PESQ, SIG, BAK, OVRL, CSIG, CBAK,
    and COVL broken down by SNR and by (environment, SNR).
    """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
