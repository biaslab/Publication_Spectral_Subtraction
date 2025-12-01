# Evaluation Script Documentation

This directory contains scripts for evaluating Virtual Hearing Aids on the VOICEBANK_DEMAND dataset.

## Quick Start

```bash
# Test with a single file first
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --single-file p257_001.wav

# Run full evaluation
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml

# Run with output files saved
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --save-output
```

## Overview

The `run_evaluation.jl` script provides a complete evaluation pipeline that:

1. **Reads metadata** from `log_testset.txt` (filename, noise category, SNR value)
2. **Loads file pairs** from `clean_testset_wav` and `noisy_testset_wav` directories
3. **Processes files** through VirtualHearingAid with a given configuration
4. **Evaluates output** using HADatasets metrics (PESQ, SIG, BAK, OVRL)
5. **Saves checkpoints** periodically to prevent data loss
6. **Optionally saves** processed output audio files
7. **Merges checkpoints** into a final results table
8. **Organizes results** in a timestamped run directory

## Prerequisites

1. **Julia environment**: Ensure Julia is installed and the project dependencies are installed:
   ```bash
   julia --project=. -e "using Pkg; Pkg.instantiate()"
   ```

2. **Dataset**: The VOICEBANK_DEMAND_resampled dataset should be available at:
   ```
   databases/VOICEBANK_DEMAND_resampled/
   ├── clean_testset_wav/
   ├── noisy_testset_wav/
   └── logfiles/
       └── log_testset.txt
   ```

3. **Configuration files**: Hearing aid configuration files should be in:
   ```
   configurations/
   ├── BaselineHearingAid/
   ├── SEMHearingAid/
   └── ExperimetalHearingAid/
   ```

## Basic Usage

### Run Full Evaluation

Evaluate all files in the test set with a specific hearing aid configuration:

```bash
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml
```

### Test with Single File

Test the pipeline with a single file before running the full evaluation:

```bash
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --single-file p257_001.wav
```

### Run with Custom Options

```bash
# Save output audio files and use custom checkpoint interval
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml \
    --checkpoint-interval 20 \
    --save-output

# Limit to first 50 samples for quick testing
julia scripts/run_evaluation.jl configurations/BaselineHearingAid/BaselineHearingAid.toml \
    --num-samples 50
```

## Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `config_path` | - | **Required.** Path to hearing aid configuration TOML file | - |
| `--single-file` | `-s` | Process only a single file (for testing) | `nothing` |
| `--checkpoint-interval` | `-c` | Save checkpoint every N files | `10` |
| `--save-output` | `-o` | Save processed output audio files | `false` |
| `--num-samples` | `-n` | Limit number of samples to process | `nothing` |

### Examples

```bash
# Basic evaluation
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml

# Test single file
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml -s p257_001.wav

# Full evaluation with output files saved
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --save-output

# Custom checkpoint interval (save every 25 files)
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml -c 25

# Quick test with limited samples
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml -n 10

# Combine options
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml \
    --checkpoint-interval 20 \
    --save-output \
    --num-samples 100
```

## Output Structure

Results are organized in timestamped directories under `results/VOICEBANK_DEMAND/<hearing_aid_type>/`:

```
results/VOICEBANK_DEMAND/
└── SEMHearingAid/
    └── run_25_10_2025_16_48/
        ├── SEMHearingAid.toml          # Configuration file copy
        ├── output/                      # Processed audio files (if --save-output)
        │   ├── p257_001.wav
        │   ├── p257_002.wav
        │   └── ...
        └── table/                       # Results tables
            ├── checkpoint_10.csv        # Checkpoint after 10 files
            ├── checkpoint_20.csv        # Checkpoint after 20 files
            ├── checkpoint_30.csv        # ...
            └── results_merged.csv       # Final merged results
```

### Results CSV Format

The results CSV contains the following columns:

- `filename`: Audio file name
- `noise_type`: Noise category (bus, cafe, living, office, psquare)
- `snr_db`: Signal-to-noise ratio in dB
- `PESQ`: Perceptual Evaluation of Speech Quality (1-5, higher is better)
- `SIG`: Signal quality from DNSMOS (1-5, higher is better)
- `BAK`: Background quality from DNSMOS (1-5, higher is better)
- `OVRL`: Overall quality from DNSMOS (1-5, higher is better)
- `processing_timestamp`: When the file was processed

## Supported Hearing Aid Types

The script supports all hearing aid types available in the `configurations/` directory:

- **BaselineHearingAid**: Unity gain processing (no noise reduction)
- **SEMHearingAid**: Speech Enhancement Model (Bayesian inference)

## Checkpointing

The script automatically saves checkpoints to prevent data loss:

- **Checkpoint interval**: Configurable via `--checkpoint-interval` (default: 10 files)
- **Checkpoint location**: `run_<timestamp>/table/checkpoint_<N>.csv`
- **Automatic merging**: All checkpoints are merged into `results_merged.csv` at the end
- **Resume capability**: If the script crashes, you can manually merge existing checkpoints

### Manual Checkpoint Merging and Summary Tables

If you need to merge checkpoints manually or create summary tables after evaluation, use the dedicated script:

```bash
# Create summary tables from a run directory
julia scripts/create_summary_tables.jl results/VOICEBANK_DEMAND/SEMHearingAid/run_25_10_2025_16_48
```

This script will:
1. Merge all checkpoint files into `results_merged.csv` (if not already merged)
2. Create `summary_by_snr.csv` - Average scores (OVRL, BAK, SIG) for each SNR level (2.5, 7.5, 12.5, 17.5)
3. Create `summary_by_environment_snr.csv` - Average per metric per environment for each SNR level
4. Create pivot tables - One per metric showing environment × SNR matrix

**Note**: The evaluation script automatically creates these summary tables at the end of a run. Use the standalone script if you need to regenerate them or if the evaluation was interrupted.

Alternatively, you can merge checkpoints manually:

```julia
using CSV, DataFrames

# Load all checkpoints
checkpoints = [CSV.read("table/checkpoint_10.csv", DataFrame),
               CSV.read("table/checkpoint_20.csv", DataFrame),
               ...]

# Merge
merged = vcat(checkpoints...)

# Save
CSV.write("table/results_merged.csv", merged)
```

## Performance Tips

1. **Start with a single file**: Always test with `--single-file` first to verify everything works
2. **Use limited samples**: Test with `--num-samples 10` before running the full dataset
3. **Adjust checkpoint interval**: For long runs, increase `--checkpoint-interval` to reduce I/O overhead
4. **Skip output saving**: Only use `--save-output` if you need the processed audio files (saves disk space)

## Troubleshooting

### Common Issues

1. **File not found errors**:
   - Verify the dataset paths are correct
   - Check that `log_testset.txt` exists and is readable
   - Ensure clean and noisy audio directories exist

2. **Configuration errors**:
   - Verify the configuration file path is correct
   - Check that the TOML file is valid (use `TOML.parsefile()` to test)

3. **Memory issues**:
   - Process in smaller batches using `--num-samples`
   - Increase checkpoint frequency to free memory

4. **Metrics evaluation errors**:
   - Ensure HADatasets dependencies are installed
   - Check that Python dependencies for DNSMOS are installed (see HADatasets README)

### Getting Help

Check the script's built-in help:

```bash
julia scripts/run_evaluation.jl --help
```

## Example Workflow

1. **Test with single file**:
   ```bash
   julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml -s p257_001.wav
   ```

2. **Test with small batch**:
   ```bash
   julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml -n 10
   ```

3. **Run full evaluation**:
   ```bash
   julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml
   ```

4. **Run with output files** (if needed):
   ```bash
   julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --save-output
   ```

## Summary Tables

After evaluation completes, summary tables are automatically created in the `table/` directory:

1. **`summary_by_snr.csv`**: Average scores (OVRL, BAK, SIG) for each SNR level across all environments
   - Columns: `SNR`, `OVRL_mean`, `BAK_mean`, `SIG_mean`
   - One row per SNR level (2.5, 7.5, 12.5, 17.5)

2. **`summary_by_environment_snr.csv`**: Average per metric per environment for each SNR level
   - Columns: `Environment`, `SNR`, `OVRL_mean`, `BAK_mean`, `SIG_mean`
   - One row per environment-SNR combination

3. **`pivot_OVRL_by_environment.csv`**: Pivot table for OVRL metric
   - Rows: Environments (bus, cafe, living, office, psquare)
   - Columns: SNR levels (SNR_2.5, SNR_7.5, SNR_12.5, SNR_17.5)

4. **`pivot_BAK_by_environment.csv`**: Same structure for BAK metric

5. **`pivot_SIG_by_environment.csv`**: Same structure for SIG metric

To regenerate these tables after evaluation:

```bash
julia scripts/create_summary_tables.jl <run_directory>
```

## Related Documentation

- **VirtualHearingAid**: See `dependencies/VirtualHearingAid/README.md`
- **HADatasets**: See `dependencies/HADatasets/README.md`
- **Configuration files**: See `configurations/` directory for example configs

## Notes

- The script processes files sequentially (not parallelized)
- Processing time depends on file length and hearing aid complexity
- SEM hearing aid uses Bayesian inference and typically takes longer than Baseline
- All audio files should be at 16 kHz sample rate (resampled dataset)

