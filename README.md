# A Probabilistic Generative Model for Spectral Speech Enhancement

This repository accompanies the paper:

> M. Hidalgo-Araya *et al.*, "A Probabilistic Generative Model for Spectral Speech Enhancement", 2025.

A comprehensive evaluation framework for virtual hearing aids using the VOICEBANK_DEMAND dataset with warped filter bank (WFB) preprocessing.

## How This Repository Relates to the Paper

This repository provides the complete implementation and evaluation framework for the spectral speech enhancement model presented in the paper. It includes:

- **Implementation**: Full codebase for the Warped-Frequency Filter Bank (WFB) front-end and Speech Enhancement Model (SEM) backend
- **Evaluation Pipeline**: Automated evaluation on the VOICEBANK_DEMAND dataset with comprehensive metrics (PESQ, DNSMOS)
- **Reproducibility**: All configurations and scripts needed to reproduce the results reported in the paper
- **Benchmark Comparisons**: Automated generation of comparison tables 

## Overview

This repository provides a complete pipeline for:
1. **Dataset Preparation**: Download, resample, and preprocess VOICEBANK_DEMAND dataset
2. **WFB Preprocessing**: Create warped filter bank processed dataset for consistent evaluation
3. **Evaluation**: Run evaluations for baseline and hearing aid algorithms using `run_evaluation.jl`
4. **Results Analysis**: Generate summary tables and metrics organized by SNR and environment
5. **Benchmark Results**: Automatically generate and update benchmark comparison tables in the README

### Quick Start - View Benchmark Results

The latest benchmark results comparing different hearing aid algorithms are automatically generated and displayed in the [Benchmark Results](#benchmark-results) section below. To update these results with the latest evaluation runs, simply run:

```bash
julia scripts/update_readme_benchmark.jl
```

This script automatically:
- Finds the latest runs for each hearing aid (excluding Baseline_clean)
- Generates comprehensive comparison tables for:
  - Overall summary across all metrics
  - Performance by SNR level (2.5, 7.5, 12.5, 17.5 dB)
  - Performance by environment and SNR (bus, cafe, living, office, psquare)
- Updates the README with the latest results and configuration details

## Prerequisites

- **Julia 1.11+**: Required for all functionality
- **Python 3.7+**: Required for metrics evaluation (PESQ, DNSMOS)
- **Git**: For cloning and submodule management

### Installation

1. **Clone the repository with submodules:**
```bash
git clone --recursive <repository-url>
cd Spectral_Subtraction
```

2. **Install Julia dependencies:**
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

3. **Install Python dependencies for metrics:**
```bash
cd dependencies/HADatasets
python install_python_deps.py
cd ../..
```

## Complete Workflow

### Step 1: Download and Resample VOICEBANK_DEMAND Dataset

#### 1.1 Download the Dataset

Download the VOICEBANK_DEMAND dataset from the official source:

1. **Visit the official dataset page**: https://datashare.ed.ac.uk/handle/10283/2791
2. **Download the dataset files**
3. **Extract and place them in the following structure:**

```
databases/VOICEBANK_DEMAND/
├── data/
│   ├── clean_testset_wav/     # Clean audio files
│   └── noisy_testset_wav/     # Noisy audio files
├── logfiles/
│   └── log_testset.txt        # SNR information
└── testset_txt/               # Text transcriptions
```

#### 1.2 Resample the Dataset

Resample the VOICEBANK_DEMAND dataset to 16 kHz using Julia:

```julia
using HADatasets

# Create dataset instance pointing to the database directory
dataset = HADatasets.VOICEBANKDEMANDDataset("databases/VOICEBANK_DEMAND")

# Resample with default settings (16kHz, 1.0s minimum duration)
HADatasets.resample_data(dataset)
```

This creates:
```
databases/VOICEBANK_DEMAND_resampled/
├── clean_testset_wav/         # Resampled clean files
├── noisy_testset_wav/         # Resampled noisy files
└── logfiles/
    └── log_testset_resampled.txt  # Updated log file
```

**Note**: The resampled dataset preserves the same directory structure as the original, with all audio files resampled to 16 kHz.

### Step 2: Create WFB-Processed Dataset

**Why WFB preprocessing is needed:**

The hearing aid processing pipeline uses a **Warped Filter Bank (WFB)** that warps the frequency domain of the audio. Since **PESQ is sensitive to changes in the data or missing samples**, we need to ensure consistent preprocessing for fair evaluation. 

The WFB preprocessing:
- Processes all audio through the BaselineHearingAid (which has unity gains, so the audio is unaltered except for the WFB warping)
- Creates a preprocessed dataset where all files have been through the same WFB pipeline
- Ensures that when we evaluate hearing aids, we compare against a consistent WFB-processed clean reference

**Create the WFB dataset:**

```bash
julia scripts/convert_to_wfb.jl
```

Or test with a limited number of samples first:

```bash
julia scripts/convert_to_wfb.jl --num-samples=10
```

This script:
1. Loads the BaselineHearingAid configuration
2. Processes all clean and noisy files from `VOICEBANK_DEMAND_resampled` through the WFB
3. Creates `VOICEBANK_DEMAND_resampled_wfb/` with the same directory structure:
   ```
   databases/VOICEBANK_DEMAND_resampled_wfb/
   ├── clean_testset_wav/      # WFB-processed clean files
   ├── noisy_testset_wav/      # WFB-processed noisy files
   ├── logfiles/               # Copied logfiles
   ```

**Note**: If the WFB dataset already exists, the script will detect it and skip processing with the following messages:
```
[Info: WFB dataset already exists and appears to be processed
[Info: Skipping conversion - dataset already processed
```

### Step 3: Run Evaluations

All evaluations, including baselines and hearing aid algorithms, are run using the `run_evaluation.jl` script:

#### 3.1 Run Baseline Evaluations

Before evaluating hearing aids, establish baseline scores for comparison:

**Baseline Best (Clean vs Clean)** - Upper bound performance:
```bash
julia scripts/run_evaluation.jl configurations/baseline_clean/baseline_clean.toml
```

**Baseline Unprocessed (Clean vs Noisy)** - Lower bound performance:
```bash
julia scripts/run_evaluation.jl configurations/baseline_noise/baseline_noise.toml
```

#### 3.2 Run Hearing Aid Evaluations

Evaluate each hearing aid algorithm on the WFB-processed dataset:

```bash
# Evaluate SEM Hearing Aid
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml

```

#### 3.3 Evaluation Options

```bash
# Test with a single file first
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --single-file p257_001.wav

# Limit number of samples for testing
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --num-samples 50

# Custom checkpoint interval (save every N files)
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --checkpoint-interval 20

# Save processed output audio files
julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --save-output
```

### Step 4: Results and Metrics

#### 4.1 Results Structure

Results are organized in timestamped directories:

```
results/VOICEBANK_DEMAND/
├── BaselineHearingAid/
│   └── run_<timestamp>/
│       ├── BaselineHearingAid.toml
│       └── table/
│           ├── results.csv                    # Complete results for all files
│           ├── overall_summary.csv           # Overall average scores
│           ├── summary_by_snr.csv            # Average scores by SNR level
│           ├── summary_by_environment_snr.csv # Average scores by environment and SNR
│           └── checkpoint_*.csv              # Optional checkpoint files (if --checkpoint-interval used)
│   └── run_<timestamp>/
│       └── ...
└── SEMHearingAid/
    └── run_<timestamp>/
        └── ...
```

#### 4.2 Metrics Computed

Each evaluation computes the following metrics:

- **PESQ** (Perceptual Evaluation of Speech Quality): 1-5 scale, higher is better
- **SIG** (Signal Quality from DNSMOS): 1-5 scale, higher is better
- **BAK** (Background Quality from DNSMOS): 1-5 scale, higher is better
- **OVRL** (Overall Quality from DNSMOS): 1-5 scale, higher is better

#### 4.3 Summary Tables

The evaluation automatically generates:

1. **`overall_summary.csv`**: Overall average scores across all conditions
2. **`summary_by_snr.csv`**: Average scores for each SNR level (2.5, 7.5, 12.5, 17.5 dB)
3. **`summary_by_environment_snr.csv`**: Average scores per environment per SNR level
4. **`results.csv`**: Complete results for all individual files

#### 4.4 Checkpointing

- **Automatic checkpoints**: Saved every N files (default: 10, configurable) - checkpoint files are created when using `--checkpoint-interval` option
- **Resume capability**: If evaluation is interrupted, checkpoints can be merged manually
- **Final results**: All results are saved to `results.csv` in the table directory

#### 4.5 Update Benchmark Results

After running evaluations for multiple hearing aids, you can automatically generate and update benchmark comparison tables in the README:

```bash
julia scripts/update_readme_benchmark.jl
```

This script:
- Finds the latest runs for each hearing aid (excluding Baseline_clean)
- Generates comprehensive comparison tables showing:
  - Overall summary across all metrics (PESQ, SIG, BAK, OVRL)
  - Performance breakdown by SNR level (2.5, 7.5, 12.5, 17.5 dB)
  - Performance breakdown by environment and SNR (bus, cafe, living, office, psquare)
- Updates the README with the latest results and configuration details

The benchmark results are displayed in the [Benchmark Results](#benchmark-results) section below.

## Evaluation Metrics

This repository uses comprehensive speech quality assessment metrics to evaluate hearing aid algorithms. All metrics are computed using the HADatasets module, which provides standardized implementations of ITU-T and IEEE/ACM standards.

### PESQ (Perceptual Evaluation of Speech Quality)

- **Type**: Intrusive (requires reference signal)
- **Scale**: 1-5 (higher is better)
- **Standard**: ITU-T P.862.2
- **Use Case**: Overall speech quality assessment
- **Description**: PESQ is a perceptual metric that predicts the subjective quality of speech as perceived by human listeners. It compares the processed/enhanced audio to the clean reference signal and provides a score that correlates with Mean Opinion Score (MOS) ratings.

**Important Note**: PESQ is sensitive to changes in the data or missing samples. This is why the evaluation pipeline uses WFB-processed clean audio as the reference, ensuring that both the processed output and reference have undergone the same WFB preprocessing for fair comparison.

### DNSMOS (Deep Noise Suppression Mean Opinion Score)

- **Type**: Non-intrusive (no reference required)
- **Scale**: 1-5 (higher is better)
- **Standard**: Microsoft DNS Challenge P.835
- **Use Case**: Noise suppression quality assessment
- **Description**: DNSMOS is a deep learning-based metric that predicts subjective quality scores without requiring a clean reference signal. It follows the ITU-T P.835 subjective test framework to measure three key quality dimensions.

**P.835 Dimensions**:

- **OVRL (Overall Quality)**: Overall audio quality assessment
  - Measures the overall perceived quality of the processed audio
  - Combines both speech and background noise quality perceptions

- **SIG (Signal Quality)**: Speech quality assessment
  - Focuses specifically on the quality of the speech signal
  - Measures how natural and clear the speech sounds

- **BAK (Background Quality)**: Background noise quality assessment
  - Evaluates the quality of the background/noise component
  - Measures how well noise is suppressed while preserving speech

### Metric Selection Rationale

The combination of PESQ and DNSMOS provides a comprehensive evaluation:

- **PESQ** provides an intrusive reference-based assessment, giving a direct comparison to the clean signal
- **DNSMOS** provides a non-intrusive assessment that doesn't require a reference, making it useful for real-world scenarios where clean references may not be available
- The three DNSMOS dimensions (OVRL, SIG, BAK) provide detailed insights into different aspects of speech enhancement performance

### Research Context

This evaluation framework adopts the **ITU-T P.835 subjective test framework** to measure speech enhancement quality across multiple dimensions, enabling comprehensive assessment of hearing aid algorithms for monaural speech enhancement tasks.

## Directory Structure

```
Spectral_Subtraction/
├── databases/
│   ├── VOICEBANK_DEMAND/              # Original dataset (downloaded)
│   ├── VOICEBANK_DEMAND_resampled/    # Resampled dataset (16 kHz)
│   └── VOICEBANK_DEMAND_resampled_wfb/ # WFB-processed dataset
├── configurations/
│   ├── BaselineHearingAid/
│   ├── SEMHearingAid/
├── results/
│   └── VOICEBANK_DEMAND/              # Evaluation results
├── scripts/
│   ├── convert_to_wfb.jl              # WFB conversion script
│   ├── run_evaluation.jl              # Evaluation script
│   └── update_readme_benchmark.jl     # Benchmark results update script
├── src/
│   └── Experiments.jl                 # Main evaluation module
└── dependencies/
    ├── HADatasets/                    # Dataset and metrics module
    └── VirtualHearingAid/             # Hearing aid processing module
```

## Key Concepts

### Speech Enhancement Module (SEM)

The SEM follows the model introduced in the paper:

![SEM Factor Graph](figures/FFG_SEM.png)

The Speech Enhancement Model (SEM) uses a probabilistic generative model for Bayesian inference of speech and noise characteristics, enabling adaptive spectral enhancement.

### Warped-Frequency Filter Bank (WFB)

The WFB front-end provides perceptually-aligned frequency warping for consistent evaluation:

![WFB Architecture](figures/WFB.png)

The input signal passes through a cascade of first-order all-pass filters, producing warped delay-line signals. A time-domain FIR structure with weights generates the output, while the warped signals are provided to the Spectral Enhancement Model for inference and synthesis.


### Evaluation Pipeline

1. **Input**: WFB-processed noisy audio (`VOICEBANK_DEMAND_resampled_wfb/noisy_testset_wav/`)
2. **Processing**: Pass through hearing aid algorithm
3. **Reference**: WFB-processed clean audio (`VOICEBANK_DEMAND_resampled_wfb/clean_testset_wav/`)
4. **Metrics**: Compare processed output to WFB-processed clean reference

## Supported Hearing Aid Types

- **BaselineHearingAid**: Unity gain processing (no noise reduction, WFB only)
- **SEMHearingAid**: Speech Enhancement Model (Bayesian inference)

## Reproducing the Paper Results

To reproduce the results reported in the paper:

1. Prepare the `VOICEBANK_DEMAND_resampled_wfb` dataset by following Steps 1 and 2 in this README.

2. Run the hearing aid configurations:

   ```bash
   julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml
   ```

3. Update the README tables:

   ```bash
   julia scripts/update_readme_benchmark.jl
   ```

4. The results used in the paper correspond to the runs in:
   ```
   results/VOICEBANK_DEMAND/<Device>/run_<timestamp>/
   ```

This reproduces the tables in the paper's results section.

## Extending the Framework

To add a new hearing aid algorithm:

1. **Implement the backend** in `dependencies/VirtualHearingAid` (create a new `<Name>Backend` type).

2. **Create a configuration file** in `configurations/<NewHearingAid>/<NewHearingAid>.toml`:
   - `[parameters.hearingaid]` with `type = "<NewHearingAid>"`
   - `[parameters.frontend]` WFB parameters (nbands, fs, etc.)
   - `[parameters.backend.*]` for algorithm-specific parameters

3. **Run evaluation**:
   ```bash
   julia scripts/run_evaluation.jl configurations/<NewHearingAid>/<NewHearingAid>.toml
   ```

4. **Update the benchmark tables**:
   ```bash
   julia scripts/update_readme_benchmark.jl
   ```

See existing configurations in `configurations/` for examples of the TOML structure.

## Runtime and Hardware Requirements

- **Tested on**: macOS / Linux, Julia 1.11+, Python 3.7+
- **GPU**: Not required. All models are CPU-friendly
- **Storage**: ~2 GB for the resampled dataset, ~4 GB for the WFB-processed dataset

## Troubleshooting

### Dataset Issues

- **Missing files**: Ensure the dataset is downloaded and extracted correctly
- **Resampling errors**: Check that audio files are valid WAV files
- **WFB conversion fails**: Verify BaselineHearingAid configuration exists

### Evaluation Issues

- **Memory errors**: Use `--num-samples` to process in smaller batches
- **Checkpoint errors**: Manually merge existing checkpoints if needed
- **Metrics errors**: Ensure Python dependencies are installed (see HADatasets README)

## Optional Dependencies

The metrics evaluation functionality relies on Python integration and the following optional dependencies:

- **PyCall**: Python integration (for full metrics functionality)
- **pesq**: Python PESQ implementation (MIT License)
- **dnsmos_wrapper**: Custom wrapper for Microsoft DNSMOS (Creative Commons Attribution 4.0 International)

These dependencies are automatically installed when running the Python installation script:
```bash
cd dependencies/HADatasets
python install_python_deps.py
```

## Third-Party Licenses

### Microsoft DNS-Challenge (DNSMOS submodule)

Licensed under **Creative Commons Attribution 4.0 International**:

- **Attribution Required**: Must give appropriate credit to Microsoft
- **Commercial Use**: Allowed
- **Modification**: Allowed
- **Distribution**: Allowed

## Citations

### DNSMOS P.835

```bibtex
@inproceedings{reddy2022dnsmos,
  title={DNSMOS P.835: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors},
  author={Reddy, Chandan KA and Gopal, Vishak and Cutler, Ross},
  booktitle={ICASSP 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2022},
  organization={IEEE}
}
```

### ICASSP 2023 Deep Noise Suppression Challenge

```bibtex
@inproceedings{dubey2023icassp,
  title={ICASSP 2023 Deep Noise Suppression Challenge},
  author={Dubey, Harishchandra and Aazami, Ashkan and Gopal, Vishak and Naderi, Babak and Braun, Sebastian and Cutler, Ross and Gamper, Hannes and Golestaneh, Mehrsa and Aichner, Robert},
  booktitle={ICASSP},
  year={2023}
}
```

### VOICEBANK DEMAND Dataset

```bibtex
@misc{Valentini-Botinhao2017NoisySpeech,
  author = {Valentini-Botinhao, Cassia},
  title = {Noisy speech database for training speech enhancement algorithms and TTS models},
  year = {2017},
  howpublished = {Edinburgh DataShare},
  doi = {10.7488/ds/2117},
  url = {https://doi.org/10.7488/ds/2117}
}
```

## Related Resources

- **[ICASSP 2023 Deep Noise Suppression Challenge](https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2023/)**: Official challenge website and resources
- **[DNSMOS Implementation](https://github.com/microsoft/DNS-Challenge)**: Microsoft's DNS Challenge repository with DNSMOS implementation
- **[VoiceBank+Demand Dataset](https://datashare.ed.ac.uk/handle/10283/2791)**: Official dataset download page
