#!/usr/bin/env julia
"""
Evaluation script for Virtual Hearing Aids on VOICEBANK_DEMAND dataset.

This script:
1. Reads metadata from log_testset.txt (filename, category, SNR)
2. Loads file pairs from clean_testset_wav and noisy_testset_wav
3. Processes files through VirtualHearingAid with a given configuration
4. Evaluates output using HADatasets metrics
5. Saves checkpoints periodically
6. Optionally saves output files
7. Merges checkpoints into final table
8. Organizes results in run_<date>_<time>/ structure

Usage:
    julia scripts/run_evaluation.jl <config_path> [--single-file <filename>] [--checkpoint-interval <N>] [--save-output] [--num-samples <N>]

Examples:
    # Run full evaluation
    julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml

    # Run single file for testing
    julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --single-file p257_001.wav

    # Run with custom checkpoint interval and save outputs
    julia scripts/run_evaluation.jl configurations/SEMHearingAid/SEMHearingAid.toml --checkpoint-interval 20 --save-output
"""

using Pkg
project_dir = joinpath(@__DIR__, "..")
Pkg.activate(project_dir)

using CSV
using DataFrames
using Dates
using Logging
using Statistics
using TOML
using WAV
using Experiments
using SampledSignals: SampleBuf
using ArgParse
using RxInfer

# Import submodules for convenience
const VirtualHearingAid = Experiments.VirtualHearingAid
const HADatasets = Experiments.HADatasets 

Logging.global_logger(SimpleLogger(stdout, Logging.Info))

# Define custom rule for ReactiveMP (must be at top level)
@rule NormalMixture{N}(:switch, Marginalisation) (q_out::Any, q_m::ManyOf{N,Any}, q_p::ManyOf{N,Any}) where {N} = begin
    return Categorical(0.5, 0.5)
end

# Parse command line arguments
function parse_args()
    s = ArgParseSettings(description="Run Virtual Hearing Aid evaluation")
    
    @add_arg_table! s begin
        "config_path"
            help = "Path to hearing aid configuration TOML file"
            required = true
        "--single-file", "-s"
            help = "Process only a single file (for testing)"
            arg_type = String
            default = nothing
        "--checkpoint-interval", "-c"
            help = "Save checkpoint every N files"
            arg_type = Int
            default = 10
        "--save-output", "-o"
            help = "Save processed output audio files"
            action = :store_true
        "--num-samples", "-n"
            help = "Limit number of samples to process"
            arg_type = Int
            default = nothing
    end
    
    return ArgParse.parse_args(s)
end

"""
    parse_log_file(log_path::String)::DataFrame

Parse log_testset.txt file to extract metadata.
Format: filename noise_type snr_value
"""
function parse_log_file(log_path::String)::DataFrame
    if !isfile(log_path)
        error("Log file not found: $log_path")
    end
    
    lines = readlines(log_path)
    data = []
    
    for line in lines
        line = strip(line)
        if isempty(line) || startswith(line, "#")
            continue
        end
        
        parts = split(line)
        if length(parts) >= 3
            filename = parts[1]
            noise_type = parts[2]
            snr_value = parse(Float64, parts[3])
            
            push!(data, Dict(
                "filename" => filename,
                "noise_type" => noise_type,
                "snr_db" => snr_value
            ))
        end
    end
    
    return DataFrame(data)
end


"""
    process_audio_file(ha, noisy_path::String)::Tuple{SampleBuf, Any}

Process audio file through hearing aid. Returns (output_signal, results).
"""
function process_audio_file(ha, noisy_path::String)::Tuple{SampleBuf, Any}
    audio = load_audio_file(noisy_path)
    result = VirtualHearingAid.process(ha, audio)
    
    if result isa Tuple
        output, results = result
    else
        output = result
        results = nothing
    end
    
    return output, results
end

"""
    save_output_file(output::SampleBuf, output_path::String)

Save processed audio output to file.
"""
function save_output_file(output::SampleBuf, output_path::String)
    # Ensure directory exists
    mkpath(dirname(output_path))
    
    # Convert SampleBuf to array and save
    audio_data = output.data
    WAV.wavwrite(audio_data, output_path, Fs=Int(output.samplerate))
end

"""
    evaluate_single_file(ha, clean_path::String, noisy_path::String, 
                        metadata::Dict, save_output::Bool, output_dir::String, ha_type_str::String)::Dict{String, Any}

Process and evaluate a single file pair.
"""
function evaluate_single_file(ha, clean_path::String, noisy_path::String,
                             metadata::Dict, save_output::Bool, output_dir::Union{String, Nothing}, ha_type_str::String)::Dict{String, Any}
    try
        # Load clean audio (always needed as reference)
        clean_audio = load_audio_file(clean_path)
        clean_vec = clean_audio.data

        # Determine output based on hearing aid type
        if ha_type_str == "baseline_clean"
            # Baseline clean: compare clean vs clean (best case)
            output = clean_audio
            output_vec = clean_audio.data
        elseif ha_type_str == "baseline_noise"
            # Baseline noise: compare clean vs noisy (worst case)
            noisy_audio = load_audio_file(noisy_path)
            output = noisy_audio
            output_vec = noisy_audio.data
        else
            # Regular hearing aid processing
            output, results = process_audio_file(ha, noisy_path)
            output_vec = output.data
        end
        
        # # Ensure same length for metrics evaluation
        # min_len = min(length(clean_vec), length(output_vec))
        # clean_vec = clean_vec[1:min_len]
        # output_vec = output_vec[1:min_len]
        
        metrics = HADatasets.evaluate_audio_metrics(clean_vec, output_vec, Int(clean_audio.samplerate))
        
        # Save output if requested
        if save_output && !isnothing(output_dir)
            filename = metadata["filename"]
            output_path = joinpath(output_dir, filename)
            save_output_file(output, output_path)
        end
        
        # Create result dictionary
        result_dict = Dict{String, Any}(
            "filename" => metadata["filename"],
            "noise_type" => metadata["noise_type"],
            "snr_db" => metadata["snr_db"],
            "PESQ" => metrics.PESQ,
            "SIG" => metrics.SIG,
            "BAK" => metrics.BAK,
            "OVRL" => metrics.OVRL,
            "processing_timestamp" => now()
        )
        
        return result_dict
        
    catch e
        @error "Error processing file" filename=metadata["filename"] error=e
        return Dict{String, Any}(
            "filename" => metadata["filename"],
            "noise_type" => metadata["noise_type"],
            "snr_db" => metadata["snr_db"],
            "error" => string(e),
            "processing_timestamp" => now()
        )
    end
end


"""
    load_config(config_path::String)::Dict{String, Any}

Load a configuration file from a path.
"""
function load_config(config_path::String)::Dict{String, Any}
    return Experiments.load_config(config_path)
end

"""
    get_hearing_aid_type(config_path::String)::String

Extract hearing aid type from configuration file path.
"""
function get_hearing_aid_type(config_path::String)::String
    # Extract directory name from path
    dir_name = basename(dirname(config_path))
    
    # Map to expected type strings
    if dir_name == "baseline_clean"
        return "baseline_clean"
    elseif dir_name == "baseline_noise"
        return "baseline_noise"
    elseif dir_name == "BaselineHearingAid"
        return "BaselineHearingAid"
    elseif dir_name == "SEMHearingAid"
        return "SEMHearingAid"
    elseif dir_name == "ExperimetalHearingAid"
        return "ExperimetalHearingAid"
    else
        return dir_name
    end
end

"""
    create_hearing_aid_instance(ha_type_str::String, config::Dict{String, Any})

Create a hearing aid instance from type string and configuration.
"""
function create_hearing_aid_instance(ha_type_str::String, config::Dict{String, Any})
    if ha_type_str in ["baseline_clean", "baseline_noise"]
        # For baseline evaluations, we don't need a hearing aid instance
        return nothing
    else
        return Experiments.create_hearing_aid_from_config(config)
    end
end

"""
    load_audio_file(file_path::String)::SampleBuf

Load audio file using HADatasets and convert to SampleBuf.
"""
function load_audio_file(file_path::String)::SampleBuf
    return Experiments.load_audio_file(file_path)
end

"""
    merge_checkpoints(table_dir::String, merged_file::String)::DataFrame

Merge all checkpoint files into a single DataFrame.
"""
function merge_checkpoints(table_dir::String, merged_file::String)::DataFrame
    checkpoint_files = filter(x -> endswith(x, ".csv") && startswith(x, "checkpoint_"), readdir(table_dir))
    
    if isempty(checkpoint_files)
        return DataFrame()
    end
    
    all_data = []
    for checkpoint_file in checkpoint_files
        checkpoint_path = joinpath(table_dir, checkpoint_file)
        df = CSV.read(checkpoint_path, DataFrame)
        append!(all_data, [df])
    end
    
    if isempty(all_data)
        return DataFrame()
    end
    
    merged_df = vcat(all_data...)
    
    # Remove duplicates based on filename
    merged_df = unique(merged_df, :filename)
    
    # Save merged file
    CSV.write(merged_file, merged_df)
    
    return merged_df
end

"""
    create_snr_summary_table(df::DataFrame, metrics::Vector{String})::DataFrame

Create summary table grouped by SNR level.
"""
function create_snr_summary_table(df::DataFrame, metrics::Vector{String})::DataFrame
    if !hasproperty(df, :snr_db)
        return DataFrame()
    end
    
    summary_data = []
    for snr in unique(df.snr_db)
        row_data = Dict(:SNR => snr)
        for metric in metrics
            if hasproperty(df, Symbol(metric))
                metric_values = df[df.snr_db .== snr, Symbol(metric)]
                if !isempty(metric_values)
                    row_data[Symbol(metric)] = Statistics.mean(skipmissing(metric_values))
                end
            end
        end
        push!(summary_data, row_data)
    end
    
    return DataFrame(summary_data)
end

"""
    create_environment_snr_summary_table(df::DataFrame, metrics::Vector{String})::DataFrame

Create summary table grouped by environment and SNR.
"""
function create_environment_snr_summary_table(df::DataFrame, metrics::Vector{String})::DataFrame
    if !hasproperty(df, :noise_type) || !hasproperty(df, :snr_db)
        return DataFrame()
    end
    
    summary_data = []
    for env in unique(df.noise_type)
        for snr in unique(df.snr_db)
            row_data = Dict(:environment => env, :SNR => snr)
            for metric in metrics
                if hasproperty(df, Symbol(metric))
                    metric_values = df[(df.noise_type .== env) .& (df.snr_db .== snr), Symbol(metric)]
                    if !isempty(metric_values)
                        row_data[Symbol(metric)] = Statistics.mean(skipmissing(metric_values))
                    end
                end
            end
            push!(summary_data, row_data)
        end
    end
    
    return DataFrame(summary_data)
end

"""
    create_overall_summary_table(df::DataFrame, metrics::Vector{String})::DataFrame

Create overall summary table with averages across all files.
"""
function create_overall_summary_table(df::DataFrame, metrics::Vector{String})::DataFrame
    row_data = Dict()
    for metric in metrics
        if hasproperty(df, Symbol(metric))
            row_data[Symbol(metric)] = Statistics.mean(skipmissing(df[:, Symbol(metric)]))
        end
    end
    
    return DataFrame(row_data)
end

"""
    create_pivot_table_by_environment(df::DataFrame, metric::String)::DataFrame

Create pivot table for a specific metric grouped by environment and SNR.
"""
function create_pivot_table_by_environment(df::DataFrame, metric::String)::DataFrame
    if !hasproperty(df, Symbol(metric)) || !hasproperty(df, :noise_type) || !hasproperty(df, :snr_db)
        return DataFrame()
    end
    
    # Get unique environments and SNR values
    environments = sort(unique(df.noise_type))
    snr_values = sort(unique(df.snr_db))
    
    # Create pivot table
    pivot_data = []
    for env in environments
        row_data = Dict(:environment => env)
        for snr in snr_values
            metric_values = df[(df.noise_type .== env) .& (df.snr_db .== snr), Symbol(metric)]
            if !isempty(metric_values)
                row_data[Symbol("$(snr)_dB")] = Statistics.mean(skipmissing(metric_values))
            else
                row_data[Symbol("$(snr)_dB")] = missing
            end
        end
        push!(pivot_data, row_data)
    end
    
    return DataFrame(pivot_data)
end

"""
    main()

Main evaluation function.
"""
function main()
    args = parse_args()
    
    # Parse arguments
    config_path = args["config_path"]
    single_file = args["single-file"]
    checkpoint_interval = args["checkpoint-interval"]
    save_output = args["save-output"]
    num_samples = args["num-samples"]
    
    @info "Starting evaluation" config_path=config_path single_file=single_file checkpoint_interval=checkpoint_interval save_output=save_output
    
    config = load_config(config_path)
    metadata = get(config, "metadata", Dict{String, Any}())
    config_name = get(metadata, "name", "unknown")
    ha_type_str = get_hearing_aid_type(config_path)
    
    @info "Configuration loaded" config_name=config_name ha_type=ha_type_str
    
    ha = create_hearing_aid_instance(ha_type_str, config)
    @info "Hearing aid created" ha_type=typeof(ha)
    
    # Setup paths
    base_dir = joinpath(@__DIR__, "..")
    log_path = joinpath(base_dir, "databases", "VOICEBANK_DEMAND_resampled", "logfiles", "log_testset.txt")

    # Clean and noisy directories for WFB-processed data
    clean_dir_wfb = joinpath(base_dir, "databases", "VOICEBANK_DEMAND_resampled_wfb", "clean_testset_wav")
    noisy_dir_wfb = joinpath(base_dir, "databases", "VOICEBANK_DEMAND_resampled_wfb", "noisy_testset_wav")

    # Noisy directories for original data (this is the one we need to filter and evaluate on)
    noisy_dir = joinpath(base_dir, "databases", "VOICEBANK_DEMAND_resampled", "noisy_testset_wav")

    results_base = joinpath(base_dir, "results", "VOICEBANK_DEMAND", ha_type_str)
    
    
    run_timestamp = Dates.format(now(), "dd_mm_yyyy_HH_MM")
    run_dir = joinpath(results_base, "run_$run_timestamp")
    mkpath(run_dir)
    
    
    table_dir = joinpath(run_dir, "table")
    mkpath(table_dir)
    output_dir = save_output ? joinpath(run_dir, "output") : nothing
    if save_output
        mkpath(output_dir)
    end
    
    
    config_copy_path = joinpath(run_dir, basename(config_path))
    cp(config_path, config_copy_path)
    @info "Configuration copied to run directory" path=config_copy_path
    
    
    log_data = parse_log_file(log_path)
    @info "Loaded log file" num_entries=nrow(log_data)
    
    # Filter to single file if requested
    if !isnothing(single_file)
        log_data = log_data[log_data.filename .== single_file, :]
        if nrow(log_data) == 0
            error("File not found in log: $single_file")
        end
        @info "Filtered to single file" filename=single_file
    end
    
    # Limit samples if requested
    if !isnothing(num_samples) && num_samples < nrow(log_data)
        log_data = log_data[1:num_samples, :]
        @info "Limited to samples" num_samples=nrow(log_data)
    end
    
    # Process files
    all_results = []
    checkpoint_counter = 0
    
    # Determine which directories to use based on hearing aid type
    is_baseline = ha_type_str in ["baseline_clean", "baseline_noise"]
    clean_dir = clean_dir_wfb  # Always use WFB clean directory
    noisy_dir = is_baseline ? noisy_dir_wfb : noisy_dir  # WFB for baseline, original for others
    
    @info "Processing $(nrow(log_data)) files..." ha_type=ha_type_str is_baseline=is_baseline clean_dir=clean_dir noisy_dir=noisy_dir
    for (idx, row) in enumerate(eachrow(log_data))
        filename = row.filename
        # Ensure filename has .wav extension
        if !endswith(filename, ".wav")
            filename = filename * ".wav"
        end
        clean_path = joinpath(clean_dir, filename)
        noisy_path = joinpath(noisy_dir, filename)
        
        @info "Processing $idx/$(nrow(log_data))" filename=filename
        
        # Check files exist
        if !isfile(clean_path)
            @warn "Clean file not found" path=clean_path
            continue
        end
        if !isfile(noisy_path)
            @warn "Noisy file not found" path=noisy_path
            continue
        end
        
        # Evaluate
        metadata = Dict(
            "filename" => filename,
            "noise_type" => row.noise_type,
            "snr_db" => row.snr_db
        )
        
        result = evaluate_single_file(ha, clean_path, noisy_path, metadata, save_output, output_dir, ha_type_str)
        push!(all_results, result)
        checkpoint_counter += 1
        
        # Save checkpoint
        if checkpoint_counter >= checkpoint_interval
            checkpoint_file = joinpath(table_dir, "checkpoint_$(idx).csv")
            df = DataFrame(all_results)
            CSV.write(checkpoint_file, df)
            @info "Checkpoint saved" file=checkpoint_file num_results=length(all_results)
            checkpoint_counter = 0
        end
    end
    
    # Save final merged results
    if !isempty(all_results)
        merged_file = joinpath(table_dir, "results_merged.csv")
        df = DataFrame(all_results)
        CSV.write(merged_file, df)
        @info "Final results saved" file=merged_file num_results=length(all_results)
        
        # Merge any remaining checkpoints if they exist
        checkpoint_files = filter(x -> endswith(x, ".csv") && startswith(x, "checkpoint_"), readdir(table_dir))
        if !isempty(checkpoint_files)
            df = merge_checkpoints(table_dir, merged_file)
            # Reload the merged file to get updated data
            df = CSV.read(merged_file, DataFrame)
        end
        
        # Remove all checkpoint files (in case any remain)
        checkpoint_files = filter(x -> endswith(x, ".csv") && startswith(x, "checkpoint_"), readdir(table_dir))
        for checkpoint_file in checkpoint_files
            rm(joinpath(table_dir, checkpoint_file))
        end
        if !isempty(checkpoint_files)
            @info "Removed $(length(checkpoint_files)) remaining checkpoint files"
        end
        
        # Rename results_merged.csv to results.csv
        results_file = joinpath(table_dir, "results.csv")
        mv(merged_file, results_file; force=true)
        @info "Renamed results_merged.csv to results.csv"
        
        # Create summary tables
        @info "Creating summary tables..."
        try
            # Include all available metrics: PESQ, OVRL, BAK, SIG
            metrics = ["PESQ", "OVRL", "BAK", "SIG"]
            available_metrics = [m for m in metrics if hasproperty(df, Symbol(m))]
            
            if !isempty(available_metrics)
                # SNR summary (averages across all environments)
                snr_summary = create_snr_summary_table(df, available_metrics)
                snr_summary_file = joinpath(table_dir, "summary_by_snr.csv")
                CSV.write(snr_summary_file, snr_summary)
                @info "SNR summary saved" file=snr_summary_file
                
                # Environment-SNR summary (per environment per SNR)
                env_snr_summary = create_environment_snr_summary_table(df, available_metrics)
                env_snr_summary_file = joinpath(table_dir, "summary_by_environment_snr.csv")
                CSV.write(env_snr_summary_file, env_snr_summary)
                @info "Environment-SNR summary saved" file=env_snr_summary_file
                
                # Overall summary (averages across all files)
                overall_summary = create_overall_summary_table(df, available_metrics)
                overall_summary_file = joinpath(table_dir, "overall_summary.csv")
                CSV.write(overall_summary_file, overall_summary)
                @info "Overall summary saved" file=overall_summary_file
                
                # Pivot tables (one per metric: environment x SNR)
                for metric in available_metrics
                    pivot_table = create_pivot_table_by_environment(df, metric)
                    pivot_file = joinpath(table_dir, "pivot_$(metric)_by_environment.csv")
                    CSV.write(pivot_file, pivot_table)
                    @info "Pivot table saved" metric=metric file=pivot_file
                end
            else
                @warn "No expected metrics found. Available columns: $(names(df))"
            end
        catch e
            @warn "Failed to create summary tables" error=e
            @info "You can create summary tables later by running:"
            @info "  julia scripts/create_summary_tables.jl $run_dir"
        end
    else
        @warn "No results to save"
    end
    
    @info "Evaluation complete" run_dir=run_dir
    println("\n" * "="^60)
    println("Evaluation Complete!")
    println("="^60)
    println("Run directory: $run_dir")
    println("Total files processed: $(length(all_results))")
    if save_output
        println("Output files saved to: $output_dir")
    end
    println("Results saved to: $(joinpath(table_dir, "results.csv"))")
    println("Summary tables saved to: $table_dir")
    println("="^60)
end

# Run if called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

