#!/usr/bin/env julia
"""
Script to update README with benchmark results and generate barplots.

This script:
1. Finds the latest runs for each hearing aid (excluding Baseline_clean)
2. Creates barplots for overall summary, by SNR, and by environment+SNR
3. Updates README with figures and configuration details
"""

using CSV
using DataFrames
using Dates
using Glob
using TOML

# Constants
RESULTS_DIR = joinpath(@__DIR__, "..", "results", "VOICEBANK_DEMAND")
FIGURES_DIR = joinpath(@__DIR__, "..", "figures")
README_PATH = joinpath(@__DIR__, "..", "README.md")

# Metrics to plot
METRICS = ["PESQ", "SIG", "BAK", "OVRL"]
METRIC_LABELS = Dict(
    "PESQ" => "PESQ (1-5)",
    "SIG" => "SIG (1-5)",
    "BAK" => "BAK (1-5)",
    "OVRL" => "OVRL (1-5)"
)


function find_latest_run(hearing_aid_dir)
    """Find the latest run directory for a hearing aid."""
    if !isdir(hearing_aid_dir)
        return nothing
    end
    
    runs = filter(x -> startswith(x, "run_"), readdir(hearing_aid_dir))
    if isempty(runs)
        return nothing
    end
    
    # Sort by timestamp (run_ format: run_DD_MM_YYYY_HH_MM)
    sorted_runs = sort(runs, rev=true)
    return joinpath(hearing_aid_dir, sorted_runs[1])
end

function get_hearing_aid_name(dir_name)
    """Get display name for hearing aid."""
    if dir_name == "Baseline_noise"
        return "Baseline Unprocessed"
    else
        return replace(dir_name, "HearingAid" => "")
    end
end

function load_overall_summary(run_dir)
    """Load overall summary CSV."""
    csv_path = joinpath(run_dir, "table", "overall_summary.csv")
    if !isfile(csv_path)
        return nothing
    end
    return CSV.read(csv_path, DataFrame)
end

function load_summary_by_snr(run_dir)
    """Load summary by SNR CSV."""
    csv_path = joinpath(run_dir, "table", "summary_by_snr.csv")
    if !isfile(csv_path)
        return nothing
    end
    return CSV.read(csv_path, DataFrame)
end

function load_summary_by_environment_snr(run_dir)
    """Load summary by environment and SNR CSV."""
    csv_path = joinpath(run_dir, "table", "summary_by_environment_snr.csv")
    if !isfile(csv_path)
        return nothing
    end
    return CSV.read(csv_path, DataFrame)
end

function find_config_file(run_dir)
    """Find the TOML config file in the run directory."""
    files = readdir(run_dir)
    toml_files = filter(f -> endswith(f, ".toml"), files)
    if !isempty(toml_files)
        return joinpath(run_dir, toml_files[1])
    end
    return nothing
end

function create_overall_summary_table(data_dict)
    """Create table for overall summary per device."""
    devices = String[]
    metric_data = Dict(m => Float64[] for m in METRICS)
    
    for (device_name, df) in data_dict
        if df === nothing
            continue
        end
        push!(devices, device_name)
        for metric in METRICS
            col_name = Symbol("$(metric)_mean")
            if hasproperty(df, col_name)
                push!(metric_data[metric], df[1, col_name])
            else
                push!(metric_data[metric], NaN)
            end
        end
    end
    
    if isempty(devices)
        return nothing
    end
    
    # Create markdown table
    table = "| Device | " * join([METRIC_LABELS[m] for m in METRICS], " | ") * " |\n"
    table *= "|" * repeat("---|", length(METRICS) + 1) * "\n"
    
    for (idx, device) in enumerate(devices)
        row = "| $(device) |"
        for metric in METRICS
            val = metric_data[metric][idx]
            if isnan(val)
                row *= " - |"
            else
                row *= " $(round(val, digits=3)) |"
            end
        end
        table *= row * "\n"
    end
    
    return table
end

function create_snr_table(data_dict)
    """Create table for summary by SNR."""
    snr_levels = [2.5, 7.5, 12.5, 17.5]
    device_names = String[]
    device_data = Dict{String, Dict{Float64, Dict{String, Float64}}}()
    
    for (device_name, df) in data_dict
        if df === nothing
            continue
        end
        push!(device_names, device_name)
        device_data[device_name] = Dict{Float64, Dict{String, Float64}}()
        
        for row in eachrow(df)
            snr = row.SNR
            device_data[device_name][snr] = Dict{String, Float64}()
            for metric in METRICS
                col_name = Symbol("$(metric)_mean")
                if hasproperty(df, col_name)
                    device_data[device_name][snr][metric] = row[col_name]
                end
            end
        end
    end
    
    if isempty(device_names)
        return nothing
    end
    
    # Create markdown table for each metric
    metric_tables = Dict{String, String}()
    for metric in METRICS
        # Check if we have data for this metric
        has_data = false
        for device in device_names
            for snr in snr_levels
                if haskey(device_data[device], snr) && haskey(device_data[device][snr], metric)
                    has_data = true
                    break
                end
            end
            has_data && break
        end
        
        if !has_data
            continue
        end
        
        table = "### $(METRIC_LABELS[metric])\n\n"
        table *= "| Device | " * join([string(snr) * " dB" for snr in snr_levels], " | ") * " |\n"
        table *= "|" * repeat("---|", length(snr_levels) + 1) * "\n"
        
        for device in device_names
            row = "| $(device) |"
            for snr in snr_levels
                if haskey(device_data[device], snr) && haskey(device_data[device][snr], metric)
                    val = device_data[device][snr][metric]
                    row *= " $(round(val, digits=3)) |"
                else
                    row *= " - |"
                end
            end
            table *= row * "\n"
        end
        table *= "\n"
        
        metric_tables[metric] = table
    end
    
    if isempty(metric_tables)
        return nothing
    end
    
    return metric_tables
end

function create_environment_snr_table(data_dict)
    """Create table for summary by environment and SNR."""
    environments = ["bus", "cafe", "living", "office", "psquare"]
    snr_levels = [2.5, 7.5, 12.5, 17.5]
    device_names = String[]
    device_data = Dict{String, Dict{String, Dict{Float64, Dict{String, Float64}}}}()
    
    for (device_name, df) in data_dict
        if df === nothing
            continue
        end
        push!(device_names, device_name)
        device_data[device_name] = Dict{String, Dict{Float64, Dict{String, Float64}}}()
        
        for row in eachrow(df)
            env = row.environment
            snr = row.SNR
            if !haskey(device_data[device_name], env)
                device_data[device_name][env] = Dict{Float64, Dict{String, Float64}}()
            end
            if !haskey(device_data[device_name][env], snr)
                device_data[device_name][env][snr] = Dict{String, Float64}()
            end
            for metric in METRICS
                col_name = Symbol("$(metric)_mean")
                if hasproperty(df, col_name)
                    device_data[device_name][env][snr][metric] = row[col_name]
                end
            end
        end
    end
    
    if isempty(device_names)
        return nothing
    end
    
    # Create table for each metric
    metric_tables = Dict{String, String}()
    for metric in METRICS
        table = "### $(METRIC_LABELS[metric])\n\n"
        
        for env in environments
            # Check if we have data for this environment and metric
            has_data = false
            for device in device_names
                if haskey(device_data[device], env)
                    for snr in snr_levels
                        if haskey(device_data[device][env], snr) && haskey(device_data[device][env][snr], metric)
                            has_data = true
                            break
                        end
                    end
                    has_data && break
                end
            end
            
            if !has_data
                continue
            end
            
            table *= "#### $(uppercasefirst(env))\n\n"
            table *= "| Device | " * join([string(snr) * " dB" for snr in snr_levels], " | ") * " |\n"
            table *= "|" * repeat("---|", length(snr_levels) + 1) * "\n"
            
            for device in device_names
                row = "| $(device) |"
                for snr in snr_levels
                    if haskey(device_data[device], env) && 
                       haskey(device_data[device][env], snr) && 
                       haskey(device_data[device][env][snr], metric)
                        val = device_data[device][env][snr][metric]
                        row *= " $(round(val, digits=3)) |"
                    else
                        row *= " - |"
                    end
                end
                table *= row * "\n"
            end
            table *= "\n"
        end
        
        metric_tables[metric] = table
    end
    
    if isempty(metric_tables)
        return nothing
    end
    
    return metric_tables
end

function update_readme_with_benchmark(overall_table, snr_tables, env_snr_tables, config_dict)
    """Update README with benchmark section."""
    readme_content = read(README_PATH, String)
    
    # Find or create benchmark section
    benchmark_start = "# Benchmark Results"
    
    # Check if benchmark section exists
    if occursin(benchmark_start, readme_content)
        # Find the section and replace it
        start_idx = findfirst(benchmark_start, readme_content)
        if start_idx !== nothing
            # Find the next section (look for next ## that's not part of the benchmark section)
            remaining = readme_content[start_idx.stop+1:end]
            next_section = findfirst(r"\n## ", remaining)
            if next_section !== nothing
                end_idx = start_idx.stop + next_section.start - 1
            else
                end_idx = length(readme_content)
            end
            
            # Generate new benchmark section
            new_section = generate_benchmark_section(overall_table, snr_tables, env_snr_tables, config_dict)
            readme_content = readme_content[1:start_idx.start-1] * new_section * "\n\n" * readme_content[end_idx+1:end]
        end
    else
        # Add benchmark section before "## Related Documentation"
        related_docs = "## Related Documentation"
        if occursin(related_docs, readme_content)
            idx = findfirst(related_docs, readme_content)
            if idx !== nothing
                new_section = generate_benchmark_section(overall_table, snr_tables, env_snr_tables, config_dict)
                readme_content = readme_content[1:idx.start-1] * new_section * "\n\n" * readme_content[idx.start:end]
            end
        else
            # Append at the end
            new_section = generate_benchmark_section(overall_table, snr_tables, env_snr_tables, config_dict)
            readme_content = readme_content * "\n\n" * new_section
        end
    end
    
    write(README_PATH, readme_content)
end

function generate_benchmark_section(overall_table, snr_tables, env_snr_tables, config_dict)
    """Generate the benchmark section content."""
    section = "# Benchmark Results\n\n"
    section *= "## Overview\n\n"
    section *= "This section presents benchmark results comparing different hearing aid algorithms on the VOICEBANK_DEMAND dataset.\n\n"
    
    # Add overall summary table
    if overall_table !== nothing
        section *= "## Overall Summary\n\n"
        section *= overall_table
        section *= "\n"
    end
    
    # Add SNR tables
    if snr_tables !== nothing && !isempty(snr_tables)
        section *= "## Summary by SNR\n\n"
        for (metric, table) in snr_tables
            section *= table
        end
    end
    
    # Add environment+SNR tables
    if env_snr_tables !== nothing && !isempty(env_snr_tables)
        section *= "## Summary by Environment and SNR\n\n"
        for (metric, table) in env_snr_tables
            section *= table
        end
    end
    
    # Add configurations
    section *= "## Configuration Details\n\n"
    section *= "The following configurations were used for each hearing aid:\n\n"
    
    for (device_name, config_path) in config_dict
        if config_path === nothing
            continue
        end
        section *= "### $(device_name)\n\n"
        section *= "```toml\n"
        config_content = read(config_path, String)
        section *= config_content
        section *= "```\n\n"
    end
    
    return section
end

function main()
    println("Starting benchmark update...")
    
    # Ensure figures directory exists
    mkpath(FIGURES_DIR)
    
    # Find all hearing aid directories (excluding Baseline_clean)
    all_dirs = readdir(RESULTS_DIR)
    hearing_aid_dirs = filter(d -> d != "Baseline_clean" && isdir(joinpath(RESULTS_DIR, d)), all_dirs)
    
    println("Found hearing aid directories: ", hearing_aid_dirs)
    
    # Find latest runs and load data
    overall_data = Dict{String, Union{DataFrame, Nothing}}()
    snr_data = Dict{String, Union{DataFrame, Nothing}}()
    env_snr_data = Dict{String, Union{DataFrame, Nothing}}()
    config_dict = Dict{String, Union{String, Nothing}}()
    
    for dir_name in hearing_aid_dirs
        device_name = get_hearing_aid_name(dir_name)
        hearing_aid_dir = joinpath(RESULTS_DIR, dir_name)
        latest_run = find_latest_run(hearing_aid_dir)
        
        if latest_run === nothing
            println("Warning: No run found for $(dir_name)")
            continue
        end
        
        println("Processing $(device_name) from $(latest_run)")
        
        # Load data
        overall_data[device_name] = load_overall_summary(latest_run)
        snr_data[device_name] = load_summary_by_snr(latest_run)
        env_snr_data[device_name] = load_summary_by_environment_snr(latest_run)
        config_dict[device_name] = find_config_file(latest_run)
    end
    
    # Create tables
    println("Creating overall summary table...")
    overall_table = create_overall_summary_table(overall_data)
    
    println("Creating SNR tables...")
    snr_tables = create_snr_table(snr_data)
    
    println("Creating environment+SNR tables...")
    env_snr_tables = create_environment_snr_table(env_snr_data)
    
    # Update README
    println("Updating README...")
    update_readme_with_benchmark(overall_table, snr_tables, env_snr_tables, config_dict)
    println("README updated successfully!")
    
    println("Benchmark update complete!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

