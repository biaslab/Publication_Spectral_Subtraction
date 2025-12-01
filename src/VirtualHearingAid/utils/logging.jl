"""
Logging utilities for hearing aid processing.

This module provides structured logging for monitoring and debugging
batch processing operations.
"""

using Logging

"""
    @log_batch_processing

Macro to log batch processing operations with timing and metadata.
"""
macro log_batch_processing(operation, ha_type, sound_length)
    return esc(
        quote
            start_time = time()
            @info "Starting batch processing" operation=$operation hearing_aid_type=$ha_type sound_length=$sound_length
            try
                result = $operation
                elapsed_time = time() - start_time
                @info "Batch processing completed" operation=$operation elapsed_time=elapsed_time
                return result
            catch e
                elapsed_time = time() - start_time
                @error "Batch processing failed" operation=$operation error=e elapsed_time=elapsed_time
                rethrow(e)
            end
        end,
    )
end

"""
    log_processing_metrics(operation::String, metrics::Dict)

Log processing metrics for monitoring and analysis.

# Arguments
- `operation`: Name of the processing operation
- `metrics`: Dictionary of metrics to log
"""
function log_processing_metrics(operation::String, metrics::Dict)
    @info "Processing metrics" operation=operation metrics=metrics
end

"""
    log_backend_processing(backend_type::String, num_blocks::Int, processing_time::Float64)

Log backend processing performance metrics.

# Arguments
- `backend_type`: Type of backend being used
- `num_blocks`: Number of blocks processed
- `processing_time`: Time taken for processing
"""
function log_backend_processing(
    backend_type::String,
    num_blocks::Int,
    processing_time::Float64,
)
    blocks_per_second = num_blocks / processing_time
    @info "Backend processing completed" backend_type=backend_type num_blocks=num_blocks processing_time=processing_time blocks_per_second=blocks_per_second
end

"""
    log_synthesis_metrics(synthesis_type::String, num_blocks::Int, synthesis_time::Float64)

Log synthesis performance metrics.

# Arguments
- `synthesis_type`: Type of synthesis being performed
- `num_blocks`: Number of blocks synthesized
- `synthesis_time`: Time taken for synthesis
"""
function log_synthesis_metrics(
    synthesis_type::String,
    num_blocks::Int,
    synthesis_time::Float64,
)
    blocks_per_second = num_blocks / synthesis_time
    @info "Synthesis completed" synthesis_type=synthesis_type num_blocks=num_blocks synthesis_time=synthesis_time blocks_per_second=blocks_per_second
end

"""
    log_memory_usage(operation::String, memory_usage::Float64)

Log memory usage for monitoring resource consumption.

# Arguments
- `operation`: Name of the operation
- `memory_usage`: Memory usage in MB
"""
function log_memory_usage(operation::String, memory_usage::Float64)
    @info "Memory usage" operation=operation memory_mb=memory_usage
end
