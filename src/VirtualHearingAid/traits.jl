"""
Processing strategy traits for hearing aid processing modes.

Processing strategies determine how audio is processed:
- `StreamProcessing`: Real-time streaming processing
- `BatchProcessingOnline`: Batch processing with online synthesis
- `BatchProcessingOffline`: Batch processing with offline synthesis (allows full result collection)
"""
abstract type ProcessingStrategy end

"""
    StreamProcessing

Real-time streaming processing strategy.
Processes audio block-by-block in real-time.
"""
struct StreamProcessing <: ProcessingStrategy end

"""
    BatchProcessingOnline

Batch processing with online synthesis.
Processes entire audio file but synthesizes block-by-block.
"""
struct BatchProcessingOnline <: ProcessingStrategy end

"""
    BatchProcessingOffline

Batch processing with offline synthesis.
Processes entire audio file and collects all results before synthesis.
Allows access to full processing history and intermediate results.
"""
struct BatchProcessingOffline <: ProcessingStrategy end

# Generic fallback for unknown hearing aid types
# Note: Type-specific implementations are defined in VirtualHearingAid.jl
# after the hearing aid types are defined to avoid forward reference issues.
processing_strategy(::Type{<:AbstractHearingAid}) = BatchProcessingOnline()

# Instance-level override storage
#
# Rationale for Dual Storage:
# The processing strategy is stored in two places:
# 1. In the hearing aid struct field (`ha.processing_strategy`) - initial/default strategy
# 2. In this override dictionary (`_ha_strategy_override`) - runtime override
#
# Why both?
# - The struct field provides a default strategy that is part of the hearing aid's
#   initial configuration and can be set during construction.
# - The override dictionary allows changing the strategy at runtime without
#   recreating the hearing aid, which is useful for testing different processing
#   modes or adapting to runtime conditions.
# - The override takes precedence when present, providing flexibility while
#   maintaining a sensible default.
#
# This design provides:
# - Flexibility: Can change strategy at runtime
# - Convenience: Default strategy stored in struct (no need to look it up)
# - Performance: Override lookup is O(1) and only checked when needed
const _ha_strategy_override = IdDict{AbstractHearingAid,ProcessingStrategy}()

"""
    set_processing_strategy!(ha, strategy)

Override processing strategy for a specific hearing aid instance at runtime.

This allows changing the processing strategy after the hearing aid is created,
which takes precedence over the strategy stored in the struct.

# Arguments
- `ha::AbstractHearingAid`: The hearing aid instance
- `strategy::ProcessingStrategy`: The processing strategy to use

# Returns
- `AbstractHearingAid`: The hearing aid instance (for method chaining)

# Example
```julia
ha = from_config(BaselineHearingAid, config)
set_processing_strategy!(ha, BatchProcessingOffline())
```
"""
function set_processing_strategy!(ha::AbstractHearingAid, strategy::ProcessingStrategy)
    _ha_strategy_override[ha] = strategy
    return ha
end

"""
    processing_strategy(ha)

Get processing strategy for a hearing aid instance.

Returns the runtime override if set, otherwise returns the default strategy
for the hearing aid type.

# Arguments
- `ha::AbstractHearingAid`: The hearing aid instance

# Returns
- `ProcessingStrategy`: The processing strategy to use

# Note
The strategy stored in `ha.processing_strategy` is the initial strategy.
Runtime overrides (via `set_processing_strategy!`) take precedence.
"""
processing_strategy(ha::AbstractHearingAid) =
    get(_ha_strategy_override, ha, processing_strategy(typeof(ha)))

# Backend traits
"""
Dispatch process to the appropriate processing strategy.
"""
function process(ha::AbstractHearingAid, sound::SampleBuf)
    # SEM hearing aids with offline processing use denoise directly
    if ha isa SEMHearingAid &&
       processing_strategy(ha) isa BatchProcessingOffline
        return denoise(ha, sound)
    end
    return process(processing_strategy(ha), ha, sound)
end
