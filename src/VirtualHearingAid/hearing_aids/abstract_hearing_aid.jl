using RxInfer

"""
    AbstractHearingAid

Abstract type for all hearing aid implementations.

# Required Methods

All hearing aid implementations must define:
- `process_block(ha::AbstractHearingAid, block::SampleBuf)` - Process a single audio block
- `backend_type(::Type{<:AbstractHearingAid})` - Return the backend type
- `getname(::Type{<:AbstractHearingAid})` - Return the hearing aid type name
- `getname(::AbstractHearingAid)` - Return the hearing aid instance name

# Structure

All hearing aid types should have at minimum:
- `frontend::AbstractFrontend` - Frontend processing component
- `backend::AbstractSPMBackend` (optional) - Backend processing component
- `processing_strategy::ProcessingStrategy` - Processing mode

# Example

```julia
struct MyHearingAid <: AbstractHearingAid
    frontend::WFBFrontend
    backend::MyBackend
    processing_strategy::ProcessingStrategy
end

backend_type(::Type{MyHearingAid}) = MyBackend
getname(::Type{MyHearingAid}) = "MyHearingAid"
getname(::MyHearingAid) = "MyHearingAid"
```
"""

"""
    getname(::Type{<:AbstractHearingAid})

Get the name of the hearing aid type as a string.

# Returns
- `String`: The name of the hearing aid type

# Example
```julia
getname(BaselineHearingAid)  # Returns "BaselineHearingAid"
```
"""
function getname end

"""
    getname(ha::AbstractHearingAid)

Get the name of a hearing aid instance.

# Arguments
- `ha::AbstractHearingAid`: The hearing aid instance

# Returns
- `String`: The name of the hearing aid

# Example
```julia
ha = from_config(BaselineHearingAid, config)
getname(ha)  # Returns "BaselineHearingAid"
```
"""
function getname(ha::AbstractHearingAid)
    getname(typeof(ha))
end

"""
    backend_type(::Type{<:AbstractHearingAid})

Get the backend type used by a hearing aid type.

# Returns
- `Type{<:AbstractSPMBackend}`: The backend type

# Example
```julia
backend_type(SEMHearingAid)  # Returns SEMBackend
```
"""
function backend_type end

"""
    process_block(ha::AbstractHearingAid, block::SampleBuf)

Process a single block of audio through the hearing aid.

This is the core processing function that handles one audio block at a time.
The default implementation uses `process_block_generic`, but specific hearing aids
can override this for custom behavior.

# Arguments
- `ha::AbstractHearingAid`: The hearing aid instance
- `block::SampleBuf`: A single block of audio samples

# Returns
- `Tuple{SampleBuf, Any}`: (output_signal, results)
  - `output_signal`: Processed audio output
  - `results`: Backend-specific results (can be `nothing`)

# Example
```julia
ha = from_config(BaselineHearingAid, config)
output, results = process_block(ha, audio_block)
```
"""
function process_block end

"""
    get_frontend(ha::AbstractHearingAid)

Get the frontend component of a hearing aid.

# Arguments
- `ha::AbstractHearingAid`: The hearing aid instance

# Returns
- `AbstractFrontend`: The frontend component

# Example
```julia
ha = from_config(BaselineHearingAid, config)
frontend = get_frontend(ha)
```
"""
function get_frontend(ha::AbstractHearingAid)
    ha.frontend
end

"""
    process_block_generic(ha::AbstractHearingAid, block::SampleBuf)

Shared per-block processing pipeline for all hearing aids.

This function implements the standard processing pipeline:
1. Extract features via frontend (`Frontends.process_frontend`)
2. Compute gains via backend (`compute_gains`)
3. Update frontend weights (`Frontends.update_weights!`)
4. Synthesize output (`Frontends.synthesize`)

# Arguments
- `ha::AbstractHearingAid`: The hearing aid instance
- `block::SampleBuf`: A single block of audio samples

# Returns
- `Tuple{SampleBuf, Any}`: (output_signal, results)
  - `output_signal`: Processed audio output
  - `results`: Backend-specific results (can be `nothing`)

# Performance
This function is marked `@inline` for performance optimization.
"""
@inline function process_block_generic(
    ha::AbstractHearingAid,
    block::SampledSignals.SampleBuf,
)
    frontend = get_frontend(ha)
    nbands = Frontends.get_nbands(frontend)

    powerdb = Frontends.process_frontend(frontend, block)
    gains, results = compute_gains(get_backend(ha), powerdb, nbands)
    Frontends.update_weights!(frontend, gains)
    output_signal = Frontends.synthesize(frontend)
    return output_signal, results
end

"""
Default process_block uses the generic pipeline. Specific hearing aids can
override if needed, but should prefer calling `process_block_generic`.
"""
process_block(ha::AbstractHearingAid, block::SampledSignals.SampleBuf) =
    process_block_generic(ha, block)

"""
    compute_gains(backend, powerdb, nbands)

Backend-specific gain computation.

This function dispatches to backend-specific implementations. The default
implementation returns unity gains (no processing).

# Arguments
- `backend`: The backend instance (can be `nothing` for baseline)
- `powerdb::AbstractVector{<:Real}`: Power spectrum in dB per frequency band
- `nbands::Integer`: Number of frequency bands

# Returns
- `Tuple{Vector{Float64}, Any}`: (gains, results)
  - `gains`: Gain values in linear scale (length `nbands`)
  - `results`: Backend-specific results (can be `nothing`)

# Performance
This function is marked `@inline` for performance optimization.

# Example
```julia
gains, results = compute_gains(backend, powerdb, 17)
```
"""
@inline function compute_gains(::Any, powerdb, nbands::Integer)
    gains = Vector{Float64}(undef, nbands)
    @inbounds for i = 1:nbands
        gains[i] = 1.0
    end
    return gains, nothing
end

"""
Get the backend component of a hearing aid.

Returns the backend for hearing aids that have one.
Type-stable implementations are provided for each hearing aid type.
"""
function get_backend end

# Type-stable implementations for each hearing aid type
# Note: These are defined in VirtualHearingAid.jl after the types are loaded
# to avoid forward reference issues.

"""
    get_processing_strategy(ha::AbstractHearingAid)

Get the processing strategy stored in the hearing aid struct.

Note: This returns the strategy stored in the struct. For the actual
strategy used (which may include runtime overrides), use `processing_strategy(ha)`.

# Arguments
- `ha::AbstractHearingAid`: The hearing aid instance

# Returns
- `ProcessingStrategy`: The processing strategy stored in the struct

# See also
- `processing_strategy(ha)`: Gets the effective processing strategy (includes overrides)
"""
function get_processing_strategy(ha::AbstractHearingAid)
    ha.processing_strategy
end

# ============================================================================
# Extension Points for New Hearing Aid Types
# ============================================================================

"""
# Creating a New Hearing Aid Type

To add a new hearing aid type:

1. **Define the struct** (in `hearing_aids/your_ha/type.jl`):
   ```julia
   struct YourHearingAid <: AbstractHearingAid
       frontend::WFBFrontend
       backend::YourBackend
       processing_strategy::ProcessingStrategy
   end
   ```

2. **Implement required methods**:
   ```julia
   backend_type(::Type{YourHearingAid}) = YourBackend
   getname(::Type{YourHearingAid}) = "YourHearingAid"
   getname(::YourHearingAid) = "YourHearingAid"
   ```

3. **Implement configuration** (in `hearing_aids/your_ha/config.jl`):
   ```julia
   function from_config(::Type{YourHearingAid}, config::AbstractDict)
       return _from_config_impl(YourHearingAid, config)
   end
   
   function init_backend(::Type{YourBackend}, config::AbstractDict)
       # Your backend initialization
   end
   ```

4. **Implement processing** (in `hearing_aids/your_ha/processing.jl`):
   ```julia
   # Use default if backend follows standard pattern
   process_block(ha::YourHearingAid, block::SampleBuf) = process_block_generic(ha, block)
   
   # Or implement compute_gains if needed:
   function compute_gains(backend::YourBackend, powerdb, nbands::Integer)
       # Your gain computation
   end
   ```

5. **Add to exports** in `VirtualHearingAid.jl`
"""
