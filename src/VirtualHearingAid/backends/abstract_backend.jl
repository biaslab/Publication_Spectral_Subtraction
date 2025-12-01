"""
Interfaces for backend implementations. 

Backends are typically implemented in the HASoundProcessing package, but
we define a common interface here.

# Backend Interface

All backend implementations must:
1. Subtype `AbstractSPMBackend` (from HASoundProcessing)
2. Implement `init_backend(::Type{BackendType}, config::AbstractDict)` - Configuration initialization
3. Implement `compute_gains(backend::BackendType, powerdb, nbands)` - Gain computation (via multiple dispatch)

The `compute_gains` function is the core backend interface. It takes:
- `powerdb::AbstractVector{<:Real}` - Power spectrum in dB per frequency band
- `nbands::Integer` - Number of frequency bands

And returns:
- `(gains::Vector{Float64}, results::Any)` - Gain values in linear scale and algorithm-specific results

# Example

```julia
struct MyBackend <: AbstractSPMBackend
    params::MyParameters
end

function init_backend(::Type{MyBackend}, config::AbstractDict)
    # Extract parameters from config
    params = MyParameters(...)
    return MyBackend(params)
end

# Gain computation (defined in processing.jl or abstract_hearing_aid.jl)
function compute_gains(backend::MyBackend, powerdb, nbands::Integer)
    # Your gain computation logic
    gains = compute_my_algorithm(backend, powerdb)
    results = MyResults(...)
    return gains, results
end
```

# Note

The `compute_gains` function is defined in `hearing_aids/abstract_hearing_aid.jl` to allow
multiple dispatch based on backend type. Each backend should implement its own version.
"""

"""
    init_backend(::Type{BackendType}, config::AbstractDict)

Abstract interface for initializing a backend from configuration.

Each backend type must implement this function to create a backend instance
from a configuration dictionary.

# Arguments
- `BackendType`: The type of backend to initialize
- `config::AbstractDict`: Configuration dictionary containing backend parameters

# Returns
- `BackendType`: An initialized backend instance

# Throws
- `ArgumentError`: If required configuration fields are missing or invalid

# Example
```julia
backend = init_backend(SEMBackend, config)
```
"""
function init_backend end

"""
    compute_gains(backend, powerdb, nbands)

Backend interface for gain computation.

All backends must implement this function (via multiple dispatch).
This is the core backend interface method that computes gain values
based on the power spectrum.

# Arguments
- `backend`: The backend instance (subtype of `AbstractSPMBackend`)
- `powerdb::AbstractVector{<:Real}`: Power spectrum in dB per frequency band
- `nbands::Integer`: Number of frequency bands

# Returns
- `Tuple{Vector{Float64}, Any}`: (gains, results)
  - `gains`: Gain values in linear scale (length `nbands`)
  - `results`: Backend-specific results (can be `nothing`)

# Implementation Note
The default implementation is defined in `hearing_aids/abstract_hearing_aid.jl`
and returns unity gains. Each backend should implement its own version via
multiple dispatch to provide algorithm-specific gain computation.

# Example
```julia
# Default implementation (unity gains)
gains, results = compute_gains(backend, powerdb, 17)

# Backend-specific implementation
function compute_gains(backend::MyBackend, powerdb, nbands::Integer)
    gains = compute_my_algorithm(backend, powerdb)
    results = MyResults(...)
    return gains, results
end
```
"""
function compute_gains end

"""
Empty backend for baseline hearing aid.
"""
struct BaselineBackend <: AbstractSPMBackend end

"""
Get the backend type for BaselineBackend.
"""
backend_type(::Type{BaselineBackend}) = BaselineBackend

"""
Initialize Baseline backend from configuration.
"""
init_backend(::Type{BaselineBackend}, config::AbstractDict) = BaselineBackend()

