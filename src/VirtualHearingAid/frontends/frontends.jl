module Frontends

using SampledSignals

# Include the abstract frontend interface
include("abstract_frontend.jl")

# Include all frontend implementations
include("wfb/wfb_frontend.jl")

# Import from WFB module (avoid importing functions that collide with our generics)
using .WFB: WFBFrontend, WFBFrontendParameters, synthesize

# Export the abstract frontend types and functions
export
    # Abstract types
    AbstractFrontend,
    AbstractFrontendParameters,

    # Interface functions
    process_frontend,
    update_weights!,
    get_params,
    synthesize,
    get_nbands

# Re-export specific frontend implementations
export WFBFrontend, WFBFrontendParameters

# Provide a uniform frontend API: delegate to WFB, supporting either `process_fronted` (new) or `process_frontend` (legacy)
function process_frontend(frontend::WFBFrontend, block::SampledSignals.SampleBuf)
    if isdefined(WFB, :process_fronted)
        return (getfield(WFB, :process_fronted))(frontend, block)
    else
        return WFB.process_frontend(frontend, block)
    end
end

# Forward update_weights! to WFB implementation to keep API uniform
function update_weights!(frontend::WFBFrontend, gains::Vector{Float64})
    return WFB.update_weights!(frontend, gains)
end

# Forward getter for number of bands to avoid reaching into WFB internals at call sites
get_nbands(frontend::WFBFrontend) = WFB.get_nbands(frontend)

end # module Frontends
