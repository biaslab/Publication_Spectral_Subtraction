"""
Required interface for all frontend implementations:

- process_frontend(frontend::AbstractFrontend, block::SampleBuf)
- update_weights!(frontend::AbstractFrontend, weights::Vector{Float64})
- get_params(frontend::AbstractFrontend)
- get_nbands(frontend::AbstractFrontend)
"""

"""
Process a block of audio through the frontend.
"""
function process_frontend end

"""
Update the weights used in synthesis.
"""
function update_weights! end

"""
Get the parameters of the frontend.
"""
function get_params end

"""
Get the number of frequency bands in the frontend.
"""
function get_nbands end

"""
Abstract type for frontend processing components.
"""
abstract type AbstractFrontend end

"""
Abstract type for frontend parameter containers.
"""
abstract type AbstractFrontendParameters end
