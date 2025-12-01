"""
Simple pass-through hearing aid implementation.
"""

struct BaselineHearingAid <: AbstractHearingAid
    frontend::WFBFrontend
    backend::BaselineBackend
    processing_strategy::ProcessingStrategy
end

"""
Get the name of the hearing aid type.
"""
getname(::Type{BaselineHearingAid}) = "BaselineHearingAid"
getname(::BaselineHearingAid) = "BaselineHearingAid"

# Add this function
backend_type(::Type{BaselineHearingAid}) = BaselineBackend
