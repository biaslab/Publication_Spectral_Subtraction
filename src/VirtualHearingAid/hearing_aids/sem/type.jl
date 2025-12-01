"""
Spectral Enhancement Model (SEM) hearing aid implementation.
"""
struct SEMHearingAid <: AbstractHearingAid
    frontend::WFBFrontend
    backend::SEMBackend
    processing_strategy::ProcessingStrategy
end

"""
Get the backend type for SEMHearingAid.
"""
backend_type(::Type{SEMHearingAid}) = SEMBackend

"""
Get the name of the hearing aid type.
"""
getname(::Type{SEMHearingAid}) = "SEMHearingAid"
getname(::SEMHearingAid) = "SEMHearingAid"
