"""
Get parameters for a component.
Generic implementation that works for any component with a params field.
"""
function get_params(component)
    hasfield(typeof(component), :params) ? component.params : nothing
end

# Also include a method for directly getting params from a hearing aid
function get_params(ha::AbstractHearingAid)
    get_params(get_frontend(ha))
end
