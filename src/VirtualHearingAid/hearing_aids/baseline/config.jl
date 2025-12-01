"""
    from_config(::Type{BaselineHearingAid}, config::AbstractDict)

Create a BaselineHearingAid from a configuration dictionary.

Uses the generic `_from_config_impl` helper from `utils/config.jl`.

# Arguments
- `config::AbstractDict`: Configuration dictionary containing hearing aid parameters

# Returns
- `BaselineHearingAid`: A configured baseline hearing aid instance

# Example
```julia
config = TOML.parsefile("config.toml")
ha = from_config(BaselineHearingAid, config)
```
"""
function from_config(::Type{BaselineHearingAid}, config::AbstractDict)
    return _from_config_impl(BaselineHearingAid, config)
end
