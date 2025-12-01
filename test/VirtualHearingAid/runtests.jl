using ReTestItems
using Experiments
using Experiments.VirtualHearingAid
using Test

# Optional test dependencies
HAS_COVERAGE = false
try
    using CpuId
    using Coverage
    global HAS_COVERAGE = true
catch
end

include("hearing_aids/test_baseline.jl")
include("hearing_aids/test_sem.jl")
include("frontends/test_wfb.jl")
include("validation/test_validation.jl")

# Generate coverage report for CI
if HAS_COVERAGE && get(ENV, "COVERAGE", "false") == "true"
    @info "Generating coverage report"
    coverage = process_folder(joinpath(pkgdir(Experiments), "src"))
    LCOV.writefile(joinpath(pkgdir(Experiments), "lcov.info"), coverage)
end
