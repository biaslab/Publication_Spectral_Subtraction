using ReTestItems
using Experiments
using Experiments.HASoundProcessing
using Test

# Optional test dependencies
HAS_CPUID = false
HAS_COVERAGE = false
try
    using CpuId
    using Coverage
    global HAS_CPUID = true
    global HAS_COVERAGE = true
catch
end

# Include all test files
include("test_sem/test_algorithm.jl")
include("nodes/sigmoid_tests.jl")
include("rules/sigmoid/out_tests.jl")
include("rules/sigmoid/in_tests.jl")
include("rules/sigmoid/zeta_tests.jl")
include("test_utils.jl")

cpu = HAS_CPUID ? cpucores() : 1

runtests(
    HASoundProcessing,
    nworkers = cpu == 0 ? 1 : cpu,
    nworker_threads = HAS_CPUID ? (cpu == 0 ? 1 : Int(cputhreads() / cpucores())) : 1,
    memory_threshold = 1.0,
)

# Generate coverage report for CI.yml
if HAS_COVERAGE && get(ENV, "COVERAGE", "false") == "true"
    @info "Generating coverage report"
    coverage = process_folder(joinpath(pkgdir(Experiments), "src"))
    LCOV.writefile(joinpath(pkgdir(Experiments), "lcov.info"), coverage)
end
