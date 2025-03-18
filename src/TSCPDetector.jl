module TSCPDetector
# Include the necessary files
include("DataHandling.jl")
include("ModelSimulation.jl")
include("ObjectiveFunction.jl")
include("ChangePointDetection.jl")
include("Visualization.jl")

using Evolutionary
using .Visualization
using .DataHandling
using .ModelSimulation
using .ObjectiveFunction
using .ChangePointDetection

# Export necessary functions
export load_data, simulate_model, example_ode_model, example_regression_model, segment_loss, objective_function, plot_results, initial_optimization, update_chromosome_bounds!, evaluate_segment, update_tau!, ChangePointDetector



end # module
