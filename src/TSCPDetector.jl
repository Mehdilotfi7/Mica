module TSCPDetector

using Evolutionary 
using DifferentialEquations
using LabelledArrays
using Plots

# Export necessary functions
export load_data, simulate_model, example_ode_model, example_regression_model, segment_loss, objective_function, plot_results, initial_optimization, update_chromosome_bounds!, evaluate_segment, update_tau!, ChangePointDetector

# Include the necessary files
include("_DataHandler.jl")
include("_ModelSimulation.jl")
include("_ModelHandling.jl")
include("_ObjectiveFunction.jl")
include("_ChangePointDetection.jl")
include("_Visualisation.jl")


end # module
