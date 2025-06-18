module TSCPDetector

using Evolutionary 
using DifferentialEquations
using LabelledArrays
using Plots


# Include the necessary files
#include("_DataHandler.jl")
include("_ModelSimulation.jl")
include("_ModelHandling.jl")
include("_ObjectiveFunction.jl")
include("_ChangePointDetection.jl")
include("penalty.jl")
#include("_Visualisation.jl")

# Export necessary functions
export AbstractModelSpec, ODEModelSpec, DifferenceModelSpec, RegressionModelSpec, simulate_model, exponential_ode_model,
example_difference_model, example_regression_model,
ModelManager, get_initial_condition, segment_model, get_model_type,
extract_parameters, objective_function, wrapped_obj_function, custom_penalty,
optimize_with_changepoints, update_bounds!, evaluate_segment, detect_changepoints
call_penalty_fn, method_argnames, default_penalty, BIC_penalty, relative_length_penalty



end # module
