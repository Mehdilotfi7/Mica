module TSCPDetector

# ========== Dependencies ==========
using Evolutionary
using DifferentialEquations
using LabelledArrays

# ========== Internal Modules ==========
include("ModelSimulation.jl")
include("ModelHandling.jl")
include("ObjectiveFunction.jl")
include("ChangePointDetection.jl")
include("penalty.jl")

# ========== Exported API ==========

# -- Model Simulation --
export AbstractModelSpec, ODEModelSpec, DifferenceModelSpec, RegressionModelSpec
export simulate_model, exponential_ode_model
export example_difference_model, example_regression_model

# -- Model Handling --
export ModelManager, get_initial_condition, update_initial_condition, segment_model, get_model_type

# -- Objective Function --
export extract_parameters
export objective_function, wrapped_obj_function

# -- Change Point Detection --
export optimize_with_changepoints, update_bounds!, evaluate_segment, detect_changepoints

# -- Penalties --
export call_penalty_fn, method_argnames, default_penalty, BIC_penalty

end # module TSCPDetector
