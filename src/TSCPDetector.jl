module TSCPDetector

# ========== Dependencies ==========
using Evolutionary
using DifferentialEquations
using LabelledArrays

# ========== Internal Modules ==========
include("_ModelSimulation.jl")
include("_ModelHandling.jl")
include("_ObjectiveFunction.jl")
include("_ChangePointDetection.jl")
include("penalty.jl")

# ========== Exported API ==========

# -- Model Specifications --
export AbstractModelSpec, ODEModelSpec, DifferenceModelSpec, RegressionModelSpec
export simulate_model, exponential_ode_model
export example_difference_model, example_regression_model

# -- Model Handling --
export ModelManager, get_initial_condition, get_model_type, update_bounds!
export extract_parameters

# -- Objective Function & Penalties --
export objective_function, wrapped_obj_function
export custom_penalty, call_penalty_fn, method_argnames
export default_penalty, BIC_penalty, relative_length_penalty

# -- Change Point Detection --
export segment_model, evaluate_segment, detect_changepoints

end # module TSCPDetector
