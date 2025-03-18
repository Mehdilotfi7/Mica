module ChangePointDetection

using Evolutionary
using ObjectiveFunction
using Visualization

"""
# ChangePointDetection Module

This module implements a framework for detecting change points in time-series or similar data
using optimization techniques. The primary function is `detect_changepoints`, which identifies
points in a dataset where the underlying model parameters change.

It is part of a package and uses the Evolutionary package for optimization, as well as 
functions provided by the `ObjectiveFunction` module.

### Main Functions:
- `detect_changepoints`: Detect change points in data.
- `optimize_with_changepoints`: Optimize the objective function with fixed change points.
- `evaluate_segment`: Evaluate potential change points within a segment.
- `update_bounds!`: Update optimization bounds dynamically.
"""
# Module definition begins here.

# ----------------------------------------------------------------------
# Function: optimize_with_changepoints
# ----------------------------------------------------------------------
"""
    optimize_with_changepoints(objective_function, chromosome, CP, bounds, ga, ...)

Optimize the given objective function with fixed change points using Evolutionary algorithms.

### Arguments:
- `objective_function`: The function to minimize.
- `chromosome`: Initial guesses for optimization parameters.
- `CP`: Current change points.
- `bounds`: Bounds for parameters (lower and upper).
- `ga`: Evolutionary algorithm configuration.
- Additional arguments: Passed directly to the `objective_function`.

### Returns:
- The minimum loss and the best parameter set achieving it.

This function wraps the `objective_function` to include fixed change points and uses the 
Evolutionary framework to perform optimization.
"""
function optimize_with_changepoints(
    objective_function, chromosome, CP, bounds, ga,
    n_global, n_segment_specific, extract_parameters, parnames,
    model_function, simulate_model, extra_data, num_steps, tspan,
    loss_function, compare_variables, data_CP, initial_conditions;
    options=Evolutionary.Options(show_trace=false)
)
    wrapped_obj = chrom -> objective_function(
        chrom, CP, n_global, n_segment_specific, extract_parameters, parnames,
        model_function, simulate_model, extra_data, num_steps, tspan,
        loss_function, compare_variables, data_CP, initial_conditions
    )
    result = Evolutionary.optimize(wrapped_obj, BoxConstraints(bounds...), chromosome, ga, options)
    return Evolutionary.minimum(result), Evolutionary.minimizer(result)
end

# ----------------------------------------------------------------------
# Function: update_bounds!
# ----------------------------------------------------------------------
"""
    update_bounds!(chromosome, bounds, n_global, n_segment_specific, extract_parameters)

Update the optimization parameter bounds and chromosome by appending segment-specific parameters.

### Arguments:
- `chromosome`: Current optimization parameters.
- `bounds`: Tuple of lower and upper bounds for parameters.
- `n_global`: Number of global parameters.
- `n_segment_specific`: Number of segment-specific parameters.
- `extract_parameters`: Function to extract global and segment-specific parameters.

This function ensures that bounds and chromosomes are dynamically updated as new change points
and corresponding parameters are added.
"""
function update_bounds!(chromosome, bounds, n_global, n_segment_specific, extract_parameters)
    global_params, seg_specific = extract_parameters(chromosome, n_global, n_segment_specific)
    global_lower, seg_lower = extract_parameters(bounds[1], n_global, n_segment_specific)
    global_upper, seg_upper = extract_parameters(bounds[2], n_global, n_segment_specific)
    
    append!(chromosome, seg_specific[1])
    append!(bounds[1], seg_lower[1])
    append!(bounds[2], seg_upper[1])
end

# ----------------------------------------------------------------------
# Function: evaluate_segment
# ----------------------------------------------------------------------
"""
    evaluate_segment(objective_function, a, b, CP, bounds, chromosome, ga, ...)

Evaluate potential change points within a segment of data by optimizing the objective function
at each candidate change point.

### Arguments:
- `objective_function`: The function to minimize.
- `a, b`: Indices defining the current segment.
- `CP`: Current set of change points.
- `bounds, chromosome, ga`: Parameters for optimization.
- Additional arguments: Passed to the `objective_function`.

### Returns:
- A vector of losses (`x`) and corresponding optimized parameters (`y`).

This function systematically evaluates potential new change points within the given segment 
and computes their associated losses.
"""
function evaluate_segment(
    objective_function, a, b, CP, bounds, chromosome, ga, pen, min_length, step,
    n_global, n_segment_specific, extract_parameters, parnames,
    model_function, simulate_model, extra_data, num_steps, tspan,
    loss_function, compare_variables, data_CP, initial_conditions
)
    x, y = Float64[], Vector{Vector{Float64}}()
    for j in (a + min_length):step:(b - min_length)
        new_cp = sort([CP; j])
        loss, best = optimize_with_changepoints(
            objective_function, chromosome, new_cp, bounds, ga,
            n_global, n_segment_specific, extract_parameters, parnames,
            model_function, simulate_model, extra_data, num_steps, tspan,
            loss_function, compare_variables, data_CP, initial_conditions
        )
        push!(x, loss + pen)
        push!(y, best)
    end
    return x, y
end

# ----------------------------------------------------------------------
# Main Function: detect_changepoints
# ----------------------------------------------------------------------
"""
    detect_changepoints(objective_function, n, n_global, n_segment_specific, ...)

Detect change points in data by iteratively optimizing the loss and splitting
segments until no significant improvements can be made.

### Arguments:
- `objective_function`: The function to minimize.
- `n`: Size of the dataset.
- `n_global, n_segment_specific`: Number of global and segment-specific parameters.
- Additional arguments: Includes data, optimization parameters, and model configuration.

### Returns:
- `CP`: The detected change points as a vector.
- `best_params`: The optimized parameters corresponding to the detected change points.

This is the main function of the module. It uses a regularization term (`pen`) to balance
model fit and complexity. The algorithm iteratively refines the set of change points 
by evaluating and optimizing each segment.
"""
function detect_changepoints(
    objective_function, n, n_global, n_segment_specific, extract_parameters, parnames,
    model_function, simulate_model, extra_data, num_steps, tspan,
    loss_function, compare_variables, data_CP, initial_conditions,
    initial_chromosome, bounds, ga, min_length, step;
    pen=log(n)
)
    tau = [(0, n)]
    CP = Int[]
    @show CP

    loss_val, best_params = optimize_with_changepoints(
        objective_function, initial_chromosome, CP, bounds, ga,
        n_global, n_segment_specific, extract_parameters, parnames,
        model_function, simulate_model, extra_data, num_steps, tspan,
        loss_function, compare_variables, data_CP, initial_conditions
    )
    @show best_params
    update_bounds!(initial_chromosome, bounds, n_global, n_segment_specific, extract_parameters)

    while !isempty(tau)
        a, b = pop!(tau)
        x, y = evaluate_segment(
            objective_function, a, b, CP, bounds, initial_chromosome, ga, pen, min_length, step,
            n_global, n_segment_specific, extract_parameters, parnames,
            model_function, simulate_model, extra_data, num_steps, tspan,
            loss_function, compare_variables, data_CP, initial_conditions
        )
        if !isempty(x)
            minval, idx = findmin(x)
            if minval < loss_val
                chpt = a + (idx * step)
                push!(CP, chpt)
                CP = sort(CP)
                loss_val = minval
                best_params = y[idx]
                update_bounds!(initial_chromosome, bounds, n_global, n_segment_specific, extract_parameters)
                if chpt != a + min_length
                    push!(tau, (a, chpt))
                end
                if chpt != b - min_length
                    push!(tau, (chpt, b))
                end
            end
        end
    end

    return CP, best_params
end

end
