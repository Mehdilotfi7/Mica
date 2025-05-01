module _ObjectiveFunction

using DataFrames
using LabelledArrays

# =============================================================================
# extract_parameters Function
# =============================================================================
"""
    extract_parameters(chromosome, n_global, n_segment_specific)

Splits a chromosome vector into global and segment-specific parameters.

# Arguments
- `chromosome`: Flat vector of all parameters.
- `n_global`: Number of global parameters.
- `n_segment_specific`: Number of segment parameters for each segment.

# Returns
- `(global_parameters, segment_parameters)`
"""
function extract_parameters(chromosome::Vector{T}, n_global::Int, n_segment_specific::Int) where T
    global_parameters = chromosome[1:n_global]
    segment_parameters = [chromosome[i:i+n_segment_specific-1] for i in n_global+1:n_segment_specific:length(chromosome)]
    return global_parameters, segment_parameters
end

# =============================================================================
# objective_function
# =============================================================================
"""
    objective_function(...)

Computes the total loss for the current chromosome, simulating each segment separately.

# Arguments
- `chromosome`: Parameters to optimize.
- `change_points`: Detected change points.
- `n_global`, `n_segment_specific`: Global and segment parameter counts.
- `extract_parameters`: Function to split the chromosome.
- `parnames`: Parameter names for building labeled arrays.
- `model_function`, `simulate_model`: Model and simulator.
- `loss_function`: User-defined function that compares data and simulation and returns a loss value.
# Note
The user-defined `loss_function(data_segment, simulated_data)` must fully handle all
required transformations, selections, or preprocessing steps (e.g., selecting variables,
normalizing, log-transforming, etc.).
- `data_CP`: Original data divided by change points.

# Keyword Arguments
- `initial_conditions`, `extra_data`, `num_steps`, `tspan`

# Returns
- `total_loss`: Sum of segment losses.
"""
function objective_function(
    chromosome, 
    change_points, 
    n_global, 
    n_segment_specific, 
    extract_parameters,
    base_model::AbstractModelSpec,
    simulate_model::Function, 
    loss_function::Function, 
    segment_loss::Function,
    data_CP,
    parnames
)
    constant_pars, segment_pars = extract_parameters(chromosome, n_global, n_segment_specific)
    num_segments = length(change_points) + 1
    total_loss = 0.0
    u0 = get_initial_condition(base_model)

    for i in 1:num_segments
        idx_start = i == 1 ? 1 : change_points[i - 1] + 1
        idx_end = i <= length(change_points) ? change_points[i] : size(data_CP[1], 1)
        data_segment = map(v -> v[idx_start:idx_end], data_CP)

        seg_pars_i = segment_pars[i]
        model_segment = segment_model(base_model, seg_pars_i, parnames, idx_start, idx_end, u0)

        simulated_data = simulate_model(model_segment)
        loss = segment_loss(data_segment, simulated_data, loss_function)
        total_loss += loss

        # Only needed if next segment depends on this output (like for ODEs)
        u0 = simulated_data[!, end][1]  # update if needed based on model
    end

    return total_loss
end


# =============================================================================
# Wrapped Objective (optional)
# =============================================================================
"""
    wrapped_obj_function(chromosome)

Convenient closure to call `objective_function` with fixed outer parameters.
"""
function wrapped_obj_function(chromosome)
    return objective_function(
        chromosome, CP, n_global, n_segment_specific, extract_parameters,
        parnames, model_function, simulate_model, loss_function, segment_loss, data_CP;
        initial_conditions=initial_conditions, extra_data=extra_data,
        num_steps=num_steps, tspan=tspan
    )
end

end # module
