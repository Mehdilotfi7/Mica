module _ObjectiveFunction

using DataFrames
using LabelledArrays

# =============================================================================
# segment_loss Function
# =============================================================================
"""
    segment_loss(segment_data, simulated_data, loss_function; compare_variables=nothing, transformation=identity)

Calculate the loss between observed and simulated segment data.

# Arguments
- `segment_data`: Actual segment data (Vector or Matrix).
- `simulated_data`: Simulated model output.
- `loss_function`: A function measuring the discrepancy (e.g., L2 loss).
- `compare_variables`: Indices of variables to compare. Defaults to all.
- `transformation`: A transformation applied before loss calculation (e.g., log).

# Returns
- Loss value.
"""
function segment_loss(segment_data, simulated_data, loss_function;
                      compare_variables=nothing, transformation=identity)
    if compare_variables === nothing
        compare_variables = ndims(segment_data) == 1 ? 1 : length(segment_data) : 1:size(segment_data, 1)
    end

    # Select and transform
    transformed_sim_data = ndims(simulated_data) == 1 ?
        transformation(simulated_data[compare_variables]) :
        transformation(simulated_data[compare_variables, :])

    transformed_segment_data = transformation(segment_data)

    return loss_function(transformed_segment_data, transformed_sim_data)
end

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
- `loss_function`, `segment_loss`: Loss calculators.
- `data_CP`: Original data divided by change points.

# Keyword Arguments
- `initial_conditions`, `extra_data`, `num_steps`, `tspan`, `compare_variables`, `transformation`

# Returns
- `total_loss`: Sum of segment losses.
"""
function objective_function(
    chromosome, change_points, n_global, n_segment_specific, extract_parameters,
    parnames, model_function, simulate_model, loss_function, segment_loss, data_CP;
    initial_conditions=nothing, extra_data=nothing, num_steps=nothing,
    tspan=nothing, compare_variables=nothing, transformation=identity
)
    num_segments = length(change_points) + 1
    global_pars, segment_pars = extract_parameters(chromosome, n_global, n_segment_specific)

    total_loss = 0.0
    u0 = initial_conditions
    segment_indices = [0; change_points; (isnothing(num_steps) ? last(tspan) : size(data_CP, 1))]

    for i in 1:num_segments
        idx_start, idx_end = segment_indices[i]+1, segment_indices[i+1]

        data_segment = isnothing(num_steps) ?
            data_CP[idx_start:idx_end] :
            [d[idx_start:idx_end] for d in data_CP]

        tspan_segment = isnothing(num_steps) ? (idx_start-1, idx_end-1) : nothing
        u_segment = isnothing(extra_data) ? nothing :
            (eltype(extra_data) <: AbstractVector ? [e[idx_start:idx_end] for e in extra_data] : extra_data[idx_start:idx_end])

        params = @LArray [global_pars; segment_pars[i]] parnames

        simulated = isnothing(num_steps) ?
            simulate_model(model_function, params, u0; tspan=tspan_segment) :
            simulate_model(model_function, params, u0; extra_data=u_segment, num_steps=idx_end - idx_start + 1)

        total_loss += segment_loss(data_segment, simulated, loss_function; compare_variables=compare_variables, transformation=transformation)

        u0 = simulated[:, end]  # Update initial condition for next segment
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
        num_steps=num_steps, tspan=tspan, compare_variables=compare_variables
    )
end

end # module
