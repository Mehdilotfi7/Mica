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
- `chromosome`: The vector of model parameters (includes global and segment-specific).
- `change_points`: Vector of indices defining segmentation boundaries.
- `n_global`: Number of global parameters.
- `n_segment_specific`: Number of parameters specific to each segment.
- `extract_parameters`: A function that extracts global and per-segment parameters from the chromosome.
- `parnames`: Names of the model parameters (used to construct LArray or Dict).
- `model_manager`: An instance of `ModelManager`, holding model type and base config.
- `simulate_model`: A function to simulate the model (multi-dispatch on model type).
- `loss_function`: Loss function applied to a single segment.
- `segment_loss`: Function that computes the loss given true vs simulated data using `loss_function`.
- `data`: The full observed data (e.g., vector, matrix, or tuple of vectors).

# Returns
- Total loss across all segments.
"""
function objective_function(
    chromosome, 
    change_points::Vector{Int}, 
    n_global::Int, 
    n_segment_specific::Int, 
    extract_parameters::Function,
    parnames::Vector{Symbol}, 
    model_manager::ModelManager, 
    simulate_model::Function,
    loss_function::Function,
    segment_loss::Function,
    data::Any
)
    constant_pars, segment_pars_list = extract_parameters(chromosome, n_global, n_segment_specific)
    total_loss = 0.0
    num_segments = length(change_points) + 1

    # For initial condition passing
    u0 = get_initial_condition(model_manager)

    for i in 1:num_segments
        idx_start = (i == 1) ? 1 : change_points[i - 1] + 1
        idx_end   = (i > length(change_points)) ? length(data[1]) : change_points[i]

        # Slice data for current segment
        segment_data = (eltype(data) <: AbstractVector) ? 
            [vec[idx_start:idx_end] for vec in data] : 
            data[idx_start:idx_end]

        seg_pars = segment_pars_list[i]
        all_pars = vcat(constant_pars, seg_pars)
        model_spec = segment_model(model_manager, all_pars, parnames, idx_start, idx_end, u0)

        sim_data = simulate_model(model_spec)
        total_loss += segment_loss(segment_data, sim_data, loss_function)

        # Update initial condition if applicable
        sim_data.state[end]
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
