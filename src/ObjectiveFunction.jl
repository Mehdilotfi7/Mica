module ObjectiveFunction

using DataFrames
using LabelledArrays  

"""
Define the segment loss function.

# Arguments
- `segment_data::AbstractArray{T}`: The actual data for the segment. Can be 1D or 2D.
- `simulated_data::AbstractArray{T}`: The simulated data for the segment. Same dimensions as `segment_data`.
- `loss_function::Function`: A user-defined function to compute the loss between `segment_data` and `simulated_data`.
- `transformation::Function = identity`: Optional transformation function applied to data before calculating loss. Default is no transformation.

# Returns
- The calculated loss for the segment, based on the user-defined loss function.
"""
function segment_loss(
    segment_data::AbstractArray{T},
    simulated_data::AbstractArray{T},
    loss_function::Function,
    compare_variables::Union{Vector{Int}, Nothing} = nothing, 
    transformation::Function = identity
) where T
    # Infer default `compare_variables` if not provided
    if compare_variables === nothing
        if ndims(segment_data) == 1
            compare_variables = 1:length(segment_data)
        else
            compare_variables = 1:size(segment_data, 1)
        end
    end
    @show length(simulated_data)
    @show compare_variables
    # Apply transformations and indexing
    if ndims(simulated_data) == 1
       transformed_simulated_data = transformation(simulated_data[compare_variables])
    else
       transformed_simulated_data = transformation(simulated_data[compare_variables, :])
    end
    transformed_segment_data = transformation(segment_data)
    return loss_function(transformed_segment_data, transformed_simulated_data)
end

"""
Extract global and segment-specific parameters from the chromosome.

# Arguments
- `chromosome::Vector{Float64}`: A vector of parameters including both global and segment-specific parameters.
- `n_global::Int`: The number of global parameters.
- `n_segment_specific::Int`: The number of segment-specific parameters per segment.

# Returns
- `global_parameters`: A vector of global parameters.
- `segment_parameters`: A vector of vectors, each containing segment-specific parameters.
"""
function extract_parameters(chromosome::Vector{T}, n_global::Int, n_segment_specific::Int) where T
    global_parameters = chromosome[1:n_global]
    segment_parameters = Vector{Vector{T}}()
    idx = n_global + 1
    while idx <= length(chromosome)
        push!(segment_parameters, chromosome[idx:idx + n_segment_specific - 1])
        idx += n_segment_specific
    end
    return global_parameters, segment_parameters
end

"""
Objective function for optimization based on change points and parameters.

# Arguments
- `chromosome::Vector{Float64}`: Vector of parameters to optimize.
- `change_points::Vector{Int}`: Locations of change points.
- `n_global::Int`: Number of global parameters.
- `n_segment_specific::Int`: Number of segment-specific parameters per segment.
- `extract_parameters::Function`: Function to extract global and segment-specific parameters.
- `parnames`: Parameter names for constructing labeled arrays.
- `model_function`: The user-defined model function.
- `simulate_model`: Function to simulate the model.
- `extra_data`: Additional data for simulation, if required.
- `num_steps::Union{Int, Nothing}`: Number of steps for discrete models or `nothing` for continuous models.
- `tspan::Tuple`: Time span for continuous models.
- `loss_function`: Loss function for comparing simulated and actual data.
- `compare_variables`: Variables to compare between segments.
- `data_CP`: Data segmented by change points.
- `initial_conditions`: Initial conditions for the model.

# Returns
- `total_loss::Float64`: Total loss over all segments.
"""
function objective_function(
    chromosome, 
    change_points, 
    n_global, 
    n_segment_specific, 
    extract_parameters, 
    parnames,
    model_function, 
    simulate_model, 
    loss_function, 
    segment_loss,
    data_CP;
    initial_conditions=nothing,
    extra_data=nothing,
    num_steps=nothing, 
    tspan=nothing,
    compare_variables=nothing,
    transformation=identity
)
    num_segments = length(change_points) + 1
    constant_pars, segment_pars = extract_parameters(chromosome, n_global, n_segment_specific)
    total_loss = 0.0
    u0 = initial_conditions
    if length(change_points)>0
    for i in 1:num_segments
        if i == 1
            data_segment = eltype(data_CP) <: AbstractVector ? 
                [vector[1:change_points[i]] for vector in data_CP] : 
                data_CP[1:change_points[i]]
            tspan_segment = isnothing(num_steps) ? (tspan[1], change_points[i]) : nothing
            u_segment = isnothing(num_steps) ? nothing : 
                (eltype(extra_data) <: AbstractVector ? 
                    [vector[1:change_points[i]] for vector in extra_data] : 
                    extra_data[1:change_points[i]])
        elseif i == num_segments
            data_segment = eltype(data_CP) <: AbstractVector ? 
                [vector[change_points[i-1]+1:end] for vector in data_CP] : 
                data_CP[change_points[i-1]+1:end]
            tspan_segment = isnothing(num_steps) ? (change_points[i-1], tspan[2]) : nothing
            u_segment = isnothing(num_steps) ? nothing : 
                (eltype(extra_data) <: AbstractVector ? 
                    [vector[change_points[i-1]+1:end] for vector in extra_data] : 
                    extra_data[change_points[i-1]+1:end])
        else
            data_segment = eltype(data_CP) <: AbstractVector ? 
                [vector[change_points[i-1]+1:change_points[i]] for vector in data_CP] : 
                data_CP[change_points[i-1]+1:change_points[i]]
            tspan_segment = isnothing(num_steps) ? (change_points[i-1], change_points[i]) : nothing
            u_segment = isnothing(num_steps) ? nothing : 
                (eltype(extra_data) <: AbstractVector ? 
                    [vector[change_points[i-1]+1:change_points[i]] for vector in extra_data] : 
                    extra_data[change_points[i-1]+1:change_points[i]])
        end

        seg_pars = segment_pars[i]
        params = @LArray [constant_pars; seg_pars] parnames
        simulated_data = isnothing(num_steps) ? 
            simulate_model(model_function, params, u0; tspan=tspan_segment) : 
            simulate_model(model_function, params, u0; extra_data=u_segment, num_steps=length(data_segment))
        loss = segment_loss(data_segment, simulated_data, loss_function, compare_variables, transformation)
        total_loss += loss
        u0 = simulated_data[:, end]
    end
else
    data_segment = data_CP
    tspan_segment = isnothing(num_steps) ? tspan : nothing
    u_segment = isnothing(num_steps) ? nothing : extra_data
    seg_pars = segment_pars[1]
    params = @LArray [constant_pars;seg_pars] parnames
    
    #@show params
    simulated_data = isnothing(num_steps) ? 
    simulate_model(model_function, params, u0; tspan=tspan_segment) : 
    simulate_model(model_function, params, u0; extra_data=u_segment, num_steps=length(data_segment))
    
    # Calculate loss for this segment
    total_loss = segment_loss(data_segment, simulated_data, loss_function, compare_variables, transformation)
    end

    return total_loss
end

"""
Wrapper for the objective function for ease of use.
"""
function wrapped_obj_function(chromosome)
    return objective_function(
        chromosome, 
        CP, 
        n_global, 
        n_segment_specific, 
        extract_parameters, 
        parnames,
        model_function, 
        simulate_model, 
        extra_data,
        num_steps, 
        tspan,
        loss_function, 
        compare_variables,
        data_CP,
        initial_conditions
    )
end

end # module
