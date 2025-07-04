module Visualization

using Plots

"""
Plot the real dataset along with simulated results and mark change points.

# Arguments
- `data::Array{Float64, 2}`: The dataset to plot. Can be 1D or 2D.
- `change_points::Vector{Int}`: The locations of change points in the data.
- `parameters::Vector{Float64}`: The list of parameters including both constant and segment-specific parameters.
- `simulate_segment::Function`: A function to simulate model outputs for each segment based on parameters and data.
- `constant_param_length::Int`: The number of constant parameters.
- `segment_param_lengths::Vector{Int}`: Lengths of the segment-specific parameters.

# Returns
- None. Plots the data and simulated results.
"""
using Plots

function plot_simulation(
    chromosome, 
    change_points, 
    n_global, 
    n_segment_specific, 
    extract_parameters, 
    parnames,
    model_function, 
    simulate_model, 
    extra_data,
    num_steps, 
    tspan,
    compare_variables,  
    data_CP,
    initial_conditions
)
    num_segments = length(change_points) + 1
    constant_pars, segment_pars = extract_parameters(chromosome, n_global, n_segment_specific)
    simulated_data_aggregated = [Vector{Float64}() for _ in compare_variables]  # Collect data for each variable
    u0 = initial_conditions

    if length(change_points) > 0
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

            # Append data for each variable specified in compare_variables
            for (j, var_idx) in enumerate(compare_variables)
                append!(simulated_data_aggregated[j], simulated_data[var_idx, :])
            end

            u0 = simulated_data[:, end]  # Update initial conditions
        end
    else
        # Case with no change points
        data_segment = data_CP
        tspan_segment = isnothing(num_steps) ? tspan : nothing
        u_segment = isnothing(num_steps) ? nothing : extra_data
        seg_pars = segment_pars[1]
        params = @LArray [constant_pars; seg_pars] parnames
        
        simulated_data = isnothing(num_steps) ? 
            simulate_model(model_function, params, u0; tspan=tspan_segment) : 
            simulate_model(model_function, params, u0; extra_data=u_segment, num_steps=length(data_segment))
        
        # Append data for each variable specified in compare_variables
        for (j, var_idx) in enumerate(compare_variables)
            append!(simulated_data_aggregated[j], simulated_data[var_idx, :])
        end
        
    end

    # Plot each variable in a subplot
    num_vars = length(compare_variables)
    p = plot(layout=(num_vars, 1), xlabel="Time")  # Create a subplot layout
    for (j, data) in enumerate(simulated_data_aggregated)
        plot!(p[j], data, label="Variable $(compare_variables[j])", ylabel="Value")
    end

    # Add vertical lines at change points
    for cp in change_points
        vline!(p[i], [cp], label="", color="black", linestyle=:dash)
    end

    display(p)
    savefig(p, "fit_result.png")

    return nothing
end


end
