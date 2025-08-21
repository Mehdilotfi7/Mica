function plot_simulation(
    chromosome, 
    change_points, 
    parnames, 
    n_global::Int, 
    n_segment_specific::Int, 
    model_manager::ModelManager, 
    loss_function::Function,
    data::Matrix{Float64}
)
    constant_pars, segment_pars_list = extract_parameters(chromosome, n_global, n_segment_specific)
    total_loss = 0.0
    num_segments = length(change_points) + 1

    # For initial condition passing
    u0 = get_initial_condition(model_manager)

    if length(change_points)>0

       for i in 1:num_segments

           idx_start = (i == 1) ? 1 : change_points[i - 1] + 1
           idx_end   = (i > length(change_points)) ? size(data, 2) : change_points[i]
           segment_data = data[:, idx_start:idx_end]
           

           seg_pars = segment_pars_list[i]
           all_pars = @LArray [constant_pars;seg_pars] parnames
           model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)

           sim_data = simulate_model(model_spec)
           total_loss += loss_function(segment_data, sim_data)
           total_loss += BIC_penalty(length(seg_pars), size(data, 2), change_points)

           # Update initial condition if applicable
           u0 = update_initial_condition(model_manager, sim_data)
       end
    else
        segment_data = data
        idx_start = 1
        idx_end = size(data, 2)

        seg_pars = segment_pars_list[1]
        
        all_pars = @LArray [constant_pars;seg_pars] parnames
        model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)

        sim_data = simulate_model(model_spec)
        total_loss += loss_function(segment_data, sim_data)  
        
        #total_loss += BIC_penalty(length(seg_pars), size(data, 2), change_points)

    end

    return total_loss
end











# ploting 
function simulate_full_model(
    chromosome,
    change_points,
    parnames,
    n_global::Int,
    n_segment_specific::Int,
    model_manager::ModelManager,
    data::Matrix{Float64}
)
    constant_pars, segment_pars_list = extract_parameters(chromosome, n_global, n_segment_specific)
    num_segments = length(change_points) + 1

    # Initial condition
    u0 = get_initial_condition(model_manager)
    full_sim = Float64[]  # storage for concatenated simulated data
    all_sims = Vector{Matrix{Float64}}()  # optional: store each segment separately

    if length(change_points) > 0
        for i in 1:num_segments
            idx_start = (i == 1) ? 1 : change_points[i - 1] + 1
            idx_end = (i > length(change_points)) ? size(data, 2) : change_points[i]
            seg_pars = segment_pars_list[i]

            all_pars = @LArray [constant_pars; seg_pars] parnames
            model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)
            sim_data = simulate_model(model_spec)
            @show sim_data

            push!(all_sims, sim_data)
            u0 = update_initial_condition(model_manager, sim_data)
        end
    else
        # Only one segment
        idx_start = 1
        idx_end = size(data, 2)

        seg_pars = segment_pars_list[1]
        all_pars = @LArray [constant_pars; seg_pars] parnames
        model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)
        sim_data = simulate_model(model_spec)

        push!(all_sims, sim_data)
    end

    # Option 1: concatenate all sims to a single matrix
    full_sim = hcat(all_sims...)

    return full_sim  # or return all_sims if you want per-segment
end

_ , pars = optimize_with_changepoints(
    objective_function, params, parnames, cp, bounds, ga,
    n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_CP,
    options=Evolutionary.Options(show_trace=true)
)

using CSV
using DataFrames

cp = CSV.read("results_detected_cp_penalty10.csv", DataFrame).detected_cp
params = CSV.read("results_params_penalty10.csv", DataFrame).params

simulated = simulate_full_model(
    pars,
    cp,
    parnames,
    n_global,
    n_segment_specific,
    model_manager,
    data_CP
)

# Now plot
using Plots
plot!(simulated[5,:], title="Simulated Output Across Segments")
plot(data_CP[1,:])


using Plots

# Time vector
T = 1:size(simulated, 2)

# Extract signals from simulated data
infected = simulated[5, :]
hospital = simulated[6, :]
icu      = simulated[7, :]
death    = simulated[9, :]
vacc     = simulated[11, :]

# Corresponding real data
infected_real = data_CP[1, :]
hospital_real = data_CP[2, :]
icu_real      = data_CP[3, :]
death_real    = data_CP[4, :]
vacc_real     = data_CP[5, :]

# Names and group data
labels = ["Infected", "Hospitalized", "ICU", "Deaths", "Vaccinated"]
sim_vals = [infected, hospital, icu, death, vacc]
real_vals = [infected_real, hospital_real, icu_real, death_real, vacc_real]

# Create multi-plot layout (5 rows × 1 column)
plt = plot(layout = (5,1), size=(800,1000))

for i in 1:5
    plot!(
        plt[i], T, real_vals[i], label = "Observed", lw = 2, color = :blue
    )
    plot!(
        plt[i], T, sim_vals[i], label = "Simulated", lw = 2, linestyle = :dash, color = :red
    )
    title!(plt[i], labels[i])
    xlabel!(plt[i], "Time")
    ylabel!(plt[i], "Count")

    # Add vertical change point lines
    for cp in cp
        vline!(plt[i], [cp], linestyle = :dot, color = :black, label = "")
    end
end

display(plt)