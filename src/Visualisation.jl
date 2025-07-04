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