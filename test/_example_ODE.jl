
using Evolutionary 
using DifferentialEquations
using LabelledArrays
using Plots

#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------


# ----------------------------
# SIR ODE model
# ----------------------------
function sirmodel!(du, u, p, t)
    S, I, R = u
    β, γ = p
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

# ----------------------------
# Data generation
# ----------------------------
function generate_toy_dataset(beta_values, change_points, γ, u0, tspan)
    data_CP = []
    all_times = []

    for i in 1:length(change_points)+1
        # Define time span for this segment
        if i == 1
            tspan_segment = (0.0, change_points[i])  # From 0 to first change point
        elseif i == length(change_points)+1
            tspan_segment = (change_points[i-1]+1.0, tspan[2])  # From last change point to the end
        else
            tspan_segment = (change_points[i-1]+1.0, change_points[i])  # Between two change points
        end

        # Set parameters for this segment
        params = [beta_values[i], γ]
        #parnames = (:γ, :β)
        #params = @LArray [beta_values[i], γ] parnames

        # Create an ODE problem
        prob = ODEProblem(sirmodel!, u0, tspan_segment, params)

        # Solve the ODE
        sol = solve(prob, saveat = 1.0)
        #@show typeof(sol)

        # Append the solution to the data
        data_CP = vcat(data_CP, sol[2,:])
        all_times = vcat(all_times, sol.t)
        #@show typeof(data_CP)

        # Update initial conditions for the next segment
        u0 = sol.u[end]
    end

    return all_times, abs.(data_CP)
end


β_values = [0.00009, 0.00014, 0.00025, 0.0005]
change_points_true = [50, 100, 150]
γ = 0.7
u0 = [9999.0, 1.0, 0.0]
tspan = (0.0, 250.0)

# Generate synthetic data
times, data = generate_toy_dataset(β_values, change_points_true, γ, u0, tspan)
data_M = reshape(Float64.(data), 1, :)
plot(data_M[1,:])
#times, sim = generate_toy_dataset(params[2:end], detected_cp, params[1], u0, tspan)
#plot!(sim) 
################################################################################################
################################################################################################


function sirmodel!(du, u, p, t)
    S, I, R = u
    β, γ = p.β , p.γ
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

# Model wrapper for TSCPDetector
function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0::Vector{Float64})
    prob = ODEProblem(sirmodel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol = 1.0e-6, reltol = 1.0e-6)
    return sol[:,:]  # returns matrix-like solution
end

function loss_function(observed, simulated)
    simulated = simulated[2:2,:]
    #@assert size(observed) == size(simulated) "Dimension mismatch in loss function."
    return sqrt(sum((observed .- simulated).^2))
end


abstract type AbstractModelSpec end


struct ModelManager{T<:AbstractModelSpec}
    base_model::T
end

struct ODEModelSpec <: AbstractModelSpec
    model_function::Function
    params
    initial_conditions::Vector{Float64}
    tspan::Tuple{Float64, Float64}
end

function simulate_model(model::ODEModelSpec)
    return model.model_function(model.params, model.tspan, model.initial_conditions)
end


get_initial_condition(manager::ModelManager{ODEModelSpec}) = manager.base_model.initial_conditions

function update_initial_condition(manager::ModelManager{ODEModelSpec}, sim_data)
    return sim_data[:,end]
end

function segment_model(
    manager::ModelManager{ODEModelSpec}, 
    pars, 
    idx_start::Int, 
    idx_end::Int,
    u0
 )
    model = manager.base_model

    return ODEModelSpec(model.model_function, pars, u0, (idx_start, idx_end))
end

get_model_type(manager::ModelManager{ODEModelSpec}) = "ODE"

function extract_parameters(chromosome::Vector{T}, n_global::Int, n_segment_specific::Int) where T
    global_parameters = chromosome[1:n_global]
    segment_parameters = [chromosome[i:i+n_segment_specific-1] for i in n_global+1:n_segment_specific:length(chromosome)]
    return global_parameters, segment_parameters
end

function objective_function(
    chromosome, 
    parnames,
    change_points::Vector{Int}, 
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
          #parnames = (:γ, :β)
          all_pars = @LArray [constant_pars;seg_pars] parnames
          model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)

          sim_data = simulate_model(model_spec)
          total_loss += loss_function(segment_data, sim_data)

          u0 = update_initial_condition(model_manager, sim_data)

        end
    else

        
        segment_data = data
        idx_start = 1
        idx_end = size(data, 2)

        seg_pars = segment_pars_list[1]
        #parnames = (:γ, :β)
        all_pars = @LArray [constant_pars;seg_pars] parnames
        model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)

        sim_data = simulate_model(model_spec)
        total_loss += loss_function(segment_data, sim_data)    
        #@show total_loss    

    end

    return total_loss
end



function wrapped_obj_function(chromosome)
    return objective_function(
        chromosome, 
        parnames,
        change_points, 
        n_global, 
        n_segment_specific, 
        model_manager, 
        loss_function,
        data
    )
end

function optimize_with_changepoints(
    objective_function, chromosome, parnames, CP, bounds, ga,
    n_global, n_segment_specific,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64};
    options=Evolutionary.Options(show_trace=false)
 )
    wrapped_obj(chrom) = objective_function(
        chrom, parnames, CP, n_global, n_segment_specific,
        model_manager, loss_function, data
    )
    #@show chromosome
    #@show bounds
    result = Evolutionary.optimize(wrapped_obj, BoxConstraints(bounds...), chromosome, ga, options)
    return Evolutionary.minimum(result), Evolutionary.minimizer(result)
end


function update_bounds!(chromosome, bounds, n_global, n_segment_specific, extract_parameters)
    _, seg_specific = extract_parameters(chromosome, n_global, n_segment_specific)
    _, seg_lower = extract_parameters(bounds[1], n_global, n_segment_specific)
    _, seg_upper = extract_parameters(bounds[2], n_global, n_segment_specific)

    append!(chromosome, seg_specific[1])
    append!(bounds[1], seg_lower[1])
    append!(bounds[2], seg_upper[1])
end


function evaluate_segment(
    objective_function, a::Int, b::Int, CP::Vector{Int}, bounds,
    chromosome, parnames, ga, pen::Float64, min_length::Int, step::Int,
    n_global::Int, n_segment_specific::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64}
 )
    x = Float64[]
    y = Vector{Vector{Float64}}()
    for j in (a + min_length):step:(b - min_length)
        new_cp = sort([CP; j])
        #@show j
        @show new_cp
        # This ensures that garbage doesn’t build up, especially when optimizing many segments.
        #GC.gc()
        loss, best = optimize_with_changepoints(
            objective_function, chromosome, parnames, new_cp, bounds, ga,
            n_global, n_segment_specific,
            model_manager, loss_function, data
        )
        @show loss
        push!(x, loss + pen)
        push!(y, best)
        #break
    end
    return x, y
end


function detect_changepoints(
    objective_function,
    n::Int, n_global::Int, n_segment_specific::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64},
    initial_chromosome,
    parnames,
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    ga, # i should define type later
    min_length::Int, step::Int;
    pen::Float64=log(n)
 )
    tau = Tuple{Int, Int}[]
    push!(tau, (0, n))
    CP = Int[]
    @show CP

    loss_val, best_params = optimize_with_changepoints(
        objective_function, initial_chromosome, parnames, CP, bounds, ga,
        n_global, n_segment_specific,
        model_manager, loss_function, data
    )
    @show best_params

    update_bounds!(initial_chromosome, bounds, n_global, n_segment_specific, extract_parameters)

    while !isempty(tau)
        #@show tau
        a, b = pop!(tau)
        x, y = evaluate_segment(
            objective_function, a, b, CP, bounds, initial_chromosome, parnames, ga, pen, min_length, step,
            n_global, n_segment_specific,
            model_manager, loss_function, data
        )
        #@show x,y
        if !isempty(x)
            minval, idx = findmin(x)
            if minval < loss_val
                chpt = a + (idx * step)
                push!(CP, chpt)
                CP = sort(CP)
                loss_val = minval
                best_params = y[idx]
                @show CP
                @show best_params
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


###########################################################################################
###########################################################################################


initial_chromosome = [0.69, 0.0002]
parnames = (:γ, :β)
initial_params = initial_chromosome
bounds = ([0.1, 0.0], [0.9, 0.1])
u0 = [9999.0, 1.0, 0.0]
ode_spec = ODEModelSpec(example_ode_model, initial_params, u0, tspan)
model_manager = ModelManager(ode_spec)
n_global = 1
n_segment_specific = 1
min_length = 10
step = 10
ga = GA(populationSize = 100, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
crossoverRate=0.6, mutation = gaussian(0.0001))
n = length(data_M)
pen = 0.0


@time detected_cp, params = detect_changepoints(
    objective_function,
    n, n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_M,
    initial_chromosome, parnames, bounds, ga,
    min_length, step
)
