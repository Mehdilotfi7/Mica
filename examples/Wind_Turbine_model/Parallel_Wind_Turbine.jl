using Distributed
addprocs(25)  # Keep 1 for the main process; use 31 for workers
#rmprocs(workers())  # remove broken workers
@everywhere using Random
@everywhere using CSV, DataFrames
@everywhere using TSCPDetector  

@everywhere using CSV, DataFrames, Smoothers
@everywhere using Evolutionary, DifferentialEquations, LabelledArrays, Plots
@everywhere using Statistics, Random

@everywhere function example_difference_model(θ, T_initial, num_steps, extra_data)
    θ1, θ2, θ3, θ4, θ5, θ6, θ7 = θ.θ1, θ.θ2, θ.θ3, θ.θ4, θ.θ5, θ.θ6, θ.θ7
    wind_speeds, ambient_temperatures = extra_data
    generator_temperatures_sim = zeros(num_steps)
    generator_temperatures_sim[1] = T_initial 
    for k in 2:num_steps
        u1, u2 = wind_speeds[k], ambient_temperatures[k]
        y_prev = generator_temperatures_sim[k-1]
        generator_temperatures_sim[k] = ((θ1*u1^3 + θ2*u1^2 + θ3*u1 + y_prev - u2)/(θ4*u1^3 + θ5*u1^2 + θ6*u1 + θ7)) + u2
    end
    return generator_temperatures_sim
end

@everywhere function loss_function(observed, simulated)
    simulated = simulated[2:2,:]
    return sum((observed .- simulated).^2)
end

csv_path = abspath("test/Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-07-01_228.csv")
isfile(csv_path)  # should return true


@everywhere csv_path = $csv_path


@everywhere begin

    #cd(dirname(@__FILE__))
    df = CSV.read(csv_path, DataFrame)
    wind_speeds            = df[:, "Wind speed (m/s)"][1:2500]
    ambient_temperatures   = df[:, "Ambient temperature (converter) (°C)"][1:2500]
    generator_temperatures_front = df[:, "Generator bearing front temperature (°C)"][1:2500]
    generator_temperatures_rear  = df[:, "Generator bearing rear temperature (°C)"][1:2500]
    generator_temperatures = (generator_temperatures_front .+ generator_temperatures_rear) ./ 2
    #generator_temperatures = hma(generator_temperatures, 21)

end



# Share global variables or functions
@everywhere begin
    initial_chromosome = [1.1, 1.1, 1.1, 1.5, 1.5, 1.5, 1.5]
    parnames = (:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7)
    lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    upper = [100.0, 100.0, 100.0, 466.0, 466.0, 466.0, 1466.0]
    bounds = (lower, upper)
    u0 = generator_temperatures[1]
    wind = wind_speeds
    temp = ambient_temperatures
    num_steps = 2500

    de_spec = DifferenceModelSpec(example_difference_model, initial_chromosome, u0, num_steps, (wind, temp))
    model_manager = ModelManager(de_spec)

    n_global = 3
    n_segment_specific = 4
    min_length = 10
    step = 10
    ga = GA(populationSize = 130, selection = tournament(2), crossover = SBX(0.7, 1), mutationRate=0.7,
        crossoverRate=0.7, mutation = gaussian(0.0001))

    data_M = reshape(Float64.(generator_temperatures), 1, :)
    n = length(data_M)
end




# Step 1: Helper function to create closures
function make_penalty(pen_val::Real)
    return (p, n) -> pen_val * p * log(n)
end

# Step 2: Build function list and values
BIC_penalty_functions = Function[]
penalty_values = Int[]

for pen in 2700:2:2760
    push!(BIC_penalty_functions, make_penalty(pen))
    push!(penalty_values, pen)
end

# Step 3: Pair penalty functions with values
penalty_tasks = [(BIC_penalty_functions[i], penalty_values[i]) for i in eachindex(BIC_penalty_functions)]


@everywhere function run_detection(task::Tuple{Function, Int})
    BIC_penalty, pen = task

    detected_cp, params = detect_changepoints(
        objective_function,
        n, n_global, n_segment_specific,
        model_manager,
        loss_function,
        data_M,
        initial_chromosome, parnames, bounds, ga,
        min_length, step, BIC_penalty
    )

    pen_id = Int(round(pen))  # safe, since 'pen' is passed in
    CSV.write("results_Turbine_detected_cp_GA130_pen$(pen_id).csv", DataFrame(detected_cp=detected_cp))
    CSV.write("results_Turbine_params_GA130_pen$(pen_id).csv", DataFrame(params=params))

    return nothing
end

results = pmap(run_detection, penalty_tasks)

# Send only Int penalty values
penalty_values = collect(2700:2:2760)

@everywhere function run_detection(pen::Int)
    # Define penalty function locally — closure is created per task
    BIC_penalty = (p, n) -> pen * p * log(n)

    detected_cp, params = detect_changepoints(
        objective_function,
        n, n_global, n_segment_specific,
        model_manager,
        loss_function,
        data_M,
        initial_chromosome, parnames, bounds, ga,
        min_length, step, BIC_penalty
    )

    pen_id = Int(round(pen))
    CSV.write("results_Turbine_detected_cp_GA130_pen$(pen_id).csv", DataFrame(detected_cp=detected_cp))
    CSV.write("results_Turbine_params_GA130_pen$(pen_id).csv", DataFrame(params=params))

    return nothing
end

# Now `penalty_values` is just a list of integers (safe to send to workers)
results = pmap(run_detection, penalty_values)







