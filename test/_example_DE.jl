using CSV

function example_difference_model(θ, T_initial, num_steps, extra_data)
    # Unpack parameters
    θ1, θ2, θ3, θ4, θ5, θ6, θ7 = θ.θ1, θ.θ2, θ.θ3, θ.θ4, θ.θ5, θ.θ6, θ.θ7
    # Unpack extra_data
    wind_speeds, ambient_temperatures = extra_data

    # Initialize arrays
    generator_temperatures_sim = zeros(num_steps)
    
    # Initial condition
    generator_temperatures_sim[1] = T_initial 
    # Simulation loop
    for k in 2:num_steps
        # Extract inputs
        u1 = wind_speeds[k]
        u2 = ambient_temperatures[k]
        y_prev = generator_temperatures_sim[k-1]
        
        # Calculate model output
       
        generator_temperatures_sim[k] = ((θ1 * u1^3 + θ2 * u1^2 + θ3 * u1 + y_prev - u2) / (θ4 * u1^3 + θ5 * u1^2 + θ6 * u1 + θ7)) + u2
        
    end

    return generator_temperatures_sim
end


function loss_function(observed, simulated)
    simulated = simulated[2:2,:]
    #@assert size(observed) == size(simulated) "Dimension mismatch in loss function."
    return sum((observed .- simulated).^2)
end

cd(dirname(@__FILE__))
df = CSV.read("Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-07-01_228.csv", DataFrame)
names(df)
findall(x -> isnan(x), df[:, "Wind speed (m/s)"][1:4660])
findall(x -> isnan(x), df[:, "Ambient temperature (converter) (°C)"])
df[:, "Date and time"][1:2500]

wind_speeds            = df[:, "Wind speed (m/s)"][1:2500]
ambient_temperatures   = df[:, "Ambient temperature (converter) (°C)"][1:2500]
generator_temperatures_front = df[:, "Generator bearing front temperature (°C)"][1:2500]
generator_temperatures_rear = df[:, "Generator bearing rear temperature (°C)"][1:2500]
generator_temperatures = (generator_temperatures_front .+ generator_temperatures_rear)./2




initial_chromosome = [1.1, 1.1, 1.1, 1.5, 1.5, 1.5, 1.5]
parnames = (:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7)
# propertynames
initial_params = initial_chromosome
lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
upper = [100.0, 100.0, 100.0, 466.0, 466.0, 466.0, 1466.0]
bounds = (lower, upper)
u0 = generator_temperatures[1]
wind = wind_speeds
temp = ambient_temperatures

de_spec = DifferenceModelSpec(example_difference_model, initial_params, u0, num_steps, (wind, temp))
model_manager = ModelManager(de_spec)

n_global = 3
n_segment_specific = 4
min_length = 10
step = 10
ga = GA(populationSize = 100, selection = tournament(2), crossover = SBX(0.7, 1), mutationRate=0.7,
    crossoverRate=0.7, mutation = gaussian(0.0001))

data_M = reshape(Float64.(generator_temperatures), 1, :)

n = length(data_M)
penlty = 57000.0

@time detected_cp, params = detect_changepoints(
    objective_function,
    n, n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_M,
    initial_chromosome, parnames, bounds, ga,
    min_length, step
)

#