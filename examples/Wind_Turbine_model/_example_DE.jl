using CSV
using DataFrames
using Evolutionary 
using DifferentialEquations
using LabelledArrays
using Plots
using Statistics
using Random



function example_difference_model(θ, T_initial, num_steps, extra_data)
    θ1, θ2, θ3, θ4, θ5, θ6, θ7 = θ.θ1, θ.θ2, θ.θ3, θ.θ4, θ.θ5, θ.θ6, θ.θ7
    wind_speeds, ambient_temperatures = extra_data
    generator_temperatures_sim = zeros(num_steps)
    generator_temperatures_sim[1] = T_initial 
    for k in 2:num_steps
        u1, u2 = wind_speeds[k], ambient_temperatures[k]
        y_prev = generator_temperatures_sim[k-1]
        generator_temperatures_sim[k] = ((θ1*u1^3 + θ2*u1^2 + θ3*u1 + y_prev - u2)/(θ4*u1^3 + θ5*u1^2 + θ6*u1 + θ7)) + u2
    end
    return reshape(Float64.(generator_temperatures_sim), 1, :)
end


function loss_function(observed, simulated)
    #simulated = simulated[2:2,:]
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
generator_temperatures = generator_temperatures_front
plot(generator_temperatures)
#using Smoothers
#generator_temperatures = hma(generator_temperatures, 21)

begin

initial_chromosome = [1.1, 1.1, 1.1, 1.5, 1.5, 1.5, 1.5]
parnames = (:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7)
# propertynames
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
stepp = 10
ga = GA(populationSize = 50, selection = tournament(2), crossover = SBX(0.7, 1), mutationRate=0.7,
    crossoverRate=0.7, mutation = gaussian(0.0001))

data_M = reshape(Float64.(generator_temperatures), 1, :)
n = length(data_M)
#penlty = 57000.0
BIC_100(p,n)= 100.0 * p * log(n)
end


@time detected_cp, params = detect_changepoints(
    objective_function,
    n, n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_M,
    initial_chromosome, parnames, bounds, ga,
    min_length, stepp, BIC_100
)
#2755,2757,2753

# 2700 --- 2750
# 
# 2800 ---- 0 
CSV.write("results_Turbine_detected_cp_correct_pen100_pop_50_frontData.csv", DataFrame(detected_cp=detected_cp))
CSV.write("results_Turbine_params_correct_pen100_pop50_frontData", DataFrame(params=params))

# data 2500
# 1700 ===> 82
# 2100 ===> 82
# 2130 ===> 82
# 2150, 2140 ===> []
# 2200 ===> []

detected_cp = CSV.read("results_Turbine_detected_cp_correct_pen0_pop_50_rearData.csv", DataFrame)[:,1]
params = CSV.read("results_Turbine_params_correct_pen0_pop50_rearData.csv", DataFrame)[:,1]

detected_cp = CSV.read("results_Turbine_detected_cp_correct_pen0_pop_50_frontData.csv", DataFrame)[:,1]
params = CSV.read("results_Turbine_params_correct_pen0_pop50_frontData.csv", DataFrame)[:,1]

detected_cp = CSV.read("results_Turbine_detected_cp_correct_pen0_pop_50.csv", DataFrame)[:,1]
params = CSV.read("results_Turbine_params_correct_pen0_pop50.csv", DataFrame)[:,1]

df[:, "Date and time"][detected_cp]
status = CSV.read("Status_Kelmarsh_1_2021-01-01_-_2021-07-01_228.csv", DataFrame)



function simulate_generator_temperature(θ, wind_speeds, ambient_temperatures, T_initial)
    # Unpack parameters
    θ1, θ2, θ3, θ4, θ5, θ6, θ7 = θ

    # Number of time steps
    num_steps = length(wind_speeds)

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

function plot_function(chromosome, change_points)
    # Number of segments
    num_segments = length(change_points) + 1
    θ₁ = chromosome[1]
    θ₂ = chromosome[2]
    θ₃ = chromosome[3]
    θ₄_values = chromosome[4:4:end]
    θ₅_values = chromosome[5:4:end]
    θ₆_values = chromosome[6:4:end]
    θ₇_values = chromosome[7:4:end]

    sim_T = Float64[]
    T_initial = generator_temperatures[1]

    # Loop through segments
  if length(change_points)>0
     for i in 1:num_segments
          # Define segment data and time span based on change points
          if i == 1
            data_segment = generator_temperatures[1:change_points[i]]
            u1_segment = wind_speeds[1:change_points[i]]
            u2_segment = ambient_temperatures[1:change_points[i]]
            
          elseif i == num_segments
            data_segment = generator_temperatures[change_points[i-1]+1:end]
            u1_segment = wind_speeds[change_points[i-1]+1:end]
            u2_segment = ambient_temperatures[change_points[i-1]+1:end]
            
          else
            data_segment = generator_temperatures[change_points[i-1]+1:change_points[i]]
            u1_segment = wind_speeds[change_points[i-1]+1:change_points[i]]
            u2_segment = ambient_temperatures[change_points[i-1]+1:change_points[i]]
            
          end

          # Extract θ values for this segment
          θ₄ = θ₄_values[i]
          θ₅ = θ₅_values[i]
          θ₆ = θ₆_values[i]
          θ₇ = θ₇_values[i]

          θ = [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆, θ₇]

          # Simulate generator temperature for the segment
          @show T_initial
          simulated_T = simulate_generator_temperature(θ, u1_segment, u2_segment, T_initial)
          append!(sim_T, simulated_T)
          T_initial = simulated_T[end]

        end
   else
       data_segment = generator_temperatures
       u1_segment   = wind_speeds
       u2_segment   = ambient_temperatures

       θ₄ = θ₄_values[1]
       θ₅ = θ₅_values[1]
       θ₆ = θ₆_values[1]
       θ₇ = θ₇_values[1]
       θ = [θ₁, θ₂, θ₃, θ₄, θ₅, θ₆, θ₇]
       # Simulate generator temperature for the segment
       simulated_T = simulate_generator_temperature(θ, u1_segment, u2_segment, T_initial)
       append!(sim_T, simulated_T)
   end
   p=plot!(sim_T, 
   label = "Simulation")
   display(p)

    return nothing
end

cps_dates = df[:, "Date and time"][detected_cp]

plot(generator_temperatures, label = "Generator temperatures", size=(1400, 700), margin = 20Plots.mm,
guidefont=font(10), legendfont=font(10), color = :skyblue, markersize = 1, titlefont=font(10), dpi=1000, xlabel = "Time", ylabel = "Generator temperature",
)

plot_function(params, detected_cp)
vline!(detected_cp, label="Found CPs")
p2 = xticks!(vcat(detected_cp, [0, 2500]), vcat(cps_dates, ["2021-01-01 00:00:00", "2021-01-18 08:30:00"]), rotation=30, tickfont=font(7))

savefig("Turbine_CPD_2sub.png")
plot(p1, p2, layout=(2, 1), margin = 8Plots.mm,dpi=100,
size=(1400, 1200), guidefont=font(12), legendfont=font(10), titlefont=font(10),tickfont=font(10)
)