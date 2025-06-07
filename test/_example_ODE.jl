using Evolutionary 
using DifferentialEquations
using LabelledArrays
using Plots
using Statistics
using Random

# ----------------------------
# SIR ODE model
# ----------------------------
function sirmodel!(du, u, p, t)
    S, I, R = u
    β, γ = p
    @show β, γ
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

# ----------------------------
# Toy data generation
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
        #params = [beta_values[i], γ]
        #parnames = (:γ, :β)
        params = @LArray [beta_values[i], γ] (:β, :γ)

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
        @show u0
    end

    return all_times, abs.(data_CP)
end


β_values = [0.00009, 0.00014, 0.00025, 0.0005]
β_values = [0.00009, 0.00014]
change_points_true = [50, 100, 150]
change_points_true = [50]
γ = 0.7
u0 = [9999.0, 1.0, 0.0]
tspan = (0.0, 250.0)
tspan = (0.0, 70.0)

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
    @show β, γ
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
    
    return sqrt(sum(abs2, (observed.- simulated).^2))
end


function loss_function(observed, simulated)
    simulated = simulated[2:2,:]    
    return sum(abs2, (observed.- simulated))
end

begin

initial_chromosome = [0.69, 0.0002]
#initial_chromosome = [0.4, 0.002]
#initial_chromosome = rand(2)
parnames = (:γ, :β)
# propertynames
initial_params = initial_chromosome
bounds = ([0.1, 0.0], [0.9, 0.1])
u0 = [9999.0, 1.0, 0.0]
ode_spec = ODEModelSpec(example_ode_model, initial_params, u0, tspan)
model_manager = ModelManager(ode_spec)
n_global = 1
n_segment_specific = 1
min_length = 10
step = 10
# this GA setting works better when the initial chromosome is defined properly close to real pars
ga = GA(populationSize = 150, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
crossoverRate=0.6, mutation = gaussian(0.0001))
#ga = GA(populationSize = 100, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.7,
#crossoverRate=0.7, mutation = gaussian(0.01))
n = length(data_M)

my_penalty4(p, n) = 0.0 * p * log(n)
#pen = 0.0
#using Random
#Random.seed!(1234)
end
# [50, 90, 100, 110, 210]
# need to choose random seed number before GA and pen coefficent 
# [50, 100, 150, 160, 190]
# [50, 60, 100, 230, 240]
#using BenchmarkTools
# @benchmark
@time detected_cp, params = detect_changepoints(
    objective_function,
    n, n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_M,
    initial_chromosome, parnames, bounds, ga,
    min_length, step, my_penalty4
)
# not penalizing and setting seed, i get the same results all the time.
# 281.751858 seconds
# [50, 100, 130, 140, 150, 190]




function sirmodel!(du, u, p, t)
    S, I, R = u
    β, γ = p
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

plot(data_M[1,:])
times, sim = generate_toy_dataset(params[2:end], detected_cp, params[1], u0, tspan)
plot!(sim) 