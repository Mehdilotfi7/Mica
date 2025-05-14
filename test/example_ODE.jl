using Test
using TSCPDetector 
using Evolutionary 
using DifferentialEquations
using Plots
using DataFrames

# Define the SIR model
function sirmodel!(du, u, p, t)
    β, γ = p
    S, I, R = u
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
  end

function example_ode_model(params, time, u0)
# Define the ODE problem
prob = ODEProblem(sirmodel!, u0, (time[1], time[end]), params)
    
# Solve the ODE problem
sol = solve(prob, Tsit5(), saveat=1)

# Return the result as a DataFrame
return sol[:,:]
end

# Function to generate toy dataset
function generate_toy_dataset(beta_values, change_points, γ, N, u0, tspan)
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

        # Create an ODE problem
        prob = ODEProblem(sirmodel!, u0, tspan_segment, params)

        # Solve the ODE
        sol = solve(prob, saveat = 1.0)

        # Append the solution to the data
        data_CP = vcat(data_CP, sol[2,:])
        all_times = vcat(all_times, sol.t)

        # Update initial conditions for the next segment
        u0 = sol.u[end]
    end

    return all_times, abs.(data_CP)
end

function loss_function(segment_data, simulated_data, compare_variables=nothing)

    @assert size(segment_data) == size(simulated_data) "Dimension mismatch between real and simulated data."

    # Calculate RMSE for the selected variables
    segment_loss = sqrt(sum((segment_data .- simulated_data).^2))

    return segment_loss
end



# Example usage
beta_values = [0.00009, 0.00014, 0.00025, 0.0005]
change_points = [50, 100, 150]
γ = 0.7
N = 10_000
u0 = [N-1, 1, 0]
tspan = (0.0, 250.0)

# Generate dataset
times, data = generate_toy_dataset(beta_values, change_points, γ, N, u0, tspan)

ga = GA(populationSize = 100, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
    crossoverRate=0.6, mutation = gaussian(0.0001))

initial_chromosome = [0.69, 0.0002]
bounds = ([0.1,0.0], [0.9,0.1])
n_global = 1
n_segment_specific = 1
n = size(data,2)
parnames=parnames = [:β, :γ] 
min_length = 10
step = 10



ode_model = ODEModelSpec(example_ode_model, params, u0, tspan)
model_manager = ModelManager(ode_model)

change_points, parameters = detect_changepoints(
    objective_function,
    n, n_global, n_segment_specific,
    parnames,
    model_manager,
    loss_function,
    data,
    initial_chromosome, bounds, ga,
    min_length, step
)



#--------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------

using Test
using TSCPDetector
using Evolutionary
using DifferentialEquations
using Plots
using DataFrames

# ----------------------------
# SIR ODE model
# ----------------------------
function sirmodel!(du, u, p, t)
    β, γ = p
    S, I, R = u
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

# Model wrapper for TSCPDetector
function example_ode_model(params::Dict, tspan::Tuple{Float64, Float64}, u0::Vector{Float64})
    β = params[:β]
    γ = params[:γ]
    prob = ODEProblem(sirmodel!, u0, tspan, [β, γ])
    sol = solve(prob, Tsit5(), saveat=1.0)
    return sol[2:2,:]  # returns matrix-like solution
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

# ----------------------------
# Loss function
# ----------------------------
function loss_function(observed, simulated)
    #simulated = simulated[2:2,:]
    #@show size(observed), size(simulated)
    #@assert size(observed) == size(simulated) "Dimension mismatch in loss function."
    return sqrt(sum((observed .- simulated).^2))
end

# ----------------------------
# Test script
# ----------------------------
@testset "Change Point Detection with SIR ODE Model" begin
    # True parameters
    β_values = [0.00009, 0.00014, 0.00025, 0.0005]
    change_points_true = [50, 100, 150]
    γ = 0.7
    u0 = [9999.0, 1.0, 0.0]
    tspan = (0.0, 250.0)

    # Generate synthetic data
    times, data = generate_toy_dataset(β_values, change_points_true, γ, u0, tspan)

    # Build model and config
    parnames = [:β, :γ]
    n_global = 1
    n_segment_specific = 1
    initial_chromosome = [γ, 0.0002]
    bounds = ([0.1, 0.00001], [0.9, 0.1])
    ga = GA(populationSize = 100, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
    crossoverRate=0.6, mutation = gaussian(0.0001))

    # Set up model manager
    ode_spec = ODEModelSpec(example_ode_model, Dict(:γ => γ, :β => 0.0002), u0, tspan)
    model_manager = ModelManager(ode_spec)

    # Detect change points
    min_length = 10
    step = 10
    data_M = reshape(Float64.(data), 1, :)
    n = length(data_M)

    simulate_model(ode_spec)

    detected_cp, params = detect_changepoints(
        objective_function,
        n, n_global, n_segment_specific,
        parnames,
        model_manager,
        loss_function,
        data_M,
        initial_chromosome, bounds, ga,
        min_length, step
    )

    @testset "Detected change points" begin
        @test length(detected_cp) ≈ length(change_points_true) atol=1
        @test all(cp -> 0 < cp < n, detected_cp)
        println("Detected change points: ", detected_cp)
    end
end
