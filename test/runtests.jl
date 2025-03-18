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
plot(data)
# adding two NA to part of dataset manually
data1 = Vector{Union{Missing, Float64}}(data)
data1[70]  = missing
data1[110] = missing
data1 = DataFrame(times =times, data1=data1)

# Test `load_data`
@testset "Data Handling Tests" begin
    # Example test data
    data_loaded = load_data(data1)  

end

# Test `simulate_model`
@testset "Model Simulation Tests" begin

  # Simulate continuous models: an ode model

    # Define model parameters
    params = Dict(:p => 0.00009)
    # Defind initial conditions
    N = 10_000
    u0 = [N-1, 1, 0]
    # Define time points
    time = (0,250)
    result_ode = simulate_model(example_ode_model, params; initial_conditions = u0, tspan = time)

  # Simulate discrete models: a difference equations model

    num_steps = 200
    initial_conditions = 0.5
    params = Dict(:θ1 => 0.5,:θ2 => 1.2,:θ3 => 0.8,:θ4 => 1.5,:θ5 => 0.9,:θ6 => 2.0,:θ7 => 1.1)
    # Create extra_data with two vectors of positive random values
    extra_data = [rand(200) * 10 for _ in 1:2]
    result_difference_equations = simulate_model(example_difference_model, params; initial_conditions=initial_conditions,
                                                 extra_data=extra_data,num_steps = num_steps)

  # Simulate time series models: a regression model

    params = Dict(:a => 0.5,:b => 1.2)
    time = 100
    result_linear_regression = simulate_model(example_regression_model, params; time = time)

end

# Test `segment_loss`
@testset "Segment loss Function Tests" begin
   
   segment_data   = data + rand(251)
   simulated_data = data
   segment_loss(segment_data, simulated_data, loss_function)   
   
end

@testset "Extraction Function Tests" begin

    chromosome = [0.7, 0.00009, 0.00014, 0.00025, 0.0005]
    n_global = 1
    n_segment_specific = 1
    extract_parameters(chromosome, n_global, n_segment_specific)
end

@testset "Objective Function Function Tests" begin


    chromosome = [0.7, 0.00009, 0.00014, 0.00025, 0.0005]
    change_points = [50, 100, 150]
    N = 10_000
    initial_conditions = [N-1, 1, 0]
    tspan = (0.0, 250.0)
    n_global = 1
    n_segment_specific = 1
    parnames = (:β,  :γ)
    data_CP = data
    compare_variables = [2]


objective_function(
    chromosome, 
    change_points, 
    n_global, 
    n_segment_specific, 
    extract_parameters, 
    parnames,
    example_ode_model, 
    simulate_model, 
    loss_function, 
    segment_loss,
    data_CP,
    initial_conditions=initial_conditions,
    tspan=tspan,
    compare_variables=compare_variables
)

end

@testset "Initial optimization Function Tests" begin

    initial_chromosome = [0.69, 0.0002]
    lower =              [0.1,     0.0]
    upper =              [0.9,     0.1]
    change_points = Array{Int64}(undef,0)
    ga = GA(populationSize = 100, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
    crossoverRate=0.6, mutation = gaussian(0.0001))
    initial_optimization(objective_function, initial_chromosome, lower, upper, ga, data_CP, tspan,
    example_ode_model, simulate_model, loss_function, n_global, n_segment_specific, initial_condition, 2)
end

# Test to update the chromosome and bounds for segment-specific parameters
@testset "update_chromosome_bounds! Tests" begin
    initial_chromosome = [0.69, 0.0002]
    lower =              [0.1,     0.0]
    upper =              [0.9,     0.1]
    n_global = 1
    n_segment_specific = 1

    update_chromosome_bounds!(
      initial_chromosome,
      lower,
      upper,
      n_global,
      n_segment_specific,
    )
end

# Test Function for evaluate potential change points in a segment
@testset "evaluate_segment Tests" begin
    # Example of calling evaluate_segment with all the required arguments
 evaluate_segment(
      objective_function, 
      0, 
      250, 
      [], 
      initial_chromosome, 
      lower, 
      upper, 
      ga, 
      0, 
      data_CP, 
      tspan, 
      example_ode_model, 
      simulate_model, 
      loss_function, 
      n_global, 
      n_segment_specific, 
      initial_condition, 
      2
    )
end

# Test for the Function to update tau (the list of segments to test)
@testset "update_tau! Tests" begin
    update_tau!(tau, a, chpt, b, min_length)
end
 

@testset "Visualization Tests" begin
    @test_throws ErrorException plot_results(simulated_data, [1, 2, 3], simulate_model)
end



# Test `plot_results`
@testset "Visualization Tests" begin
    @test_throws ErrorException plot_results(simulated_data, [1, 2, 3], simulate_model)
end

# Test `initial_optimization`
@testset "Initial Optimization Tests" begin
    loss_val_CP, best_pars = initial_optimization(objective_function, initial_chromosome, lower, upper, ga, [])
    @test typeof(loss_val_CP) == Float64
    @test length(best_pars) == length(initial_chromosome)
end

# Test `update_chromosome_bounds!`
@testset "Update Chromosome Bounds Tests" begin
    update_chromosome_bounds!(initial_chromosome, lower, upper, 3, 4)
    @test length(initial_chromosome) == 3 + 4
end

# Test `evaluate_segment`
@testset "Evaluate Segment Tests" begin
    x, y = evaluate_segment(objective_function, 1, 5, [], initial_chromosome, lower, upper, ga, 1)
    @test length(x) > 0
    @test length(y) > 0
end

# Test `update_tau!`
@testset "Update Tau Tests" begin
    tau = [(0, 5)]
    update_tau!(tau, 0, 3, 5)
    @test length(tau) == 2
end

# Test `ChangePointDetector`
@testset "Change Point Detection Tests" begin
    CP, best_y = ChangePointDetector(objective_function, n; pen=log(n), initial_chromosome=initial_chromosome, lower=lower, upper=upper, ga=ga)
    @test length(CP) >= 0
    @test length(best_y) == length(initial_chromosome)
end
