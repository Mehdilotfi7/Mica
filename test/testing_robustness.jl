using LabelledArrays
using DifferentialEquations
using Statistics


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

# Function to generate toy dataset with different conditions
function generate_toy_dataset(beta_values, change_points, γ, u0, tspan, noise_level, noise=randn)
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
        
        params = @LArray [beta_values[i], γ] (:γ, :β)

        # Create an ODE problem
        prob = ODEProblem(sirmodel!, u0, tspan_segment, params)
        # Solve the ODE
        sol = solve(prob, saveat = 1.0)

        # Append the solution to the data
        data_CP = vcat(data_CP, sol[2,:] + noise_level * noise(length(sol.t)))
        all_times = vcat(all_times, sol.t)

        # Update initial conditions for the next segment
        u0 = sol.u[end]
    end

    return all_times, abs.(data_CP)
end

# Example usage
beta_values = [0.00009, 0.000167, 0.00042]
change_points = [80.0, 150.0]
beta_values = [0.00009, 0.00014, 0.00025, 0.0005]
change_points = [50.0, 100.0, 150.0]
γ = 0.7
N = 10_000
u0 = [N-1, 1, 0]
tspan = (0.0, 250.0)
noise_level = 0.0
times, data = generate_toy_dataset(beta_values, change_points, γ, N, u0, tspan, noise_level)
data

###########################################
# Function to calculate precision
function calculate_precision(detected_cps, true_cps, tolerance=0)
    TP = 0
    FP = 0
    FN = 0
  
    for detected_cp in detected_cps
        if any(abs(detected_cp - true_cp) <= tolerance for true_cp in true_cps)
            TP += 1
        else
            FP += 1
        end
    end
  
    FN = length(true_cps) - TP
  
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
  
    return precision, recall, f1_score
end
  
function BIC_penalty1(p,n)
    pen = 0.0 * p * log(n)
    return pen
end

function BIC_penalty2(p,n)
    pen = 1.0 * p * log(n)
    return pen
end

function BIC_penalty3(p,n)
    pen = 10.0 * p * log(n)
    return pen
end

function BIC_penalty4(p,n)
    pen = 100.0 * p * log(n)
    return pen
end
  
  ###########################################
  

  function benchmark_noise_types(
    # benchnarking arguments
    beta_values, change_points, γ, u0, noise_levels,
    noise_types, detect_changepoints, change_point_counts,
    data_lengths,
    # detect_changepoints arguments
    objective_function,
    n_global, n_segment_specific,
    model_manager,
    loss_function,
    initial_chromosome, parnames, bounds, ga,
    min_length, step)
      results = []
          for change_point_count in change_point_counts
              for data_length in data_lengths
                  # Ensure the chosen data length is sufficient for the change points
                  valid_change_points = change_points[1:change_point_count]
                  if data_length >= maximum(valid_change_points)
                      for noise_level in noise_levels
                          for noise_type in noise_types
                              if noise_type == "Gaussian"
                                  noise = randn
                              elseif noise_type == "Uniform"
                                  noise = rand
                              else
                                  throw(ArgumentError("Unsupported noise type"))
                              end
                              for penalty in penalty_values
                                  # Generate dataset with the specified length
                                  times, data = generate_toy_dataset(beta_values, valid_change_points, γ, u0, (0.0, data_length), noise_level, noise)
                                  global data_CP = reshape(Float64.(data), 1, :)
                                  n = length(data_CP)
                                  @show noise_level
                                  @show noise_type 
                                  @show penalty
                                  @show change_point_count 
                                  @show data_length
                                  @show beta_values 
                                  @show valid_change_points
                                  @show tspan = (0.0, data_length)
                                  global tspan = (0.0, data_length)
                                  # Measure the runtime of the change point detection method
                                  start_time = time()
                                  detected_cps, pars_cps = detect_changepoints(
                                    objective_function,
                                    n, n_global, n_segment_specific,
                                    model_manager,
                                    loss_function,
                                    data_CP,
                                    initial_chromosome, parnames, bounds, ga,
                                    min_length, step, penalty
                                 )
                                  end_time = time()
                                  run_time = end_time - start_time
                                  
      
                                  # Calculate precision, recall, and F1 score
                                  precision, recall, f1_score = calculate_precision(detected_cps, valid_change_points)
                                  # Extracting changepoints and their parameters for each case 
                                  cps_pars = (detected_cps, pars_cps)
      
                                  # Store results
                                  push!(results, (change_point_count, data_length, noise_level, noise_type, penalty, run_time, precision, recall, f1_score, cps_pars))
                              end
                          end
                      end
                  end
              end
          end
      return results
  end
      
  
  ######
  
  
# Example usage
begin

noise_levels = [1, 10, 20]
noise_types = ["Gaussian", "Uniform"]
penalty_values = [BIC_penalty1, BIC_penalty2, BIC_penalty3, BIC_penalty4]
change_point_counts = [1, 2, 3]
data_lengths = [70, 130, 160, 200, 250]
beta_values = [0.00009, 0.00014, 0.00025, 0.0005]
change_points = [50.0, 100.0, 150.0]
γ = 0.7



tspan = (0.0, 250.0)
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
ga = GA(populationSize = 150, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
crossoverRate=0.6, mutation = gaussian(0.0001))
# n = length(data_M)

end

@time results_noise = benchmark_noise_types(
    # benchnarking arguments
    beta_values, change_points, γ, u0, noise_levels,
    noise_types, detect_changepoints, change_point_counts,
    data_lengths,
    # detect_changepoints arguments
    objective_function,
    n_global, n_segment_specific,
    model_manager,
    loss_function,
    initial_chromosome, parnames, bounds, ga,
    min_length, step)
