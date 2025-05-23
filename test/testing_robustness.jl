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




###########################################
# Function to calculate precision
function calculate_precision(detected_cps, true_cps, tolerance=10)
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
  
  
  
  ###########################################
  using Statistics
  
  function benchmark_noise_types(beta_values, change_points, γ, N, u0, tspan, noise_levels, noise_types, penalty_values, Bi_S)
    results = []
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
                times, data = generate_toy_dataset(beta_values, change_points, γ, N, u0, tspan, noise_level, noise)
                global data_CP = data
                start_time = time()
                detected_cps = Bi_S(objective_function, length(data_CP), pen=penalty)
                end_time = time()
                run_time = end_time - start_time
                precision, recall, f1_score = calculate_precision(detected_cps, change_points)
                push!(results, (noise_level, noise_type, penalty, run_time, precision, recall, f1_score))
            end
        end
    end
    return results
  end
  
  using Random  # Ensure you have this package imported for the `randn` and `rand` functions
  # integrating number of change points and data size to previous critera
  # To do:
  # 1. putting manually the change point positions and their parameters in the other file and see 
  # if it can detect all chage points. if it detects change points then add number of change points and 
  # data size to the benchmark function 
  # 2. if it cannot detect the change points stick on 2 change points as before and add only data size
  # criteria to the benchmark function.
  
  function benchmark_noise_types(beta_values, change_points, γ, N, u0, noise_levels, noise_types, penalty_values, Bi_S, change_point_counts, data_lengths)
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
                                  times, data = generate_toy_dataset(beta_values, valid_change_points, γ, N, u0, (0.0, data_length), noise_level, noise)
                                  global data_CP = data
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
                                  detected_cps, pars_cps = Bi_S(objective_function, length(data_CP), pen=penalty)
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
  noise_levels = [1.0, 10.0, 20.0]
  noise_types = ["Gaussian", "Uniform"]
  penalty_values = [1.0, 10.0, 100.0, 1000.0, 10000.0, 95460.0]
  change_point_counts = [1, 2, 3]
  data_lengths = [70, 130, 160, 200, 250]
  beta_values = [0.00009, 0.00014, 0.00025, 0.0005]
  change_points = [50.0, 100.0, 150.0]
  
  #results_noise = benchmark_noise_types(beta_values, change_points, γ, N, u0, tspan, noise_levels, noise_types, Bi_S)
  #results_noise = benchmark_noise_types(beta_values, change_points, γ, N, u0, tspan, noise_levels, noise_types, penalty_values, Bi_S)
  @time results_noise = benchmark_noise_types(beta_values, change_points, γ, N, u0, noise_levels, noise_types, penalty_values, Bi_S, change_point_counts, data_lengths)
  