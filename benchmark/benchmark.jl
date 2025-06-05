using Distributed
addprocs()
@everywhere using DifferentialEquations, Statistics, Random, CSV, DataFrames, LabelledArrays

# --- Your change point detection function is assumed available in environment ---
@everywhere using MyChangePointPackage  # Replace with actual module name if needed

# ODE definition
@everywhere function sirmodel!(du, u, p, t)
    S, I, R = u
    β, γ = p.β , p.γ
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

@everywhere function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0::Vector{Float64})
    prob = ODEProblem(sirmodel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol=1e-6, reltol=1e-6)
    return sol[:,:]
end

@everywhere function generate_toy_dataset(beta_values, change_points, γ, u0, tspan, noise_level, noise=randn)
    data_CP = []
    all_times = []
    for i in 1:length(change_points)+1
        tspan_segment = if i == 1
            (0.0, change_points[i])
        elseif i == length(change_points)+1
            (change_points[i-1]+1.0, tspan[2])
        else
            (change_points[i-1]+1.0, change_points[i])
        end
        params = @LArray [beta_values[i], γ] (:γ, :β)
        prob = ODEProblem(sirmodel!, u0, tspan_segment, params)
        sol = solve(prob, saveat = 1.0)
        data_CP = vcat(data_CP, sol[2,:] + noise_level * noise(length(sol.t)))
        all_times = vcat(all_times, sol.t)
        u0 = sol.u[end]
    end
    return all_times, abs.(data_CP)
end

@everywhere function loss_function(observed, simulated)
    simulated = simulated[2:2,:]
    return sqrt(sum((observed .- simulated).^2))
end

@everywhere function calculate_precision(detected_cps, true_cps, tolerance=0)
    TP = sum(any(abs(d - t) <= tolerance for t in true_cps) for d in detected_cps)
    FP = length(detected_cps) - TP
    FN = length(true_cps) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1
end

@everywhere function benchmark_one(noise_level, noise_type, penalty, change_point_count, data_length)
    beta_values = [0.00009, 0.00014, 0.00025, 0.0005]
    change_points = [50.0, 100.0, 150.0]
    valid_change_points = change_points[1:change_point_count]
    γ = 0.7
    u0 = [9999.0, 1.0, 0.0]
    tspan = (0.0, data_length)
    initial_chromosome = [0.69, 0.0002]
    parnames = (:γ, :β)
    bounds = ([0.1, 0.0], [0.9, 0.1])
    ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan)
    model_manager = ModelManager(ode_spec)
    n_global = 1
    n_segment_specific = 1
    min_length = 10
    step = 10
    noise = noise_type == "Gaussian" ? randn : rand
    times, data = generate_toy_dataset(beta_values, valid_change_points, γ, u0, tspan, noise_level, noise)
    data_CP = reshape(Float64.(data), 1, :)
    n = length(data_CP)
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
    runtime = time() - start_time
    precision, recall, f1 = calculate_precision(detected_cps, valid_change_points)
    return (;change_point_count, data_length, noise_level, noise_type, penalty=string(penalty),
        runtime, precision, recall, f1)
end

# Define parameters
noise_levels = [1, 10, 20]
noise_types = ["Gaussian", "Uniform"]
penalty_values = [BIC_penalty1, BIC_penalty2, BIC_penalty3, BIC_penalty4]
change_point_counts = [1, 2, 3]
data_lengths = [70, 130, 160, 200, 250]

# Generate all combinations
combos = [(n, t, p, c, l) for n in noise_levels, t in noise_types, p in penalty_values, c in change_point_counts, l in data_lengths]

# Filter out invalid data lengths
valid_combos = filter(x -> x[5] >= [50.0, 100.0, 150.0][x[4]], combos)

# Run benchmarks in parallel
@time results = pmap(x -> benchmark_one(x...), valid_combos)

# Save to CSV
CSV.write("benchmark_results.csv", DataFrame(results))
