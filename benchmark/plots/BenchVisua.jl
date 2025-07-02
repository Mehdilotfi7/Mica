using CSV, DataFrames, StatsPlots, DifferentialEquations, LinearAlgebra, Random
using LabelledArrays  # For @LArray

# SIR model definition
function sirmodel!(du, u, p, t)
    S, I, R = u
    β, γ = p.β, p.γ
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

# Toy dataset generation
function generate_toy_dataset(beta_values, change_points, γ, u0, tspan, noise_level, noise)
    data_CP = []
    all_times = []

    if isempty(change_points)
        # Single segment: no change points
        params = @LArray [beta_values[1], γ] (:β, :γ)
        prob = ODEProblem(sirmodel!, u0, tspan, params)
        sol = solve(prob, saveat=1.0)
        data_CP = sol[2,:] + noise_level * noise(length(sol.t))
        all_times = sol.t
        return all_times, abs.(data_CP)
    end

    for i in 1:length(change_points)+1
        tspan_segment = if i == 1
            (0.0, change_points[i])
        elseif i == length(change_points)+1
            (change_points[i-1]+1.0, tspan[2])
        else
            (change_points[i-1]+1.0, change_points[i])
        end

        params = @LArray [beta_values[i], γ] (:β, :γ)
        prob = ODEProblem(sirmodel!, u0, tspan_segment, params)
        sol = solve(prob, saveat=1.0)
        data_CP = vcat(data_CP, sol[2,:] + noise_level * noise(length(sol.t)))
        all_times = vcat(all_times, sol.t)
        u0 = sol.u[end]
    end

    return all_times, abs.(data_CP)
end

# cps_pars parser
function parse_cps_pars(s)
    if isnothing(s) || s === "" || ismissing(s)
        return ([], [], NaN)
    end
    tup = Meta.parse(s) |> eval
    change_points, params = tup
    γ = params[1]
    betas = params[2:end]
    return (change_points, betas, γ)
end

# Load CSV
df = CSV.read("benchmark/results/benchmark_results_all_configs.csv", DataFrame)


true_β_values = [0.00009, 0.00014, 0.00025, 0.0005]
true_γ = 0.7
true_change_points = [50, 100, 150]
u0 = [9999.0, 1.0, 0.0]


plots = []

for row in eachrow(df)
    len = row.data_length
    noise_level = row.noise_level
    noise_type = row.noise_type
    change_point_count = row.change_point_count
    penalty = row.penalty
    noise_fn = noise_type == "Gaussian" ? randn : rand

    # TRUE simulation setup
    valid_change_points = true_change_points[1:change_point_count]
    valid_betas = true_β_values[1:(change_point_count + 1)]
    tspan = (0.0, len)

    # Simulate true noisy dataset
    times, infections = generate_toy_dataset(valid_betas, valid_change_points, true_γ, copy(u0), tspan, noise_level, noise_fn)

    # DETECTED simulation setup
    detected_cps, est_betas, est_gamma = parse_cps_pars(row.cps_pars)
    detected_tspan = (0.0, len)

    # Simulate estimated clean ODE solution
    recon_times, recon_infections = generate_toy_dataset(est_betas, detected_cps, est_gamma, copy(u0), detected_tspan, 0.0, noise_fn)

    # Plot both
    p = plot(times, infections;
        lw = 1.5, label = "Noisy Toy Data",
        title = "nt=$(noise_type), noise=$noise_level, len=$len, pen = $penalty",
        xlabel = "Time", ylabel = "Infected", legend = :topright)

    if !isempty(recon_times)
        plot!(p, recon_times, recon_infections; lw = 2, label = "ODE from Estimated Params", color = :blue)
    end

    vline!(p, valid_change_points; color = :green, linestyle = :dash, label = "True CPs")
    vline!(p, detected_cps .+ 0.5; color = :red, linestyle = :solid, label = "Detected CPs")

    push!(plots, p)
end

ncols = 2
nrows = ceil(Int, length(plots) / ncols)
layout = grid(nrows, ncols)


using IterTools  # For partition
chunks = partition(plots, 30)
    
    for (i, page) in enumerate(chunks)
        n = length(page)
        ncols = 5
        nrows = ceil(Int, n / ncols)
        layout = grid(nrows, ncols)
    
        final_plot = plot(page...;
            layout = layout,
            size = (ncols * 300, nrows * 250),
            dpi = 300,
            tickfont = font(6),
            guidefont = font(6),
            legendfont = font(4),
            titlefont = font(5),
            bottom_margin = 1mm,
            left_margin = 4mm
        )
    
        savefig(final_plot, "benchmark/plots/cp_results_page_$i.pdf")

    end
    