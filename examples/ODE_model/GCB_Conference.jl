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
    #@show β, γ
    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

# ----------------------------
# Toy data generation
# ----------------------------
function generate_toy_dataset(beta_values, change_points, γ, u0, tspan, noise_level, noise)
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
        data_CP = vcat(data_CP, sol[2,:] + noise_level * noise(length(sol.t)))
        all_times = vcat(all_times, sol.t)
        #@show typeof(data_CP)

        # Update initial conditions for the next segment
        u0 = sol.u[end]
        @show u0
    end

    return all_times, abs.(data_CP)
end

begin
β_values = [0.00009, 0.00014, 0.00025, 0.0005]
#β_values = [0.00009, 0.00014]
change_points_true = [50, 100, 150]
#change_points_true = [50]
γ = 0.7
u0 = [9999.0, 1.0, 0.0]
tspan = (0.0, 250.0)
#tspan = (0.0, 70.0)
Random.seed!(1234)
noise_level = 20
noise = rand
end
# Generate synthetic data
times, data = generate_toy_dataset(β_values, change_points_true, γ, u0, tspan, noise_level, noise)
data_M = reshape(Float64.(data), 1, :)
using Plots
scatter(data_M[1,:])
#times, sim = generate_toy_dataset(params[2:end], detected_cp, params[1], u0, tspan)
#plot!(sim) 
################################################################################################
################################################################################################


function sirmodel!(du, u, p, t)
    S, I, R = u
    β, γ = p.β , p.γ
    #@show β, γ
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


#function loss_function(observed, simulated)
#    simulated = simulated[2:2,:]    
#    return sum(abs2, (observed.- simulated))
#end

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
ga = GA(populationSize = 100, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
crossoverRate=0.6, mutation = gaussian(0.0001))
#ga = GA(populationSize = 100, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.7,
#crossoverRate=0.7, mutation = gaussian(0.01))
n = length(data_M)

my_penalty4(p, n) = 20.0 * p * log(n)
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
[50, 100, 240] # noise=30
[50, 60, 90, 100, 180] # noise=20
[50, 100, 130, 150, 200, 220, 240] # noise = 10


##################################################
# using R packages for change points detection


# ----------------------------
# Run existing CPD in R
# ----------------------------
@rput data
R"""
library(changepoint.np)
changePoints = cpt.np(
  data,
  penalty = "BIC",
  pen.value = 20,        # Large penalty => fewer CPs
  method = "PELT",
  test.stat = "empirical_distribution",
  class = TRUE,
  minseglen = 10,         # Forces segments to be long
  nquantiles = 10
)
cp_locations <- cpts(changePoints)
"""

@rget cp_locations

# ----------------------------
# Run Mocha on toy data
# ----------------------------
# Assuming you have a function run_mocha returning detected CPs and fitted parameters:
# detected_CPs, est_params, sim_times, sim_data = run_mocha(times, data, sirmodel!, ...)

# ----------------------------
# Plot results
# ----------------------------
plot_true = scatter(times, data, color=:black, label="Toy data")
vline!(change_points_true, color=:green, lw=2, label="Change points")
title!(plot_true, "Ground truth")

plot_existing = scatter(times, data, color=:black, label="Toy data")
vline!(cp_locations, color=:red, lw=2, label="Change points")
title!(plot_existing, "PELT result")

# Placeholder for Mocha simulation
# Replace sim_data and mocha_CPs with your actual output
mocha_CPs = detected_cp  
sim_time, sim_data = generate_toy_dataset(params[2:end], detected_cp, params[1], u0, tspan, 0, noise)  

plot_mocha = scatter(times, data, color=:black, label="Toy data")
plot!(times, sim_data, color=:blue, lw=2, label="Mocha simulation")
vline!(mocha_CPs, color=:green, lw=2, label="Change points")
title!(plot_mocha, "Mocha result")

# Combine into a single figure
pp = plot(plot_true, plot_existing, plot_mocha, layout=(3,1), size=(800,800))
savefig(pp, "ex_vs_mocha.png")



#########################################

using Plots, Random

# --- Generate toy signal with structural shifts ---
Random.seed!(123)
t = 0:0.1:100
y = similar(t)

for (i, ti) in enumerate(t)
    if ti < 30
        y[i] = sin(0.4*ti) + 0.4*randn()          # low frequency
    elseif ti < 60
        y[i] = 4.5*sin(0.4*ti) + 0.4*randn()      # higher amplitude & freq
    elseif ti < 85
        y[i] = 0.5*sin(0.4*ti + 2) + 0.4*randn()  # phase shift + lower amp
    else
        y[i] = 3.5*sin(0.4*ti) + 0.4*randn()        # strong oscillations
    end
end

# --- Change points ---
change_points = [30, 60, 85]

# --- Segment colors (pastel palette) ---
segment_colors = ["#AED6F1", "#F9E79F", "#A9DFBF", "#F5B7B1"]

# --- Plot ---
p = plot(; legend=false, grid=false, framestyle=:none, size=(1000,300))

# Shade background segments with colored rectangles
ymin, ymax = minimum(y) - 1, maximum(y) + 1
for (i, (x0, x1)) in enumerate(zip([0; change_points], [change_points; maximum(t)]))
    plot!([x0, x1, x1, x0], [ymin, ymin, ymax, ymax],
          seriestype=:shape, color=segment_colors[i], alpha=0.3, lw=0)
end

# Overlay signal
plot!(t, y, color="#AFDDFF", lw=2)

# Mark change points
vline!(change_points, color="#FFDDAB", lw=2, ls=:dash)

savefig(p, "title_hook_fixed.png")
p


#############################################################
# comparision of Pelt with Mocha

using Plots, Random, RCall

# --- Generate toy signal ---
Random.seed!(123)
t = 0:1.0:100
y = similar(t)

for (i, ti) in enumerate(t)
    if ti < 30
        y[i] = sin(0.4*ti) + 0.0*randn()
    elseif ti < 60
        y[i] = 4.5*sin(0.4*ti) + 0.0*randn()
    elseif ti < 85
        y[i] = 0.5*sin(0.4*ti + 2) + 0.0*randn()
    else
        y[i] = 3.5*sin(0.4*ti) + 0.0*randn()
    end
end

# --- Run PELT in R ---
data = y
@rput data
R"""
library(changepoint.np)
cp_model <- cpt.np(
  data,
  penalty = "BIC",
  pen.value = 20,
  method = "PELT",
  test.stat = "empirical_distribution",
  class = TRUE,
  minseglen = 10,
  nquantiles = 10
)
cp_locations <- cpts(cp_model)
"""
@rget cp_locations

# --- Plot function for segments ---
function plot_segments(t, y, cps; title_str="")
    cps = sort(cps)
    all_cps = [0; cps; maximum(t)]
    colors = ["#AED6F1", "#F9E79F", "#A9DFBF", "#F5B7B1", "#D7BDE2"]

    ymin, ymax = minimum(y) - 1, maximum(y) + 1
    p = plot(; legend=false, grid=false, framestyle=:box, size=(800,300),
             title=title_str)

    for (i, (x0, x1)) in enumerate(zip(all_cps[1:end-1], all_cps[2:end]))
        plot!([x0, x1, x1, x0], [ymin, ymin, ymax, ymax],
              seriestype=:shape, color=colors[i], alpha=0.3, lw=0)
    end

    plot!(t, y, color="black", lw=1.5)
    vline!(cps, color="red", lw=2, ls=:dash)
    return p
end

# --- Comparison plots ---
p1 = plot_segments(t, y, cp_locations, title_str="PELT: all CPs detected")
p2 = plot_segments(t, y, change_points[1:1], title_str="Mocha mimic: step 1")
p3 = plot_segments(t, y, change_points[1:2], title_str="Mocha mimic: step 2")
p4 = plot_segments(t, y, change_points, title_str="Mocha mimic: step 3")

plot(p1, p2, p3, p4, layout=(4,1), size=(900,900))











using Plots, Random

# --- Generate toy signal with true change points ---
Random.seed!(123)
t = 0:1.0:100
y = similar(t)

for (i, ti) in enumerate(t)
    if ti < 30
        y[i] = sin(0.4*ti) + 0.4*randn()
    elseif ti < 60
        y[i] = 4.5*sin(0.4*ti) + 0.4*randn()
    elseif ti < 85
        y[i] = 0.5*sin(0.4*ti + 2) + 0.4*randn()
    else
        y[i] = 3.5*sin(0.4*ti) + 0.4*randn()
    end
end

true_cps = [30, 60, 85]  # true CPs

# --- Ground-truth sine params for each true segment ---
# [ (amplitude, freq, phase, range) ]
segments_truth = [
    (1.0, 0.4, 0.0, (0,30)),     # before 30
    (4.5, 0.4, 0.0, (30,60)),    # 30–60
    (0.5, 0.4, 2.0, (60,85)),    # 60–85
    (3.5, 0.4, 0.0, (85,100))    # 85–100
]

function simulate_signal(t, cps)
    cps_sorted = sort(cps)
    cps_full = [0; cps_sorted; 100.0]  # full intervals
    y_sim = similar(t)

    for (i, (t0,t1)) in enumerate(zip(cps_full[1:end-1], cps_full[2:end]))
        # use the true parameters for this interval
        (A, f, ph, _) = segments_truth[i]
        for (j, ti) in enumerate(t)
            if ti ≥ t0 && ti < t1
                y_sim[j] = A * sin(f*ti + ph)
            end
        end
    end
    return y_sim
end


# --- Iterative plots ---
iter_cps = [[30], [30, 60], [30, 60, 85]]
plots = []

for (i, cps) in enumerate(iter_cps)
    y_sim = simulate_signal(t, cps)
    p = plot(t, y, color="black", lw=1, label="Data",
             title="Mocha Iteration $i", legend=:topright,
             size=(800,300))
    plot!(t, y_sim, color="blue", lw=2, label="Simulation")
    vline!(cps, color="red", lw=2, ls=:dash, label="")
    push!(plots, p)
end

plot(plots..., layout=(3,1), size=(900,800))




# GCB conference plot generation

using Plots, Random

# --- Ground-truth sine params ---
segments_truth = [
    (1.0, 0.4, 0.0, (0,30)),     # before 30
    (4.5, 0.4, 0.0, (30,60)),    # 30–60
    (0.5, 0.4, 2.0, (60,85)),    # 60–85
    (3.5, 0.4, 0.0, (85,100))    # 85–100
]

# --- Generate noisy signal ---
Random.seed!(123)
t = 0:0.1:100
y = [ti < 30  ? sin(0.4*ti) + 0.4*randn() :
     ti < 60  ? 4.5*sin(0.4*ti) + 0.4*randn() :
     ti < 85  ? 0.5*sin(0.4*ti + 2) + 0.4*randn() :
                 3.5*sin(0.4*ti) + 0.4*randn() for ti in t]

# --- Function: simulate given CPs ---
function simulate_signal(t, cps)
    cps_sorted = sort(cps)
    cps_full = [0; cps_sorted; 100.0]
    y_sim = similar(t)

    for (i, (t0,t1)) in enumerate(zip(cps_full[1:end-1], cps_full[2:end]))
        (A, f, ph, _) = segments_truth[i]
        for (j, ti) in enumerate(t)
            if ti ≥ t0 && ti < t1
                y_sim[j] = A * sin(f*ti + ph)
            end
        end
    end
    return y_sim, cps_full
end

# --- Iterations (0 = no CP) ---
iter_cps = [[], [30], [30, 60], [30, 60, 85]]
colors = ["#AED6F1", "#F9E79F", "#A9DFBF", "#F5B7B1"]

titles = [
    "Iter 0: (Loss + Penalty) ⬆⬆",
    "Iter 1: (Loss + Penalty) ⬇",
    "Iter 2: (Loss + Penalty) ⬇⬇",
    "Iter 3: (Loss + Penalty) ⬇⬇⬇ (Optimal)"
]


plots = []
for (i, cps) in enumerate(iter_cps)
    if isempty(cps)
        y_sim = [2*sin(0.3*ti) for ti in t]  # arbitrary poor fit
        cps_full = [0, 100.0]
    else
        y_sim, cps_full = simulate_signal(t, cps)
    end

    # enable legend only for the first subplot
    show_legend = (i == 1)

    p = plot(; legend=show_legend, title=titles[i], framestyle=:none,
              xticks=false, yticks=false, size=(900,200), legendfontsize=7)

    ymin, ymax = minimum(y) - 1, maximum(y) + 1
    for (j, (x0, x1)) in enumerate(zip(cps_full[1:end-1], cps_full[2:end]))
        plot!([x0, x1, x1, x0], [ymin, ymin, ymax, ymax],
              seriestype=:shape, color=colors[j], alpha=0.3, lw=0,
              label="")   # no legend for shading
    end

    # data
    plot!(t, y, color="black", lw=1.0, label=(show_legend ? "Data" : ""))
    # model
    plot!(t, y_sim, color="blue", lw=2, label=(show_legend ? "Model" : ""))

    if !isempty(cps)
        # real CPs
        vline!(cps, color="red", lw=2, ls=:dash, label=(show_legend ? "Change Points" : ""))
    elseif show_legend
        # dummy CP legend entry (invisible line)
        plot!([NaN], [NaN], color="red", lw=2, ls=:dash, label="Change Points")
    end

    push!(plots, p)
end

p_mocha = plot(plots..., layout=(4,1), size=(900,800))
savefig(p_mocha, "p_mocha.pdf")




using Plots, Random

# --- Generate toy signal with true CPs ---
Random.seed!(123)
t = 0:0.1:100
y = [ti < 30  ? sin(0.4*ti) + 0.4*randn() :
     ti < 60  ? 4.5*sin(0.4*ti) + 0.4*randn() :
     ti < 85  ? 0.5*sin(0.4*ti + 2) + 0.4*randn() :
                 3.5*sin(0.4*ti) + 0.4*randn() for ti in t]

true_cps = [30, 60, 85]
cps_full = [0; true_cps; 100.0]
colors = ["#AED6F1", "#F9E79F", "#A9DFBF", "#F5B7B1"]

p_all = plot(; label="Data", title="All Change Points Detected Simultaneously",
              framestyle=:none, xticks=false, yticks=false, size=(500,300), legendfontsize=7)

ymin, ymax = minimum(y) - 1, maximum(y) + 1
for (i, (x0, x1)) in enumerate(zip(cps_full[1:end-1], cps_full[2:end]))
    plot!([x0, x1, x1, x0], [ymin, ymin, ymax, ymax],
          seriestype=:shape, color=colors[i], alpha=0.3, lw=0, label="")
end


plot!(t, y, color="black", lw=1.5, label="Data")
p_cp_AllOncee = vline!(true_cps, color="red", lw=2, ls=:dash, label ="Change Points")

savefig(p_cp_AllOncee, "p_cp_AllOncee.pdf")

signal = plot(t, y, color="black", lw=1.5, framestyle=:none, label = "Data", xticks=false, yticks=false, size=(500,300))
savefig(signal, "signal.pdf")

#####################
# plotting parameters 
using Plots
gr()

# Data
alpha = [1.0, 4.5, 0.5, 3.5]
beta  = [1.4, 1.4, 1.4, 1.4]
segment_edges = 1:5
colors = ["#AED6F1", "#F9E79F", "#A9DFBF", "#F5B7B1"]

# Create base plot (minimal, tight layout)
p = plot(legend = false, size = (800, 400),
         xlabel = "", ylabel = "",
         xticks = false, yticks = false,
         framestyle = :box,
         xlims = (1, 5),
         ylims = (0, maximum([alpha; beta]) + 0.5))

# Background segment colors
for i in 1:4
    plot!([segment_edges[i], segment_edges[i+1]], [0, 0], fillrange = [6, 6],
          fillcolor = colors[i], fillalpha = 0.4, linealpha = 0, label = false)
end

# Stepify helper
function stepify(values)
    x, y = Float64[], Float64[]
    for i in 1:length(values)
        push!(x, segment_edges[i])
        push!(x, segment_edges[i+1])
        push!(y, values[i])
        push!(y, values[i])
    end
    return x, y
end

# Step plots
xα, yα = stepify(alpha)
xβ, yβ = stepify(beta)

plot!(xα, yα, lw = 2.5, color = :blue)
plot!(xβ, yβ, lw = 2.5, linestyle = :dash, color = :black)

display(p)

savefig(pars_shaded, "pars_shaded.pdf")
###################
# stepwise adding CPs to Covid Model

using Evolutionary 
using DifferentialEquations
using LabelledArrays
using Plots
using Statistics
using Random
using CSV
using DataFrames
using Dates

include("src/Mocha.jl")
using .Mocha


detected_cp = CSV.read("examples/Covid-model/results_detected_cp_penalty40_ts10_pop150.csv", DataFrame)[:,1] 
params = CSV.read("examples/Covid-model/results_params_penalty40__ts10_pop150.csv", DataFrame)[:,1] 
parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω, :ᴺp₁, :ᴺβ,:ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν) 
u0 = [83129285-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
tspan = (0.0, 399.0) 
ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan) 
model_manager = ModelManager(ode_spec) 
n_global = 8 
n_segment_specific = 8

# Simulate without plotting
sim = simulate_full_model(params, detected_cp, parnames,
                          n_global, n_segment_specific,
                          model_manager, data_CP)

# Simulate and plot all compartments with change points and data overlay
sim = simulate_full_model(params, detected_cp, parnames,
                          n_global, n_segment_specific,
                          model_manager, data_CP;
                          plot_results=true,
                          show_change_points=true,
                          show_data=true,
                          data_indices=[5, 6, 7, 9, 11])

