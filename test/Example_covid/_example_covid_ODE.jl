using Evolutionary 
using DifferentialEquations
using LabelledArrays
using Plots
using Statistics
using Random
using CSV
using DataFrames




# seasonal factor
function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ*cos(2*π*((t - t₀)/365))
end
  
  
function log_transform(data, threshold=1e-3)
    return [val > threshold ? log(val) : val for val in data]
end
  

# ODE model
function CovModel!(du,u,p,t)
  
    # states
    (ᴺS, ᴺE₀, ᴺE₁, ᴺI₀, ᴺI₁, ᴺI₂, ᴺI₃,ᴺR, D, Cases, V) = u[1:11]
    N = ᴺS + ᴺE₀ + ᴺE₁ + ᴺI₀ + ᴺI₁ + ᴺI₂ + ᴺI₃ + ᴺR + D 
  
    # params
  
    ᴺε₀  = p.ᴺε₀
    ᴺε₁  = p.ᴺε₁
    ᴺγ₀  = p.ᴺγ₀
    ᴺγ₁  = p.ᴺγ₁
    ᴺγ₂  = p.ᴺγ₂
    ᴺγ₃  = p.ᴺγ₃ 
    ᴺp₁  = p.ᴺp₁ 
    ᴺp₁₂ = p.ᴺp₁₂ 
    ᴺp₂₃ = p.ᴺp₂₃
    ᴺp₁D = p.ᴺp₁D
    ᴺp₂D = p.ᴺp₂D
    ᴺp₃D = p.ᴺp₃D
    δ    = p.δ
    δₜ = fδ(t,δ)
    ᴺβ   = p.ᴺβ  
    ω    = p.ω 
    if t < 330
      ν    = 0
    else
      ν    = p.ν  
    end
  
     
    
  
    # ODE System
        ᴺβᴺSI = ᴺβ * δₜ * ᴺS * (ᴺE₁ + ᴺI₀ + ᴺI₁)
        N = ᴺS + ᴺE₀ + ᴺE₁ + ᴺI₀ + ᴺI₁ + ᴺI₂ + ᴺI₃ + ᴺR + D
        # ᴺαₜ * δₜ * 
        du[1]  = - (ᴺβᴺSI)/N + ω * ᴺR - ν * ᴺS # S
        du[2]  =  (ᴺβᴺSI/N) - (ᴺε₀ * ᴺE₀) # E0
        du[3]  =  (ᴺε₀ * ᴺE₀) - (ᴺε₁ * ᴺE₁) # E1
        du[4]  =  ((1 - ᴺp₁) * ᴺε₁ * ᴺE₁) - (ᴺγ₀ * ᴺI₀) # I0
        du[5]  =  (ᴺp₁ * ᴺε₁ * ᴺE₁) - (ᴺγ₁ * ᴺI₁) # I1
        du[6]  =  (ᴺp₁₂ * ᴺγ₁ * ᴺI₁) - (ᴺγ₂ * ᴺI₂) # I2
        du[7]  =  (ᴺp₂₃ * ᴺγ₂ * ᴺI₂) - (ᴺγ₃ * ᴺI₃) # I3
        du[8]  =  ᴺγ₀ * ᴺI₀ + (1 - ᴺp₁₂ - ᴺp₁D) * ᴺγ₁ * ᴺI₁ + (1 - ᴺp₂₃ - ᴺp₂D) * ᴺγ₂ * ᴺI₂ +(1 - ᴺp₃D)* ᴺγ₃ * ᴺI₃ - ω * ᴺR + ν * ᴺS  # R
        du[9]  =  (ᴺp₁D * ᴺγ₁ * ᴺI₁) + (ᴺp₂D * ᴺγ₂ * ᴺI₂) + ( ᴺp₃D * ᴺγ₃ * ᴺI₃) # D
        du[10] =  (ᴺp₁ * ᴺε₁ * ᴺE₁) # Cases
        du[11] =  ν * ᴺS  # Vaccination
        #du[11] =  0 # Vaccination
  
end

function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0::Vector{Float64})
    prob = ODEProblem(CovModel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol = 1.0e-6, reltol = 1.0e-6,
                isoutofdomain = (u,p,t)->any(x->x<0,u))
    return sol[:,:]  # returns matrix-like solution
end

function loss_function(observed, simulated)
    infected =  simulated[5,:]
    hospital =  simulated[6,:]
    icu      =  simulated[7,:]
    death    =  simulated[9,:]
    vacc     =  simulated[11,:] 
    #@assert size(observed) == size(simulated) "Dimension mismatch in loss function."
    loss =
    sum(abs, log_transform(infected).- log_transform(observed[1]))+
    sum(abs, log_transform(hospital).- log_transform(observed[2]))+
    sum(abs, log_transform(icu).- log_transform(observed[3]))+
    sum(abs, log_transform(death).- log_transform(observed[4]))+
    sum(abs, log_transform(vacc).- log_transform(observed[5]))
    return loss
end



####################################################
# simulation to date 2021-03-27
# 2020-12-26 before starting vaccination. I:335, D:295, H:303, icu:284
# Data
cd(dirname(@__FILE__))
cases_CP = CSV.read("case_rki_daily.csv", DataFrame)
cases_CP_date = cases_CP.date
cases_CP = cases_CP.total
#plot(cases_CP[1:250])
hospital_CP = CSV.read("Hospitalization_rki_daily.csv", DataFrame)
hospital_CP = hospital_CP.total
#plot(hospital_CP[1:600])
death_CP = CSV.read("death_rki_daily.csv", DataFrame)
death_CP = death_CP.Todesfaelle_neu
death_CP = cumsum(death_CP)
#death_CP = (death_CP)
#plot(death_CP[1:600])
icu_CP = CSV.read("icu_rki_daily.csv", DataFrame)
icu_CP = icu_CP.total
#plot(icu_CP[1:600])
vacc_CP = CSV.read("vaccination_rki_daily_allShots.csv", DataFrame)
vacc_CP = vacc_CP.Total
vacc_CP = cumsum(vacc_CP)
#plot(vacc_CP)
# Select columns 4 to 12 (inclusive)
#columns_to_sum = Matrix(icu_CP[:, 4:12])
#icu_CP = sum(columns_to_sum, dims=2)[:,1]


##############
# Put all datasets into an array
data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
#length(data_CP[1])
# Determine the maximum length among all datasets
max_length = maximum(length, data_CP)

# Pad the smaller datasets with zeros to match the maximum length (to the left)
using Smoothers
data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
data_CP = [vector[1:400] for vector in data_CP]
data_CP[1] = hma(data_CP[1], 21)
data_CP[4] = hma(data_CP[4], 21)
data_CP[5] = hma(data_CP[5], 21)

data_CP = reduce(hcat, data_CP)
data_CP = data_CP'
data_CP = Matrix(data_CP)


########################################

parnames = (:ω, :δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃,
           :ᴺp₁, :ᴺβ,:ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
# propertynames
#                     δ,   ᴺε₀,   ᴺε₁,   ᴺγ₀, ᴺγ₁,     ᴺγ₂, ᴺγ₃,   ω,      ᴺp₁, ᴺβ,  ᴺp₁₂,  ᴺp₂₃,   ᴺp₁D,  ᴺp₂D,  ᴺp₃D,    ν
initial_chromosome = [0.1, 1/7, 1/11.4, 1/14, 1/13.4, 1/9,  1/16, 0.0055,  0.2, 0.05, 0.17, 0.144, 0.01,   0.017, 0.173,   0.01]
lower =              [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,   0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001,   10e-5]
upper =              [0.3, 1/3,  1/11.2, 1/5,  1/10.9, 1/5,  1/8,  0.012,   0.8, 8.0, 0.5,   0.5,   0.5,   0.5,   0.5,     0.1]
bounds = (lower, upper)
u0 = [83129285-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tspan = (0.0, 399.0)
ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan)
model_manager = ModelManager(ode_spec)
n_global = 8
n_segment_specific = 8
min_length = 10
step = 10
ga = GA(populationSize = 100, selection = tournament(2), crossover = SBX(0.7, 1), mutationRate=0.7,
crossoverRate=0.7, mutation = gaussian(0.0001))
n = size(data_CP,2)
my_penalty4(p, n) = 860.0 * p * log(n)

@time detected_cp, params = detect_changepoints(
    objective_function,
    n, n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_CP,
    initial_chromosome, parnames, bounds, ga,
    min_length, step
)



# ploting 
function simulate_full_model(
    chromosome,
    change_points,
    parnames,
    n_global::Int,
    n_segment_specific::Int,
    model_manager::ModelManager,
    data::Matrix{Float64}
)
    constant_pars, segment_pars_list = extract_parameters(chromosome, n_global, n_segment_specific)
    num_segments = length(change_points) + 1

    # Initial condition
    u0 = get_initial_condition(model_manager)
    full_sim = Float64[]  # storage for concatenated simulated data
    all_sims = Vector{Matrix{Float64}}()  # optional: store each segment separately

    if length(change_points) > 0
        for i in 1:num_segments
            idx_start = (i == 1) ? 1 : change_points[i - 1] + 1
            idx_end = (i > length(change_points)) ? size(data, 2) : change_points[i]
            seg_pars = segment_pars_list[i]

            all_pars = @LArray [constant_pars; seg_pars] parnames
            model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)
            sim_data = simulate_model(model_spec)
            @show sim_data

            push!(all_sims, sim_data)
            u0 = update_initial_condition(model_manager, sim_data)
        end
    else
        # Only one segment
        idx_start = 1
        idx_end = size(data, 2)

        seg_pars = segment_pars_list[1]
        all_pars = @LArray [constant_pars; seg_pars] parnames
        model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)
        sim_data = simulate_model(model_spec)

        push!(all_sims, sim_data)
    end

    # Option 1: concatenate all sims to a single matrix
    full_sim = hcat(all_sims...)

    return full_sim  # or return all_sims if you want per-segment
end

_ , pars = optimize_with_changepoints(
    objective_function, params, parnames, cp, bounds, ga,
    n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_CP,
    options=Evolutionary.Options(show_trace=true)
)

using CSV
using DataFrames

cp = CSV.read("results_detected_cp_penalty10.csv", DataFrame).detected_cp
params = CSV.read("results_params_penalty10.csv", DataFrame).params

simulated = simulate_full_model(
    pars,
    cp,
    parnames,
    n_global,
    n_segment_specific,
    model_manager,
    data_CP
)

# Now plot
using Plots
plot!(simulated[5,:], title="Simulated Output Across Segments")
plot(data_CP[1,:])


using Plots

# Time vector
T = 1:size(simulated, 2)

# Extract signals from simulated data
infected = simulated[5, :]
hospital = simulated[6, :]
icu      = simulated[7, :]
death    = simulated[9, :]
vacc     = simulated[11, :]

# Corresponding real data
infected_real = data_CP[1, :]
hospital_real = data_CP[2, :]
icu_real      = data_CP[3, :]
death_real    = data_CP[4, :]
vacc_real     = data_CP[5, :]

# Names and group data
labels = ["Infected", "Hospitalized", "ICU", "Deaths", "Vaccinated"]
sim_vals = [infected, hospital, icu, death, vacc]
real_vals = [infected_real, hospital_real, icu_real, death_real, vacc_real]

# Create multi-plot layout (5 rows × 1 column)
plt = plot(layout = (5,1), size=(800,1000))

for i in 1:5
    plot!(
        plt[i], T, real_vals[i], label = "Observed", lw = 2, color = :blue
    )
    plot!(
        plt[i], T, sim_vals[i], label = "Simulated", lw = 2, linestyle = :dash, color = :red
    )
    title!(plt[i], labels[i])
    xlabel!(plt[i], "Time")
    ylabel!(plt[i], "Count")

    # Add vertical change point lines
    for cp in cp
        vline!(plt[i], [cp], linestyle = :dot, color = :black, label = "")
    end
end

display(plt)
