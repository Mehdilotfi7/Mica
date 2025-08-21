using Distributed

# Remove any previously broken workers
rmprocs(workers())

if !isempty(workers())
    rmprocs(workers())
end

# Add all available cores (excluding main)
addprocs(3)


@everywhere begin
    if !isdefined(Main, :Random)
        Base.require(:Random)
    end
end
@everywhere begin
    try
        using Random
    catch e
        @warn "Random module failed on worker $(myid()). Attempting manual require..."
        Base.require(:Random)
    end

    # Load other modules
    using CSV, DataFrames, Smoothers
    using Evolutionary, DifferentialEquations, LabelledArrays
    using Statistics, Plots
end
@everywhere begin
    try
        using Random
    catch e
        @warn "Random module failed on worker $(myid()). Attempting manual require..."
        Base.require(:Random)
    end

    # Load other modules
    using CSV, DataFrames, Smoothers
    using Evolutionary, DifferentialEquations, LabelledArrays
    using Statistics, Plots
end



# Load required packages on all workers
@everywhere begin
 using CSV, DataFrames, Smoothers
 using Evolutionary, DifferentialEquations, LabelledArrays, Plots
 using Statistics
 using TSCPDetector

end
@everywhere function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ*cos(2π*(t - t₀)/365)
end

@everywhere function log_transform(data, threshold=1)
    return [val >= threshold ? log(val) : 0 for val in data]
end

@everywhere function CovModel!(du,u,p,t)
  
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

@everywhere function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0)
    prob = ODEProblem(CovModel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0, abstol = 1e-6, reltol = 1e-6,
                isoutofdomain = (u,p,t)->any(x->x<0,u))
    return sol[:,:]
end

@everywhere function loss_function(observed, simulated)
    infected =  simulated[5,:]
    hospital =  simulated[6,:]
    icu      =  simulated[7,:]
    death    =  simulated[9,:]
    vacc     =  simulated[11,:] 

    return sum(abs, log_transform(infected) .- log_transform(observed[1,:])) +
           sum(abs, log_transform(hospital) .- log_transform(observed[2,:])) +
           sum(abs, log_transform(icu) .- log_transform(observed[3,:])) +
           sum(abs, log_transform(death) .- log_transform(observed[4,:])) +
           sum(abs, log_transform(vacc) .- log_transform(observed[5,:]))
end



# === STEP 1: Load data on main ===
cases_path = abspath("test/case_rki_daily.csv")
hosp_path  = abspath("test/Hospitalization_rki_daily.csv")
death_path = abspath("test/death_rki_daily.csv")
icu_path   = abspath("test/icu_rki_daily.csv")
vacc_path  = abspath("test/vaccination_rki_daily_allShots.csv")

@assert isfile(cases_path)
@assert isfile(hosp_path)
@assert isfile(death_path)
@assert isfile(icu_path)
@assert isfile(vacc_path)

# Broadcast file paths to workers
@everywhere cases_path = $cases_path
@everywhere hosp_path  = $hosp_path
@everywhere death_path = $death_path
@everywhere icu_path   = $icu_path
@everywhere vacc_path  = $vacc_path

# === STEP 2: Load and preprocess data on all workers ===
@everywhere begin
    cases_CP = CSV.read(cases_path, DataFrame).total[1:400]
    hosp_CP  = CSV.read(hosp_path, DataFrame).total[1:400]
    death_CP = cumsum(CSV.read(death_path, DataFrame).Todesfaelle_neu[1:400])
    icu_CP   = CSV.read(icu_path, DataFrame).total[1:400]
    vacc_CP  = cumsum(CSV.read(vacc_path, DataFrame).Total[1:400])


    data_CP = [cases_CP, hosp_CP, icu_CP, death_CP, vacc_CP]
    #length(data_CP[1])
    # Determine the maximum length among all datasets
    max_length = maximum(length, data_CP)

    # Pad the smaller datasets with zeros to match the maximum length (to the left)
    data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
    data_CP = [vector[1:400] for vector in data_CP]
    data_CP[1] = hma(data_CP[1], 21)
    

    data_CP = reduce(hcat, data_CP)
    data_CP = data_CP'
    data_CP = Matrix(data_CP)

end

# === STEP 3: Shared configuration ===
@everywhere begin
    parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
                :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)

    initial_chromosome = [0.1, 1/7, 1/11.4, 1/14, 1/13.4, 1/9,  1/16, 0.0055,
                          0.2, 0.05, 0.17, 0.144, 0.01, 0.017, 0.173, 0.01]
    lower = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
             0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 1e-4]
    upper = [0.3, 1/3,  1/11.2, 1/5,  1/10.9, 1/5,  1/8,  0.012,
             0.8, 8.0, 0.5,   0.5,   0.5,   0.5,   0.5,   0.1]

    bounds = (lower, upper)
    u0 = [83129285-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    tspan = (0.0, 399.0)

    ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan)
    model_manager = ModelManager(ode_spec)

    n_global = 8
    n_segment_specific = 8
    min_length = 10
    step = 10
    ga = GA(populationSize = 150, selection = tournament(2), crossover = SBX(0.7, 1),
            mutationRate=0.7, crossoverRate=0.7, mutation = gaussian(0.0001))

    n = size(data_CP, 2)
end

# === STEP 4: Generate and share BIC functions ===
penalty_range = 41:49  

for pen in penalty_range
    ex = quote
        function $(Symbol("BIC_penalty", pen))(p, n)
            return $(pen) * p + log(n)
        end
    end
    @everywhere @eval $ex
end

BIC_penalty_functions = []
penalty_values = []

for pen in penalty_range
    push!(BIC_penalty_functions, getfield(Main, Symbol("BIC_penalty", pen)))
    push!(penalty_values, pen)
end

penalty_tasks = [(BIC_penalty_functions[i], penalty_values[i]) for i in 1:length(BIC_penalty_functions)]

# === STEP 5: Detection function ===
@everywhere function run_detection(task::Tuple{Function, Int})
BIC_penalty, pen = task

parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
            :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)

initial_chromosome = [0.1, 1/7, 1/11.4, 1/14, 1/13.4, 1/9,  1/16, 0.0055,
              0.2, 0.05, 0.17, 0.144, 0.01, 0.017, 0.173, 0.01]
lower = [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,
 0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001, 1e-4]
upper = [0.3, 1/3,  1/11.2, 1/5,  1/10.9, 1/5,  1/8,  0.012,
 0.8, 8.0, 0.5,   0.5,   0.5,   0.5,   0.5,   0.1]

bounds = (lower, upper)
u0 = [83129285-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tspan = (0.0, 399.0)

ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan)
model_manager = ModelManager(ode_spec)

n_global = 8
n_segment_specific = 8
min_length = 10
step = 10
ga = GA(populationSize = 150, selection = tournament(2), crossover = SBX(0.7, 1),
mutationRate=0.7, crossoverRate=0.7, mutation = gaussian(0.0001))

n = size(data_CP, 2)

    detected_cp, params = detect_changepoints(
        objective_function,
        n, n_global, n_segment_specific,
        model_manager,
        loss_function,
        data_CP,
        initial_chromosome, parnames, bounds, ga,
        min_length, step, BIC_penalty
    )

    CSV.write("results_detected_cp_penalty$(pen)_ts10_pop150.csv", DataFrame(detected_cp=detected_cp))
    CSV.write("results_params_penalty$(pen)_ts10_pop150.csv", DataFrame(params=params))

    return pen
end

# === STEP 6: Run in parallel ===
results = pmap(run_detection, penalty_tasks)
