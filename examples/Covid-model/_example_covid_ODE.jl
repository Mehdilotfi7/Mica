# Covid Model Definition

# seasonality factor
function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ*cos(2*π*((t - t₀)/365))
end
  
# log transformation
function log_transform(data, threshold=1)
    return [val >= threshold ? log(val) : 0 for val in data]
end

  

# ODE model
function CovModel!(du, u, p, t)
  # State unpacking
  (ᴺS, ᴺE₀, ᴺE₁, ᴺI₀, ᴺI₁, ᴺI₂, ᴺI₃, ᴺR, D, Cases, V) = u

  # Total population
  N = ᴺS + ᴺE₀ + ᴺE₁ + ᴺI₀ + ᴺI₁ + ᴺI₂ + ᴺI₃ + ᴺR + D

  # Parameters
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
  δₜ   = fδ(t, δ)
  ᴺβ   = p.ᴺβ
  ω    = p.ω
  ν    = t < 330 ? 0.0 : p.ν  # vaccination starts at t = 330

  # Infection term
  ᴺβᴺSI = ᴺβ * δₜ * ᴺS * (ᴺE₁ + ᴺI₀ + ᴺI₁)

  # ODEs
  du[1]  = -(ᴺβᴺSI)/N + ω * ᴺR - ν * ᴺS                          # Susceptible
  du[2]  =  (ᴺβᴺSI/N) - (ᴺε₀ * ᴺE₀)                                      # E0
  du[3]  =  (ᴺε₀ * ᴺE₀) - (ᴺε₁ * ᴺE₁)                                    # E1
  du[4]  =  ((1 - ᴺp₁) * ᴺε₁ * ᴺE₁) - (ᴺγ₀ * ᴺI₀)                        # I0
  du[5]  =  (ᴺp₁ * ᴺε₁ * ᴺE₁) - (ᴺγ₁ * ᴺI₁)                              # I1
  du[6]  =  (ᴺp₁₂ * ᴺγ₁ * ᴺI₁) - (ᴺγ₂ * ᴺI₂)                             # I2
  du[7]  =  (ᴺp₂₃ * ᴺγ₂ * ᴺI₂) - (ᴺγ₃ * ᴺI₃)                             # I3
  du[8]  =  ᴺγ₀ * ᴺI₀ +
            (1 - ᴺp₁₂ - ᴺp₁D) * ᴺγ₁ * ᴺI₁ +
            (1 - ᴺp₂₃ - ᴺp₂D) * ᴺγ₂ * ᴺI₂ +
            (1 - ᴺp₃D) * ᴺγ₃ * ᴺI₃ - ω * ᴺR + ν * ᴺS                      # Recovered
  du[9]  =  (ᴺp₁D * ᴺγ₁ * ᴺI₁) + (ᴺp₂D * ᴺγ₂ * ᴺI₂) + (ᴺp₃D * ᴺγ₃ * ᴺI₃)   # Cumulative Deaths
  du[10] =  ᴺp₁ * ᴺε₁ * ᴺE₁                                               # Cumulative Cases
  du[11] =  ν * ᴺS                                                # Cumulative Vaccinated
end


function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0)

    # Force ν = 0 before t = 330
    if tspan[2] < 330
        params[:ν] = 0.0
    end
    prob = ODEProblem(CovModel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0 , abstol = 1.0e-6, reltol = 1.0e-6,
                isoutofdomain = (u,p,t)->any(x->x<0,u))

    # return matrix if successful, otherwise NaN filler
    if SciMLBase.successful_retcode(sol)
        return sol[:,:]
    else
        return fill(NaN, length(u0), Int(tspan[2]-tspan[1])+1)
    end
end



function loss_function(observed, simulated)

  # return big loss if sol is unsuccessful
    if any(isnan, simulated)
       return Inf
    end

    infected =  simulated[5,:]
    hospital =  simulated[6,:]
    icu      =  simulated[7,:]
    death    =  simulated[9,:]
    vacc     =  simulated[11,:] 

    loss =
    sum(abs, log_transform(infected).- log_transform(observed[1,:]))+
    sum(abs, log_transform(hospital).- log_transform(observed[2,:]))+
    sum(abs, log_transform(icu).- log_transform(observed[3,:]))+
    sum(abs, log_transform(death).- log_transform(observed[4,:]))+
    sum(abs, log_transform(vacc).- log_transform(observed[5,:]))    

    return loss
end





####################################################
# simulation from 2020-01-27 to date 2021-03-27

# Data loading
begin
cd(dirname(@__FILE__))
cases_CP = CSV.read("case_rki_daily.csv", DataFrame)
cases_CP_date = cases_CP.date
cases_CP = cases_CP.total

hospital_CP = CSV.read("Hospitalization_rki_daily.csv", DataFrame)
hospital_CP = hospital_CP.total

death_CP = CSV.read("death_rki_daily.csv", DataFrame)
death_CP = death_CP.Todesfaelle_neu
death_CP = cumsum(death_CP)

icu_CP = CSV.read("icu_rki_daily.csv", DataFrame)
icu_CP = icu_CP.total

vacc_CP = CSV.read("vaccination_rki_daily_allShots.csv", DataFrame)
vacc_CP = vacc_CP.Total
vacc_CP = cumsum(vacc_CP)

end


##############
begin
# Put all datasets into an array
data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
#length(data_CP[1])
# Determine the maximum length among all datasets
max_length = maximum(length, data_CP)


data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
data_CP = [vector[1:400] for vector in data_CP]
data_CP[1] = hma(data_CP[1], 14)
data_CP[4] = (hma(data_CP[4], 14))
data_CP[5] = (hma(data_CP[5], 14))

data_CP = reduce(hcat, data_CP)
data_CP = data_CP'
data_CP = Matrix(data_CP)

end

########################################
# Prepare setting for using Mocha
begin

# Model parameter names [non-segment specific parameters; segment parameters]
parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
           :ᴺp₁, :ᴺβ,:ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
           

# Initial guess, upper and lower bounds for non-segment and segment specific parameters           
#                     δ,   ᴺε₀,   ᴺε₁,   ᴺγ₀, ᴺγ₁,     ᴺγ₂, ᴺγ₃,   ω,      ᴺp₁, ᴺβ,  ᴺp₁₂,  ᴺp₂₃,   ᴺp₁D,  ᴺp₂D,  ᴺp₃D,    ν
initial_chromosome = [0.1, 1/7, 1/11.4, 1/14, 1/13.4, 1/9,  1/16, 0.0055,  0.2, 0.05, 0.17, 0.144, 0.01,   0.017, 0.173,   0.01]
lower =              [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,   0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001,   10e-5]
upper =              [0.3, 1/3,  1/11.2, 1/5,  1/10.9, 1/5,  1/8,  0.012,   0.8, 8.0, 0.5,   0.5,   0.5,   0.5,   0.5,     0.1]
bounds = (lower, upper)
u0 = [83129285-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
tspan = (0.0, 399.0)

# constructing ode model using ModelSimulation module
ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan)
model_manager = ModelManager(ode_spec)

# number of segment and non segment specific parameters
n_global = 8
n_segment_specific = 8

# minimum length of sliding window and window elongation step
min_length = 10
step = 10

# GA setting
ga = GA(populationSize = 150, selection = tournament(2), crossover = SBX(0.7, 1), mutationRate=0.7,
crossoverRate=0.7, mutation = gaussian(0.0001))
n = size(data_CP,2)

# Different penalty functions 
BIC_0(p, n) = 0.0 * p * log(n)
BIC_12(p, n) = 12.0 * p * log(n)
BIC_40(p, n) = 40.0 * p * log(n)

# indeces for available data for state variables
data_indices = [5, 6, 7, 9, 11]
end
@time detected_cp, params = detect_changepoints(
    objective_function,
    n, n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_CP, 
    initial_chromosome, parnames, bounds, ga,
    min_length, step, BIC_0, data_indices
)

