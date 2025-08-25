using Evolutionary 
using DifferentialEquations
using LabelledArrays
using Plots
using Statistics
using Random
using CSV
using DataFrames




# seasonal factor
# https://www.cdc.gov/ncird/whats-new/covid-19-can-surge-throughout-the-year.html
# It says two peaks but i tested did not like the simulation , create second peak where there is no need.
# but maybe change the t0

function fδ(t::Number, δ::Number, t₀::Number=0.0)
    return 1 + δ*cos(2*π*((t - t₀)/365))
end
  
  
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
  du[1]  = -(ᴺβᴺSI)/N + ω * ᴺR - ν * ᴺS + ω * V                          # Susceptible
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
  du[11] =  ν * ᴺS - ω * V                                                # Cumulative Vaccinated
end


function reset_daily!(integrator)
  integrator.u[12] = 0.0  # daily deaths
  integrator.u[13] = 0.0  # daily vaccinations
end

condition(u, t, integrator) = isapprox(t, round(t); atol=1e-8)
reset_cb = DiscreteCallback(condition, reset_daily!,
                            save_positions = (false, false)
                           )
#


function example_ode_model(params, tspan::Tuple{Float64, Float64}, u0)

    # Force ν = 0 before t = 330
    if tspan[2] < 330
        params[:ν] = 0.0
    end
    prob = ODEProblem(CovModel!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=1.0 , abstol = 1.0e-6, reltol = 1.0e-6,
                isoutofdomain = (u,p,t)->any(x->x<0,u))
    return sol[:,:]  # returns matrix-like solution
end

using SciMLBase

function loss_function(observed, simulated)

  if SciMLBase.successful_retcode(simulated)
    infected =  simulated[5,:]
    hospital =  simulated[6,:]
    icu      =  simulated[7,:]
    death    =  simulated[9,:]
    vacc     =  simulated[11,:] 

    # Convert to daily by differencing
    #death = vcat(0.0, diff(death_cum))
    #vacc  = vcat(0.0, diff(vacc_cum))

    #=

    w_1=1/var(log_transform(observed[1,:]))
    w_2=1/var(log_transform(observed[2,:]))
    w_3=1/var(log_transform(observed[3,:]))
    w_4=1/var(log_transform(observed[4,:]))
    w_5=1/var(log_transform(observed[5,:]))


    ϵ = 1e-8  # to avoid division by zero
    loss =
        sum(abs2.(infected .- observed[1, :]))  / (mean(observed[1, :])^2 + ϵ) +
        sum(abs2.(hospital .- observed[2, :]))  / (mean(observed[2, :])^2 + ϵ) +
        sum(abs2.(icu .- observed[3, :]))      / (mean(observed[3, :])^2 + ϵ) +
        sum(abs2.(death .- observed[4, :]))    / (mean(observed[4, :])^2 + ϵ) +
        sum(abs2.(vacc .- observed[5, :]))     / (mean(observed[5, :])^2 + ϵ)
    


  =#

    loss =
    sum(abs, log_transform(infected).- log_transform(observed[1,:]))+
    sum(abs, log_transform(hospital).- log_transform(observed[2,:]))+
    sum(abs, log_transform(icu).- log_transform(observed[3,:]))+
    sum(abs, log_transform(death).- log_transform(observed[4,:]))+
    sum(abs, log_transform(vacc).- log_transform(observed[5,:]))

  


    # ToDo: scaled loss to mean or max + big error for bad parameter and solutions 


    #@assert size(observed) == size(simulated) "Dimension mismatch in loss function."
    

    
    return loss
  else
    return Inf
  end
end





####################################################
# simulation to date 2021-03-27
# 2020-12-26 before starting vaccination. I:335, D:295, H:303, icu:284
# Data
begin
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
end

plot(death_CP)

##############
begin
# Put all datasets into an array
data_CP = [cases_CP, hospital_CP, icu_CP, death_CP, vacc_CP]
#length(data_CP[1])
# Determine the maximum length among all datasets
max_length = maximum(length, data_CP)

# Pad the smaller datasets with zeros to match the maximum length (to the left)
using Smoothers
data_CP = [vcat(zeros(Int, max_length - length(data)), data) for data in data_CP]
data_CP = [vector[1:400] for vector in data_CP]
data_CP[1] = hma(data_CP[1], 14)
data_CP[4] = (hma(data_CP[4], 14))
data_CP[5] = (hma(data_CP[5], 14))

data_CP = reduce(hcat, data_CP)
data_CP = data_CP'
data_CP = Matrix(data_CP)

end
p = scatter(data_CP[1,:])
#savefig(p,"p.png")


########################################
begin
parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω,
           :ᴺp₁, :ᴺβ,:ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)
# propertynames
#                     δ,   ᴺε₀,   ᴺε₁,   ᴺγ₀, ᴺγ₁,     ᴺγ₂, ᴺγ₃,   ω,      ᴺp₁, ᴺβ,  ᴺp₁₂,  ᴺp₂₃,   ᴺp₁D,  ᴺp₂D,  ᴺp₃D,    ν
initial_chromosome = [0.1, 1/7, 1/11.4, 1/14, 1/13.4, 1/9,  1/16, 0.0055,  0.2, 0.05, 0.17, 0.144, 0.01,   0.017, 0.173,   0.01]
lower =              [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,   0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001,   0.05]
upper =              [0.3, 1/3,  1/11.2, 1/5,  1/10.9, 1/5,  1/8,  0.012,   0.8, 8.0, 0.5,   0.5,   0.5,   0.5,   0.5,     0.1]


bounds = (lower, upper)
#u0 = [83129285-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
BIC_0(p, n) = 0.0 * p * log(n)
BIC_12(p, n) = 12.0 * p * log(n)
BIC_40(p, n) = 40.0 * p * log(n)
BIC_1000(p, n) = 1000.0 * p * log(n)
end
@time detected_cp, params = detect_changepoints(
    objective_function,
    n, n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_CP,
    initial_chromosome, parnames, bounds, ga,
    min_length, step,BIC_0
)

CSV.write("results_detected_cp_penalty0_ts10_pop150_nolog.csv", DataFrame(detected_cp=detected_cp))
CSV.write("results_params_penalty0__ts10_pop150_nolog.csv", DataFrame(params=params))


CSV.write("results_detected_cp_penalty15.csv", DataFrame(detected_cp=detected_cp))
CSV.write("results_params_penalty15.csv", DataFrame(params=params))

CSV.write("results_detected_cp_penalty13.csv", DataFrame(detected_cp=detected_cp))
CSV.write("results_params_penalty13.csv", DataFrame(params=params))
########################################################################################################################

using CSV
using DataFrames
detected_cp = CSV.read("results_detected_cp_penalty40_ts10_pop150.csv", DataFrame)[:,1]
params = CSV.read("results_params_penalty40__ts10_pop150.csv", DataFrame)[:,1]


function plot_function(chromosome, change_points)
  # Extract number of segments and beta values
  num_segments = length(change_points) + 1
  
  δ           = chromosome[1]
  ᴺε₀         = chromosome[2]
  ᴺε₁         = chromosome[3]
  ᴺγ₀         = chromosome[4]
  ᴺγ₁         = chromosome[5]
  ᴺγ₂         = chromosome[6]
  ᴺγ₃         = chromosome[7]
  ω           = chromosome[8]
  ᴺp₁_values  = chromosome[9:8:end]
  ᴺβ_values   = chromosome[10:8:end]
  ᴺp₁₂_values = chromosome[11:8:end]
  ᴺp₂₃_values = chromosome[12:8:end]
  ᴺp₁D_values = chromosome[13:8:end]
  ᴺp₂D_values = chromosome[14:8:end]
  ᴺp₃D_values = chromosome[15:8:end]
  ν_values    = chromosome[16:8:end]
  
  # Initialize variables
 
  u0 = [83129285-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  infct = Float64[]
  hos   = Float64[]
  ic    = Float64[]
  dth   = Float64[]
  

  # Loop through segments
  if length(change_points)>0
    for i in 1:num_segments
     # Define segment data and time span based on change points
     if i == 1
       tspan_segment = (0.0, change_points[i]-1)
      elseif i == num_segments
       tspan_segment = (change_points[i-1], tspan[2])
      else
       tspan_segment = (change_points[i-1], change_points[i]-1)
     end
  
     # Extract beta value for this segment
     #beta = beta_values[i]
     #params = [beta, gamma]  # Use extracted gamma for each segment

     ᴺp₁  = ᴺp₁_values[i]
     ᴺβ   = ᴺβ_values[i]
     ᴺp₁₂ = ᴺp₁₂_values[i]
     ᴺp₂₃ = ᴺp₂₃_values[i]
     ᴺp₁D = ᴺp₁D_values[i]
     ᴺp₂D = ᴺp₂D_values[i]
     ᴺp₃D = ᴺp₃D_values[i]
     #ω    = ω_values[i]
     ν    = ν_values[i]
     theta_est = [ᴺp₁, ᴺβ, ᴺp₁₂, ᴺp₂₃, ᴺp₁D, ᴺp₂D, ᴺp₃D, ν]
     theta_fix = [ ω, δ, ᴺε₀, ᴺε₁, ᴺγ₀, ᴺγ₁, ᴺγ₂, ᴺγ₃]

     parnames = (:ω, :δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, 
                 :ᴺp₁, :ᴺβ,:ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)  
     params = @LArray [theta_fix;theta_est] parnames
  
     # Solve ODE for the segment
     problem = ODEProblem(CovModel!, u0, tspan_segment, params)
     #CVODE_BDF(linear_solver=:GMRES)
     sol = solve(problem, Tsit5(), saveat=1.0, #maxiters = 10000000
     abstol = 1.0e-6, reltol = 1.0e-6, isoutofdomain = (u,p,t)->any(x->x<0,u))#, callback = cb)
     
    
     append!(infct, sol[5,:])
     append!(hos, sol[6,:])
     append!(ic, sol[7,:])
     append!(dth, sol[9,:])
     
    
     # Update initial conditions for next segment
     u0 = sol.u[end]
    end
  else
    tspan_segment = tspan
    # Extract beta value for this segment 
    #ᴺp₁  = ᴺp₁_values[1]
     ᴺp₁  = ᴺp₁_values[1]
     ᴺβ   = ᴺβ_values[1]
     ᴺp₁₂ = ᴺp₁₂_values[1]
     ᴺp₂₃ = ᴺp₂₃_values[1]
     ᴺp₁D = ᴺp₁D_values[1]
     ᴺp₂D = ᴺp₂D_values[1]
     ᴺp₃D = ᴺp₃D_values[1]
     #ω    = ω_values[1]
     ν    = ν_values[1]
    theta_est = [ᴺp₁, ᴺβ, ᴺp₁₂, ᴺp₂₃, ᴺp₁D, ᴺp₂D, ᴺp₃D, ν]
    theta_fix = [ ω, δ, ᴺε₀, ᴺε₁, ᴺγ₀, ᴺγ₁, ᴺγ₂, ᴺγ₃]

    parnames = (:ω, :δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃,
                :ᴺp₁, :ᴺβ,:ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν)  
    params = @LArray [theta_fix;theta_est] parnames
    problem = ODEProblem(CovModel!, u0, tspan_segment, params)
    #CVODE_BDF(linear_solver=:GMRES)
    sol = solve(problem, Tsit5(), saveat=1.0, #maxiters = 10000000
    abstol = 1.0e-6, reltol = 1.0e-6, isoutofdomain = (u,p,t)->any(x->x<0,u))#, callback = cb)
    append!(infct, sol[5,:])
    append!(hos, sol[6,:])
    append!(ic, sol[7,:])
    append!(dth, sol[9,:])
  end
  p=plot!([infct, hos, ic, dth], label = ["Simulation" "Simulation" "Simulation" "Simulation"],layout = 4)
  #p=plot!([log_transform(abs.(infct)), log_transform(abs.(hos)), log_transform(abs.(ic)), log_transform(abs.(dth))], label = ["Simulation" "Simulation" "Simulation" "Simulation"],layout = 4)
  display(p)
  return nothing
end

using Plots, Measurements
gr()  # or another backend
pyplot()


p = scatter([data_CP[1,:], data_CP[2,:], data_CP[3,:], data_CP[4,:]], m = (0.3, [:dot :dot :dot :dot], 1),
        label = ["Real data" "Real data" "Real data" "Real data"], layout = 4,
        xticks = (1:50:450,cases_CP_date[1:50:450]) ,seriestype=:steppre, xtickfontrotation=30,
        title=["Infected" "Hospitalized" "ICU" "Death"],titlefont = font(10), 
        legendfont = font(6), legend=:topleft, tickfont=font(7),
        markershape = :circle,
        markersize = 1.5,
        color = :blue,
        size = (900,500),
        bottom_margin = 5Plots.mm,
        guidefont = font(8)
        )

#
p = plot([log_transform(data_CP[1,:]), log_transform(data_CP[2,:]), log_transform(data_CP[3,:]), log_transform(data_CP[4,:])], 
        m = (0.3, [:dot :dot :dot :dot :dot], 1),
        label = ["Real data" "Real data" "Real data" "Real data"], layout = 4,
        xticks = (1:50:450,cases_CP_date[1:50:450]) ,seriestype=:steppre, xrotation=20,
        title=["Infected" "Hospitalized" "ICU" "Death"],titlefont = font(10), 
        legendfont = font(7), legend=:bottomright, tickfont=font(7),
        markershape = :circle,
        markersize = 1.5,
        color = :black,
        guidefont = font(8)
        )
#
        
p = plot_function(params, detected_cp)
p = vline!([detected_cp,detected_cp,detected_cp,detected_cp], label = "Change points")

savefig(p, "CovSim_pen40.pdf")


# Dates for results_detected_cp_penalty7_ts10_pop150
segment_dates = cases_CP.date[detected_cp]
#2020-02-05
#2020-02-15
#2020-03-06
#2020-03-26
#2020-04-05
#2020-07-04
#2020-08-03
#2020-10-02
#2020-10-12
#2020-11-11
#2020-12-11
#2020-12-21


#####################################
# visualization of parameters 
using Dates

segment_dates = cases_CP_date[detected_cp]
segment_labels = vcat(Date("2020-01-27"), segment_dates, Date("2021-03-02"))
segment_labels = Date.(segment_labels)


# Convert to strings for xticks
xtick_labels = string.(segment_labels)

segment_edges = segment_labels
segment_indices = 1:length(segment_labels_str)

cps = ["2020-01-27", "cp1", "cp2", "cp3", "cp4", "cp5", "cp6", "cp7", "cp8", "cp9", "cp10", "cp11","2021-03-02"]
# 4 parameters × 12 segments

param_labels = ["p₁ \n(Detection rate)", "β \n(Infection rate)", "p₁₂ \n(Hospitalization rate)", 
"p₂₃ \n(ICU admision rate)", "p₁D \n(Infection death rate)", 
"p₂D \n(Hospital death rate)","p₃D \n(ICU death rate)"]
param1 = params[9:8:end]
param2 = params[10:8:end]
param3 = params[11:8:end]
param4 = params[12:8:end]
param5 = params[13:8:end]
param6 = params[14:8:end]
param7 = params[15:8:end]
data = [param1 param2 param3 param4 param5 param6 param7]'  
data_rel = data ./ data[:, 1]
#data_rel = hcat(ones(size(data, 1)), data[:, 2:end] ./ data[:, 1:end-1])


# We need 5 y-edges for 4 parameters (just space them uniformly)
yedges = 0:1:7  # 5 edges for 4 rows


heatmaps = []

using Printf
gr()
#pyplot()
#plotly()
#gaston()

for i in 1:size(data_rel, 1)

    vmin = minimum(data_rel[i:i, :])
    vmax = maximum(data_rel[i:i, :])
    tickvals = [vmin, vmax]
    ticklabels = round.(tickvals; digits=2)

    show_xticks = (i == size(data_rel, 1))
    show_colorbar_title = (i == size(data_rel, 1))

    hm = heatmap(
        segment_edges,
        [0, 1],
        data_rel[i:i, :],
        xlabel = show_xticks ? "Change Points" : "",
        ylabel = "",
        yticks = ([0.5], [param_labels[i]]),
        #xticks = show_xticks ? (segment_edges, segment_edges) : false,  
        xticks = show_xticks ? (segment_edges, cps) : false,  
        #xticks=false,
        #xtickfontrotation = show_xticks ? 30 : 0,
        xrotation = show_xticks ? 30 : 0,
        c = :blues,
        #clims = (vmin, vmax),
        colorbar = [vmin, vmax],
        colorbar_title = show_colorbar_title ? "RCPS" : "",  # ← Only for last plot
        #colorbar_tickvals = tickvals,
        #colorbar_ticklabels = ticklabels,
        #colorbar_ticks = (tickvals, ticklabels),
        #colorbar_ticks = ([vmin, vmax], round.([vmin, vmax]; digits=2)),
        size = (900, 130),
        bottom_margin = show_xticks ? 15Plots.mm : 2Plots.mm,
        top_margin = 1Plots.mm,
        left_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        xtickfont = font(7),
        #xtickfont = font(7, halign=:left),
        ytickfont = font(9),
        guidefont = font(7),
        #tickfont = font(9),
        framestyle = :box
    )

    push!(heatmaps, hm)
end

# Combine plots
p2 = plot(heatmaps..., layout = @layout([a;b;c;d;e;f;g]), size = (900, 900))
p4 = plot(p,p2, layout = (2,1))
layout = @layout [a{0.6h} ;b{0.4h}]  # 30% height for p, 70% for p2
p4 = plot(p, p2, layout = layout,  size = (900, 900))
savefig(p2, "relative_pars_Previous_Segment_pen40_gr_backend.pdf")
savefig(p4, "covsim_relative_pars_Previous_Segment_pen40.pdf")


plt = plot(heatmaps..., layout = (length(heatmaps), 1))

segment_edges = collect(1:length(segment_labels))
using Plots: xticks!

xticks!(
  p2[end],  # Apply only to the last subplot
    segment_edges[1:end-1].-0.5 ,            # Shift to left edge
    string.(segment_labels[1:end])
)
