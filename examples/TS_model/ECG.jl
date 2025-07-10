using StateSpaceModels

function simulate_model(model::ARIMAModelSpec, params, t_start::Float64, t_end::Float64, Δ::Float64)
    T = Int(round((t_end - t_start) / Δ))  # number of steps

    μ = params[1]
    ϕ = params[2 : 1 + model.p]
    θ = params[2 + model.p : end]

    dummy_data = randn(T)  # just to initialize SARIMA
    arima_model = SARIMA(dummy_data; order=(model.p, model.d, model.q), seasonal=(0, 0, 0, 0))

    arima_model.coefficients.ar .= ϕ
    arima_model.coefficients.ma .= θ
    arima_model.coefficients.c = μ
    arima_model.σ² = 1.0  # set to constant noise

    sim = simulate(arima_model, T)

    return reshape(sim, 1, :)
end




# Parameter naming (μ, ar1-ar5, ma1-ma4) for ARIMA(5,2,4)
parnames = (:ma1, :ma2, :ma3, :ma4, :μ, :ar1, :ar2, :ar3, :ar4, :ar5)

# Initial guess
initial_chromosome = [0.0, 0.2, -0.3, 0.1, 0.0, 0.05, 0.4, -0.2, 0.1, 0.0]

# Bounds
lower = [-10.0; fill(-0.9, 5); fill(-0.9, 4)]
upper = [10.0; fill(0.9, 5); fill(0.9, 4)]
bounds = (lower, upper)

# Segment/non-segment counts
n_global = 4     # MA  global
n_segment_specific = 6  # AR/μ segment-specific

# Segment length
min_length = 10
step = 10

# GA setup
ga = GA(populationSize = 150, selection = uniformranking(20), crossover = MILX(0.01, 0.17, 0.5), mutationRate=0.3,
        crossoverRate=0.6, mutation = gaussian(0.0001))

# Time and dummy input
tspan = (0.0, length(data_M))
u0 = nothing  # unused for ARIMA

# ARIMA ModelSpec
arima_spec = RegressionModelSpec(example_arima_model, initial_chromosome, Int(tspan[2]))
model_manager = ModelManager(arima_spec)

# Penalty
my_penalty4(p, n) = 0.0 * p * log(n)


@time detected_cp, params = detect_changepoints(
    objective_function,
    length(data_M), n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_M,
    initial_chromosome, parnames, bounds, ga,
    min_length, step, my_penalty4
)
