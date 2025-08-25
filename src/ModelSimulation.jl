# module ModelSimulation

# =============================================================================
# Model Specification Types
# =============================================================================

"""
Abstract base type for all model specifications.
"""
abstract type AbstractModelSpec end


"""
Specification for an ODE (Ordinary Differential Equation) model.

# Fields
- `model_function::Function`: A function that defines the ODE dynamics.
- `params::Dict{Symbol, Any}`: Parameters needed for the model.
- `initial_conditions::Vector{Float64}`: Initial conditions for the ODE system.
- `tspan::Tuple{Float64, Float64}`: Time span over which to simulate the model.
"""
struct ODEModelSpec <: AbstractModelSpec
    model_function::Function
    params
    initial_conditions::Vector{Float64}
    tspan::Tuple{Float64, Float64}
end

"""
Specification for a discrete Difference Equation model.

# Fields
- `model_function::Function`: A function that defines the difference dynamics.
- `params::Dict{Symbol, Any}`: Parameters needed for the model.
- `initial_conditions::Float64`: Initial state of the system.
- `num_steps::Int`: Number of time steps for the simulation.
- `extra_data::Tuple{Vector{Float64}, Vector{Float64}}`: Additional inputs (e.g., external variables).
"""
struct DifferenceModelSpec <: AbstractModelSpec
    model_function::Function
    params
    initial_conditions::Float64
    num_steps::Int
    extra_data::Tuple{Vector{Float64}, Vector{Float64}}
end

"""
Specification for a simple Regression model (e.g., linear).

# Fields
- `model_function::Function`: A function that defines the regression output.
- `params::Dict{Symbol, Any}`: Parameters needed for the model.
- `time_steps::Int`: Number of time steps or observations.
"""
struct RegressionModelSpec <: AbstractModelSpec
    model_function::Function
    params
    time_steps::Int
end

"""
    ARIMAModelSpec <: AbstractModelSpec

A specification type for defining ARIMA model behavior in the Mocha changepoint detection framework.

# Fields
- `model_function::Function`: A function that simulates the ARIMA model given parameters, time span, and optional inputs. Typically returns a time series array of shape `(1, T)` where `T` is the number of time steps.
- `params::Vector`: The initial parameter vector, containing:
    - `μ` (drift term),
    - `p` AR coefficients (segment-specific),
    - `q` MA coefficients (segment-specific).
- `time_steps::Int`: The total number of time points over which the model is simulated.
- `p::Int`: The number of autoregressive (AR) terms.
- `d::Int`: The order of differencing (integration).
- `q::Int`: The number of moving average (MA) terms.

# Usage
Used as input to `ModelManager` for segment-based simulation and parameter estimation via Mocha's genetic algorithm-based changepoint detection pipeline.

# Example
```julia
spec = ARIMAModelSpec(
    simulate_model,     # ARIMA simulation function
    [0.1, 0.2, -0.1, 0.05],  # μ, AR, MA coefficients
    200,                # simulate over 200 time steps
    1,                  # AR order p
    1,                  # differencing d
    1                   # MA order q
)
"""

struct ARIMAModelSpec <: AbstractModelSpec
    model_function::Function
    params::Vector
    time_steps::Int
    p::Int
    d::Int
    q::Int
end
# =============================================================================
# Model Simulation Functions
# =============================================================================

"""
Simulates an ODEModelSpec by solving the ODE system.

# Arguments
- `model::ODEModelSpec`: An ODE model specification.

# Returns
- Simulated results over time.
"""
function simulate_model(model::ODEModelSpec)
    return model.model_function(model.params, model.tspan, model.initial_conditions)
end

"""
Simulates a DifferenceModelSpec by iterating the discrete equation.

# Arguments
- `model::DifferenceModelSpec`: A Difference model specification.

# Returns
- Simulated results over discrete time steps.
"""
function simulate_model(model::DifferenceModelSpec)
    return model.model_function(model.params, model.initial_conditions, model.num_steps, model.extra_data)
end

"""
Simulates a RegressionModelSpec by evaluating the regression model.

# Arguments
- `model::RegressionModelSpec`: A Regression model specification.

# Returns
- Simulated outputs.
"""
function simulate_model(model::RegressionModelSpec)
    return model.model_function(model.params, model.time_steps)
end

# =============================================================================
# Example Model Functions
# =============================================================================

"""
Example: Simple exponential decay ODE model.

Defines the dynamics `du/dt = -p * u`.

# Arguments
- `params::Dict`: Model parameters, expects key `:p`.
- `tspan::Tuple`: (start_time, end_time).
- `u0::Vector{Float64}`: Initial condition vector.

# Returns
- `Matrix`: state variable evolution.
"""
function exponential_ode_model(p, tspan, u0)
    function ode!(du, u, p, t)
        du[1] = -p[1] * u[1]
    end

    prob = ODEProblem(ode!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0)

    return sol[:,:]
end

"""
Example: Discrete difference equation model.

Simulates a difference equation influenced by external variables.

# Arguments
- `params::Dict`: Model parameters.
- `initial_conditions::Float64`: Initial state.
- `num_steps::Int`: Number of steps.
- `extra_data::Tuple`: (wind_speeds, ambient_temperatures).

# Returns
- `DataFrame`: Time and state variable evolution.
"""
function example_difference_model(params, initial_conditions, num_steps, extra_data)
    wind_speeds, ambient_temperatures = extra_data

    state_values = zeros(num_steps)
    state_values[1] = initial_conditions

    for k in 2:num_steps
        u1 = wind_speeds[k]
        u2 = ambient_temperatures[k]
        y_prev = state_values[k - 1]

        state_values[k] = (params[:θ1] * u1^3 + params[:θ2] * u1^2 + params[:θ3] * u1 + y_prev - u2) /
                          (params[:θ4] * u1^3 + params[:θ5] * u1^2 + params[:θ6] * u1 + params[:θ7]) + u2
    end

    time = 1:num_steps
    return state_values[:,:]
end

"""
Example: Simple linear regression model.

Simulates a linear trend `y = a * t + b`.

# Arguments
- `params::Dict`: Model parameters, expects `:a` and `:b`.
- `time_steps::Int`: Number of time steps.

# Returns
- `DataFrame`: Time and simulated values.
"""
function example_regression_model(params, time_steps::Int)
    time = 1:time_steps
    simulated_values = params[1] .* time .+ params[2]

    return simulated_values[:,:]
end

#end # module
