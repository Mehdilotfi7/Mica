# Supported Problem Types

Mica.jl is designed to detect changepoints in **model-based time series**. Instead of relying on statistical shifts (e.g., mean/variance), Mica tracks **changes in the parameters** of an explicit model.

This section outlines the types of models currently supported by the package.

---

## Model Categories

### 1. **Ordinary Differential Equation (ODE) Models**

These models describe the system as a set of continuous-time equations:

* Suitable for: epidemiology, population dynamics, physics-based systems
* Requires: a function defining the ODE, a time span `(t₀, t₁)`, and initial conditions

```julia
struct ODEModelSpec <: AbstractModelSpec
    model_function::Function
    params
    initial_conditions::Vector{Float64}
    tspan::Tuple{Float64, Float64}
end
```

Use when your system evolves continuously and smoothly over time.

---

### 2. **Difference Equation Models**

These simulate discrete-time systems where the state evolves via recurrence relations.

* Suitable for: digital control systems, econometrics, thermal systems
* Requires: initial value, number of steps, external inputs (optional)

```julia
struct DifferenceModelSpec <: AbstractModelSpec
    model_function::Function
    params
    initial_conditions::Float64
    num_steps::Int
    extra_data::Tuple{Vector{Float64}, Vector{Float64}}
end
```

Use when the system is updated at regular discrete time intervals.

---

### 3. **Regression Models**

Simple predictive models based on a parametric function.

* Suitable for: trend fitting, linear/nonlinear regression, control baselines
* Requires: a function, number of time steps, and parameters

```julia
struct RegressionModelSpec <: AbstractModelSpec
    model_function::Function
    params
    time_steps::Int
end
```

Use for interpretable baselines or when data relationships are simple but nonlinear.

---

## Future Extensions

While the current version focuses on ODEs, difference, and regression models, Mica.jl is designed to be extensible.

Planned or possible extensions include:

* State-space models (e.g., Kalman filters)
* Agent-based models
* Hybrid continuous/discrete systems
* Neural differential equations

If you'd like to contribute new model types or request support, visit the [GitHub repository](https://github.com/mehdilotfi7/Mica.jl).

---

Next: Learn how to define and simulate your own model in the Tutorials section.
