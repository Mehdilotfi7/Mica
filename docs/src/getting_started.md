# Getting Started

Once you've installed and loaded `Mica.jl`, you're ready to begin modeling and detecting changepoints.

---

## First Steps

Try a simple built-in model using the `simulate_model` function:

```julia
using Mica

model = exponential_ode_model()
sol = simulate_model(model)
```

This simulates a basic exponential decay model defined by an ordinary differential equation (ODE), using default parameters and initial conditions.

---

## Core Workflow

A typical workflow in Mica.jl looks like this:

1. **Define or select a model**
   Use one of the built-in examples (`exponential_ode_model`, `example_difference_model`, etc.) or define your own model structure using the provided `ODEModelSpec`, `DifferenceModelSpec`, or `RegressionModelSpec` types.

2. **Simulate the model**
   Use `simulate_model(model)` to generate synthetic or fitted outputs.

3. **Format data and objective**
   Prepare your observed data in a matrix format and define a loss function to quantify the difference between model predictions and data.

4. **Run changepoint detection**
   Use `detect_changepoints(...)` to estimate both the changepoint locations and the model parameters in each segment. The algorithm applies evolutionary optimization to jointly minimize the simulation error and a penalty function.

---

## Next Steps

To learn more:

* Explore Tutorials: Guided examples for ODEs and discrete systems
* Review Problem Types: Understand which models are supported
* Try Examples Realistic use cases and model setups
* See Reference: API documentation for all core types and functions
