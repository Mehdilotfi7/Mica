# Getting Started

This page helps you get started with TSCPDetector.

## Simulating a Model

```julia
using TSCPDetector

model = exponential_ode_model()
sol = simulate_model(model, 0.0, 10.0, 0.1)
