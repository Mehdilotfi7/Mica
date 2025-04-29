# TSCPDetector
=======

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Mehdilotfi7.github.io/FirstPkg.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Mehdilotfi7.github.io/FirstPkg.jl/dev/)
[![Build Status](https://github.com/Mehdilotfi7/FirstPkg.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Mehdilotfi7/FirstPkg.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Mehdilotfi7/FirstPkg.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Mehdilotfi7/FirstPkg.jl)

![My Algorithm Icon](TSCPDetector2.png)



## Table of Contents
1. [Introduction](#introduction)
2. [Models](#models)
   - [Time Series Models](#time-series-models)
   - [Differential Equation Models](#differential-equation-models)
3. [Model Input Requirements](#Model-Input-Requirements)
4. [Simulation](#simulation)
5. [Installation](#installation)
6. [Usage](#usage)
   - [COVID-19 Simulation Using ODEs](#covid-19-simulation-using-odes)
   - [Wind Turbine Cooling System Using Difference Equations](#wind-turbine-cooling-system-using-difference-equations)
7. [Package_development](#Package_development)

## Introduction
This project focuses on simulating different types of mathematical models, including time series, differential equations, and more. You can use it to analyze and forecast various systems.

```markdown
# TSCPDetector

TSCPDetector/
├── Project.toml
├── src/
│   ├── TSCPDetector.jl
│   ├── DataHandling.jl
│   ├── ModelSimulation.jl
│   ├── Optimization.jl
│   ├── Visualization.jl
└── test/
    └── runtests.jl
```

## Models
This project supports multiple models for time series and dynamic systems.

### Time Series Models: These are standard models used for univariate or multivariate time series data.
- AR (Autoregressive)
- MA (Moving Average)
- ARIMA (Autoregressive Integrated Moving Average)
- GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
- HMM (Hidden Markov Models)

### Differential Equation Models: These models involve continuous dynamics described by differential equations, commonly used for modeling physical, biological, or economic systems.
- **Lotka-Volterra**: Predator-prey model
- **SIR**: Epidemiological model
- **Reaction-Diffusion**: Used for chemical processes or biological pattern formation

### Difference Equation Models: First-order and higher-order difference equations, logistic growth, ARMA.
- First-order
- higher-order difference equations
- logistic growth
- ARMA

### Hybrid and Advanced Models: These models combine elements of multiple frameworks or represent complex systems.
- Agent-based models
- PDEs
- neural networks

#### State Space Models: State space models are used for both time series and control systems where the system evolves over time, and parameters can change.
- Kalman Filters
- Nonlinear State Space Models

## Model Input Requirements

This table summarizes the input requirements for different types of models, specifying whether the model requires `tspan` (for continuous models) or `num_steps` (for discrete models or agent-based models).

| **Main Category**                | **Subcategory**                | **Examples**                                 | **Input Requirement** | **Explanation**                                                                                                                                              |
|-----------------------------------|--------------------------------|---------------------------------------------|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Time Series Models**            | Univariate Time Series         | AR, MA, ARIMA                               | `num_steps`           | Time series models use discrete time steps, requiring `num_steps` to specify how many iterations the model will simulate (e.g., forecasting future values).  |
|                                   | Multivariate Time Series       | VAR, VECM, multivariate ARIMA               | `num_steps`           | Like univariate, but for multiple time series. Requires `num_steps` to specify the number of steps for the multivariate process.                              |
| **Differential Equation Models**  | Ordinary Differential Equations (ODEs) | Lotka-Volterra, SIR, Simple Harmonic Motion | `tspan`               | ODE models describe continuous processes and require `tspan` (start and end time) to simulate continuous evolution over a given time interval.                |
|                                   | Partial Differential Equations (PDEs) | Heat Equation, Wave Equation                | `tspan`               | PDEs involve continuous variables and need a time span (`tspan`) to simulate continuous space and time evolution.                                             |
| **Difference Equation Models**    | First-Order Difference Equations | AR, MA, Logistic Growth                     | `num_steps`           | Discrete-time systems, requiring `num_steps` to specify how many steps the model will run (e.g., iterating over time).                                      |
|                                   | Higher-Order Difference Equations | ARMA, Fibonacci Sequence                    | `num_steps`           | Like first-order, but with more complex dependencies. Requires `num_steps` to define the number of iterations (steps).                                       |
| **State Space Models**            | Linear State Space Models      | Kalman Filter, Linear Discrete Systems      | `tspan` or `num_steps`| Linear state space models can be continuous (requiring `tspan`) or discrete (requiring `num_steps`). The input depends on whether the system is continuous or discrete. |
|                                   | Nonlinear State Space Models   | Extended Kalman Filter, Particle Filters    | `tspan` or `num_steps`| Nonlinear state space models can also be continuous or discrete, with inputs depending on whether the system is modeled with continuous dynamics or discrete steps. |
| **Agent-Based Models**            | Spatial Agent-Based Models     | Ecological Systems, Social Dynamics         | `num_steps`           | Agent-based models require `num_steps` to specify how many iterations the simulation will run (representing agent interactions over time).                   |
|                                   | Non-Spatial Agent-Based Models | Traffic Simulation, Consumer Behavior       | `num_steps`           | Similar to spatial ABMs, but focusing on interactions between agents without spatial constraints. Requires `num_steps` for the simulation length.           |
| **Hybrid/Advanced Models**        | Neural Network Models          | Deep Learning, RNNs, LSTMs                  | `num_steps` or `tspan`| Hybrid models like neural networks can evolve in discrete time steps (requiring `num_steps`) or be modeled continuously (requiring `tspan`), depending on how the model is trained and simulated. |
|                                   | Hybrid Continuous/Discrete Models | Agent-based models with continuous dynamics | `num_steps` or `tspan`| These hybrid models may have both continuous dynamics (requiring `tspan`) and discrete interactions (requiring `num_steps`). The input depends on the model structure. |



## Simulation
To simulate the models, follow these steps:
1. Choose a model.
2. Input your initial parameters (e.g., initial population, rate constants).
3. Run the model with a numerical solver (e.g., Runge-Kutta for ODEs).

## Installation

This project requires Julia version 1.0.5 or above. To install this package, follow these steps:

1. Open Julia and enter the **package mode** by typing `]` in the Julia REPL.

2. Run the following command to install the package:

```julia
(v1.0.5) pkg> add TSCPDetector
```
## Usage

### COVID-19 Simulation Using ODEs

This example demonstrates how to use an **Ordinary Differential Equation (ODE)** model to simulate the spread of COVID-19. The model considers parameters like infection rate and recovery rate. The goal is to detect change points where government interventions or human behavior changes occur, affecting the spread of the virus.

```julia
julia>  using TSCPDetector
julia> penlty = 0.0
n_global = 8
n_segment_specific = 8
parnames = (:ω , :δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃,
                :ᴺp₁, :ᴺβ, :ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν) 
initial_chromosome = [0.1, 1/7, 1/11.4, 1/14, 1/13.4, 1/9,  1/16, 0.0055,  0.2, 0.05, 0.17, 0.144, 0.01,   0.017, 0.173,   0.01]
lower_bound =              [0.1, 1/10, 1/11.7, 1/24, 1/15.8, 1/19, 1/27, 0.003,   0.0, 0.0, 0.001, 0.001, 0.001, 0.001, 0.001,   10e-5]
upper_bound =              [0.3, 1/3,  1/11.2, 1/5,  1/10.9, 1/5,  1/8,  0.012,   0.8, 8.0, 0.5,   0.5,   0.5,   0.5,   0.5,     0.1]

ga = GA(populationSize = 100, selection = tournament(2), crossover = SBX(0.7, 1), mutationRate=0.7,
  crossoverRate=0.7, mutation = gaussian(0.0001))

min_length  = 10
step        = 10
const Nₚ = 83129285
begin
    Sinit = (Nₚ - 1) 
    E0init = 1
    E1init = 0 
    I0init = 0 
    I1init = 0 
    I2init = 0
    I3init = 0
    Rinit = 0 
    Dinit = 0
    Casesinit = 0 
    Vinit = 0
    end
    N_COMPARTMENTS = 11
    u0 = zeros(N_COMPARTMENTS)
    # intially infected with first variant
    
    begin
    u0[1] = Sinit
    u0[2] = E0init
    u0[3] = E1init
    u0[4] = I0init
    u0[5] = I1init
    u0[6] = I2init
    u0[7] = I3init
    u0[8] = Rinit
    u0[9] = Dinit
    u0[10] = Casesinit
    u0[11] = Vinit

    end
tspan = (0.0, 399.0)
initial_conditions = u0
compare_vars = [5, 6, 7, 9, 11]
@time cps = Bi_S(objective_function, length(data_CP[1]), n_global, n_segment_specific, extract_parameters, parnames,
simulate_model, 
example_ode_model, 
loss_function, compare_vars,
data_CP,
initial_conditions, tspan, 
initial_chromosome, lower_bound, upper_bound, ga, update_chromosome_bounds!,
min_length, step, penlty)
```

![My Algorithm Icon](Covsim.png)

### Wind Turbine Cooling System Using Difference Equations

This example demonstrates how to model the temperature dynamics of a **wind turbine generator** cooling system using **Difference Equations**. The primary goal is to detect **change points** that indicate significant shifts in the system, such as modifications to the cooling system, operational changes, or external environmental conditions affecting the generator's temperature. The temperature is modeled as a difference equation, and the change points are detected by monitoring changes in the model parameters, which represent operational shifts in the turbine's cooling system.

```julia
julia>  using TSCPDetector
julia>  penlty = 0.0
n_global = 3
n_segment_specific = 4
parnames = (:θ1, :θ2, :θ3, :θ4, :θ5, :θ6, :θ7) 


initial_chromosome = [1.1, 1.1, 1.1, 1.5, 1.5, 1.5, 1.5]
lower_bound =        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
upper_bound =        [100.0, 100.0, 100.0, 466.0, 466.0, 466.0, 1466.0]

ga = GA(populationSize = 100, selection = tournament(2), crossover = SBX(0.7, 1), mutationRate=0.7,
crossoverRate=0.7, mutation = gaussian(0.0001))

min_length  = 10
step        = 10

num_steps = length(wind_speeds)
initial_conditions = generator_temperatures[1]
data_CP = generator_temperatures
#compare_vars = [5, 6, 7, 9, 11]
extra_data = [wind_speeds, ambient_temperatures, ]

julia>  cps = Bi_S(objective_function, length(generator_temperatures),  n_global, 
n_segment_specific, 
extract_parameters, parnames,
simulate_model, 
generator_temperature_model,
loss_function, 
data_CP,
initial_conditions,
initial_chromosome, lower_bound, upper_bound, ga, update_chromosome_bounds!,
min_length, step;
pen = penlty,
extra_data = extra_data,
num_steps = num_steps)
```

![My Algorithm Icon](Turbine_CPD_2sub.png)

## Package_development
This package was originally developed by Mehdi Lotfi (@mehdilotfi7) in 2024. It is currently being maintained and extended by Mehdi Lotfi.
