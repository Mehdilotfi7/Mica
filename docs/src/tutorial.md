# Tutorials


## [Model Type Definitions] (@id ModType)

Mocha.jl supports a variety of time series model types, which are organized under a common abstract interface.

### Abstract Model Specification

All model types inherit from the abstract type 'AbstractModelSpec'

```@docs; canonical=false
AbstractModelSpec
```

### ODE-Based Models
For modeling systems with continuous dynamics using ordinary differential equations, use:

```@docs; canonical=false
ODEModelSpec
```
### Difference Equation Models
For discrete-time models based on difference equations, use:

```docs; canonical=false
DifferenceModelSpec
```
### Regression Models
For standard time series regression-based models, use:

and 'RegressionModelSpec' 
```docs; canonical=false
DifferenceModelSpec
```

Use the function [evaluate_segment] (@ref) for referencing and [title] (@ref ModType)