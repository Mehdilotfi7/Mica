# Mocha - Model-based Changepoints Analysis
![TSCPDetector.jl](fig/mocha3.png)

Welcome to the documentation for Mocha.jl.

Mocha.jl provides a robust and model-driven approach to changepoint detection in time series data.

Key Features
- Unlike traditional methods that detect changepoints based solely on shifts in data characteristics (e.g., mean or variance), Mocha.jl incorporates both the data and an explicit model of the system.
- Rather than analyzing changes in statistical properties of the dataset, our algorithm detects changepoints by tracking shifts in the model parameters over time.

## Using

Like other Julia packages, you can install Mocha.jl using the Julia package manager:

```julia-repl
julia> using Pkg
julia> Pkg.add("Mocha")
```

After installing the package, load it into your working environment with:

```julia-repl
julia> using Mocha
```



