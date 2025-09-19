# Mica.jl
*Model-Informed Change-point Analysis in Time Series Models* 

<p align="center">
<img src="images/mocha3.png" width="200" />
</p>

<p align="center">

[![Stable Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://changepointdetection.com/)
[![Dev Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://changepointdetection.com/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![JuliaHub](https://juliahub.com/docs/Mica/version.svg)](https://juliahub.com/ui/Packages/Mica)
[![PkgEval](https://juliahub.com/docs/Mica/pkgeval.svg)](https://juliahub.com/ui/Packages/Mica)


</p>

---

## Repository Structure

```text
Mica/
├── Project.toml
├── src/
│   ├── Mica.jl                 # Main entry point of the package
│   ├── ChangePointDetection.jl # Change-point detection framework
│   ├── ModelHandling.jl        # Model manager
│   ├── ModelSimulation.jl      # Wrappers for ODE/difference/regression models
│   ├── ObjectiveFunction.jl    # Objective function and parameter optimization
│   ├── Visualization.jl        # Plotting and visualization tools
│   ├── penalty.jl              # Penalty functions (e.g., BIC)
├── test/
│   └── runtests.jl             # Unit tests
├── examples/
│   ├── Covid-model/            # Example: epidemiological case study
│   ├── Wind_Turbine_model/     # Example: engineering case study
├── benchmark/                  # Benchmarking experiments
└── docs/                       # Documentation (in progress)
```
## Overview

**Mica.jl** provides a **model-driven approach** to changepoint detection in time series.  
Unlike statistical-only methods (which detect shifts in mean/variance), Mica integrates **system models** (ODEs, difference equations, regression models) to identify **structural changes in dynamics**.

Key features:
- Works with ODEs, difference equations, and regression-based models
- Detects *when* and *how* model parameters change
- Segment-wise simulation and optimization
- Supports information criteria (BIC, AIC) for regularization
- Built on [`DifferentialEquations.jl`](https://diffeq.sciml.ai/stable/) and [`Evolutionary.jl`](https://wildart.github.io/Evolutionary.jl/stable/)

 **[📘 Full Documentation](https://changepointdetection.com/)**

---

## Installation

```julia
using Pkg
Pkg.add("Mica")
```

## Presentations

- *TSCPDetector: A Comprehensive Approach to Change Point Detection in Time Series Models*  
  Mehdi Lotfi, Lars Kaderali – Statistical Computing 2024, Günzburg, Germany  

- Upcoming: **Mica.jl** at German Conference on Bioinformatics (GCB) 2025


## Acknowledgments

The segmentation module in **Mica.jl** was inspired by [Changepoints.jl](https://github.com/STOR-i/Changepoints.jl).  
This package is developed and maintained at the [**Kaderali Lab**](https://wordpress.kaderali.org),  
Institute of Bioinformatics, University Medicine Greifswald.  


## Package_development
This package was originally developed by Mehdi Lotfi (@mehdilotfi7) in 2025. It is currently being maintained and extended by Mehdi Lotfi.
