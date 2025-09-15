# Mica.jl

<p align="center">
<img src="images/mocha3.png" width="200" />
</p>

<p align="center">

   ![Build Status](https://travis-ci.org/Mehdilotfi7/Mica.svg?branch=main)
   [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Mehdilotfi7.github.io/Mica.jl/stable/)
   [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Mehdilotfi7.github.io/Mica.jl/dev/)
   [![Coverage](https://codecov.io/gh/Mehdilotfi7/Mica.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Mehdilotfi7/Mica.jl)

</p>

<p align="center">
![Build Status](https://travis-ci.org/Mehdilotfi7/Mica.svg?branch=main)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Mehdilotfi7.github.io/Mica.jl/stable/)
[![Coverage](https://codecov.io/gh/Mehdilotfi7/Mica.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Mehdilotfi7/Mica.jl)
</p>

---

## Overview

**Mica.jl** is a Julia package for simulating time-dependent models (ODEs, difference equations) and detecting **change points** in their dynamics using evolutionary optimization.

Supports:
- Differential and difference equation models
- Segment-wise simulation and optimization
- Automatic model management and loss evaluation
- BIC/AIC-style regularization

 **[📘 Full Documentation](https://changepointdetection.com/)**

---

## Installation

```julia
using Pkg
Pkg.add("Mica")
```

## Acknowledgments

The segmentation module in **Mica.jl** was inspired by the excellent 
[Changepoints.jl](https://github.com/STOR-i/Changepoints.jl) package. 
Many thanks to its authors for their contribution to the Julia ecosystem.


## Package_development
This package was originally developed by Mehdi Lotfi (@mehdilotfi7) in 2024. It is currently being maintained and extended by Mehdi Lotfi.
