using Pkg
Pkg.activate(".")
Pkg.add([
    "CSV", "DataFrames", "Distributed", "DifferentialEquations",
    "LabelledArrays", "Statistics", "BenchmarkTools", "Random", "DelimitedFiles"
])
Pkg.instantiate()
