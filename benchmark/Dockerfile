FROM julia:1.10

RUN julia -e 'using Pkg; Pkg.add(["CSV", "DataFrames", "Distributed", "DifferentialEquations", "LabelledArrays", "Statistics", "BenchmarkTools", "Random", "DelimitedFiles"])'

WORKDIR /benchmark

COPY . .

CMD ["julia", "--project=.", "benchmark.jl"]
