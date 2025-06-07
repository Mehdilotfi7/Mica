#!/bin/bash
#SBATCH --job-name=changepoint_benchmark
#SBATCH --output=benchmark_output.txt
#SBATCH --error=benchmark_error.txt
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

module load julia/1.10

julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. benchmark.jl



