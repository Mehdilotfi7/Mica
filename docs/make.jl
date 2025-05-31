using Documenter, TSCPDetector


makedocs(;
  modules = [TSCPDetector],
  doctest = true,
  linkcheck = false, # Rely on Lint.yml/lychee for the links
  authors = "Mehdi Lotfi <mehdilotfi.tabrizu@gmail.com>",
  repo = "https://github.com/mehdilotfi7/TSCPDetector.jl/blob/{commit}{path}#{line}",
  sitename = "TSCPDetector.jl",
  format = Documenter.HTML(;
    prettyurls = false,
    canonical = "https://mehdilotfi7.github.io/TSCPDetector.jl",
    assets = ["assets/style.css"],
  ),
  pages = [
    "Home" => "index.md",
    "Getting started" => "getting_started.md",
    "Problem types" => "problems.md",
    "Solver algorithms for ODE models" => "solver_hints_ODE.md",
    "Genetic algorithms hints" => "GA_hints.md",
    "Reference" => "references.md"

  ],
  checkdocs=:warn
)

deploydocs(; repo = "github.com/mehdilotfi7/TSCPDetector.jl", push_preview = false)


# Getting started

# Problem types

# Examples 

# Solver algorithms for ODE models

# Genetic algorithms hints

# Future features 