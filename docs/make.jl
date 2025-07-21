using Pkg

# Activate the docs environment and ensure Mocha is available
Pkg.activate(@__DIR__)
#Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Mocha
using Documenter

makedocs(;
  modules = [Mocha],
  doctest = true,
  linkcheck = false,
  authors = "Mehdi Lotfi <Mehdilotfi.tabrizu@gmail.com>",
  repo = "https://github.com/Mehdilotfi7/Mocha.jl/blob/{commit}{path}#{line}",
  sitename = "Mocha.jl",
  format = Documenter.HTML(;
    prettyurls = false,
    canonical = "https://Mehdilotfi7.github.io/Mocha.jl",
    assets = ["assets/style.css"],
  ),
  pages = [
    "Home" => "index.md",
    "Getting started" => "getting_started.md",
    "Problem types" => "problems.md",
    "Tutorials" => "tutorial.md",
    "Examples" => "examples.md",
    "Custom Loss Functions for Segment Evaluation" => "loss_function.md",
    "Reference" => "references.md"
  ],
  checkdocs = :warn
)

deploydocs(
  repo = "github.com/Mehdilotfi7/Mocha",
  devbranch = "main",
  push_preview = false,
  versions = ["stable" => "v1.0.0", "v1.0.0", "dev" => "dev"]
)
