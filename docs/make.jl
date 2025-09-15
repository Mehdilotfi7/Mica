using Pkg

# Activate the docs environment and ensure Mica is available
Pkg.activate(@__DIR__)
#Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")))
Pkg.instantiate()

using Mica
using Documenter

makedocs(;
  modules = [Mica],
  doctest = true,
  linkcheck = false,
  authors = "Mehdi Lotfi <Mehdilotfi.tabrizu@gmail.com>",
  repo = "https://github.com/Mehdilotfi7/Mica.jl/blob/{commit}{path}#{line}",
  sitename = "Mica.jl",
  format = Documenter.HTML(;
    prettyurls = false,
    canonical = "https://Mehdilotfi7.github.io/Mica.jl",
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
  repo = "github.com/Mehdilotfi7/Mica",
  devbranch = "main",
  push_preview = false,
  versions = ["stable" => "v1.0.0", "v1.0.0", "dev" => "dev"]
)
