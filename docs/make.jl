using TSCPDetector
using Documenter

#DocMeta.setdocmeta!(TSCPDetector, :DocTestSetup, :(using TSCPDetector); recursive=true)

makedocs(;
    modules=[TSCPDetector],
    authors="Mehdi Lotfi",
    sitename="TSCPDetector.jl",
    format=Documenter.HTML(),
    remotes = Dict(),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "API" => "api.md",
    ],
)

#deploydocs(;
#    repo=nothing,
#    devbranch="main",
#)
