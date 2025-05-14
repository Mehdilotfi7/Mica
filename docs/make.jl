using TSCPDetector
using Documenter

DocMeta.setdocmeta!(TSCPDetector, :DocTestSetup, :(using TSCPDetector); recursive=true)

makedocs(;
    modules=[TSCPDetector],
    authors="Mehdi Lotfi",
    sitename="TSCPDetector.jl",
    format=Documenter.HTML(;
        canonical="https://Mehdilotfi7.github.io/TSCPDetector.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Mehdilotfi7/TSCPDetector.jl",
    devbranch="main",
)
