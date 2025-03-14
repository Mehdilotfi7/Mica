using FirstPkg
using Documenter

DocMeta.setdocmeta!(FirstPkg, :DocTestSetup, :(using FirstPkg); recursive=true)

makedocs(;
    modules=[FirstPkg],
    authors="Mehdi Lotfi",
    sitename="FirstPkg.jl",
    format=Documenter.HTML(;
        canonical="https://Mehdilotfi7.github.io/FirstPkg.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Mehdilotfi7/FirstPkg.jl",
    devbranch="main",
)
