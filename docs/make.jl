using Documenter, Subspaces

DocMeta.setdocmeta!(Subspaces, :DocTestSetup, :( using Subspaces; using LinearAlgebra ); recursive=true)

makedocs(
    sitename="Subspaces.jl",
    modules=[Subspaces],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Home" => "index.md",
        "Usage" => "usage.md",
        "Reference" => "reference.md",
    ],
)
