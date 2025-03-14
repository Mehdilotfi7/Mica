module FirstPkg

export greet_your_package_name

include("functions.jl")
using .Functions  # Import the submodule

end
