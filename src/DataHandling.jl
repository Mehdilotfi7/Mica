module DataHandling

using Interpolations
using DataFrames

"""
    load_data(df::DataFrame, method::Symbol=:linear) -> DataFrame

Load a dataset and handle missing values using interpolation.

# Arguments
- `df::DataFrame`: The dataset in a DataFrame format.
- `method::Symbol`: The interpolation method. Must be one of the methods supported by `Interpolations.jl`.
  - Supported methods:
    - `:linear` - Linear interpolation.
    - `:quadratic` - Quadratic interpolation.
    - `:cubic` - Cubic interpolation.
    - `:lagrange` - Lagrange interpolation.

# Description
This function checks for missing values in the dataset and interpolates them using the specified method.

# Example
```julia
using DataFrames, DataHandling

df = DataFrame(x=[1, 2, 3, 4, missing, 6], y=[10, missing, 30, 40, 50, 60])
df_clean = load_data(df, :linear)
Notes

Ensure that the method you select is appropriate for your data's characteristics.
The function assumes the data is sorted and continuous in the domain where interpolation is applied.
"""
function load_data(df::DataFrame, method::Union{Symbol, Function}=:linear) :: DataFrame
  if any(col -> any(ismissing, col), eachcol(df)) || any(col -> any(cell -> cell === NA, col), eachcol(df))
           println("Missing or NA values detected. Applying interpolation using the $method method.")
           interpolated_df = DataFrame()
     for col in eachcol(df)
         if eltype(col) <: Union{Missing, Number}  # Check if the column has missing numerical data
             x = findall(!ismissing, col)             # Indices of non-missing values
             y = col[x]                               # Non-missing values
        
             # Determine the interpolation method
             #ToDo: add a condition to accept custom function  
                 itp = if isa(method, Function)
                    # Use the custom user-provided function
                    method(y, x)
                 elseif method == :linear
                    interpolate(y, BSpline(Linear()))
                 elseif method == :quadratic
                  interpolate(y, BSpline(Quadratic()))
                 elseif method == :cubic
                  interpolate(y, BSpline(Cubic()))
                 elseif method == :lagrange
                  interpolate(y, Lagrange())
                 else
                  error("Unsupported interpolation method: $method")
                end
        
             extrap = extrapolate(itp)
             interpolated_col = [ismissing(v) || v === NA ? extrap(i) : v for (i, v) in enumerate(col)]
             interpolated_df[!, nameof(col)] = interpolated_col
           else
            interpolated_df[!, nameof(col)] = col
           end
       end
  return interpolated_df
  else
  println("No missing values detected. Returning the original dataset.")
  return df
  end
end # End of function
end # End of module
