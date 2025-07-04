module DataHandler

using DataFrames
using Interpolations
using Tables

export preprocess_data, DataInfo, interpolate_missing

"""
    struct DataInfo

Stores metadata about the input dataset.
- `num_vars`: Number of variables (columns)
- `num_points`: Number of time points (rows)
- `is_data_vector`: Whether the original input was a single `Vector`
"""
struct DataInfo
    num_vars::Int
    num_points::Int
    is_data_vector::Bool
end

"""
    preprocess_data(data; interpolate_missing_values=true, method=:linear) -> (data_matrix, info)

Processes input data into a matrix format and returns a `DataInfo` object with metadata.

Accepted input types:
- `Vector{T}` → interpreted as a single time series
- `Vector{Vector{T}}` → converted to matrix if all vectors are the same length
- `Matrix`
- `DataFrame`

# Arguments
- `data`: Input dataset. Can be one of:
  - `Vector{<:Number}`
  - `Vector{<:Vector}`
  - `Matrix`
  - `DataFrame`
- `interpolate_missing_values::Bool`: Whether to interpolate missing values (default = `true`)
- `method::Symbol | Function`: Interpolation method for missing data. Supported symbols:
  - `:linear`, `:quadratic`, `:cubic`, `:lagrange`

# Returns
- `data_matrix::Matrix`: Converted matrix of cleaned data
- `info::DataInfo`: Struct containing number of variables, points, and whether input was a vector

# Example
```julia
using DataFrames, DataHandler

df = DataFrame(x=[1, 2, 3, 4, missing, 6], 
               y=[10, missing, 30, 40, 50, 60])

data_matrix, info = preprocess_data(df; method=:linear)

@show size(data_matrix)
@show info
"""
function preprocess_data(data)
    df = nothing
    is_vec = false

    if data isa Vector{<:Number}
        df = DataFrame(var1 = data)
        is_vec = true
    elseif data isa Vector{<:AbstractVector}
        lengths = map(length, data)
        if length(unique(lengths)) != 1
            error("All inner vectors must have the same length.")
        end
        df = DataFrame([getindex.(data, i) for i in 1:length(data[1])], :auto, copycols=false)
    elseif data isa Matrix
        df = DataFrame(data, :auto)
    elseif data isa DataFrame
        df = copy(data)
    else
        error("Unsupported data type. Must be Vector, Vector of Vectors, Matrix, or DataFrame.")
    end

    #if interpolate_missing_values
    #    df = interpolate_missing(df, method)
    #end

    mat = Tables.matrix(df)
    info = DataInfo(size(mat, 2), size(mat, 1), is_vec)
    return mat, info
end

"""
    interpolate_missing(df::DataFrame, method::Symbol=:linear) -> DataFrame

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
df_clean = interpolate_missing(df, :linear)
Notes

Ensure that the method you select is appropriate for your data's characteristics.
The function assumes the data is sorted and continuous in the domain where interpolation is applied.
"""
# for now i skip this function

function interpolate_missing(df::DataFrame, method::Union{Symbol, Function}=:linear) :: DataFrame
    if any(col -> any(ismissing, col), eachcol(df)) || any(col -> any(cell -> cell === NA, col), eachcol(df))
        println("Missing or NA values detected. Applying interpolation using the $method method.")
        interpolated_df = DataFrame()
        for col in eachcol(df)
            if eltype(col) <: Union{Missing, Number}
                x = findall(!ismissing, col)
                y = col[x]

                itp = if isa(method, Function)
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
                @show itp

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
end

end # module
