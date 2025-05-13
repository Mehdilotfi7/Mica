module _ChangePointDetection

using Evolutionary
using _ObjectiveFunction
using _Visualization

"""
# ChangePointDetection Module

This module implements a framework for detecting change points in time-series or similar data
using optimization techniques. It relies on a generic `ModelManager` interface and uses
Evolutionary.jl for optimization.

### Main Functions:
- `detect_changepoints`: Detect change points in data.
- `optimize_with_changepoints`: Optimize the model parameters with fixed change points.
- `evaluate_segment`: Evaluate all possible change points in a given segment.
- `update_bounds!`: Dynamically update bounds and chromosome.
"""

# ----------------------------------------------------------------------
# Function: optimize_with_changepoints
# ----------------------------------------------------------------------
"""
    optimize_with_changepoints(
        objective_function, chromosome, CP, bounds, ga,
        n_global, n_segment_specific, parnames,
        model_manager, loss_function, data; options=...
    )

Optimize model parameters for a fixed set of change points using Evolutionary.jl.

Returns the minimum loss and the best parameter set.
"""
function optimize_with_changepoints(
    objective_function, chromosome, CP, bounds, ga,
    n_global, n_segment_specific, parnames::Vector{Symbol},
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64};
    options=Evolutionary.Options(show_trace=false)
)
    wrapped_obj = chrom -> objective_function(
        chrom, CP, n_global, n_segment_specific, parnames,
        model_manager, loss_function, data
    )
    result = Evolutionary.optimize(wrapped_obj, BoxConstraints(bounds...), chromosome, ga, options)
    return Evolutionary.minimum(result), Evolutionary.minimizer(result)
end

# ----------------------------------------------------------------------
# Function: update_bounds!
# ----------------------------------------------------------------------
"""
    update_bounds!(chromosome, bounds, n_global, n_segment_specific, extract_parameters)

Update bounds and chromosome by appending segment-specific parameters.
"""
function update_bounds!(chromosome, bounds, n_global, n_segment_specific, extract_parameters)
    _, seg_specific = extract_parameters(chromosome, n_global, n_segment_specific)
    _, seg_lower = extract_parameters(bounds[1], n_global, n_segment_specific)
    _, seg_upper = extract_parameters(bounds[2], n_global, n_segment_specific)

    append!(chromosome, seg_specific[1])
    append!(bounds[1], seg_lower[1])
    append!(bounds[2], seg_upper[1])
end

# ----------------------------------------------------------------------
# Function: evaluate_segment
# ----------------------------------------------------------------------
"""
    evaluate_segment(...)

Evaluate candidate change points in a segment [a, b] by inserting change points
and computing the new loss for each.

Returns a tuple of loss values and corresponding best chromosomes.
"""
function evaluate_segment(
    objective_function, a::Int, b::Int, CP::Vector{Int}, bounds,
    chromosome::Vector{Float64}, ga, pen::Float64, min_length::Int, step::Int,
    n_global::Int, n_segment_specific::Int, parnames::Vector{Symbol},
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64}
)
    x = Float64[]
    y = Vector{Vector{Float64}}()
    for j in (a + min_length):step:(b - min_length)
        new_cp = sort([CP; j])
        loss, best = optimize_with_changepoints(
            objective_function, chromosome, new_cp, bounds, ga,
            n_global, n_segment_specific, parnames,
            model_manager, loss_function, data
        )
        push!(x, loss + pen)
        push!(y, best)
    end
    return x, y
end

# ----------------------------------------------------------------------
# Main Function: detect_changepoints
# ----------------------------------------------------------------------
"""
    detect_changepoints(
        objective_function, n, n_global, n_segment_specific, parnames,
        model_manager, loss_function, data,
        initial_chromosome, bounds, ga, min_length, step; pen=log(n)
    )

Detect optimal change points using a greedy search strategy and regularized loss.

Returns:
- Vector of change point indices (CP)
- Best parameter vector found
"""
function detect_changepoints(
    objective_function,
    n::Int, n_global::Int, n_segment_specific::Int,
    parnames::Vector{Symbol},
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64},
    initial_chromosome::Vector{Float64},
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    ga::Evolutionary.AbstractGA,
    min_length::Int, step::Int;
    pen::Float64=log(n)
)
    tau = [(0, n)]
    CP = Int[]
    @show CP

    loss_val, best_params = optimize_with_changepoints(
        objective_function, initial_chromosome, CP, bounds, ga,
        n_global, n_segment_specific, parnames,
        model_manager, loss_function, data
    )
    @show best_params

    update_bounds!(initial_chromosome, bounds, n_global, n_segment_specific, _ObjectiveFunction.extract_parameters)

    while !isempty(tau)
        a, b = pop!(tau)
        x, y = evaluate_segment(
            objective_function, a, b, CP, bounds, initial_chromosome, ga, pen, min_length, step,
            n_global, n_segment_specific, parnames,
            model_manager, loss_function, data
        )
        if !isempty(x)
            minval, idx = findmin(x)
            if minval < loss_val
                chpt = a + (idx * step)
                push!(CP, chpt)
                CP = sort(CP)
                loss_val = minval
                best_params = y[idx]
                update_bounds!(initial_chromosome, bounds, n_global, n_segment_specific, _ObjectiveFunction.extract_parameters)
                if chpt != a + min_length
                    push!(tau, (a, chpt))
                end
                if chpt != b - min_length
                    push!(tau, (chpt, b))
                end
            end
        end
    end

    return CP, best_params
end

end
