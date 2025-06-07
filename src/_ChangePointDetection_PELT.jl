module ChangePointDetection

using Evolutionary, Random

"""
# PELT-Based Change Point Detection

This module implements PELT (Pruned Exact Linear Time) for detecting change points 
in time-series data using an ODE-based model. Unlike greedy search, PELT efficiently 
optimizes change point selection while balancing accuracy and computational speed.

### Main Functions:
- `detect_changepoints_PELT`: Detect change points in data using PELT.
- `evaluate_segment_PELT`: Evaluate candidate change points adaptively.
- `prune_candidates!`: Remove weak candidates dynamically.

"""

# ----------------------------------------------------------------------
# Function: optimize_with_changepoints
# ----------------------------------------------------------------------
function optimize_with_changepoints(
    objective_function, chromosome, parnames, CP, bounds, ga,
    n_global, n_segment_specific,
    model_manager, loss_function, data;
    options=Evolutionary.Options(show_trace=false)
)
    wrapped_obj(chrom) = objective_function(
        chrom, CP, parnames, n_global, n_segment_specific,
        model_manager, loss_function, data
    )
    Random.seed!(1234)
    result = Evolutionary.optimize(wrapped_obj, BoxConstraints(bounds...), chromosome, ga, options)
    return Evolutionary.minimum(result), Evolutionary.minimizer(result)
end

# ----------------------------------------------------------------------
# Function: prune_candidates!
# ----------------------------------------------------------------------
function prune_candidates!(R, F, valid_cpts, seg_costs, t)
    ineq_prune = (F[valid_cpts .+ 1] .+ seg_costs) .< F[t+1]
    R = valid_cpts[ineq_prune]
    return R
end

# ----------------------------------------------------------------------
# Function: evaluate_segment_PELT
# ----------------------------------------------------------------------
function evaluate_segment_PELT(
    objective_function, a::Int, b::Int, CP::Vector{Int}, bounds,
    chromosome::Vector{Float64}, parnames, ga, min_length::Int,
    n_global::Int, n_segment_specific::Int, n::Int,
    model_manager, loss_function, data, penalty_fn, residual_thresh::Float64
)
    x = Float64[]
    y = Vector{Vector{Float64}}()

    for j in (a + min_length):(b - min_length)
        new_cp = sort([CP; j])
        loss, best = optimize_with_changepoints(
            objective_function, chromosome, parnames, new_cp, bounds, ga,
            n_global, n_segment_specific, model_manager, loss_function, data
        )
        segment_lengths = diff([0; new_cp; n])
        pen = call_penalty_fn(penalty_fn, p=n_segment_specific, n=n, CP=new_cp,
                              segment_lengths=segment_lengths, num_segments=length(new_cp) + 1)

        push!(x, loss + pen)
        push!(y, best)
    end

    return x, y
end

# ----------------------------------------------------------------------
# Main Function: detect_changepoints_PELT
# ----------------------------------------------------------------------
function detect_changepoints_PELT(
    objective_function, n::Int, n_global::Int, n_segment_specific::Int,
    model_manager, loss_function, data, initial_chromosome::Vector{Float64},
    parnames, bounds::Tuple{Vector{Float64}, Vector{Float64}},
    ga, min_length::Int, penalty_fn::Function = default_penalty, residual_thresh::Float64 = 0.1
)
    F = fill(Inf, n+1)
    F[1] = -log(n)
    F[2] = 0
    chpts = fill(0, n)
    chpts[1] = 0
    R = Int64[0]  # Candidate change points
    CP = Int[]

    loss_val, best_params = optimize_with_changepoints(
        objective_function, initial_chromosome, parnames, CP, bounds, ga,
        n_global, n_segment_specific, model_manager, loss_function, data
    )

    update_bounds!(initial_chromosome, bounds, n_global, n_segment_specific, extract_parameters)

    t = 2
    while t <= n
        valid_cpts = filter(x -> (t - x >= min_length), R)
        seg_costs = [evaluate_segment_PELT(objective_function, x, t, CP, bounds, initial_chromosome,
                                           parnames, ga, min_length, n_global, n_segment_specific, n,
                                           model_manager, loss_function, data, penalty_fn, residual_thresh)[1] for x in valid_cpts]

        F[t+1], tau = findmin(F[valid_cpts .+ 1] .+ seg_costs .+ log(n))
        chpts[t] = valid_cpts[tau]
        R = prune_candidates!(R, F, valid_cpts, seg_costs, t)

        step_size = (evaluate_segment_PELT(objective_function, t - min_length, t, CP, bounds,
                                           initial_chromosome, parnames, ga, min_length, n_global,
                                           n_segment_specific, n, model_manager, loss_function,
                                           data, penalty_fn, residual_thresh)[1] > residual_thresh) ? 1 : 10

        t += step_size
    end

    CP = Int[]
    last = chpts[n]
    while last > 0
        push!(CP, last)
        last = chpts[last]
    end
    sort!(CP)

    return CP, F[n+1]
end

end  # Module End
