function segment_cost_pelt(
    τ::Int, t::Int,
    chromosome::Vector{Float64},
    parnames,
    n_global::Int,
    n_segment_specific::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64}
)
    # Extract parameters
    constant_pars, segment_pars_list = extract_parameters(chromosome, n_global, n_segment_specific)
    idx_start = τ + 1
    idx_end = t
    segment_data = data[:, idx_start:idx_end]

    seg_pars = segment_pars_list[1]  # Assume single segment setup
    all_pars = @LArray [constant_pars; seg_pars] parnames

    model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, get_initial_condition(model_manager))
    sim_data = simulate_model(model_spec)

    return loss_function(segment_data, sim_data)
end

segment_cost_pelt(
    0, 10,
    initial_chromosome,
    parnames,
    n_global,
    n_segment_specific,
    model_manager,
    loss_function,
    data_M
)

function optimize_on_segment(
    segment_cost_pelt::Function,
    τ::Int, t::Int,
    chromosome::Vector{Float64},
    parnames,
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    ga,
    n_global::Int, n_segment_specific::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64};
    options = Evolutionary.Options(show_trace = false)
)
    wrapped_obj(chrom) = segment_cost_pelt(
        τ, t, chrom, parnames, n_global, n_segment_specific,
        model_manager, loss_function, data
    )
    Random.seed!(1234)
    result = Evolutionary.optimize(wrapped_obj, BoxConstraints(bounds...), chromosome, ga, options)
    return Evolutionary.minimum(result), Evolutionary.minimizer(result)
end


optimize_on_segment(
    segment_cost_pelt,
    0, 10,
    initial_chromosome,
    parnames,
    bounds,
    ga,
    n_global, n_segment_specific,
    model_manager,
    loss_function,
    data_M
)



function PELT_model_based(
    segment_cost_pelt::Function,
    optimize_on_segment::Function,
    chromosome::Vector{Float64},
    parnames,
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    ga,
    n_global::Int,
    n_segment_specific::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64},
    n::Int;
    pen::Float64 = log(n),
    min_length::Int = 10,
    step::Int = 10
)
    F = fill(Inf, n + 1)
    F[1] = -pen
    chpts = Array{Int}(undef, n)
    R = [0]

    for t in min_length:step:n
        cpt_cands = R
        seg_costs = Float64[]

        valid_cands = filter(τ -> (t - τ) ≥ min_length, cpt_cands)

        for τ in valid_cands
            cost, _ = optimize_on_segment(
                segment_cost_pelt,
                τ, t,
                chromosome,
                parnames,
                bounds,
                ga,
                n_global,
                n_segment_specific,
                model_manager,
                loss_function,
                data
            )
            push!(seg_costs, cost)
        end

        if isempty(valid_cands)
            continue
        end

        costs = F[valid_cands .+ 1] .+ seg_costs .+ pen
        F[t + 1], tau_idx = findmin(costs)
        chpts[t] = valid_cands[tau_idx]

        ineq_prune = (F[valid_cands .+ 1] .+ seg_costs) .< F[t + 1]
        R = push!(valid_cands[ineq_prune], t - step)
        @show R
    end
    @show chpts

    CP = Int[]
    last = chpts[n]
    while last > 0
        push!(CP, last)
        last = chpts[last]
    end
    sort!(CP)

    return CP, F[n + 1]
end


PELT_model_based(
    segment_cost_pelt,
    optimize_on_segment,
    initial_chromosome,
    parnames,
    bounds,
    ga,
    n_global,
    n_segment_specific,
    model_manager,
    loss_function,
    data_M,
    n
)

function PELT_model_based(
    segment_cost_pelt::Function,
    optimize_on_segment::Function,
    chromosome::Vector{Float64},
    parnames,
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    ga,
    n_global::Int,
    n_segment_specific::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64},
    n::Int;
    pen::Float64 = log(n),
    min_length::Int = 10,
    step::Int = 10
)
    # Initialize cost function values
    F = fill(Inf, n + 1)
    F[1] = -pen

    # Initialize change point tracker
    chpts = fill(-1, n + 1)

    # Initial candidate change points
    R = [0]

    for t in min_length:step:n
        cpt_cands = R
        seg_costs = Float64[]

        # Filter to enforce minimum segment length
        valid_cands = filter(τ -> (t - τ) ≥ min_length, cpt_cands)

        for τ in valid_cands
            cost, _ = optimize_on_segment(
                segment_cost_pelt,
                τ, t,
                chromosome,
                parnames,
                bounds,
                ga,
                n_global,
                n_segment_specific,
                model_manager,
                loss_function,
                data
            )
            @show τ, t, cost
            push!(seg_costs, cost)
        end

        if isempty(valid_cands)
            continue  # No valid segments for this t
        end

        # Compute full cost and update
        costs = F[valid_cands .+ 1] .+ seg_costs .+ pen
        F[t + 1], tau_idx = findmin(costs)
        chpts[t] = valid_cands[tau_idx]
        @show t, chpts[t], F[t + 1]
        # Prune candidate set
        ineq_prune = (F[valid_cands .+ 1] .+ seg_costs) .< F[t + 1]
        R = copy(valid_cands[ineq_prune])
        push!(R, t - step)
        @show R
    end

    # Trace back change points
    CP = Int[]
    last = chpts[n]
    while last > 0
        push!(CP, last)
        last = chpts[last]
    end
    sort!(CP)

    return CP, F[n + 1]
end



function PELT_model_based(
    segment_cost_pelt::Function,
    optimize_on_segment::Function,
    chromosome::Vector{Float64},
    parnames,
    bounds::Tuple{Vector{Float64}, Vector{Float64}},
    ga,
    n_global::Int,
    n_segment_specific::Int,
    model_manager::ModelManager,
    loss_function::Function,
    data::Matrix{Float64},
    n::Int;
    pen::Float64 = log(n),
    min_length::Int = 10,
    step::Int = 10
)

    F = fill(Inf, n + 1)
    F[1] = -pen
    chpts = fill(-1, n + 1)
    R = [0]  # initial candidate changepoints

    for t in min_length:step:n
        cpt_cands = R
        seg_costs = Float64[]
        valid_cands = filter(τ -> (t - τ) ≥ min_length, cpt_cands)

        if isempty(valid_cands)
            continue
        end

        for τ in valid_cands
            cost, _ = optimize_on_segment(
                segment_cost_pelt,
                τ, t,
                chromosome,
                parnames,
                bounds,
                ga,
                n_global,
                n_segment_specific,
                model_manager,
                loss_function,
                data
            )
            push!(seg_costs, cost)

            @show τ, t, cost
        end

        costs = F[valid_cands .+ 1] .+ seg_costs .+ pen
        F[t + 1], tau_idx = findmin(costs)
        chpts[t] = valid_cands[tau_idx]

        @show t, chpts[t], F[t + 1]

        # Pruning step: allow slight tolerance for numerical stability
        ϵ = 1e-6
        ineq_prune = (F[valid_cands .+ 1] .+ seg_costs) .< F[t + 1] + ϵ
        new_R = valid_cands[ineq_prune]

        if (t - step) ∉ new_R
            push!(new_R, t - step)
        end

        R = new_R
        @show R
    end

    # Trace back changepoints
    CP = Int[]
    last = chpts[n]
    while last > 0
        push!(CP, last)
        last = chpts[last]
    end
    sort!(CP)

    return CP, F[n + 1]
end
