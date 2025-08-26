# module Visualization

"""
# Visualization.jl

Utilities for simulating and visualizing change-point models.

This module provides two main functionalities:

1. **`simulate_full_model`**  
   Runs an model with both global and segment-specific parameters across multiple
   change-point intervals, concatenates the simulations, and optionally visualizes
   the trajectories alongside observed data and detected change points.

2. **`plot_parameter_changes`**  
   Produces heatmaps of parameter values relative to the first segment, enabling
   comparison of how model parameters evolve across detected change points.

# Features
- Seamless integration with change-point detection pipelines.
- Plotting of model trajectories by compartment.
- Optional overlay of observed data.
- Visualization of detected change points via vertical lines.
- Parameter heatmaps for relative changes across segments.
- Configurable colormaps and layout.
"""


"""
    simulate_full_model(chromosome, change_points, parnames,
                        n_global, n_segment_specific,
                        model_manager, data;
                        plot_results=false,
                        compartments=nothing,
                        show_change_points=false,
                        show_data=false,
                        data_indices=nothing)

Simulate a model across multiple change-point segments and optionally
visualize the results.

# Arguments
- `chromosome::AbstractVector`: Flattened parameter vector containing both global and 
  segment-specific parameters.
- `change_points::Vector{Int}`: Detected change point positions (time indices).
- `parnames`: Names of parameters (e.g. `(:β, :γ, ...)`).
- `n_global::Int`: Number of global parameters (shared across segments).
- `n_segment_specific::Int`: Number of parameters that are specific to each segment.
- `model_manager::ModelManager`: Wrapper managing the model type from ModelSimulation module and simulation setup.
- `data::Matrix{Float64}`: Observed data matrix (rows = compartments, columns = time steps).

# Keyword Arguments
- `plot_results::Bool=false`: Whether to generate plots of the simulation.
- `compartments::Union{Nothing,Vector{String}}`: Subset of compartments to plot by name. 
  If `nothing`, all compartments are plotted.
- `show_change_points::Bool=false`: If true, vertical dashed lines mark detected change points.
- `show_data::Bool=false`: If true, overlays observed data (from `data`) where available.
- `data_indices::Union{Nothing,Vector{Int}}=nothing`: Row indices mapping `data` to model
  compartments. For example, `[5,6,7]` means rows 1–3 of `data` correspond to compartments 5,6,7.

# Returns
- `Matrix{Float64}`: Simulation results with rows = compartments, columns = time steps 
  concatenated across all segments.

# Notes
- Simulation segments are concatenated with `hcat`, so time is continuous across change points.
- If plotting is enabled, a grid of subplots is shown with `"Simulated"`, `"Data"`, 
  and `"Change points"` in the legend.


# Examples
```julia

detected_cp = CSV.read("examples/Covid-mode/results_detected_cp_penalty40_ts10_pop150.csv", DataFrame)[:,1] 
params = CSV.read("examples/Covid-mode/results_params_penalty40__ts10_pop150.csv", DataFrame)[:,1] 
parnames = (:δ, :ᴺε₀, :ᴺε₁, :ᴺγ₀, :ᴺγ₁, :ᴺγ₂, :ᴺγ₃, :ω, :ᴺp₁, :ᴺβ,:ᴺp₁₂, :ᴺp₂₃, :ᴺp₁D, :ᴺp₂D, :ᴺp₃D, :ν) 
u0 = [83129285-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
tspan = (0.0, 399.0) 
ode_spec = ODEModelSpec(example_ode_model, initial_chromosome, u0, tspan) 
model_manager = ModelManager(ode_spec) 
n_global = 8 
n_segment_specific = 8

# Simulate without plotting
sim = simulate_full_model(params, detected_cp, parnames,
                          n_global, n_segment_specific,
                          model_manager, data_CP)

# Simulate and plot all compartments with change points and data overlay
sim = simulate_full_model(params, detected_cp, parnames,
                          n_global, n_segment_specific,
                          model_manager, data_CP;
                          plot_results=true,
                          show_change_points=true,
                          show_data=true,
                          data_indices=[5, 6, 7, 9, 11])

```
"""
function simulate_full_model(
    chromosome,
    change_points,
    parnames,
    n_global::Int,
    n_segment_specific::Int,
    model_manager,
    data::Matrix{Float64};
    plot_results::Bool=false,
    compartments::Union{Nothing,Vector{String}}=nothing,
    show_change_points::Bool=false,
    show_data::Bool=false,
    data_indices::Union{Nothing,Vector{Int}}=nothing
)
    constant_pars, segment_pars_list = extract_parameters(chromosome, n_global, n_segment_specific)
    num_segments = length(change_points) + 1

    # Initial condition
    u0 = get_initial_condition(model_manager)
    all_sims = nothing

    if length(change_points) > 0
        for i in 1:num_segments
            idx_start = (i == 1) ? 1 : change_points[i - 1] + 1
            idx_end   = (i > length(change_points)) ? size(data, 2) : change_points[i]
            seg_pars  = segment_pars_list[i]

            all_pars  = @LArray [constant_pars; seg_pars] parnames
            model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)
            sim_data   = simulate_model(model_spec)

            if all_sims === nothing
                all_sims = sim_data
            else
                all_sims = hcat(all_sims, sim_data)
            end

            u0 = update_initial_condition(model_manager, sim_data)
        end
    else
        idx_start = 1
        idx_end   = size(data, 2)
        seg_pars  = segment_pars_list[1]
        all_pars  = @LArray [constant_pars; seg_pars] parnames
        model_spec = segment_model(model_manager, all_pars, idx_start, idx_end, u0)
        sim_data   = simulate_model(model_spec)

        all_sims = (all_sims === nothing) ? sim_data : hcat(all_sims, sim_data)
    end

    # ----------------- Plotting -----------------
    if plot_results
        comp_ids = if isnothing(compartments)
            1:size(all_sims,1)
        else
            findall(x -> x in compartments, parnames)
        end

        ncomp = length(comp_ids)
        nrow  = ceil(Int, sqrt(ncomp))
        ncol  = ceil(Int, ncomp/nrow)
        plt   = plot(layout=(nrow,ncol), size=(1000,700), legend=:topleft, legendfont=font(6))

        for (k, ci) in enumerate(comp_ids)
            t = 1:size(all_sims,2)

            # Always plot simulation
            plot!(plt[k], t, all_sims[ci,:], label="Simulated", lw=2)

            # Plot data if this compartment index is in data_indices
            if show_data && !isnothing(data_indices)
                pos = findfirst(==(ci), data_indices)
                if pos !== nothing && pos <= size(data,1)
                    plot!(plt[k], t, data[pos,:], seriestype=:scatter,
                          label="Data", ms=1, alpha=0.4)
                end
            end

            # Change points
            if show_change_points && !isempty(change_points)
                for (j, cp) in enumerate(change_points)
                    lbl = (j == 1) ? "Change points" : false
                    vline!(plt[k], [cp], c=:red, lw=1, ls=:dash, label=lbl)
                end
            end
        end

        display(plt)
    end
    # --------------------------------------------

    return all_sims
end




"""
    plot_parameter_changes(params, param_labels, change_points,
                           base_date, end_date;
                           n_segment_specific, n_global, color=:blues)

Visualize relative changes of segment-specific parameters compared to the first segment,
using heatmaps stacked vertically.

# Arguments
- `params::Vector{Float64}`: Flattened parameter vector from optimization.
- `param_labels::Vector{String}`: Human-readable labels for each parameter (e.g. 
  `"β (Infection rate)"`).
- `change_points::Vector{Date}`: Detected change point dates.
- `base_date::Date`: Start date of the first segment.
- `end_date::Date`: End date of the last segment.

# Keyword Arguments
- `n_segment_specific::Int`: Number of segment-specific parameters per segment.
- `n_global::Int`: Number of global parameters (to skip over at the start of `params`).
- `color=:blues`: Colormap for the heatmaps.

# Returns
- `Plots.Plot`: A composite plot object with one heatmap per parameter.

# Details
- Each row corresponds to one parameter, and columns correspond to segments.
- Parameter values are normalized relative to the first segment (`data_rel = data ./ data[:,1]`).
- X-axis ticks mark change points, labeled as `"cp1"`, `"cp2"`, …, plus start and end dates.
- Colorbar ranges are adjusted per parameter, and only the last subplot includes the legend.

# Examples

```julia
param_labels = ["p₁ \n(Detection rate)", "β \n(Infection rate)",
 "p₁₂ \n(Hospitalization rate)", "p₂₃ \n(ICU admission rate)",
 "p₁D \n(Infection death rate)", "p₂D \n(Hospital death rate)",
 "p₃D \n(ICU death rate)"] 
p2 = plot_parameter_changes(params, param_labels, 
cases_CP_date[detected_cp], 
Date("2020-01-27"), 
Date("2021-03-02") 
n_global=8, n_segment_specific=8)
```

"""
function plot_parameter_changes(params::Vector{Float64},
                                param_labels::Vector{String},
                                change_points::Vector{Date},
                                base_date::Date,
                                end_date::Date,
                                n_segment_specific::Int,
                                n_global::Int,
                                color=:blues)

    # Construct segment edges (dates of CPs + start & end)
    segment_edges = vcat(base_date, change_points, end_date)

    # Labels for xticks
    cps = ["$(base_date)", ["cp$(i)" for i in 1:length(change_points)]..., "$(end_date)"]

    # Extract parameters for each segment
    n_params = length(param_labels)
    param_values = [params[n_global + i : n_segment_specific : end] for i in 1:n_params]
    data = hcat(param_values...)'
    data_rel = data ./ data[:, 1]

    heatmaps = []
    for i in 1:n_params
        vmin = minimum(data_rel[i, :])
        vmax = maximum(data_rel[i, :])
        tickvals = [vmin, vmax]
        ticklabels = round.(tickvals; digits=2)

        show_xticks = (i == n_params)
        show_colorbar_title = (i == n_params)

        hm = heatmap(
            segment_edges,
            [0, 1],
            data_rel[i:i, :],
            xlabel = show_xticks ? "Change Points" : "",
            ylabel = "",
            yticks = ([0.5], [param_labels[i]]),
            xticks = show_xticks ? (segment_edges, cps) : false,
            xrotation = show_xticks ? 30 : 0,
            c = color,
            colorbar = true,
            colorbar_title = show_colorbar_title ? "Rel. Change" : "",
            colorbar_ticks = (tickvals, ticklabels),
            size = (900, 130),
            bottom_margin = show_xticks ? 15Plots.mm : 2Plots.mm,
            top_margin = 1Plots.mm,
            left_margin = 10Plots.mm,
            right_margin = 10Plots.mm,
            xtickfont = font(7),
            ytickfont = font(9),
            guidefont = font(7),
            framestyle = :box
        )
        push!(heatmaps, hm)
    end

    # Combine into one plot
    layout_spec = grid(n_params, 1, heights=fill(1/n_params, n_params))
    plot(heatmaps..., layout=layout_spec, size=(900, 150*n_params))
end

#end # module
