using CSV, DataFrames, CategoricalArrays, StatsPlots, Plots, Statistics

# Load and clean
df = CSV.read("benchmark/results/benchmark_results_all_configs.csv", DataFrame)
df = CSV.read("benchmark/results/benchmark_results_all_configs_sorted.csv", DataFrame)


df = CSV.read("benchmark/results/benchmark_results_all_configs_sorted_added_cp_pars-columns.csv", DataFrame)

df = filter(row -> 
 (row.noise_level == 10 || row.noise_level == 30 || row.noise_level == 10) &&
 row.noise_type == "Uniform" &&
 (row.data_length == 250 || row.data_length == 250 )&&
 row.change_point_count == 3,
 df
)



df = filter(row -> !isnan(row.precision), df)
df = filter(row -> row.noise_type == "Uniform", df)



# Extract numeric penalty score
df.penalty_score = parse.(Int, replace.(string.(df.penalty), r"[^\d]" => ""))

# Create composite label
df.combo = string.("N=", df.noise_level, ", P=", df.penalty_score)

# Compute mean F1 per combination
df_summary = combine(groupby(df, [:noise_level, :penalty_score, :combo]), 
                     :precision => mean => :mean_precision)

# Sort by noise and penalty
df_summary = sort(df_summary, [:noise_level, :penalty_score])
df_summary.combo = categorical(df_summary.combo, levels=unique(df_summary.combo))

# Create position mapping
combo_pos = Dict(lbl => i for (i, lbl) in enumerate(levels(df_summary.combo)))

# Group label positions by noise level
noise_to_x = Dict{Int, Vector{Int}}()
for row in eachrow(df_summary)
    push!(get!(noise_to_x, row.noise_level, Int[]), combo_pos[row.combo])
end

# Plot the F1 trend line
plt = @df df_summary Plots.plot(:combo, :mean_precision,
    seriestype = :line,
    xlabel = "Penalty",
    ylabel = "Mean precision",
    title = "Mean precision vs Noise and Penalty",
    marker = :circle,
    linewidth = 2,
    color = :black,
    rotation = 45,
    legend = false,
    ylims = (0, 1.1),
    xtickfont = font(8)
)



# Get y-limits
ymin, ymax = Plots.ylims(plt)

# Define region colors (adjust as needed)
noise_colors = Dict(
    0   => :deepskyblue,     # vivid blue
    1   => :goldenrod,       # rich yellow
    10  => :mediumseagreen,  # medium green
    20  => :orchid,          # medium purple
    30  => :sandybrown,      # soft orange-brown
    40  => :mediumaquamarine,# bluish-green
    100 => :lightcoral       # soft red
)

# Add background regions for each noise level
for (noise, xpos) in sort(collect(noise_to_x))
    x0 = minimum(xpos) - 0.5
    x1 = maximum(xpos) + 0.5
    Plots.plot!(plt, [x0, x1, x1, x0], [ymin, ymin, ymax, ymax],
        seriestype = :shape,
        fillcolor = get(noise_colors, noise, :lightgray),
        fillalpha = 0.2,
        linecolor = :transparent,
        label = "")
    # Optional: annotate noise level
    annotate!(plt, mean(xpos), ymax - 0.05, Plots.text("Noise=$(noise)", :black, 8, :center))
end

# Generate a sequential x position for plotting (repeated penalty values for each noise level)
penalty_values = [0, 1, 10, 20, 30, 100]
n_levels = length(unique(df_summary.noise_level))
x_positions = repeat(penalty_values, n_levels)[1:end-1]

df_summary.x = 1:length(x_positions)

# Customize x-ticks: repeat penalty values
xticks = (1:length(x_positions), string.(repeat(penalty_values, n_levels)))
xtick_positions = 1:length(x_positions)

Plots.plot!(plt, xticks = (xtick_positions, xticks[2]), xtickfont = font(8), rotation = 60)


# Show or save
savefig(plt, "benchmark/plots/mean_precision_shaded_by_noise.pdf")





using CSV, DataFrames, CategoricalArrays, StatsPlots, Plots, Statistics

# Load and clean
df = CSV.read("benchmark/results/benchmark_results_all_configs.csv", DataFrame)
df = filter(row -> !isnan(row.precision), df)

# Extract numeric penalty score
df.penalty_score = parse.(Int, replace.(string.(df.penalty), r"[^\d]" => ""))

# Compute mean precision grouped by noise and penalty
df_summary = combine(groupby(df, [:penalty_score, :noise_level]), 
                     :precision => mean => :mean_precision)

# Sort by penalty then noise
sort!(df_summary, [:penalty_score, :noise_level])

# Create x-axis based on noise_level
df_summary.x = 1:nrow(df_summary)

# Group positions by penalty for shading
penalty_to_x = Dict{Int, Vector{Int}}()
for row in eachrow(df_summary)
    push!(get!(penalty_to_x, row.penalty_score, Int[]), row.x)
end

# Define colors per penalty value
penalty_colors = Dict(
    0   => :deepskyblue,
    1   => :goldenrod,
    10  => :mediumseagreen,
    20  => :orchid,
    30  => :sandybrown,
    100 => :lightcoral
)

# Plot line: x is noise level index, y is mean precision
plt = @df df_summary plot(:x, :mean_precision,
    seriestype = :line,
    marker = :circle,
    linewidth = 2,
    color = :black,
    xlabel = "Noise Level",
    ylabel = "Mean Precision",
    title = "Mean Precision vs Noise (shaded by Penalty)",
    legend = false,
    ylims = (0, 1.1),
    xtickfont = font(8)
)

# Add colored regions by penalty
ymin, ymax = Plots.ylims(plt)
for (penalty, xpos) in sort(collect(penalty_to_x))
    x0 = minimum(xpos) - 0.5
    x1 = maximum(xpos) + 0.5
    Plots.plot!(plt, [x0, x1, x1, x0], [ymin, ymin, ymax, ymax],
        seriestype = :shape,
        fillcolor = get(penalty_colors, penalty, :lightgray),
        fillalpha = 0.2,
        linecolor = :transparent,
        label = "")
    # Optional: add penalty annotation
    annotate!(plt, mean(xpos), ymax - 0.05, Plots.text("Penalty=$(penalty)", :black, 8, :center))
end

# Customize x-ticks: show noise level values
xticks = (df_summary.x, string.(df_summary.noise_level))
plot!(plt, xticks = xticks, rotation = 60)

# Save or display
savefig(plt, "benchmark/plots/mean_precision_shaded_by_penalty.pdf")
display(plt)


