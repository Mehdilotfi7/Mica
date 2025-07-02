using DataFrames, CSV, StatsPlots, Statistics
using CategoricalArrays

df = CSV.read("benchmark/results/benchmark_results_all_configs.csv", DataFrame)

df_clean = filter(row -> !isnan(row.precision) && !isnan(row.recall), df)

df = df_clean

# Create a combined label
df.label = string.("CP=", df.change_point_count, ", L=", df.data_length)

# Compute mean runtime per label
mean_runtimes = combine(groupby(df, :label), :runtime => mean => :mean_runtime)

# Sort labels by mean runtime
sorted_labels = sort(mean_runtimes, :mean_runtime).label

# Convert label to categorical with ordered levels
df.label = categorical(df.label, levels=sorted_labels)


# Plot sorted boxplot
runtime_vs_CP_and_L = @df df boxplot(:label, :runtime,
    xlabel="Configuration (sorted)",
    ylabel="Runtime (s)",
    title="Runtime Distribution per Configuration (Sorted)",
    rotation=45,
    legend=false,
    linewidth=0.8,
    boxstyle = :auto
)





using CategoricalArrays, StatsPlots, DataFrames, Statistics

df = df_clean
# 1. Label per configuration
df.label = string.("CP=", df.change_point_count, ", L=", df.data_length)

# 2. Sort labels by median runtime
medians = combine(groupby(df, :label), :runtime => median => :median_runtime)
sorted_labels = sort(medians, :median_runtime).label
df.label = categorical(df.label, levels=sorted_labels)

# 3. Plot
@df df boxplot(:label, :runtime,
    xlabel="Sorted Configurations",
    ylabel="Runtime (s)",
    title="Runtime Distribution by Change Point Count and Data Length",
    rotation=45,
    legend=false
)

# 4. Optional overlay: mean
groupmeans = combine(groupby(df, :label), :runtime => mean => :mean_runtime)
scatter!(groupmeans.label, groupmeans.mean_runtime,
    color=:red, marker=:star5, label="Mean")







    using CategoricalArrays, DataFrames, CSV

    df = CSV.read("benchmark/results/benchmark_results_all_configs.csv", DataFrame)
    
    # Filter out NaN precision values
    df_clean = filter(row -> !isnan(row.precision), df)
    
    # Create label per configuration
    df_clean.label = string.("CP=", df_clean.change_point_count, ", L=", df_clean.data_length)
    
    # Sort DataFrame by CP, then data length
    df_clean = sort(df_clean, [:change_point_count, :data_length])
    
    # Use unique ordered labels to preserve this structure
    ordered_labels = unique(df_clean.label)
    df_clean.label = categorical(df_clean.label, levels=ordered_labels)
    
    # Plot
    using StatsPlots
    runtime_vs_CP_and_L_Lsorting = @df df_clean boxplot(:label, :runtime,
    group = :change_point_count,               
    label = ["CP=1" "CP=2" "CP=3"],
    xlabel = "Configuration (CP, Length)",
    ylabel = "Runtime (s)",
    rotation = 20,
    title = "Runtime by Change Point Count and Data Length",
    titlefont = 11,
    tickfont = font(7),
    guidefont = font(9),
    legend = :topleft                          # show legend for CP groups
)

savefig(runtime_vs_CP_and_L_Lsorting, "benchmark/plots/runtime_vs_CP_and_L_Lsorting.pdf")





#
using CategoricalArrays, DataFrames, CSV, StatsPlots

df = CSV.read("benchmark/results/benchmark_results_all_configs.csv", DataFrame)

# Filter out invalid rows
df_clean = filter(row -> !isnan(row.precision), df)

# Reorder: data length first, then change points
df_clean = sort(df_clean, [:data_length, :change_point_count])

# New label format: data length first
df_clean.label = string.("L=", df_clean.data_length, ", CP=", df_clean.change_point_count)

# Define ordered label levels
ordered_labels = unique(df_clean.label)
df_clean.label = categorical(df_clean.label, levels=ordered_labels)

# Plot
runtime_vs_CP_and_L_CPsorting = @df df_clean boxplot(:label, :runtime,
    group = :change_point_count,
    label = ["CP=1" "CP=2" "CP=3"],
    xlabel="Configuration (Length, CP)",
    ylabel="Runtime (s)",
    rotation=20,
    titlefont = 11,
    tickfont = font(7),
    guidefont = font(9),
    legend = :topleft,
    title="Runtime by Data Length and Change Point Count"
)

savefig(runtime_vs_CP_and_L_CPsorting, "benchmark/plots/runtime_vs_CP_and_L_CPsorting.pdf")



df.label = string.("L=", df.data_length, "\nCP=", df.change_point_count)

# Sort in logical order
df = sort(df, [:data_length, :change_point_count])
df.label = categorical(df.label, levels=unique(df.label))

@df df boxplot(:label, :runtime,
    group = :change_point_count,
    label = ["CP=1" "CP=2" "CP=3"],
    xlabel = "Data Length + CP",
    ylabel = "Runtime (s)",
    rotation = 60,
    title = "Runtime vs Data Length and CP (Grouped Labels)",
    legend = :topleft,
    boxstyle = :auto,
    linewidth = 0.8,
    tickfont = font(7)
)







#############################################

using CSV, DataFrames, CategoricalArrays, StatsPlots, Plots

# Load and clean data
df = CSV.read("benchmark/results/benchmark_results_all_configs.csv", DataFrame)
df = filter(row -> !isnan(row.precision), df)
df.label = string.("CP=", df.change_point_count, ", L=", df.data_length)
df = sort(df, [:change_point_count, :data_length])
df.label = categorical(df.label, levels=unique(df.label))

# Numeric x-axis mapping for labels
label_pos = Dict(lbl => i for (i, lbl) in enumerate(levels(df.label)))

# Group x positions by CP
cp_to_x = Dict{Int, Vector{Int}}()
for lbl in levels(df.label)
    cp = parse(Int, match(r"CP=(\d+)", lbl).captures[1])
    push!(get!(cp_to_x, cp, Int[]), label_pos[lbl])
end

# Plot boxplot: use same color for all boxes, no legend
plt = @df df boxplot(:label, :runtime,
    fillcolor = :white,
    linecolor = :black,
    legend = false,
    xlabel = "Configuration (CP, Length)",
    ylabel = "Runtime (s)",
    rotation = 20,
    title = "Runtime by Change Point Count and Data Length",
    titlefont = 11,
    tickfont = font(7),
    guidefont = font(9)
)

# Get y-limits of current plot
ymin, ymax = Plots.ylims(plt)

# Region colors for CP values
cp_colors = Dict(1 => :royalblue, 2 => :darkorange, 3 => :darkolivegreen)

# Add shaded regions per CP group
for cp in sort(collect(keys(cp_to_x)))
    inds = cp_to_x[cp]
    x0 = minimum(inds) - 0.5
    x1 = maximum(inds) + 0.5
    Plots.plot!(plt, [x0, x1, x1, x0], [ymin, ymin, ymax, ymax],
        seriestype = :shape,
        fillcolor = cp_colors[cp],
        fillalpha = 0.25,
        linecolor = :transparent,
        label = "")
end

# Display and save
display(plt)
annotate!(2, 460, Plots.text("CP=1", :red, 10, :center))
annotate!(7, 460, Plots.text("CP=2", :red, 10, :center))
annotate!(11, 460, Plots.text("CP=3", :red, 10, :center))

display(plt)
savefig(plt, "benchmark/plots/runtime_shaded_by_CP_no_legend_same_box_color.pdf")







