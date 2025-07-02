using CSV, DataFrames, StatsPlots, Statistics, CategoricalArrays, Measures


# Load the data
df = CSV.read("benchmark/results/benchmark_results_all_configs.csv", DataFrame)

# Clean missing or NaN values in accuracy metrics
for col in [:precision, :recall, :f1]
    df[!, col] .= coalesce.(df[!, col], 0.0)
    df[!, col] .= ifelse.(isnan.(df[!, col]), 0.0, df[!, col])
end
df
# Make sure penalty values are strings
df.penalty = string.(df.penalty)

# (Optional) define desired penalty order if needed
penalty_levels = ["BIC_0", "BIC_1", "BIC_10", "BIC_20", "BIC_30", "BIC_100"]
#penalty_levels = sort(unique(df.penalty))  # or specify manually
df.penalty = CategoricalArray(df.penalty; ordered=true, levels=penalty_levels)

# Filter for Gaussian noise only
#df_gaussian = filter(:noise_type => ==("Gaussian"), df)
# Filter for Gaussian noise only and exclude noise_level=100 and penalty="BIC_100"
#df_gaussian = filter(row -> row.noise_type == "Gaussian" &&
#                              row.noise_level != 100 &&
#                              row.penalty != "BIC_100", df)


# Group by change_point_count and noise_level
grouped = groupby(df, [:change_point_count, :noise_level, :noise_type])



begin
# Layout setup
n = length(grouped)
ncols = 5
nrows = ceil(Int, n / ncols)
layout = grid(nrows, ncols)




# Collect plots
plots = []

for g in grouped
    cpc = g.change_point_count[1]
    nl = g.noise_level[1]
    nt = g.noise_type[1]
    g = sort!(g, [:data_length, :penalty])

    ######################
    # add small numbers to make the plots readible 

    # Get unique data_lengths
    unique_lengths = sort(unique(g.data_length))

   # Create a mapping: data_length => small shift
   offset_map = Dict(dl => 0.01 * i for (i, dl) in enumerate(unique_lengths))

   # Add a new column for precision with slight offset per group
   g.recall_offset = [row.recall + offset_map[row.data_length] for row in eachrow(g)]
 
    p = @df g plot(:penalty, :recall,
        group = :data_length,
        xlabel = "Penalty",
        ylabel = "recall",
        title = "nt=$nt,cpc=$cpc, noise=$nl",
        lw = 1,
        #marker = :auto,
        #markersize = 0.2,
        legend = :bottomleft,
        xrotation = 45,
        ylim = (0, 1.1)
        )

    push!(plots, p)
end

# Combine and display all plots
final_plot = plot(plots...,
    layout = layout,
    tickfont = font(4),
    guidefont = font(4),
    legendfont = font(3),
    titlefont = font(4),
    bottom_margin=1mm,
    left_margin = 5mm,
    right_margin = 5mm,
    size = (1000, 1200)
    )
end
# Save the figure
savefig(final_plot, "benchmark/plots/recall_vs_Penalty_all_configs_original.pdf")
#savefig(final_plot, "/Users/mehdilotfi/Desktop/recall_vs_Penalty.pdf")


###########################################################################################
# Heatmap of Runtime by Data Length and Change Point Count

using CSV, DataFrames, StatsPlots, Statistics, ColorSchemes, Measures

# Load your data
df = CSV.read("benchmark/results/benchmark_results_all_configs.csv", DataFrame)

# Clean NaN/missing runtime
df.runtime .= coalesce.(df.runtime, 0.0)
df.runtime .= ifelse.(isnan.(df.runtime), 0.0, df.runtime)

# Sort keys for consistent heatmap layout
data_lengths = sort(unique(df.data_length))
change_points = sort(unique(df.change_point_count))

# Group by all other variables
grouped = groupby(df, [:noise_level, :noise_type, :penalty])
length(grouped)
# Prepare plot list
begin
plots = []

for g in grouped
    nl = g.noise_level[1]
    nt = g.noise_type[1]
    p = g.penalty[1]

    # Build runtime matrix: rows = change_point_count, cols = data_length
    runtime_matrix = [
        begin
            rt = g[(g.data_length .== len) .& (g.change_point_count .== cp), :runtime]
            isempty(rt) ? NaN : rt[1]
        end
        for cp in change_points, len in data_lengths
    ]

    h = heatmap(
        data_lengths,
        change_points,
        runtime_matrix;
        xlabel = "Data Length",
        ylabel = "Change Point Count",
        yticks = ([1, 2, 3], ["1", "2", "3"]),
        title = "$nt, noise=$nl, $p",
        color = :viridis,
        clims = (minimum(df.runtime), maximum(df.runtime)),
        colorbar = false,
        colorbar_title = "Runtime (s)",
        colorbar_titlefontsize = 6,
        colorbar_tickfontsize = 6,
        guidefont = font(7),
        titlefont = 7,
        tickfont = font(7),
        size = (300, 250)
    )

    push!(plots, h)
end

min_runtime, max_runtime = (minimum(df.runtime), maximum(df.runtime))
# Global colorbar as a fake heatmap




# Plot layout (e.g., 3 cols)
ncols = 5
nrows = ceil(Int, length(plots) / ncols)
layout = grid(nrows, ncols)
#layout = @layout [grid(nrows, ncols) a{0.05w}]

#
# Combine all into one figure
#final_plot = plot(plots..., heatmap((0:0.01:1).*ones(101,1),title = "Runtime (s)", color = :viridis, legend=:none, xticks=:none, yticks=(1:10:101, string.(round.(range(min_runtime, max_runtime; length=11)))));
final_plot = plot(plots...,
    layout = layout,
    size = (300 * ncols, 200 * nrows),
    #margin = 2mm,
    left_margin = 30mm,
    right_margin = 2mm,
    top_margin = 5mm,
    bottom_margin = 5mm)
end
savefig(final_plot, "benchmark/plots/runtime_heatmaps_all_configs.pdf")
#savefig(final_plot, "/Users/mehdilotfi/Desktop/runtime_heatmaps_all_configs.pdf")


