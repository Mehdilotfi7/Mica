using CSV, DataFrames, StatsPlots, Plots, Statistics

# Load and clean
df = CSV.read("benchmark/results/benchmark_results_all_configs_sorted_added_cp_pars-columns.csv", DataFrame)

for col in [:precision, :recall, :f1]
    df[!, col] .= coalesce.(df[!, col], 0.0)
    df[!, col] .= ifelse.(isnan.(df[!, col]), 0.0, df[!, col])
end

# Group and compute mean runtime
df_summary = combine(groupby(df, [:change_point_count, :data_length]),
                     :runtime => mean => :mean_runtime)
sort!(df_summary, [:change_point_count, :data_length])

# All unique data lengths (for tick labels)
all_lengths = unique(df.data_length) |> sort

# Prepare plot
plt = plot(
    xlabel = "Data Length",
    ylabel = "Mean Runtime (s)",
    title = "Runtime vs Data Length for Different Change Point Counts",
    xticks = (1:length(all_lengths), string.(all_lengths)),
    legend = :topleft,
    xtickfont = font(8),
    titlefont = font(10),
    ylims = (0, maximum(df_summary.mean_runtime) * 1.1)
)

# Plot one line per change point count
for cp in sort(unique(df_summary.change_point_count))
    # Filter only valid rows for this CP
    cp_df = filter(row -> row.change_point_count == cp, df_summary)
    sorted_lengths = cp_df.data_length
    # Get index positions of valid data_lengths
    x_vals = findall(x -> x in sorted_lengths, all_lengths)
    plot!(plt, x_vals, cp_df.mean_runtime;
          label = "CP=$(cp)", lw=2, marker=:circle)
end
display(plt)
# Save and display
savefig(plt, "benchmark/plots/runtime_vs_data_length_trimmed_per_cp.pdf")






df = filter(row -> parse(Int, replace(string(row.penalty), r"[^\d]" => "")) != 1, df)


# Extract numeric penalty scores
df.penalty_score = parse.(Int, replace.(string.(df.penalty), r"[^\d]" => ""))

# Group by penalty and noise level, compute mean precision
df_summary = combine(groupby(df, [:penalty_score, :noise_level]),
                     :f1 => mean => :mean_precision)

# Sort for consistency
sort!(df_summary, [:penalty_score, :noise_level])

# Prepare x-axis: unique sorted noise levels
all_noises = unique(df_summary.noise_level) |> sort

# Setup plot
plt = plot(
    xlabel = "Noise Level",
    ylabel = "Mean F1 Score",
    title = "Effect of BIC Penalty Coefficients on F1 Score Across Noise Levels",
    xticks = (1:length(all_noises), string.(all_noises)),
    ylims = (0, 1.2),
    legend = :topright,
    titlefont = font(10),
    xtickfont = font(8)
)

# Plot one line per penalty value
for penalty in sort(unique(df_summary.penalty_score))
    p_df = filter(row -> row.penalty_score == penalty, df_summary)
    sorted_noises = p_df.noise_level
    x_vals = findall(x -> x in sorted_noises, all_noises)
    plot!(plt, x_vals, p_df.mean_precision;
          label = "Penalty=$(penalty)", lw=2, marker=:circle)
end

# Save and display
display(plt)
savefig(plt, "benchmark/plots/f1_vs_noise_trimmed_per_penalty.pdf")

