using JLD2, Glob, Plots
gr()
default(fontfamily="Computer Modern", guidefontsize=12, tickfontsize=10, legendfontsize=10)

folder = "results/"
files = Glob.glob("*.jld2", folder)

# Load and organize file metadata
plot_entries = []
for file in files
    @load file data times detected_cps valid_change_points noise_level noise_type change_point_count data_length f1
    push!(plot_entries, (
        file=file,
        data=data,
        times=times,
        detected_cps=detected_cps,
        valid_change_points=valid_change_points,
        noise_level=noise_level,
        noise_type=noise_type,
        change_point_count=change_point_count,
        data_length=data_length,
        f1=f1
    ))
end

# Sort by data_length to allow grouping by common x-axis
sort!(plot_entries, by = x -> x.data_length)

# Determine layout
n = length(plot_entries)
ncols = 3
nrows = ceil(Int, n / ncols)

plot_list = []

for (i, entry) in enumerate(plot_entries)
    p = plot(entry.times, entry.data, label="Simulated Data", lw=1.5, color=:black)

    if !isempty(entry.valid_change_points)
        vline!(p, entry.valid_change_points, label="True Change Points", color=:green, linestyle=:dot, lw=2)
    end

    if !isempty(entry.detected_cps)
        vline!(p, entry.detected_cps, label="Detected Change Points", color=:red, linestyle=:dash, lw=2)
    end

    # Title with full metadata
    combo_str = "Noise=$(entry.noise_level), Type=$(entry.noise_type), CPs=$(entry.change_point_count), Len=$(entry.data_length), F1=$(round(entry.f1, digits=2))"
    plot!(p, title=combo_str, xlabel="", ylabel="", legend=false)

    push!(plot_list, p)
end

# Add axis labels only to bottom row and leftmost column
for (i, p) in enumerate(plot_list)
    row = ceil(Int, i / ncols)
    col = i % ncols == 0 ? ncols : i % ncols

    if row == nrows
        xlabel!(p, "Time")
    end
    if col == 1
        ylabel!(p, "Value")
    end
end

# Add legend separately
legend_plot = plot(legend=:outerright)
plot!(legend_plot, [0], [0], label="Simulated Data", lw=2, color=:black)
plot!(legend_plot, [0], [0], label="True Change Points", lw=2, color=:green, linestyle=:dot)
plot!(legend_plot, [0], [0], label="Detected Change Points", lw=2, color=:red, linestyle=:dash)

# Combine into final layout
final_plot = plot(plot_list..., layout=(nrows, ncols), size=(ncols*500, nrows*350))
plot_with_legend = plot(final_plot, legend_plot, layout=@layout([a; b]), size=(ncols*500, nrows*350 + 100))

# Save to PDF
mkpath("plots/")
savefig(plot_with_legend, "plots/combined_results.pdf")









