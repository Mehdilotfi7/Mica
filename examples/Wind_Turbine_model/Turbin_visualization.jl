#####################################
# visualization of parameters 
using Dates
using CSV
using DataFrames
using Plots


df = CSV.read("Turbine_Data_Kelmarsh_1_2021-01-01_-_2021-07-01_228.csv", DataFrame)
detected_cp = CSV.read("results_Turbine_detected_cp_correct_pen0_pop_50.csv", DataFrame)[:,1]
params = CSV.read("results_Turbine_params_correct_pen0_pop50.csv", DataFrame)[:,1]
df[:, "Date and time"][1]

segment_dates = df[:, "Date and time"][detected_cp]
segment_labels = vcat(("2021-01-01 00:00:00"), segment_dates, ("2021-01-18 08:30:00"))
segment_labels = Date.(segment_labels)

segment_labels_str = string.(segment_labels)
segment_indices = 1:length(segment_labels_str)

# Convert to strings for xticks
xtick_labels = string.(segment_labels)

segment_edges = segment_labels

# Dummy data: 4 parameters × 12 segments

param_labels = ["θ1 \n(Thermal resistance Coeff1)", "θ2 \n(Thermal resistance Coeff2)", "θ3 \n(Thermal resistance Coeff3)", "θ4 \n(Thermal resistance Coeff4)"]


param1 = params[4:4:end]
param2 = params[5:4:end]
param3 = params[6:4:end]
param4 = params[7:4:end]

data = [param1 param2 param3 param4]'  
data_rel = data ./ data[:, 1]
data_rel = hcat(ones(size(data, 1)), data[:, 2:end] ./ data[:, 1:end-1])


# We need 5 y-edges for 4 parameters (just space them uniformly)
yedges = 0:1:4  # 5 edges for 4 rows



heatmaps = []

for i in 1:size(data_rel, 1)
    vmin = minimum(data_rel[i:i, :])
    vmax = maximum(data_rel[i:i, :])
    show_xticks = (i == size(data_rel, 1))
    show_colorbar_title = (i == size(data_rel, 1))
    tickvals = [vmin, vmax]
    ticklabels = round.(tickvals; digits=2)
    tick_pairs = Pair.(tickvals, ticklabels)


    hm = heatmap(
        segment_indices,
        [0, 1],
        data_rel[i:i, :],
        xlabel = show_xticks ? "Change Points" : "",
        ylabel = "",
        yticks = ([0.5], [param_labels[i]]),
        xticks = show_xticks ? (segment_indices, segment_labels_str) : false,
        xrotation = show_xticks ? 40 : 0,
        c = :blues,
        #clim = (vmin, vmax),
        colorbar = true,
        colorbar_title = show_colorbar_title ? "RCFS" : "",
        colorbar_ticks = tick_pairs,
        size = (950, 130),
        bottom_margin = show_xticks ? 15Plots.mm : 2Plots.mm,
        top_margin = 1Plots.mm,
        left_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        xtickfont = font(7),
        ytickfont = font(9),
        guidefont = font(10),
        framestyle = :box,
    )

    push!(heatmaps, hm)
end

p3 = plot(heatmaps..., layout = @layout([a;b;c;d]), size = (950, 950))
p4 = plot(p2, p3, layout=(2, 1), margin = 8Plots.mm,dpi=100,
size=(1400, 1200), guidefont=font(12), legendfont=font(10), titlefont=font(10),tickfont=font(10)
)
savefig(p4, "Turbine_relative_pars_Previous_Segment.pdf")


layout = @layout [a{0.6h} ;b{0.4h}]  # 30% height for p, 70% for p2















heatmaps = []
using Plots
using Printf
pyplot()

for i in 1:size(data_rel, 1)

    vmin = minimum(data_rel[i:i, :])
    vmax = maximum(data_rel[i:i, :])
    tickvals = [vmin, vmax]
    ticklabels = round.(tickvals; digits=2)

    show_xticks = (i == size(data_rel, 1))
    show_colorbar_title = (i == size(data_rel, 1))
    

    hm = heatmap(
        segment_edges,
        [0, 1],
        data_rel[i:i, :],
        xlabel = show_xticks ? "Change Points" : "",
        ylabel = "",
        yticks = ([0.5], [param_labels[i]]),
        #xticks = show_xticks ? (segment_edges, string.(segment_edges)) : false,  # ← Hides xticks if not last
        xticks=false,
        xtickfontrotation = show_xticks ? 30 : 0,
        c = :blues,
        #clim = (vmin, vmax),
        colorbar = true,
        colorbar_title = show_colorbar_title ? "RCPS" : "",  # ← Only for last plot
        #colorbar_tickvals = tickvals,
        #colorbar_ticklabels = ticklabels,
        colorbar_ticks = (tickvals, ticklabels),
        size = (900, 130),
        bottom_margin = show_xticks ? 15Plots.mm : 2Plots.mm,
        top_margin = 1Plots.mm,
        left_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        xtickfont = font(7),
        ytickfont = font(9),
        guidefont = font(10),
        #tickfont = font(9),
        framestyle = :box
    )

    push!(heatmaps, hm)
end

# Combine plots
p2 = plot(heatmaps..., layout = @layout([a;b;c;d;e;f;g]), size = (900, 400))
p4 = plot(p,p2, layout = (2,1))
layout = @layout [a{0.6h} ;b{0.4h}]  # 30% height for p, 70% for p2
p4 = plot(p, p2, layout = layout,  size = (900, 900))
savefig(p2, "relative_pars_Previous_Segment_pen40.pdf")
savefig(p4, "covsim_relative_pars_Previous_Segment_pen40.pdf")


plt = plot(heatmaps..., layout = (length(heatmaps), 1))

segment_edges = collect(1:length(segment_labels))
using Plots: xticks!

xticks!(
  p2[end],  # Apply only to the last subplot
    segment_edges[1:end].-0.5 ,            # Shift to left edge
    string.(segment_labels[1:end]),

    framestyle = :none     # Convert dates to strings
)