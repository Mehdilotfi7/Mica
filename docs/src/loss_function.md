# Custom Loss Functions for Segment Evaluation

In Mica, the cost of each segment is calculated using a **loss function** that compares the simulated model output to the observed data. This per-segment cost is then aggregated by the **objective function**, which drives changepoint detection and optimization.

This guide explains how to define and use custom loss functions within Mica, how they are used in segment evaluation, and how they differ from the internal objective function.

---

## Segment Loss vs. Global Objective

* **Loss Function**: Computes the discrepancy between model predictions and real data **within a single segment**.
* **Objective Function**: Combines the segment losses, adds penalty terms (e.g. BIC), and evaluates the **overall model fit** across all segments.

This separation allows you to plug in domain-specific error measures for each segment while letting Mica handle optimization at the global level.

---

## Defining a Custom Loss Function

A loss function in Mica must follow this interface:

```julia
function my_loss_function(sim_output::Vector, observed_data::Vector)::Float64
```

It should return a scalar loss value.

### Example 1: Mean Squared Error (MSE)

```julia
function mse_loss(sim, data)
    return sum((sim .- data).^2)
end
```

### Example 2: Logarithmic Loss

```julia
function log_loss(sim, data)
    return sum(log.(abs.(sim .- data) .+ 1e-6))  # Avoid log(0)
end
```

### Example 3: Normalized RMSE

```julia
function nrmse_loss(sim, data)
    rmse = sqrt(mean((sim .- data).^2))
    return rmse / (maximum(data) - minimum(data))
end
```

---

## Using a Custom Loss Function in Mica

To apply your custom loss during changepoint detection:

```julia
obj_fn = wrapped_obj_function(model, data, loss_fn = my_loss_function)
```

Then pass `obj_fn` to `optimize_with_changepoints` or other relevant calls.

This gives you full control over how error is computed for each segment while leveraging Mica’s optimization engine.

---

## Additional Notes

* The loss function should be **efficient**, as it's called frequently during GA optimization.
* Mica expects **vector output** from the model; ensure your simulation returns the correct shape.
* You can log or visualize segment-wise losses for interpretability.

---

By customizing your loss function, you tailor the segmentation process to your modeling priorities — whether it's prediction accuracy, stability, or a domain-specific criterion.
