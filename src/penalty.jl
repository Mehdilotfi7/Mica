

"""
    call_penalty_fn(f::Function; kwargs...) -> Real

Safely call a user-provided penalty function `f` with only the arguments it accepts, using the *last defined method* of `fn`.

This allows users to define custom penalty functions that take any subset of the following:

- `p`: number of segment-specific parameters
- `n`: total data length
- `CP`: vector of change points
- `segment_lengths`: vector of segment lengths (computed as `diff([0; CP; n])`)
- `num_segments`: number of segments (`length(CP) + 1`)

# Important Note

Due to Julia's multiple dispatch behavior, defining multiple methods for the same function name
accumulates methods rather than replacing them. This means:

- `call_penalty_fn` uses the *last method* returned by `methods(fn)`.
- To test different penalty functions, **use different function names** 
  (e.g., `my_penalty1`, `my_penalty2`, `my_penalty3`) to avoid ambiguity.

# Examples

```julia
my_penalty(p, n) = 2 * p * log(n)

call_penalty_fn(my_penalty;
    p=3, n=250, CP=[50, 100], segment_lengths=[50, 50, 150], num_segments=3
)
# => 33.21
More complex example:
function imbalance_penalty(p, n, CP)
    seg_lengths = diff([0; CP; n])
    imbalance = std(seg_lengths)
    return 3.3 * p * length(CP) * log(n) + 0.12 * imbalance
end

call_penalty_fn(imbalance_penalty;
    p=3, n=250, CP=[60, 130], segment_lengths=[60, 70, 120], num_segments=3
)
If the user omits some arguments, only those required by their function are passed.
"""
function call_penalty_fn(fn::Function; kwargs...)
    ms = collect(methods(fn))
    argnames = method_argnames(last(ms))[2:end]
    args = Vector{Any}()
    for arg in argnames
        if haskey(kwargs, arg)
            push!(args, kwargs[arg])
        else
            error("Missing argument `$(arg)` needed by penalty function.")
        end
    end

    return fn(args...)  # Call with positional arguments
end

function method_argnames(m::Method)
    argnames = ccall(:jl_uncompress_argnames, Vector{Symbol}, (Any,), m.slot_syms)
    isempty(argnames) && return argnames
    return argnames[1:m.nargs]
end

#------------------------

# Example penalty functions

#------------------------

"""
default_penalty(p, n)

A basic penalty proportional to BIC.
"""
default_penalty(p, n) = 2 * p * log(n)

"""
BIC_penalty(p, n)

Bayesian Information Criterion-style penalty.
"""
BIC_penalty(p, n) = 100.0 * p * log(n)



