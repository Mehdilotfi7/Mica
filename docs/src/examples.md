# Examples

```@example 
using TSCPDetector
extract_parameters([1, 2, 3, 4, 5], 1, 2)

```

```@example ex1
using TSCPDetector
pars = extract_parameters([1, 2, 3, 4, 5], 1, 2)

```

```@example ex1
using TSCPDetector # hide
global_pars = pars[1]
segment_pars = pars[2]

```