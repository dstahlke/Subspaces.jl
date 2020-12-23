Subspaces.jl - Vector subspaces in Julia
========================================

Subspaces.jl provides a Subspace type for representing linear subspaces of a finite
dimensional vector space.  Subspaces of matrices or tensors are also supported, as these
are vectors in the mathematical sense of the word.  Both real and complex base fields are
supported.

## Example

```
julia> using Subspaces

julia> S = Subspace([ [1,0,0], [0,1,1] ])
Subspace{Float64} size (3,) dim 2

julia> [1,2,2] ∈ S
true

julia> [1,1,2] ∈ S
false

julia> S | Subspace([ [0,0,1] ])
Subspace{Float64} size (3,) dim 3

julia> S & Subspace([ [1,1,1] ])
Subspace{Float64} size (3,) dim 1

julia> Subspace([ [1 0; 0 1] ])
Subspace{Float64} size (2, 2) dim 1
```
