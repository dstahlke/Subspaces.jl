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

julia> S.basis
3×2 Array{Float64,2}:
  0.0       1.0
 -0.707107  0.0
 -0.707107  0.0

julia> S | Subspace([ [0,0,1] ])
Subspace{Float64} size (3,) dim 3

julia> S & Subspace([ [1,1,1] ])
Subspace{Float64} size (3,) dim 1

julia> kron(S, S)
Subspace{Float64} size (9,) dim 4

julia> S * S'
Subspace{Float64} size (3, 3) dim 4

julia> Subspace([ ones(3,4,5) ])
Subspace{Float64} size (3, 4, 5) dim 1
```
