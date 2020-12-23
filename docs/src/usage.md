Usage
=====

## Constructors

| Function                                        | Description                                                                |
| :--------                                       | :-----------                                                               |
| `Subspace(basis::Array{<:Number, N+1})`         | Subspace from basis, where last index of array indexes the basis elements. |
| `Subspace(basis::Array{Array{<:Number, N}, 1})` | Subspace from list of basis elements.                                      |
| `random_subspace(T, dim, size)`                 | Random subspace of the given size and dimension, on base field `T`.        |
| `random_hermitian_subspace`                     | Random subspace satisfying `S == S'`                                       |
| `empty_subspace`                                | Empty subspace                                                             |
| `full_subspace`                                 | Full subspace                                                              |

## Operations

| Function                   | Description                                                                                                   |
| :--------                  | :-----------                                                                                                  |
| `size(S)`                  | Size of the elements of `S`.                                                                                  |
| `dim(S)`                   | Dimension of subspace `S`.                                                                                    |
| `S.basis`                  | Array representing an orthonormal basis of `S`, with the last index of the array indexing the basis elements. |
| `each_basis_element(S)`    | Generator for iterating over orthonormal basis elements.                                                      |
| `random_element(S)`        | Returns a random element of subspace `S`.                                                                     |
| `perp(S)`, `~S`            | Perpendicular (orthogonal) subspace.                                                                          |
| `tobasis(S, x)`            | Transform a vector into basis coordinates.  Projects `x` onto `S` if `x` is not an element of `S`.            |
| `frombasis(S, x)`          | Returns a vector given the basis coordinates.                                                                 |
| `projection(S, x)`         | Projects vector `x` onto subspace `S`.                                                                        |
| `hermitian_basis(S)`       | For a subspace satisfying `S == S'`, returns a basis consisting of Hermitian operators.                       |
| `hcat(S1, S2, ...)`        | Direct sum of subspaces.                                                                                      |
| `vcat(S1, S2, ...)`        | Direct sum of subspaces.                                                                                      |
| `hvcat(rows, S1, S2, ...)` | Direct sum of subspaces.                                                                                      |
| `cat(S1, S2, ...; dims)`   | Direct sum of subspaces.                                                                                      |
| `kron(S, T)`               | Direct product of subspaces.                                                                                  |
| `adjoint(S)`, `S'`         | Subspace consisting of adjoints of vectors, $\{ x' : x \in S \}$.                                             |
| `S ⟂ T`                    | Check whether subspace `S` is orthogonal to `T`.                                                              |
| `S == T`                   | Check for equality of subspaces.                                                                              |
| `x in S`, `x ∈ S`          | Check membership of `x` in subspace `S`.                                                                      |
| `S ⊆ T`                    | Check whether subspace `S` is contained in `T`.                                                               |
| `S ⊇ T`                    | Check whether subspace `S` contains `T`.                                                                      |
| `S + T`, `S &#124; T`      | Linear span of union of subspaces, $\textrm{span}\{ x y : x \in S, y \in T \}$.                               |
| `S & T`, `S ∩ T`           | Intersection of subspaces.                                                                                    |
| `S * T`                    | Linear span of products of elements of `S` and of `T`.                                                        |
| `S / T`                    | Vector space quotient.  Requires `T ⊆ S`.                                                                     |

## Convex.jl integration

| Function               | Description                                                        |
| :--------              | :-----------                                                       |
| `variable_in_space(S)` | Creates a variable constrained to space `S`.                       |
| `tobasis(S, x)`        | Transform variale from vector to basis coordinates.                |
| `frombasis(S, x)`      | Transform variale from basis coordinates to vector.                |
| `x in S`               | Creates a constraint requiring variable `x` to be in subspace `S`. |

As an example, here is how one could do a positive semidefinite matrix completion.

```jldoctest psdcompletion
using Convex, SCS, Subspaces

n = 3
A = [ 1 2 0
      0 0 6
      0 0 9 ]

basis_element(x, y) = [ i==x && j == y for i in 1:n, j in 1:n ]

S = Subspace([ basis_element(I[1], I[2]) for I in findall(A .== 0) ])

X = variable_in_space(S)

problem = minimize(tr(X), [ X+A ⪰ 0 ])
solve!(problem, () -> SCS.Optimizer(verbose=0, eps=1e-9))
evaluate(X+A)

# output

3×3 Array{Float64,2}:
 1.0  2.0  3.0
 2.0  4.0  6.0
 3.0  6.0  9.0
```

Alternately, instead of `X = variable_in_space(S)` we could use `X ∈ S`.

```jldoctest psdcompletion
X = Variable(n, n)
problem = minimize(tr(X), [ X ∈ S, X+A ⪰ 0 ])

solve!(problem, () -> SCS.Optimizer(verbose=0, eps=1e-9))
evaluate(X+A)

# output

3×3 Array{Float64,2}:
 1.0  2.0  3.0
 2.0  4.0  6.0
 3.0  6.0  9.0
```
