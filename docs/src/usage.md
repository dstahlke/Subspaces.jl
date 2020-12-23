Usage
=====

TODO

## Constructors

| Function                    | Description |
| --------                    | ----------- |
| `Subspace`                  |             |
| `random_subspace`           |             |
| `random_hermitian_subspace` |             |
| `empty_subspace`            |             |
| `full_subspace`             |             |

## Operations

| Function                   | Description |
| --------                   | ----------- |
| `shape(S)`                 |             |
| `dim(S)`                   |             |
| `S.basis`                  |             |
| `each_basis_element(S)`    |             |
| `random_element`           |             |
| `perp(S)`, `~S`            |             |
| `tobasis(S, x)`            |             |
| `frombasis(S, x)`          |             |
| `projection(S, x)`         |             |
| `hermitian_basis(S)`       |             |
| `S ⟂ T`                    |             |
| `hcat(S1, S2, ...)`        |             |
| `vcat(S1, S2, ...)`        |             |
| `hvcat(rows, S1, S2, ...)` |             |
| `cat(S1, S2, ...; dims)`   |             |
| `S + T`                    |             |
| `S * T`                    |             |
| `kron(S, T)`               |             |
| `S == T`                   |             |
| `x in S`, `x ∈ S`          |             |
| `adjoint(S)`               |             |
| `S & T`                    |             |
| `S / T`                    |             |
| `S ⊆ T`                    |             |
| `S ⊇ T`                    |             |

## Convex.jl integration

| Function               | Description |
| `variable_in_space(S)` |             |
| `tobasis(S, x)`        |             |
| `frombasis(S, x)`      |             |
| `x in S`               |             |
