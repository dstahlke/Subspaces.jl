module Subspaces

using DocStringExtensions

import Base.hcat, Base.vcat, Base.hvcat, Base.cat, Base.+, Base.*, Base.kron, Base.show, Base.iterate, Base.==, Base.in, Base.adjoint
import Base.|, Base.&, Base.~, Base./, Base.⊆, Base.⊇
import Base.size
using LinearAlgebra
using Convex

export Subspace, dim, each_basis_element
export random_subspace, random_hermitian_subspace, empty_subspace, full_subspace
export tobasis, frombasis, random_element, projection, perp
export hermitian_basis
export variable_in_space
export ⟂

default_tol = 1e-6

"""
    Subspace(basis::AbstractArray{AbstractArray{N}, 1}; tol::Real=$(default_tol))

Create a subspace from the given basis, expressed as a list of basis vectors.

    Subspace(basis::AbstractArray{N+1}; tol::Real=$(default_tol))

Create a subspace from a basis given as a multi-dimensional array.  The last array index
enumerates the basis elements.  E.g., if given a matrix this constructor will create a
subspace representing the column span of that matrix.

The `tol` parameter sets the tolerance for determining whether vectors are linearly
dependent.

$(TYPEDFIELDS)

```jldoctest
julia> Subspace([[1, 2, 3], [4, 5, 6]]) == Subspace([ 1 4; 2 5; 3 6])
true
```
"""
struct Subspace{T, N}
    """An orthonormal basis for this subspace.  The final index of this array indexes the
    basis vectors."""
    basis::Array # FIXME can it somehow be Array{T, N+1}?
    """An orthonormal basis for the perpendicular subspace."""
    perp::Array # FIXME can it somehow be Array{T, N+1}?
    """The tolerance for determining whether vectors are linearly dependent."""
    tol::AbstractFloat

    function Subspace(basis::AbstractArray{<:Number}; tol::Real=default_tol)
        shape = size(basis)[1:end-1]
        d = size(basis)[end]
        mat = reshape(basis, prod(shape), d)
        if true
            q = qr(mat, Val(true))
            take = [norm(x) >= tol for x in eachrow(q.R)]
            # This is the most efficient way to get the Q matrix.
            # Multiply by I is slow: https://github.com/JuliaLang/julia/issues/38972
            # Cast to Array changes size: https://github.com/JuliaLang/julia/issues/37102
            qQ = q.Q * Matrix(1.0*I, (size(q.Q)[2], size(q.Q)[2]))
            resize!(take, size(qQ)[2])
            good = qQ[:,take]
            perp = qQ[:,.!take]
        else
            s = svd(mat; full=true)
            lastgood = findlast(s.S .>= tol)
            if typeof(lastgood) == Nothing
                lastgood = 0
            end
            good = s.U[:,1:lastgood]
            perp = s.U[:,lastgood+1:end]
        end
        good = reshape(good, shape..., size(good)[2])
        perp = reshape(perp, shape..., size(perp)[2])
        return new{eltype(good), length(shape)}(good, perp, tol)
    end

    function Subspace(basis::AbstractArray{<:AbstractArray{<:Number}, 1}; tol::Real=default_tol)
        shape = size(basis[1])
        return Subspace(cat(basis...; dims=length(shape)+1); tol)
    end
end

function show(io::IO, S::Subspace)
    print(io, "Subspace{$(eltype(S.basis))} size $(size(S)) dim $(dim(S))")
end

"""
$(TYPEDSIGNATURES)

Returns the size of the elements of a subspace.

```jldoctest
julia> size(Subspace([ [1,2,3], [4,5,6] ]))
(3,)
```
"""
size(S::Subspace)::Tuple{Vararg{<:Integer}} = size(S.basis)[1:end-1]

"""
$(TYPEDSIGNATURES)

Returns the linear dimension of this subspace.

```jldoctest
julia> dim(Subspace([ [1,2,3], [4,5,6] ]))
2
```
"""
dim(S::Subspace)::Integer = size(S.basis)[end]

"""
$(TYPEDSIGNATURES)

Returns the orthogonal subspace.  Can also be written as `~S`.

```jldoctest
julia> S = Subspace([ [1,2,3], [4,5,6] ])
Subspace{Float64} size (3,) dim 2

julia> perp(S)
Subspace{Float64} size (3,) dim 1

julia> perp(S) == ~S
true

julia> perp(S) ⟂ S
true
```
"""
perp(S::Subspace)::Subspace = Subspace(S.perp, tol=S.tol)

"""
$(TYPEDSIGNATURES)

Returns the perpendicular subspace.  Can also be written as `perp(S)`.
"""
(~)(S::Subspace) = perp(S)

# It'd be nice for these to all take SubspaceOrArray but then our overloads seem to be selected even
# when all args are Array.
#const SubspaceOrArray{T, N} = Union{Subspace{T, N}, AbstractArray{T, N}}

"""
$(TYPEDSIGNATURES)

Create a generator that iterates over orthonormal basis vectors of a subspace.
"""
each_basis_element(S::Subspace) = eachslice(S.basis; dims=length(size(S.basis)))

each_basis_element(arr::AbstractArray) = [arr]

function each_basis_element_or_zero(S::Subspace{T, N}) where {T, N}
    if dim(S) == 0
        return [ zeros(T, size(S)) ]
    else
        return each_basis_element(S)
    end
end

"""
    +(a::Subspace, b::Subspace)
    +(a::AbstractArray, b::Subspace)
    +(a::Subspace, b::AbstractArray)
    +(a::UniformScaling, b::Subspace)
    +(a::Subspace, b::UniformScaling)

Linear span of two subspaces, or of a subspace an and array.  Equivalent to |(a, b).
"""
function +(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N}
    if size(a) != size(b)
        throw(DimensionMismatch("Array size mismatch: $(size(a)) vs $(size(b))"))
    end
    return Subspace(cat(a.basis, b.basis; dims=N+1), tol=max(a.tol, b.tol))
end

(+)(a::Subspace{T, N}, b::AbstractArray{U, N}) where {T,U,N} = a + Subspace([b], tol=a.tol)
(+)(a::AbstractArray{T, N}, b::Subspace{U, N}) where {T,U,N} = Subspace([a], tol=b.tol) + b
(+)(S::Subspace{T, 2}, x::UniformScaling) where T = S + Subspace([ Array{T}(I, size(S)) ], tol=S.tol)
(+)(x::UniformScaling, S::Subspace{T, 2}) where T = S + I

"""
    |(a::Subspace, b::Subspace)
    |(a::AbstractArray, b::Subspace)
    |(a::Subspace, b::AbstractArray)
    |(a::UniformScaling, b::Subspace)
    |(a::Subspace, b::UniformScaling)

Linear span of two subspaces, or of a subspace an and array.  Equivalent to |(a, b).
"""
(|)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = a + b
(|)(a::Subspace{T, N}, b::AbstractArray{U, N}) where {T,U,N} = a + b
(|)(a::AbstractArray{T, N}, b::Subspace{U, N}) where {T,U,N} = a + b
(|)(S::Subspace{T, 2}, x::UniformScaling) where T = S + I
(|)(x::UniformScaling, S::Subspace{T, 2}) where T = S + I

"""
    *(a::Subspace, b::Subspace)
    *(a::AbstractArray, b::Subspace)
    *(a::Subspace, b::AbstractArray)

Linear span of products of elements of space `a` with elements of space `b`.
"""
function *(a::Subspace, b::Subspace)
    #if dim(a) == 0 || dim(b) == 0
    #    return Subspace([ zeros(T, size(a)) * zeros(U, size(b)) ], tol=max(a.tol, b.tol))
    #else
        return Subspace([ x*y for x in each_basis_element_or_zero(a) for y in each_basis_element_or_zero(b) ], tol=max(a.tol, b.tol))
    #end
end

(*)(a::Subspace, b::AbstractArray) = a * Subspace([b], tol=a.tol)
(*)(a::AbstractArray, b::Subspace) = Subspace([a], tol=b.tol) * b

"""
$(TYPEDSIGNATURES)

Linear span of Kronecker products of elements of space `a` with elements of space `b`.
"""
function kron(a::Subspace{T,N}, b::Subspace{U,N}) where {T,U,N}
    return Subspace([
        kron(x, y)
        for x in each_basis_element_or_zero(a)
        for y in each_basis_element_or_zero(b)
    ], tol=max(a.tol, b.tol))
end

kron(a::Subspace, b::AbstractArray) = kron(a, Subspace([b], tol=a.tol))
kron(a::AbstractArray, b::Subspace) = kron(Subspace([a], tol=b.tol), b)

"""
$(TYPEDSIGNATURES)

Vertical concatenations of vector subspaces (direct sum).
"""
vcat(S::Subspace...) = cat(S...; dims=1)

"""
$(TYPEDSIGNATURES)

Horizontal concatenations of vector subspaces (direct sum).
"""
hcat(S::Subspace...) = cat(S...; dims=2)

"""
$(TYPEDSIGNATURES)

Concatenations of vector subspaces (direct sum).
"""
function cat(S::Subspace...; dims)
    n = length(S)
    # FIXME doesn't work well with heterogenous types
    #T = promote_type(map((x)->eltype(x.basis), [S...])...)
    Subspace([
        cat([ i==j ? x : zeros(size(S[i])) for i in 1:n ]...; dims=dims)
        for j in 1:n
        for x in each_basis_element_or_zero(S[j])
    ], tol=maximum([ s.tol for s in S ]))
end

"""
$(TYPEDSIGNATURES)

Concatenations of vector subspaces (direct sum).
"""
function hvcat(rows::Tuple{Vararg{Int}}, S::Subspace{T, N}...) where {T, N}
    n = length(S)
    basis = Array{Array{T, N}, 1}()
    for j in 1:n
        for x in each_basis_element_or_zero(S[j])
            push!(basis, hvcat(rows, [ i==j ? x : zeros(T, size(S[i])) for i in 1:n ]...))
        end
    end
    if isempty(basis)
        push!(basis, hvcat(rows, [ zeros(T, size(S[i])) for i in 1:n ]...))
    end
    return Subspace(basis, tol=maximum([ s.tol for s in S ]))
end

"""
$(TYPEDSIGNATURES)

Adjoint of vector subspace (linear span of adjoints of members of a space).
"""
adjoint(S::Subspace) =
    Subspace([ x' for x in each_basis_element_or_zero(S) ], tol=S.tol)

"""
$(TYPEDSIGNATURES)

Check for membership of a vector in a vector subspace.
"""
function in(x::AbstractArray{<:Number, N}, S::Subspace{<:Number, N}) where N
    return norm(x - projection(S, x)) <= S.tol
end

function in(x::UniformScaling, S::Subspace{T, 2}) where T
    return Matrix{T}(I, size(S)) in S
end

"""
$(TYPEDSIGNATURES)

Check whether `a` is a subspace of `b`.
"""
function in(a::Subspace{<:Number, N}, b::Subspace{<:Number, N}) where N
    shp = size(a)
    if dim(a) > dim(b)
        return false
    end
    Ma = reshape(a.basis, prod(shp), dim(a))
    Mb = reshape(b.basis, prod(shp), dim(b))
    s = svdvals(Mb' * Ma)
    tol = max(a.tol, b.tol)
    return all(s .> (1.0 - tol))
end

"""
$(TYPEDSIGNATURES)

Check whether two vector subspaces are equal.
"""
function ==(a::Subspace{<:Number, N}, b::Subspace{<:Number, N}) where N
    return dim(a) == dim(b) && a in b
end

"""
$(TYPEDSIGNATURES)

Intersection of vector subspaces.
"""
(&)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = perp(perp(a) + perp(b))

"""
$(TYPEDSIGNATURES)

Intersection of vector subspaces.
"""
(∩)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = a & b

"""
$(TYPEDSIGNATURES)

Quotient of vector subspaces.  Throws error unless `b ⊆ a`.

```jldoctest
julia> a = random_subspace(ComplexF64, 2, 10)
Subspace{Complex{Float64}} size (10,) dim 2

julia> b = random_subspace(ComplexF64, 3, 10)
Subspace{Complex{Float64}} size (10,) dim 3

julia> a / b
ERROR: ArgumentError: divisor must be a subspace of dividend for subspace quotient

julia> (a+b) / a
Subspace{Complex{Float64}} size (10,) dim 3

julia> (a+b) / a == Subspace([ projection(~a, x) for x in each_basis_element(b) ])
true

julia> ((a+b) / b) ⟂ b
true

```
"""
function (/)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N}
    if !(b ⊆ a)
        throw(ArgumentError("divisor must be a subspace of dividend for subspace quotient"))
    end
    return perp(perp(a) + b)
end

(/)(a::Subspace{T, N}, b::AbstractArray{U, N}) where {T,U,N} = a / Subspace([b], tol=a.tol)
(/)(a::Subspace{T, 2}, b::UniformScaling) where T = a / Array{T}(I, size(a))

"""
$(TYPEDSIGNATURES)

Check whether `a` is a subspace of `b`.
"""
(⊆)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = a in b

"""
$(TYPEDSIGNATURES)

Check whether `b` is a subspace of `a`.
"""
(⊇)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = b in a

"""
$(TYPEDSIGNATURES)

Check whether `a` is orthogonal to `b`.
"""
(⟂)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = a ⊆ perp(b)

################
### Math
################

"""
$(TYPEDSIGNATURES)

Returns the basis components of `x` in the basis `S.basis`.

See also: [`frombasis`](@ref).

```jldoctest
julia> S = random_subspace(ComplexF64, 2, 10)
Subspace{Complex{Float64}} size (10,) dim 2

julia> x = randn(10);

julia> size(tobasis(S, x))
(2,)

julia> norm( frombasis(S, tobasis(S, x)) - projection(S, x) ) < 1e-8
true
```
"""
function tobasis(S::Subspace{<:Number, N}, x::AbstractArray{<:Number, N}) where N
    shp = size(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return basis_mat' * vec(x)
end

"""
$(TYPEDSIGNATURES)

Returns a vector from the given basis components in the basis `S.basis`.

See also: [`tobasis`](@ref).

```jldoctest
julia> S = random_subspace(ComplexF64, 2, 10)
Subspace{Complex{Float64}} size (10,) dim 2

julia> x = randn(2);

julia> size(frombasis(S, x))
(10,)

julia> norm( tobasis(S, frombasis(S, x)) - x ) < 1e-8
true
```
"""
function frombasis(S::Subspace, x::AbstractArray{<:Number, 1})
    shp = size(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return reshape(basis_mat * x, shp)
end

"""
$(TYPEDSIGNATURES)

Projection of vector `x` onto suspace `S`.
"""
function projection(S::Subspace{<:Number, N}, x::AbstractArray{<:Number, N}) where N
    return frombasis(S, tobasis(S, x))
end

"""
$(TYPEDSIGNATURES)

Returns a random element of a subspace.
"""
random_element(S::Subspace{T}) where T <: Number = frombasis(S, randn(T, dim(S)))

################
### Constructors
################

"""
$(TYPEDSIGNATURES)

Random dimension-`d` subspace of dimension-`siz` vector space, on base field `T`.

```jldoctest
julia> S = random_subspace(ComplexF64, 2, (3, 4))
Subspace{Complex{Float64}} size (3, 4) dim 2
```
"""
function random_subspace(T::Type, d::Int, siz, tol=default_tol)
    if d < 0
        throw(ArgumentError("subspace dimension was negative: $d"))
    elseif d == 0
        return empty_subspace(siz)
    else
        b = [ randn(T, siz) for i in 1:d ]
        return Subspace(b, tol=tol)
    end
end

"""
$(TYPEDSIGNATURES)

Random dimension-`d` subspace of `n × n` matrices on base field `T`, satisfying
`x ∈ S => x' ∈ S`.

```jldoctest
julia> S = random_hermitian_subspace(ComplexF64, 2, 3)
Subspace{Complex{Float64}} size (3, 3) dim 2

julia> S == S'
true
```
"""
function random_hermitian_subspace(T::Type, d::Int, n::Int, tol=default_tol)
    if d < 0
        throw(ArgumentError("subspace dimension was negative: $d"))
    elseif d == 0
        return empty_subspace((n, n))
    else
        b = [ randn(T, n, n) for i in 1:d ]
        b = [ x + x' for x in b ]
        return Subspace(b, tol=tol)
    end
end

"""
$(TYPEDSIGNATURES)

Create an empty subspace of dimension-`siz` vector space, on base field `T`.

```jldoctest
julia> S = empty_subspace(ComplexF64, (3, 4))
Subspace{Complex{Float64}} size (3, 4) dim 0
```
"""
empty_subspace(T::Type, siz::Tuple, tol=default_tol) = Subspace([zeros(T, siz)], tol=tol)

"""
$(TYPEDSIGNATURES)

Create an full subspace of dimension-`siz` vector space, on base field `T`.

```jldoctest
julia> S = full_subspace(ComplexF64, (3, 4))
Subspace{Complex{Float64}} size (3, 4) dim 12
```
"""
full_subspace(T::Type, siz::Tuple) = perp(empty_subspace(T, siz))

#############
### Hermitian
#############

function hermit_to_vec(M::AbstractArray{Complex{T}, 2}) where T
    n = size(M)[1]
    size(M)[2] == n || throw(ArgumentError("matrix was not square: $(size(M))"))
    v = zeros(T, n*n)
    k = 0
    for i in 1:n
        v[k+1:k+i] = real(M[1:i, i])
        k += i
        v[k+1:k+i-1] = imag(M[1:i-1, i])
        k += i-1
    end
    return v
end

function vec_to_hermit(v::AbstractArray{T, 1}, n::Integer) where T
    @assert n*n == size(v)[1]
    M = zeros(Complex{T}, n, n)
    k = 0
    for i in 1:n
        M[1:i, i] = v[k+1:k+i]
        M[i, 1:i] = v[k+1:k+i]
        k += i
        M[1:i-1, i] += v[k+1:k+i-1] * 1im
        M[i, 1:i-1] -= v[k+1:k+i-1] * 1im
        k += i-1
    end
    return Hermitian(M)
end

"""
$(TYPEDSIGNATURES)

Given a subspace satisfying `S == S'`, returns a basis in which each basis element is Hermitian.
"""
function hermitian_basis(S::Subspace{Complex{T}})::Array{Hermitian{Complex{T},Array{Complex{T},2}},1} where T
    if dim(S) == 0
        return []
    end
    n = size(S)[1]
    size(S)[2] == n || throw(ArgumentError("subspace size was not square: $(size(S))"))
    M = hcat(
        [ hermit_to_vec( x     + x'     ) for x in each_basis_element(S) ]...,
        [ hermit_to_vec((x*1im)+(x*1im)') for x in each_basis_element(S) ]...
    )
    hb = [ vec_to_hermit(x, n) for x in each_basis_element(Subspace(M)) ]
    @assert (Subspace(hb, tol=S.tol) == S) "Hermitian basis didn't equal original space"
    return hb
end

#########################
### Support for Convex.jl
#########################

"""
$(TYPEDSIGNATURES)

Returns the basis components of `x` in the basis `S.basis`.
"""
function tobasis(S::Subspace{<:Number, N}, x::Convex.AbstractExpr) where N
    shp = size(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return basis_mat' * vec(x)
end

"""
$(TYPEDSIGNATURES)

Returns a vector from the given basis components in the basis `S.basis`.
"""
function frombasis(S::Subspace{<:Number, 1}, x::Convex.AbstractExpr)
    shp = size(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return basis_mat * x
end

"""
$(TYPEDSIGNATURES)

Returns a vector from the given basis components in the basis `S.basis`.
"""
function frombasis(S::Subspace{<:Number, 2}, x::Convex.AbstractExpr)
    shp = size(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return reshape(basis_mat * x, shp...)
end

"""
$(TYPEDSIGNATURES)

Projection of vector `x` onto suspace `S`.
"""
function projection(S::Subspace{<:Number, N}, x::Convex.AbstractExpr) where N
    return frombasis(S, tobasis(S, x))
end

"""
$(TYPEDSIGNATURES)

Constrains the variable `x` to be in the subspace `S`.
"""
function in(x::Convex.AbstractExpr, S::Subspace{<:Number, N}) where N
    return tobasis(perp(S), x) == 0
end

"""
$(TYPEDSIGNATURES)

Creates a `Convex.jl` variable ranging over the given subspace.
"""
function variable_in_space(S::Subspace{<:Complex{<:Real}, N}) where N
    x = Convex.ComplexVariable(dim(S))
    return frombasis(S, x)
end

"""
$(TYPEDSIGNATURES)

Creates a `Convex.jl` variable ranging over the given subspace.
"""
function variable_in_space(S::Subspace{<:Real, N}) where N
    x = Convex.Variable(dim(S))
    return frombasis(S, x)
end

end # module
