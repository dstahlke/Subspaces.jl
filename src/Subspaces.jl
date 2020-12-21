module Subspaces

using DocStringExtensions

# FIXME all functions need to propagate tol
# FIXME rename shape to size

import Base.hcat, Base.vcat, Base.hvcat, Base.cat, Base.+, Base.*, Base.kron, Base.show, Base.iterate, Base.==, Base.in, Base.adjoint
import Base.|, Base.&, Base.~, Base./, Base.⊆
using LinearAlgebra
using Convex

export Subspace, shape, dim, each_basis_element
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
        s = svd(mat; full=true)
        lastgood = findlast(s.S .>= tol)
        if typeof(lastgood) == Nothing
            lastgood = 0
        end
        good = reshape(s.U[:,1:lastgood], shape..., lastgood)
        perp = reshape(s.U[:,lastgood+1:end], shape..., prod(shape)-lastgood)
        return new{eltype(good), length(shape)}(good, perp, tol)
    end

    function Subspace(basis::AbstractArray{<:AbstractArray{<:Number}, 1}; tol::Real=default_tol)
        shape = size(basis[1])
        return Subspace(cat(basis...; dims=length(shape)+1); tol)
    end
end

function show(io::IO, S::Subspace)
    print(io, "Subspace{$(eltype(S.basis))} shape $(shape(S)) dim $(dim(S))")
end

"""
$(TYPEDSIGNATURES)
Returns the size of the elements of a subspace.

```jldoctest
julia> shape(Subspace([ [1,2,3], [4,5,6] ]))
(3,)
```
"""
shape(S::Subspace)::Tuple{Integer} = size(S.basis)[1:end-1]

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
Returns the perpendicular subspace.  Can also be written as ~S.

```jldoctest
julia> S = Subspace([ [1,2,3], [4,5,6] ])
Subspace{Float64} shape (3,) dim 2

julia> perp(S)
Subspace{Float64} shape (3,) dim 1

julia> perp(S) == ~S
true

julia> perp(S) ⟂ S
true
```
"""
perp(S::Subspace)::Subspace = Subspace(S.perp)

# It'd be nice for these to all take SubspaceOrArray but then our overloads seem to be selected even
# when all args are Array.
#const SubspaceOrArray{T, N} = Union{Subspace{T, N}, AbstractArray{T, N}}

each_basis_element(S::Subspace) = eachslice(S.basis; dims=length(size(S.basis)))

each_basis_element(arr::AbstractArray) = [arr]

function each_basis_element_or_zero(S::Subspace{T, N}) where {T, N}
    if dim(S) == 0
        return [ zeros(T, shape(S)) ]
    else
        return each_basis_element(S)
    end
end

raw"""
    +(a::Subspace, b::Subspace)
    +(a::AbstractArray, b::Subspace)
    +(a::Subspace, b::AbstractArray)
    +(a::UniformScaling, b::Subspace)
    +(a::Subspace, b::UniformScaling)

Linear span of two subspaces, or of a subspace an and array.  Equivalent to |(a, b).
"""
function +(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N}
    if shape(a) != shape(b)
        throw(DimensionMismatch("Array shape mismatch: $(shape(a)) vs $(shape(b))"))
    end
    return Subspace(cat(a.basis, b.basis; dims=N+1))
end

function *(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N}
    #if dim(a) == 0 || dim(b) == 0
    #    return Subspace([ zeros(T, shape(a)) * zeros(U, shape(b)) ])
    #else
        return Subspace([ x*y for x in each_basis_element_or_zero(a) for y in each_basis_element_or_zero(b) ])
    #end
end

function kron(a::Subspace{T,N}, b::Subspace{U,N}) where {T,U,N}
    return Subspace([
        kron(x, y)
        for x in each_basis_element_or_zero(a)
        for y in each_basis_element_or_zero(b)
    ])
end

vcat(S::Subspace...) = cat(S...; dims=1)

hcat(S::Subspace...) = cat(S...; dims=2)

function cat(S::Subspace...; dims)
    n = length(S)
    # FIXME doesn't work well with heterogenous types
    #T = promote_type(map((x)->eltype(x.basis), [S...])...)
    Subspace([
        cat([ i==j ? x : zeros(shape(S[i])) for i in 1:n ]...; dims=dims)
        for j in 1:n
        for x in each_basis_element_or_zero(S[j])
    ])
end

function hvcat(rows::Tuple{Vararg{Int}}, S::Subspace{T, N}...) where {T, N}
    n = length(S)
    basis = Array{Array{T, N}, 1}()
    for j in 1:n
        for x in each_basis_element_or_zero(S[j])
            push!(basis, hvcat(rows, [ i==j ? x : zeros(T, shape(S[i])) for i in 1:n ]...))
        end
    end
    if isempty(basis)
        push!(basis, hvcat(rows, [ zeros(T, shape(S[i])) for i in 1:n ]...))
    end
    return Subspace(basis)
end

adjoint(S::Subspace) =
    Subspace([ x' for x in each_basis_element_or_zero(S) ])

function in(x::UniformScaling, S::Subspace{T, 2}) where T
    return Matrix{T}(I, shape(S)) in S
end

function in(x::AbstractArray{<:Number, N}, S::Subspace{<:Number, N}) where N
    return norm(x - projection(S, x)) <= S.tol
end

function in(a::Subspace{<:Number, N}, b::Subspace{<:Number, N}) where N
    shp = shape(a)
    if dim(a) > dim(b)
        return false
    end
    Ma = reshape(a.basis, prod(shp), dim(a))
    Mb = reshape(b.basis, prod(shp), dim(b))
    s = svdvals(Mb' * Ma)
    tol = max(a.tol, b.tol)
    return all(s .> (1.0 - tol))
end

function ==(a::Subspace{<:Number, N}, b::Subspace{<:Number, N}) where N
    return dim(a) == dim(b) && a in b
end

(+)(a::Subspace{T, N}, b::AbstractArray{U, N}) where {T,U,N} = a + Subspace([b])
(+)(a::AbstractArray{T, N}, b::Subspace{U, N}) where {T,U,N} = Subspace([a]) + b
(+)(S::Subspace{T, 2}, x::UniformScaling) where T = S + Subspace([ Array{T}(I, shape(S)) ])
(+)(x::UniformScaling, S::Subspace{T, 2}) where T = S + I

(*)(a::Subspace, b::AbstractArray) = a * Subspace([b])
(*)(a::AbstractArray, b::Subspace) = Subspace([a]) * b

(|)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = a + b
(|)(a::Subspace{T, N}, b::AbstractArray{U, N}) where {T,U,N} = a + b
(|)(a::AbstractArray{T, N}, b::Subspace{U, N}) where {T,U,N} = a + b

(~)(S::Subspace) = perp(S)

(&)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = perp(perp(a) + perp(b))

kron(a::Subspace, b::AbstractArray) = kron(a, Subspace([b]))
kron(a::AbstractArray, b::Subspace) = kron(Subspace([a]), b)

function (/)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N}
    if !(b ⊆ a)
        throw(ArgumentError("divisor must be a subspace of dividend for subspace quotient"))
    end
    return perp(perp(a) + b)
end

(/)(a::Subspace{T, N}, b::AbstractArray{U, N}) where {T,U,N} = a / Subspace([b])
(/)(a::Subspace{T, 2}, b::UniformScaling) where T = a / Array{T}(I, shape(a))

(⊆)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = a in b
(⊇)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = b in a

(⟂)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = a ⊆ perp(b)

################
### Math
################

function tobasis(S::Subspace{<:Number, N}, x::AbstractArray{<:Number, N}) where N
    shp = shape(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return basis_mat' * vec(x)
end

function frombasis(S::Subspace, x::AbstractArray{<:Number, 1})
    shp = shape(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return reshape(basis_mat * x, shp)
end

function projection(S::Subspace{<:Number, N}, x::AbstractArray{<:Number, N}) where N
    return frombasis(S, tobasis(S, x))
end

random_element(S::Subspace{T}) where T <: Number = frombasis(S, randn(T, dim(S)))

################
### Constructors
################

function random_subspace(T::Type, d::Int, dims)
    if d < 0
        throw(ArgumentError("subspace dimension was negative: $d"))
    elseif d == 0
        return empty_subspace(dims)
    else
        b = [ randn(T, dims) for i in 1:d ]
        return Subspace(b)
    end
end

function random_hermitian_subspace(T::Type, d::Int, n::Int)
    if d < 0
        throw(ArgumentError("subspace dimension was negative: $d"))
    elseif d == 0
        return empty_subspace((n, n))
    else
        b = [ randn(T, n, n) for i in 1:d ]
        b = [ x + x' for x in b ]
        return Subspace(b)
    end
end

empty_subspace(T::Type, dims::Tuple) = Subspace([zeros(T, dims)])

full_subspace(T::Type, dims::Tuple) = perp(empty_subspace(T, dims))

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

function hermitian_basis(S::Subspace{Complex{T}})::Array{Hermitian{Complex{T},Array{Complex{T},2}},1} where T
    if dim(S) == 0
        return []
    end
    n = shape(S)[1]
    shape(S)[2] == n || throw(ArgumentError("subspace shape was not square: $(shape(S))"))
    M = hcat(
        [ hermit_to_vec( x     + x'     ) for x in each_basis_element(S) ]...,
        [ hermit_to_vec((x*1im)+(x*1im)') for x in each_basis_element(S) ]...
    )
    hb = [ vec_to_hermit(x, n) for x in each_basis_element(Subspace(M)) ]
    @assert (Subspace(hb) == S) "Hermitian basis didn't equal original space"
    return hb
end

#########################
### Support for Convex.jl
#########################

function tobasis(S::Subspace{<:Number, N}, x::Convex.AbstractExpr) where N
    shp = shape(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return basis_mat' * vec(x)
end

function frombasis(S::Subspace{<:Number, 1}, x::Convex.AbstractExpr)
    shp = shape(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return basis_mat * x
end

function frombasis(S::Subspace{<:Number, 2}, x::Convex.AbstractExpr)
    shp = shape(S)
    basis_mat = reshape(S.basis, prod(shp), dim(S))
    return reshape(basis_mat * x, shp...)
end

function projection(S::Subspace{<:Number, N}, x::Convex.AbstractExpr) where N
    return frombasis(S, tobasis(S, x))
end

function in(x::Convex.AbstractExpr, S::Subspace{<:Number, N}) where N
    return tobasis(perp(S), x) == 0
end

function variable_in_space(S::Subspace{<:Complex{<:Real}, N}) where N
    x = Convex.ComplexVariable(dim(S))
    return frombasis(S, x)
end

function variable_in_space(S::Subspace{<:Real, N}) where N
    x = Convex.Variable(dim(S))
    return frombasis(S, x)
end

end # module
