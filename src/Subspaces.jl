module Subspaces

import Base.hcat, Base.vcat, Base.hvcat, Base.cat, Base.+, Base.*, Base.kron, Base.show, Base.iterate, Base.==, Base.in, Base.adjoint
import Base.|, Base.&, Base.~
using LinearAlgebra

export Subspace, shape, dim, each_basis_element
export random_subspace, random_hermitian_subspace, empty_subspace, full_subspace
export tobasis, frombasis, random_element, projection, perp

struct Subspace{T, N}
    basis::AbstractArray
    perp::AbstractArray
    tol::AbstractFloat

    function Subspace(basis::AbstractArray{<:Number})
        shape = size(basis)[1:end-1]
        d = size(basis)[end]
        mat = reshape(basis, prod(shape), d)
        s = svd(mat; full=true)
        tol = 1e-6
        lastgood = findlast(s.S .>= tol)
        if typeof(lastgood) == Nothing
            lastgood = 0
        end
        good = reshape(s.U[:,1:lastgood], shape..., lastgood)
        perp = reshape(s.U[:,lastgood+1:end], shape..., prod(shape)-lastgood)
        return new{eltype(good), length(shape)}(good, perp, tol)
    end

    function Subspace(basis::AbstractArray{<:AbstractArray{<:Number}, 1})
        shape = size(basis[1])
        return Subspace(cat(basis...; dims=length(shape)+1))
    end
end

function show(io::IO, ss::Subspace)
    print(io, "Subspace{$(eltype(ss.basis))} shape $(shape(ss)) dim $(dim(ss))")
end

shape(ss::Subspace) = size(ss.basis)[1:end-1]

dim(ss::Subspace) = size(ss.basis)[end]

perp(ss::Subspace) = Subspace(ss.perp)

const SubspaceOrArray = Union{Subspace, AbstractArray}

each_basis_element(ss::Subspace) = eachslice(ss.basis; dims=length(size(ss.basis)))

each_basis_element(arr::AbstractArray) = [arr]

# FIXME all these need to propagate tol

# FIXME take SubspaceOrArray
# FIXME try this instead of da=...
#function +(a::Subspace{Any, N}, b::Subspace{Any, N}) where N
function +(a::Subspace, b::Subspace)
    da = length(size(a.basis))
    db = length(size(b.basis))
    if da != db
        throw(DimensionMismatch("Array rank mismatch: $da vs $db"))
    end
    return Subspace(cat(a.basis, b.basis; dims=da))
end

# It'd be nice for these to all take SubspaceOrArray but then our overloads seem to be selected even
# when all args are Array.

*(a::Subspace, b::Subspace) =
    Subspace([ x*y for x in each_basis_element(a) for y in each_basis_element(b) ])

kron(a::Subspace, b::Subspace) =
    Subspace([ kron(x, y) for x in each_basis_element(a) for y in each_basis_element(b) ])

vcat(ss::Subspace...) = cat(ss...; dims=1)

hcat(ss::Subspace...) = cat(ss...; dims=2)

function cat(ss::Subspace...; dims)
    n = length(ss)
    Subspace([
        cat([ i==j ? x : zeros(shape(ss[i])) for i in 1:n ]...; dims=dims)
        for j in 1:n
        for x in each_basis_element(ss[j])
    ])
end

function hvcat(rows::Tuple{Vararg{Int}}, ss::Subspace...)
    n = length(ss)
    Subspace([
        hvcat(rows, [ i==j ? x : zeros(shape(ss[i])) for i in 1:n ]...)
        for j in 1:n
        for x in each_basis_element(ss[j])
    ])
end

adjoint(ss::Subspace{<:Number, 2}) =
    Subspace([ x' for x in each_basis_element(ss) ])

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

(+)(a::Subspace, b::AbstractArray) = a + Subspace([b])
(+)(a::AbstractArray, b::Subspace) = Subspace([a]) + b

(*)(a::Subspace, b::AbstractArray) = a * Subspace([b])
(*)(a::AbstractArray, b::Subspace) = Subspace([a]) * b

(|)(a::Subspace, b::Subspace) = a + b
(|)(a::Subspace, b::AbstractArray) = a + b
(|)(a::AbstractArray, b::Subspace) = a + b

(~)(ss::Subspace) = perp(ss)

(&)(a::Subspace, b::Subspace) = perp(perp(a) + perp(b))

kron(a::Subspace, b::AbstractArray) = kron(a, Subspace([b]))
kron(a::AbstractArray, b::Subspace) = kron(Subspace([a]), b)

################
### Math
################

function tobasis(ss::Subspace{<:Number, N}, x::AbstractArray{<:Number, N}) where N
    shp = shape(ss)
    basis_mat = reshape(ss.basis, prod(shp), dim(ss))
    return basis_mat' * reshape(x, prod(shp))
end

function frombasis(ss::Subspace, x::AbstractArray{<:Number, 1})
    shp = shape(ss)
    basis_mat = reshape(ss.basis, prod(shp), dim(ss))
    return reshape(basis_mat * x, shp)
end

function projection(ss::Subspace{<:Number, N}, x::AbstractArray{<:Number, N}) where N
    return frombasis(ss, tobasis(ss, x))
end

random_element(ss::Subspace{T}) where T <: Number = frombasis(ss, randn(T, dim(ss)))

################
### Constructors
################

# FIXME take datatype parameter
# FIXME should use a unitarily invariant distribution
function random_subspace(d::Int, dims)
    b = [ randn(ComplexF64, dims) for i in 1:d ]
    return Subspace(b)
end

# FIXME take datatype parameter
# FIXME should use a unitarily invariant distribution
function random_hermitian_subspace(d::Int, n::Int)
    b = [ randn(ComplexF64, n, n) for i in 1:d ]
    b = [ x + x' for x in b ]
    return Subspace(b)
end

# FIXME take datatype parameter
empty_subspace(dims::Tuple) = Subspace([zeros(dims)])

# FIXME take datatype parameter
full_subspace(dims::Tuple) = perp(empty_subspace(dims))

end # module
