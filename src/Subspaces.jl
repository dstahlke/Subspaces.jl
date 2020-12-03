module Subspaces

# FIXME support real space of complex arrays, e.g., a space of Hermitian matrices
# FIXME could use promote_rule to promote Array to Subspace
#       https://erik-engheim.medium.com/defining-custom-units-in-julia-and-python-513c34a4c971

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

struct Subspace{T, N}
    basis::AbstractArray
    perp::AbstractArray
    tol::AbstractFloat

    function Subspace(basis::AbstractArray{<:Number}; tol=1e-6)
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

    function Subspace(basis::AbstractArray{<:AbstractArray{<:Number}, 1}; tol=1e-6)
        shape = size(basis[1])
        return Subspace(cat(basis...; dims=length(shape)+1); tol)
    end
end

function show(io::IO, ss::Subspace)
    print(io, "Subspace{$(eltype(ss.basis))} shape $(shape(ss)) dim $(dim(ss))")
end

shape(ss::Subspace) = size(ss.basis)[1:end-1]

dim(ss::Subspace) = size(ss.basis)[end]

perp(ss::Subspace) = Subspace(ss.perp)

const SubspaceOrArray{T, N} = Union{Subspace{T, N}, AbstractArray{T, N}}

each_basis_element(ss::Subspace) = eachslice(ss.basis; dims=length(size(ss.basis)))

each_basis_element(arr::AbstractArray) = [arr]

function each_basis_element_or_zero(ss::Subspace{T, N}) where {T, N}
    if dim(ss) == 0
        return [ zeros(T, shape(ss)) ]
    else
        return each_basis_element(ss)
    end
end

# FIXME all these need to propagate tol

# FIXME all functions should use T,U,N where possible
function +(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N}
    if shape(a) != shape(b)
        throw(DimensionMismatch("Array shape mismatch: $(shape(a)) vs $(shape(b))"))
    end
    return Subspace(cat(a.basis, b.basis; dims=N+1))
end

# It'd be nice for these to all take SubspaceOrArray but then our overloads seem to be selected even
# when all args are Array.

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

vcat(ss::Subspace...) = cat(ss...; dims=1)

hcat(ss::Subspace...) = cat(ss...; dims=2)

function cat(ss::Subspace...; dims)
    n = length(ss)
    Subspace([
        cat([ i==j ? x : zeros(shape(ss[i])) for i in 1:n ]...; dims=dims)
        for j in 1:n
        for x in each_basis_element_or_zero(ss[j])
    ])
end

function hvcat(rows::Tuple{Vararg{Int}}, ss::Subspace{T, N}...) where {T, N}
    n = length(ss)
    basis = Array{Array{T, N}, 1}()
    for j in 1:n
        for x in each_basis_element_or_zero(ss[j])
            push!(basis, hvcat(rows, [ i==j ? x : zeros(T, shape(ss[i])) for i in 1:n ]...))
        end
    end
    if isempty(basis)
        push!(basis, hvcat(rows, [ zeros(T, shape(ss[i])) for i in 1:n ]...))
    end
    return Subspace(basis)
end

adjoint(ss::Subspace) =
    Subspace([ x' for x in each_basis_element_or_zero(ss) ])

function in(x::UniformScaling, ss::Subspace{T, 2}) where T
    return Matrix{T}(I, shape(ss)) in ss
end

function in(x::AbstractArray{<:Number, N}, ss::Subspace{<:Number, N}) where N
    return norm(x - projection(ss, x)) <= ss.tol
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

(+)(a::Subspace, b::AbstractArray) = a + Subspace([b])
(+)(a::AbstractArray, b::Subspace) = Subspace([a]) + b
(+)(ss::Subspace{T, 2}, x::UniformScaling) where T = ss + Subspace([ Array{T}(I, shape(ss)) ])
(+)(x::UniformScaling, ss::Subspace{T, 2}) where T = ss + I

(*)(a::Subspace, b::AbstractArray) = a * Subspace([b])
(*)(a::AbstractArray, b::Subspace) = Subspace([a]) * b

(|)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = a + b
(|)(a::Subspace, b::AbstractArray) = a + b
(|)(a::AbstractArray, b::Subspace) = a + b

(~)(ss::Subspace) = perp(ss)

(&)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = perp(perp(a) + perp(b))

kron(a::Subspace, b::AbstractArray) = kron(a, Subspace([b]))
kron(a::AbstractArray, b::Subspace) = kron(Subspace([a]), b)

function (/)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N}
    if !(b in a)
        throw(ArgumentError("divisor must be a subspace of dividend for subspace quotient"))
    end
    return perp(perp(a) + b)
end

(/)(a::Subspace, b::AbstractArray) = a / Subspace([b])

(/)(ss::Subspace{T, 2}, x::UniformScaling) where T = ss / Array{T}(I, shape(ss))

(⊆)(a::Subspace      , b::Subspace) = a in b
(⊆)(a::AbstractArray , b::Subspace) = a in b
(⊆)(a::UniformScaling, b::Subspace) = a in b
(⊇)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = b in a
(⊇)(a::Subspace, b::AbstractArray ) = b in a
(⊇)(a::Subspace, b::UniformScaling) = b in a

(⟂)(a::Subspace{T, N}, b::Subspace{U, N}) where {T,U,N} = a ⊆ perp(b)

################
### Math
################

function tobasis(ss::Subspace{<:Number, N}, x::AbstractArray{<:Number, N}) where N
    shp = shape(ss)
    basis_mat = reshape(ss.basis, prod(shp), dim(ss))
    return basis_mat' * vec(x)
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

# FIXME type should not be optional
random_subspace(d::Int, dims) = random_subspace(ComplexF64, d, dims)
random_hermitian_subspace(d::Int, dims) = random_hermitian_subspace(ComplexF64, d, dims)
empty_subspace(dims) = empty_subspace(ComplexF64, dims)
full_subspace(dims) = full_subspace(ComplexF64, dims)

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

function hermitian_basis(ss::Subspace{Complex{T}})::Array{Hermitian{Complex{T},Array{Complex{T},2}},1} where T
    if dim(ss) == 0
        return []
    end
    n = shape(ss)[1]
    shape(ss)[2] == n || throw(ArgumentError("subspace shape was not square: $(shape(ss))"))
    M = hcat(
        [ hermit_to_vec( x     + x'     ) for x in each_basis_element(ss) ]...,
        [ hermit_to_vec((x*1im)+(x*1im)') for x in each_basis_element(ss) ]...
    )
    hb = [ vec_to_hermit(x, n) for x in each_basis_element(Subspace(M)) ]
    @assert (Subspace(hb) == ss) "Hermitian basis didn't equal original space"
    return hb
end

#########################
### Support for Convex.jl
#########################

function tobasis(ss::Subspace{<:Number, N}, x::Convex.AbstractExpr) where N
    shp = shape(ss)
    basis_mat = reshape(ss.basis, prod(shp), dim(ss))
    return basis_mat' * vec(x)
end

function frombasis(ss::Subspace{<:Number, 1}, x::Convex.AbstractExpr)
    shp = shape(ss)
    basis_mat = reshape(ss.basis, prod(shp), dim(ss))
    return basis_mat * x
end

function frombasis(ss::Subspace{<:Number, 2}, x::Convex.AbstractExpr)
    shp = shape(ss)
    basis_mat = reshape(ss.basis, prod(shp), dim(ss))
    return reshape(basis_mat * x, shp...)
end

function projection(ss::Subspace{<:Number, N}, x::Convex.AbstractExpr) where N
    return frombasis(ss, tobasis(ss, x))
end

function in(x::Convex.AbstractExpr, ss::Subspace{<:Number, N}) where N
    return tobasis(perp(ss), x) == 0
end

function variable_in_space(ss::Subspace{<:Complex{<:Real}, N}) where N
    x = Convex.ComplexVariable(dim(ss))
    return frombasis(ss, x)
end

function variable_in_space(ss::Subspace{<:Real, N}) where N
    x = Convex.Variable(dim(ss))
    return frombasis(ss, x)
end

end # module
