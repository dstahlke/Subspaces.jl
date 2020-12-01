using Subspaces
#using Convex, SCS
using LinearAlgebra
using Random, RandomMatrices
using Test

function random_bounded(n)
    U = rand(Haar(2), n)
    return Hermitian(U' * Diagonal(rand(n)) * U)
end

function basis_vec(dims::Tuple, idx::Tuple)
    m = zeros(dims)
    m[idx...] = 1
    return m
end

eye(n) = Matrix(1.0*I, (n,n))

@testset "Basic tests" begin
    a = random_subspace(1, 10)
    b = random_subspace(2, 10)
    c = random_subspace(3, 10)
    @test dim(a) == 1
    @test dim(b) == 2
    @test dim(c) == 3
    @test dim(a+b) == 3
    @test dim(a+c) == 4
    @test dim(b+c) == 5
    @test dim(a+b+c) == 6
    @test dim((a+b) + (b+c)) == 6
    @test (a+b) + (b+c) == a+b+c
    @test (a+b) / a == Subspace([ projection(~a, x) for x in each_basis_element(b) ])
    @test (a+b) / b == Subspace([ projection(~b, x) for x in each_basis_element(a) ])
    @test (c+b) / b == Subspace([ projection(~b, x) for x in each_basis_element(c) ])
    @test (c+b) / b ⟂ b
    @test a ⊆ a
    @test a ⊇ a
    @test a ⊆ a+b
    @test a+b ⊇ a
    @test !(a ⊆ b)
    @test !(a ⊇ b)
    @test random_element(c) in c
    @test random_element(c) != random_element(c)
    x = random_element(c)
    @test projection(c, x) ≈ x
    @test frombasis(c, tobasis(c, x)) ≈ x
    @test dim(kron(b, c)) == 6
    @test shape(kron(b, c)) == (100,)
    @test dim(hcat(b, c)) == 5
    @test shape(hcat(b, c)) == (10, 2)
    @test dim(vcat(b, c)) == 5
    @test shape(vcat(b, c)) == (20,)
    @test shape(b') == (1, 10)
    @test shape(b'') == (10, 1)
    @test dim(b + ones(10)) == 3
    @test dim(empty_subspace(Float64, (10,))) == 0
    @test dim(full_subspace(Float64, (10,))) == 10
    @test dim(full_subspace(Float64, (3, 4))) == 12
    @test shape(full_subspace(Float64, (3, 4))) == (3, 4)
end
