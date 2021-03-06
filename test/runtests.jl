using Subspaces
#using Convex, SCS
using LinearAlgebra
using Random
using Test

eye(n) = Matrix(1.0*I, (n,n))

@testset "Basic tests" begin
    a = random_subspace(ComplexF64, 1, 10)
    b = random_subspace(ComplexF64, 2, 10)
    c = random_subspace(ComplexF64, 3, 10)
    @test dim(a) == 1
    @test dim(b) == 2
    @test dim(c) == 3
    @test dim(a+b) == 3
    @test dim(a+c) == 4
    @test dim(b+c) == 5
    @test dim(b*c') == 6
    @test size(b*c') == (10, 10)
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
    @test size(kron(b, c)) == (100,)
    @test dim(hcat(b, c)) == 5
    @test size(hcat(b, c)) == (10, 2)
    @test dim(vcat(b, c)) == 5
    @test size(vcat(b, c)) == (20,)
    @test size(b') == (1, 10)
    @test size(b'') == (10, 1)
    @test dim(b + ones(10)) == 3
    @test dim(empty_subspace(Float64, (10,))) == 0
    @test dim(full_subspace(Float64, (10,))) == 10
    @test dim(full_subspace(Float64, (3, 4))) == 12
    @test size(full_subspace(Float64, (3, 4))) == (3, 4)
    @test dim(full_subspace(Float64, (4, 4)) / I) == 15
end

@testset "Empty spaces" begin
    a = empty_subspace(Float64, (3,4))
    b = empty_subspace(Float64, (4,5))
    f = full_subspace(Float64, (3,4))
    @test dim(a) == 0
    @test dim(~a) == 12
    @test dim(f) == 12
    @test dim(~f) == 0
    @test size(a') == (4,3)
    @test size(a+a) == (3,4)
    @test size(a*b) == (3,5)
    @test size(a&a) == (3,4)
    @test size(vcat(a)) == (3,4)
    @test size(vcat(a,a)) == (6,4)
    @test size(hcat(a)) == (3,4)
    @test size(hcat(a,a)) == (3,8)
    @test size(kron(a,b)) == (12,20)
    @test size(a/a) == (3,4)
    @test dim(a/a) == 0
    @test dim(f/a) == 12
    @test dim(f/f) == 0
    @test a ⊆ a
    @test a ⊆ f
    @test zeros((3,4)) in a
    @test !(ones((3,4)) in a)
    @test dim(random_subspace(ComplexF64, 0, (3,3))) == 0
end
