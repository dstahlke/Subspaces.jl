var documenterSearchIndex = {"docs":
[{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"usage/#Constructors","page":"Usage","title":"Constructors","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"Function Description\nSubspace(basis::Array{<:Number, N+1}) Subspace from basis, where last index of array indexes the basis elements.\nSubspace(basis::Array{Array{<:Number, N}, 1}) Subspace from list of basis elements.\nrandom_subspace(T, dim, size) Random subspace of the given size and dimension, on base field T.\nrandom_hermitian_subspace Random subspace satisfying S == S'\nempty_subspace Empty subspace\nfull_subspace Full subspace","category":"page"},{"location":"usage/#Operations","page":"Usage","title":"Operations","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"Function Description\nsize(S) Size of the elements of S.\ndim(S) Dimension of subspace S.\nS.basis Array representing an orthonormal basis of S, with the last index of the array indexing the basis elements.\neach_basis_element(S) Generator for iterating over orthonormal basis elements.\nrandom_element(S) Returns a random element of subspace S.\nperp(S), ~S Perpendicular (orthogonal) subspace.\ntobasis(S, x) Transform a vector into basis coordinates.  Projects x onto S if x is not an element of S.\nfrombasis(S, x) Returns a vector given the basis coordinates.\nprojection(S, x) Projects vector x onto subspace S.\nhermitian_basis(S) For a subspace satisfying S == S', returns a basis consisting of Hermitian operators.\nhcat(S1, S2, ...) Direct sum of subspaces.\nvcat(S1, S2, ...) Direct sum of subspaces.\nhvcat(rows, S1, S2, ...) Direct sum of subspaces.\ncat(S1, S2, ...; dims) Direct sum of subspaces.\nkron(S, T) Direct product of subspaces.\nadjoint(S), S' Subspace consisting of adjoints of vectors,  x  x in S .\nS ⟂ T Check whether subspace S is orthogonal to T.\nS == T Check for equality of subspaces.\nx in S, x ∈ S Check membership of x in subspace S.\nS ⊆ T Check whether subspace S is contained in T.\nS ⊇ T Check whether subspace S contains T.\nS + T, S &#124; T Linear span of union of subspaces, textrmspan x y  x in S y in T .\nS & T, S ∩ T Intersection of subspaces.\nS * T Linear span of products of elements of S and of T.\nS / T Vector space quotient.  Requires T ⊆ S.","category":"page"},{"location":"usage/#Convex.jl-integration","page":"Usage","title":"Convex.jl integration","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"Function Description\nvariable_in_space(S) Creates a variable constrained to space S.\ntobasis(S, x) Transform variale from vector to basis coordinates.\nfrombasis(S, x) Transform variale from basis coordinates to vector.\nx in S Creates a constraint requiring variable x to be in subspace S.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"As an example, here is how one could do a positive semidefinite matrix completion.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using Convex, SCS, Subspaces\n\nn = 3\nA = [ 1 2 0\n      0 0 6\n      0 0 9 ]\n\nbasis_element(x, y) = [ i==x && j == y for i in 1:n, j in 1:n ]\n\nS = Subspace([ basis_element(I[1], I[2]) for I in findall(A .== 0) ])\n\nX = variable_in_space(S)\n\nproblem = minimize(tr(X), [ X+A ⪰ 0 ])\nsolve!(problem, () -> SCS.Optimizer(verbose=0, eps=1e-9))\nevaluate(X+A)\n\n# output\n\n3×3 Array{Float64,2}:\n 1.0  2.0  3.0\n 2.0  4.0  6.0\n 3.0  6.0  9.0","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Alternately, instead of X = variable_in_space(S) we could use X ∈ S.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"X = Variable(n, n)\nproblem = minimize(tr(X), [ X ∈ S, X+A ⪰ 0 ])\n\nsolve!(problem, () -> SCS.Optimizer(verbose=0, eps=1e-9))\nevaluate(X+A)\n\n# output\n\n3×3 Array{Float64,2}:\n 1.0  2.0  3.0\n 2.0  4.0  6.0\n 3.0  6.0  9.0","category":"page"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [Subspaces]\nPrivate = false","category":"page"},{"location":"reference/#Subspaces.Subspace","page":"Reference","title":"Subspaces.Subspace","text":"Subspace(basis::AbstractArray{AbstractArray{N}, 1}; tol::Real=1.0e-6)\n\nCreate a subspace from the given basis, expressed as a list of basis vectors.\n\nSubspace(basis::AbstractArray{N+1}; tol::Real=1.0e-6)\n\nCreate a subspace from a basis given as a multi-dimensional array.  The last array index enumerates the basis elements.  E.g., if given a matrix this constructor will create a subspace representing the column span of that matrix.\n\nThe tol parameter sets the tolerance for determining whether vectors are linearly dependent.\n\nbasis::Array\nAn orthonormal basis for this subspace.  The final index of this array indexes the basis vectors.\nperp::Array\nAn orthonormal basis for the perpendicular subspace.\ntol::AbstractFloat\nThe tolerance for determining whether vectors are linearly dependent.\n\njulia> Subspace([[1, 2, 3], [4, 5, 6]]) == Subspace([ 1 4; 2 5; 3 6])\ntrue\n\n\n\n\n\n","category":"type"},{"location":"reference/#Subspaces.:⟂-Union{Tuple{N}, Tuple{U}, Tuple{T}, Tuple{Subspace{T,N},Subspace{U,N}}} where N where U where T","page":"Reference","title":"Subspaces.:⟂","text":"⟂(a::Subspace{T,N}, b::Subspace{U,N}) -> Any\n\n\nCheck whether a is orthogonal to b.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.dim-Tuple{Subspace}","page":"Reference","title":"Subspaces.dim","text":"dim(S::Subspace) -> Integer\n\n\nReturns the linear dimension of this subspace.\n\njulia> dim(Subspace([ [1,2,3], [4,5,6] ]))\n2\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.each_basis_element-Tuple{Subspace}","page":"Reference","title":"Subspaces.each_basis_element","text":"each_basis_element(S::Subspace) -> Base.Generator{_A,_B} where _B where _A\n\n\nCreate a generator that iterates over orthonormal basis vectors of a subspace.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.empty_subspace","page":"Reference","title":"Subspaces.empty_subspace","text":"empty_subspace(T::Type, siz::Tuple) -> Subspace{_A,_B} where _B where _A\nempty_subspace(T::Type, siz::Tuple, tol::Any) -> Subspace{_A,_B} where _B where _A\n\n\nCreate an empty subspace of dimension-siz vector space, on base field T.\n\njulia> S = empty_subspace(ComplexF64, (3, 4))\nSubspace{Complex{Float64}} size (3, 4) dim 0\n\n\n\n\n\n","category":"function"},{"location":"reference/#Subspaces.frombasis-Tuple{Subspace,AbstractArray{var\"#s25\",1} where var\"#s25\"<:Number}","page":"Reference","title":"Subspaces.frombasis","text":"frombasis(S::Subspace, x::AbstractArray{var\"#s25\",1} where var\"#s25\"<:Number) -> Any\n\n\nReturns a vector from the given basis components in the basis S.basis.\n\nSee also: tobasis.\n\njulia> S = random_subspace(ComplexF64, 2, 10)\nSubspace{Complex{Float64}} size (10,) dim 2\n\njulia> x = randn(2);\n\njulia> size(frombasis(S, x))\n(10,)\n\njulia> norm( tobasis(S, frombasis(S, x)) - x ) < 1e-8\ntrue\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.frombasis-Tuple{Subspace{var\"#s25\",1} where var\"#s25\"<:Number,Convex.AbstractExpr}","page":"Reference","title":"Subspaces.frombasis","text":"frombasis(S::Subspace{var\"#s25\",1} where var\"#s25\"<:Number, x::Convex.AbstractExpr) -> Convex.MultiplyAtom\n\n\nReturns a vector from the given basis components in the basis S.basis.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.frombasis-Tuple{Subspace{var\"#s25\",2} where var\"#s25\"<:Number,Convex.AbstractExpr}","page":"Reference","title":"Subspaces.frombasis","text":"frombasis(S::Subspace{var\"#s25\",2} where var\"#s25\"<:Number, x::Convex.AbstractExpr) -> Convex.ReshapeAtom\n\n\nReturns a vector from the given basis components in the basis S.basis.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.full_subspace-Tuple{Type,Tuple}","page":"Reference","title":"Subspaces.full_subspace","text":"full_subspace(T::Type, siz::Tuple) -> Subspace{_A,_B} where _B where _A\n\n\nCreate an full subspace of dimension-siz vector space, on base field T.\n\njulia> S = full_subspace(ComplexF64, (3, 4))\nSubspace{Complex{Float64}} size (3, 4) dim 12\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.hermitian_basis-Union{Tuple{Subspace{Complex{T},N} where N}, Tuple{T}} where T","page":"Reference","title":"Subspaces.hermitian_basis","text":"Given a subspace satisfying S == S', returns a basis in which each basis element is Hermitian.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.perp-Tuple{Subspace}","page":"Reference","title":"Subspaces.perp","text":"perp(S::Subspace) -> Subspace{_A,_B} where _B where _A\n\n\nReturns the orthogonal subspace.  Can also be written as ~S.\n\njulia> S = Subspace([ [1,2,3], [4,5,6] ])\nSubspace{Float64} size (3,) dim 2\n\njulia> perp(S)\nSubspace{Float64} size (3,) dim 1\n\njulia> perp(S) == ~S\ntrue\n\njulia> perp(S) ⟂ S\ntrue\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.projection-Union{Tuple{N}, Tuple{Subspace{var\"#s23\",N} where var\"#s23\"<:Number,AbstractArray{var\"#s22\",N} where var\"#s22\"<:Number}} where N","page":"Reference","title":"Subspaces.projection","text":"Projection of vector x onto suspace S.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.projection-Union{Tuple{N}, Tuple{Subspace{var\"#s25\",N} where var\"#s25\"<:Number,Convex.AbstractExpr}} where N","page":"Reference","title":"Subspaces.projection","text":"Projection of vector x onto suspace S.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.random_element-Union{Tuple{Subspace{T,N} where N}, Tuple{T}} where T<:Number","page":"Reference","title":"Subspaces.random_element","text":"random_element(S::Subspace{T<:Number,N} where N) -> Any\n\n\nReturns a random element of a subspace.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.random_hermitian_subspace","page":"Reference","title":"Subspaces.random_hermitian_subspace","text":"random_hermitian_subspace(T::Type, d::Int64, n::Int64) -> Subspace{_A,_B} where _B where _A\nrandom_hermitian_subspace(T::Type, d::Int64, n::Int64, tol::Any) -> Subspace{_A,_B} where _B where _A\n\n\nRandom dimension-d subspace of n × n matrices on base field T, satisfying x ∈ S => x' ∈ S.\n\njulia> S = random_hermitian_subspace(ComplexF64, 2, 3)\nSubspace{Complex{Float64}} size (3, 3) dim 2\n\njulia> S == S'\ntrue\n\n\n\n\n\n","category":"function"},{"location":"reference/#Subspaces.random_subspace","page":"Reference","title":"Subspaces.random_subspace","text":"random_subspace(T::Type, d::Int64, siz::Any) -> Subspace{_A,_B} where _B where _A\nrandom_subspace(T::Type, d::Int64, siz::Any, tol::Any) -> Subspace{_A,_B} where _B where _A\n\n\nRandom dimension-d subspace of dimension-siz vector space, on base field T.\n\njulia> S = random_subspace(ComplexF64, 2, (3, 4))\nSubspace{Complex{Float64}} size (3, 4) dim 2\n\n\n\n\n\n","category":"function"},{"location":"reference/#Subspaces.tobasis-Union{Tuple{N}, Tuple{Subspace{var\"#s23\",N} where var\"#s23\"<:Number,AbstractArray{var\"#s22\",N} where var\"#s22\"<:Number}} where N","page":"Reference","title":"Subspaces.tobasis","text":"tobasis(S::Subspace{var\"#s23\",N} where var\"#s23\"<:Number, x::AbstractArray{var\"#s22\",N} where var\"#s22\"<:Number) -> Any\n\n\nReturns the basis components of x in the basis S.basis.\n\nSee also: frombasis.\n\njulia> S = random_subspace(ComplexF64, 2, 10)\nSubspace{Complex{Float64}} size (10,) dim 2\n\njulia> x = randn(10);\n\njulia> size(tobasis(S, x))\n(2,)\n\njulia> norm( frombasis(S, tobasis(S, x)) - projection(S, x) ) < 1e-8\ntrue\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.tobasis-Union{Tuple{N}, Tuple{Subspace{var\"#s25\",N} where var\"#s25\"<:Number,Convex.AbstractExpr}} where N","page":"Reference","title":"Subspaces.tobasis","text":"tobasis(S::Subspace{var\"#s25\",N} where var\"#s25\"<:Number, x::Convex.AbstractExpr) -> Convex.MultiplyAtom\n\n\nReturns the basis components of x in the basis S.basis.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.variable_in_space-Union{Tuple{Subspace{var\"#s23\",N} where var\"#s23\"<:(Complex{var\"#s22\"} where var\"#s22\"<:Real)}, Tuple{N}} where N","page":"Reference","title":"Subspaces.variable_in_space","text":"variable_in_space(S::Subspace{var\"#s23\",N} where var\"#s23\"<:(Complex{var\"#s22\"} where var\"#s22\"<:Real)) -> Union{Convex.MultiplyAtom, Convex.ReshapeAtom}\n\n\nCreates a Convex.jl variable ranging over the given subspace.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Subspaces.variable_in_space-Union{Tuple{Subspace{var\"#s25\",N} where var\"#s25\"<:Real}, Tuple{N}} where N","page":"Reference","title":"Subspaces.variable_in_space","text":"Creates a Convex.jl variable ranging over the given subspace.\n\n\n\n\n\n","category":"method"},{"location":"#Subspaces.jl-Vector-subspaces-in-Julia","page":"Home","title":"Subspaces.jl - Vector subspaces in Julia","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Subspaces.jl provides a Subspace type for representing linear subspaces of a finite dimensional vector space.  Subspaces of matrices or tensors are also supported, as these are vectors in the mathematical sense of the word.  Both real and complex base fields are supported.","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"julia> using Subspaces\n\njulia> S = Subspace([ [1,0,0], [0,1,1] ])\nSubspace{Float64} size (3,) dim 2\n\njulia> [1,2,2] ∈ S\ntrue\n\njulia> [1,1,2] ∈ S\nfalse\n\njulia> S.basis\n3×2 Array{Float64,2}:\n  0.0       1.0\n -0.707107  0.0\n -0.707107  0.0\n\njulia> S | Subspace([ [0,0,1] ])\nSubspace{Float64} size (3,) dim 3\n\njulia> S & Subspace([ [1,1,1] ])\nSubspace{Float64} size (3,) dim 1\n\njulia> kron(S, S)\nSubspace{Float64} size (9,) dim 4\n\njulia> S * S'\nSubspace{Float64} size (3, 3) dim 4\n\njulia> Subspace([ ones(3,4,5) ])\nSubspace{Float64} size (3, 4, 5) dim 1","category":"page"}]
}
