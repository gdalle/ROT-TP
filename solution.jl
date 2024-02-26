### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 7b1cb992-a5ce-11ec-3eb2-ef7b56d0b3f6
begin
	using Graphs
	using Ipopt
	using JuMP
	using PlutoUI
	using Plots
	using ProgressLogging
end

# ╔═╡ a4d76f51-2801-4cb3-9929-a93dc4b2132b
md"""
!!! note "Getting started"
	The tutorial you are reading is written in the [Julia](https://julialang.org/) programming language and presented as a [Pluto](https://github.com/fonsp/Pluto.jl) notebook.

	The [JuMP documentation](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/introduction/) is also a great resource for syntax basics, and it will help you understand how optimization problems are formulated and solved below.
"""

# ╔═╡ cebaf96f-09f9-4ce5-bec5-fb33fd4fabe2
md"""
We begin by importing the packages we need.
"""

# ╔═╡ 077d1eef-fdf9-4327-ac1b-7189129bead5
md"""
And we display a nice TOC on the side.
"""

# ╔═╡ a544ef87-05d9-4bda-9e93-551903b4b993
PlutoUI.TableOfContents()

# ╔═╡ bc43ebee-2ed0-4255-bf2b-3352e53b5d92
md"""
# Preliminaries
"""

# ╔═╡ 67a5a5d1-e9b2-4800-a71e-a8f631d480f3
md"""
## Notations
"""

# ╔═╡ c546cd21-700f-4c2a-931b-9647b345b454
md"""
Consider a (finite) directed graph $G=(V,E)$ endowed with $K$ origin-destination vertex pairs $(o^k,d^k)_{k\in [1,K]}$. Let us denote by:

- ``r^k`` the (continuous) flow of users entering the network in ``o^k`` and exiting in ``d^k`` (which is fixed);
- ``\mathcal{P}_k`` the set of all simple paths from ``o^k`` to ``d^k``, and ``\mathcal{P} = \bigcup_{k=1}^K \mathcal{P}_k``;
- ``f_p`` the flow of users taking path ``p \in \mathcal{P}``, and ``f = (f_p)_{p\in\mathcal{P}}``;
- ``x_e`` the flow of users taking edge ``e \in E``, and ``x = (x_e)_{e\in E}``;
- ``\ell_e(x_e)`` the cost paid by a user taking edge ``e`` when the edge flow is ``x_e``
- ``L_e(x_e) = \int_0^{x_e}\ell_e(u) \mathrm{d}u``.
"""

# ╔═╡ 304456e3-8ddd-496e-80aa-4df3e46d772c
md"""
## Optimization problems
"""

# ╔═╡ de311c92-b521-404a-b42c-e39fdfbfae63
md"""
Minimizing the total cost of the system (i.e. finding the social optimum) is an optimization problem that reads:

```math
\begin{align}
(P^{SO}) \quad \min_{x,f} \quad
& \sum_{e\in E} x_e \ell_e(x_e) \\
s.t. \quad
& r_k = \sum_{p\in\mathcal{P}_k} f_p & k \in [1, K] \\
& x_e = \sum_{p \ni e} f_p & e \in E \\
& f_p \geq 0 & p \in \mathcal{P}
\end{align}
```

Here, the first constraint ensures that the flow going from $o^k$ to $d^k$ is split among the different possible paths, and the second constraint is the definition of $x_e$ as the sum of the flows on all paths containing edge $e$.
"""

# ╔═╡ 2b69da06-fb23-4268-a9d4-38525a7ac20c
md"""
Recall from the course that if $\ell_e$ is non-decreasing, we can guarantee that $f$ is a user equilibrium if and only if it is a solution of

```math
\begin{align}
(P^{UE}) \quad \min_{x,f} \quad
& \sum_{e\in E} L_e(x_e) \\
s.t. \quad
& r_k = \sum_{p\in\mathcal{P}_k}f_p & k \in [1, K] \\
& x_e = \sum_{p \ni e} f_p & e \in E \\
& f_p \geq 0 & p \in \mathcal{P}
\end{align}
```

where the only difference with $(P^{SO})$ is the objective function.
"""

# ╔═╡ 59128377-7a05-4b40-9a1b-91cea30d7a5b
md"""
## Price of anarchy
"""

# ╔═╡ 927b5dde-31fd-4b3c-bb38-346f013673e5
md"""
Let $C(x)=\sum_{e\in E}x_e \ell_e(x_e)$ be the total cost associated with the admissible flow $x$.
We denote by $x^{UE}$ the user equilibrium, and by $x^{SO}$ the social optimum.
The price of anarchy is defined as:

```math
PoA = \frac{C(x^{UE})}{C(x^{SO})} \geq 1
```
"""

# ╔═╡ b4ed607f-af8d-49fe-b19e-a207083245a9
md"""
# Model reformulation
"""

# ╔═╡ a7b1d5df-9c7c-4ee0-90a5-f41d94c32dad
md"""
## Question 1

Prove that if the cost function $\ell_e$ is constant for every edge $e$, then the social optimum and the user equilibrium are equivalent.
"""

# ╔═╡ 1a2e2b25-8103-4c64-863a-bab9377531c1
md"""
## Question 2

A complete directed graph $(V,E)$ is a directed graph such that for all pairs of vertices $(v_1,v_2)$, there exists an edge with origin $v_1$ and destination $v_2$.

Consider a complete directed graph with $n$ nodes, and assume that there is only one origin-destination pair $(o, d)$.
What are the dimensions of the vectors $f$ and $x$ ?
How many variables and constraints are there in the problems $(P^{SO})$ and $(P^{UE})$?
Don't count positivity constraints.
"""

# ╔═╡ f6db4c0a-a17c-4996-82d8-91bf1decd492
md"""
## Question 3

Rewrite both $(P^{SO})$ and $(P^{UE})$ using only the vector $f$.
"""

# ╔═╡ 9be537f0-9740-4af6-9367-6db7f99ae89e
md"""
## Question 4

Assume that there is only one origin-destination pair $(o, d)$.
Rewrite both problems using only the vector $x$.
We can use, for any vertex $v$ the notation $\delta^-(v)$ for the set of edges with destination $v$, and $\delta^+(v)$ for the set of edges with origin $v$.
"""

# ╔═╡ e9072bbe-439c-4939-92fd-9ba80a4856d0
md"""
## Question 5

Let $N$ be a matrix with $|V|$ rows and $|E|$ columns, such that

```math
N_{v,e} = \begin{cases}
-1 & \text{if the origin of $e$ is $v$} \\
+1 & \text{if the destination of $e$ is $v$} \\
0 & \text{otherwise}
\end{cases}
```

Thus, each column of $N$ corresponds to an edge, and indicates its origin with a $-1$ and its destination with a $+1$. This object is called the incidence matrix of the graph.

If $K=1$, what is the interpretation of each coordinate of the vector $Nx$ ? 
Deduce a more compact formulation of both problems using only the vector $x$.
"""

# ╔═╡ 600475b1-d054-48ac-b162-b28af92b713b
md"""
# Julia implementation
"""

# ╔═╡ a1dff897-fee1-40fb-8710-4c3ae3abdd4f
md"""
## Graph structure
"""

# ╔═╡ ef34ce22-b494-4ec6-940a-7a6c1869c518
md"""
Each vertex $v$ must contain the inflow of users, which is equal to $\sum_{k:v=o_k} r_k - \sum_{k:v=d_k} r_k$.
"""

# ╔═╡ e4002f3b-46cd-406b-8ce1-22c5d40495fe
Base.@kwdef struct VertexData
	inflow::Float64
end

# ╔═╡ 3e7e2935-5d7a-4072-a5b2-305ec7722afa
md"""
Each edge $e$ must contain the parameters of the cost function $\ell_e$, which we take to be linear: $\ell_e(x_e) = a_e x_e + b_e$.
"""

# ╔═╡ f10a9ab7-14d5-4a34-a208-f64496262b2f
Base.@kwdef struct EdgeData
	a::Float64
	b::Float64
end

# ╔═╡ 1cc81642-b941-42ff-9a7d-36e7ffb9617f
md"""
The `Network` structure below is just a convenience object meant to store a graph and the associated metadata.
"""

# ╔═╡ d9560fd1-57c7-46b8-91b1-ae3613df000d
Base.@kwdef struct Network
	G::SimpleDiGraph = SimpleDiGraph()
	vertex_data::Dict{Int,VertexData} = Dict{Int,VertexData}()
	edge_data::Dict{Tuple{Int,Int},EdgeData} = Dict{Tuple{Int,Int},EdgeData}()
end

# ╔═╡ 67d180c5-8e40-4167-8a23-4e4bf49bd72b
md"""
If the variable `network` has type `Network`, then you cann access:
- the underlying graph with `network.G`
- the metadata of vertex `v` with `network.vertex_data[v]`
- the metadata of edge `(u, v)` with `network.edge_data[(u, v)]`
"""

# ╔═╡ 32780d79-3a9e-4124-9ade-eb2b5756303e
md"""
## Graph operations
"""

# ╔═╡ a317b192-f6cf-4b9c-b75b-7eaad1c372ec
"""
	add_vertex_with_data!(network; inflow)

Modify the network by adding a vertex with a given `inflow`.
"""
function add_vertex_with_data!(network::Network; inflow)
	add_vertex!(network.G)  # append a new vertex to the underlying graph
	v = nv(network.G)  # its index is equal to the number of vertices in the graph (the new vertex is the last one)
	network.vertex_data[v] = VertexData(inflow=inflow)  # define its metadata
	return true
end

# ╔═╡ f95e3ddc-c187-4a7c-9604-c656f15967bb
"""
	add_edge_with_data!(network, s, d; a, b)

Modify the network by adding the edge `(s, d)` with cost coefficients `a` and `b`.
"""
function add_edge_with_data!(network::Network, s, d; a, b)
	add_edge!(network.G, s, d)  # add a new edge to the underlying graph
	network.edge_data[(s, d)] = EdgeData(a=a, b=b)  # define its metadata
	return true
end

# ╔═╡ 59f864ae-01ef-4225-be06-622740766c74
function modify_edge_data!(network, s, d; a, b)
	network.edge_data[(s, d)] = EdgeData(a=a, b=b)
end

# ╔═╡ 315c0821-442d-4a20-ac42-1442e1f37142
md"""
Since we know that the incidence matrix $N$ is sparse (with few non-zero coefficients), a dedicated structure allows for faster computations.
"""

# ╔═╡ eaeef73e-846f-4da9-9510-52f38bcf0976
"""
	build_incidence_matrix(network)

Compute the binary incidence matrix of the network graph.
"""
function build_incidence_matrix(network::Network)
	G = network.G
    N = zeros(Int,(nv(G), ne(G)))
    for (j, e) in enumerate(edges(G))
		s, d = src(e), dst(e)  #  edge e goes from source s to destination d
        N[s, j] = -1  # negative coefficient for the source
        N[d, j] = 1  # positive coefficient for the destination
    end
    return N
end

# ╔═╡ 4ed50233-cc44-49f1-a82f-d2cf2d7df034
md"""
## Visualization
"""

# ╔═╡ 4b75fb3b-9cd1-4f1d-96f2-ea3f7fd79daf
"""
	plot_network(network; vx, vy)

Visualize the network with vertex coordinates `vx` and `vy`.
"""
function plot_network(network::Network; vx, vy)
	G = network.G
	vertex_data = network.vertex_data
	# Vertices
    plt = scatter(vx, vy, label=nothing, text = 1:nv(G))
	# Edges
	for e in edges(G)
		s, d = src(e), dst(e)
        x_s, x_d, y_s, y_d = vx[s], vx[d], vy[s], vy[d]
        plot!(
			plt, [x_s, x_d], [y_s, y_d], color="black", arrow=true, label=nothing
		)
    end
	return plt
end

# ╔═╡ f78a3ac2-c2ae-4e0f-9538-27ed72105c05
md"""
## Optimization
"""

# ╔═╡ e26709a3-3e44-4b8b-9c03-6cdd720144c7
md"""
To solve nonlinear optimization problems, we will use the [JuMP.jl](https://jump.dev/JuMP.jl/stable/) package with the [Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) backend.
"""

# ╔═╡ 29c0a166-0aaa-4cbb-8f5c-69f5f5c03e00
"""
	solve_optimization(network; goal)

If `goal == :social_optimum`, find the globally optimal flow by minimizing the sum of user costs.
If `goal == :user_equilibrium`, find the game-theoretical equilibrium by minimizing the Wardrop potential.
"""
function solve_optimization(network::Network; goal)
	G = network.G
	vertex_data = network.vertex_data
	edge_data = network.edge_data

	# We start by creating a model and choosing the solver
    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
	
	# We add decision variables by specifying their dimension
    @variable(model, x[1:ne(G)])
    @variable(model, edge_costs[1:ne(G)])

	# We add constraints on these variables
	@constraint(model, x[1:ne(G)] .>= 0)  #  positive flow
    
	N = build_incidence_matrix(network)
    inflow = [vertex_data[v].inflow for v in 1:nv(G)]
	@constraint(model, N * x .+ inflow .== 0)  # Kirchhoff's law

	for (j, e) in enumerate(edges(G))
		s, d = src(e), dst(e)
		a, b = edge_data[(s, d)].a, edge_data[(s, d)].b
		if goal == :social_optimum
			# The objective is the sum of individual costs
        	@NLconstraint(model, edge_costs[j] == a * x[j]^2 + b * x[j])
		elseif goal == :user_equilibrium
			# The objective is the Wardrop potential
        	@NLconstraint(model, edge_costs[j] == a * x[j]^2 / 2 + b * x[j])
		end
	end

	# We define the objective as a sum of costs to minimize
    @NLobjective(model, Min, sum(edge_costs[j] for j in 1:ne(G)))

	# Here is where the computation actually happens
    optimize!(model)

	# We recover the variable value and round it for display purposes
    x_opt = round.(value.(x), digits=3)
    
    return x_opt
end

# ╔═╡ 38910d08-ab48-46bc-a0ec-814058c6e114
"""
	social_cost(network, x)

Compute the social cost based on a flow solution `x` to the optimization problem
"""
function social_cost(network::Network, x)
	G = network.G
	edge_data = network.edge_data
    cost = 0.
	for (j, e) in enumerate(edges(G))
		s, d = src(e), dst(e)
		a, b = edge_data[(s, d)].a, edge_data[(s, d)].b
		cost += a * x[j]^2 + b * x[j]
    end
    return cost
end

# ╔═╡ 427e627e-cc6c-4fb0-9f54-f89d7827791e
"""
	price_of_anarchy(network; verbose)

Compare the social optimum and user equilibrium on the network to deduce the price of anarchy.
"""
function price_of_anarchy(network::Network; verbose=true)
	x_so = solve_optimization(network; goal=:social_optimum)
	x_ue = solve_optimization(network; goal=:user_equilibrium)
	C_so = social_cost(network, x_so)
	C_ue = social_cost(network, x_ue)
	poa = C_ue / C_so
	verbose && @info "User equilibrium" x_ue collect(edges(network.G))
	return poa
end

# ╔═╡ 56d0e067-ab56-4fc2-9d78-2d971b8f1455
md"""
# Braess paradox
"""

# ╔═╡ 68411bd0-bbbd-4058-beca-103fc9e79e75
"""
	diamond_network()

Build the basic network on which the Braess paradox can be observed.
"""
function diamond_network()
	network = Network()

	add_vertex_with_data!(network, inflow=1)  # add vertex n°1 (left)
	add_vertex_with_data!(network, inflow=0)  # add vertex n°2 (top)
	add_vertex_with_data!(network, inflow=0)  # add vertex n°3 (bottom)
	add_vertex_with_data!(network, inflow=-1)  # add vertex n°4 (right)

	add_edge_with_data!(network, 1, 2, a=1, b=0)  # add edge (1, 2)
	add_edge_with_data!(network, 1, 3, a=0, b=1)  # add edge (1, 3)
	add_edge_with_data!(network, 2, 4, a=0, b=1)  # add edge (2, 4)
	add_edge_with_data!(network, 3, 4, a=1, b=0)  # add edge (3, 4)

	return network
end

# ╔═╡ bae0d516-6db6-4d41-9e3c-ce89d4e9dfaa
md"""
## Question 6

Compute the price of anarchy in a diamond network. Comment the result.
"""

# ╔═╡ 50ac4e5d-11cb-4caa-b0dd-dabad26d87d6
network6 = diamond_network()

# ╔═╡ fd786978-3ea2-482f-80c5-604c72decbe4
plot_network(network6, vx=[0, 1, 1, 2], vy=[0, 1, -1, 0])

# ╔═╡ 82dd92aa-d117-4088-abfb-dd7f5dfaa305
x_so = solve_optimization(network6; goal=:social_optimum)

# ╔═╡ 17c2368a-d2d1-4fda-a1f6-aeb45b84853e
x_ue = solve_optimization(network6; goal=:user_equilibrium)

# ╔═╡ 30231615-1b22-4a26-a0a8-5900ac98d7e9
social_cost(network6, x_so)

# ╔═╡ 5d1f806a-b9f6-405e-bc0b-baddeec76613
social_cost(network6, x_ue)

# ╔═╡ 3fa1fbc8-f675-4f9a-ab0b-15a1521a614a
price_of_anarchy(network6)

# ╔═╡ 0e0a3d5b-fcc1-46f2-be12-3e9333e5c9bc
md"""
## Question 7

- Add one edge form node $2$ to $3$ with null cost. Compute the price of anarchy.
- Add one edge form node $3$ to $2$ with null cost. Compute the price of anarchy.
- Add both edges. Compute the price of anarchy.
- Look at the intensities you obtain, and comment the result.
"""

# ╔═╡ 62d58c18-312e-4572-acad-9efb00b43a94
begin
	network7a = diamond_network()
	add_edge_with_data!(network7a, 2, 3, a=0, b=0)
	price_of_anarchy(network7a)
end

# ╔═╡ d95c1cf4-5cbf-483d-9927-ae523ede6dc5
begin
	network7b = diamond_network()
	add_edge_with_data!(network7b, 3, 2, a=0, b=0)
	price_of_anarchy(network7b)
end

# ╔═╡ afd6c239-00cc-415c-b225-ec70105fd5bb
begin
	network7c = diamond_network()
	add_edge_with_data!(network7c, 2, 3, a=0, b=0)
	add_edge_with_data!(network7c, 3, 2, a=0, b=0)
	price_of_anarchy(network7c)
end

# ╔═╡ 7d3c6b05-6449-4ca0-bc0c-2acccdce6584
md"""
# Variants
"""

# ╔═╡ 4f827a72-49e7-43e0-9745-c63bdbf29c4b
md"""
## Question 8

Create another version of the previous graph (the one with $6$ edges), in which the constant cost functions $\ell_e(x_e) = 1$ of edges $(1,3)$ and $(2, 4)$ become affine cost functions of the form $\ell_e(x_e) = x_e/2 + 1$.
What is the impact on the optimal solutions $x^{UE}$ and $x^{S0}$ ? Comment.
"""

# ╔═╡ 3d0d3306-b9c2-48cb-9646-d4d8a6550ccc
function create_network_Q8()
	network = Network()

	add_vertex_with_data!(network, inflow=1)
	add_vertex_with_data!(network, inflow=0)
	add_vertex_with_data!(network, inflow=0)
	add_vertex_with_data!(network, inflow=-1)

	add_edge_with_data!(network, 1, 2, a=1, b=0)
	add_edge_with_data!(network, 1, 3, a=0.5, b=1)  # a changed
	add_edge_with_data!(network, 2, 4, a=0.5, b=1)  # a changed
	add_edge_with_data!(network, 3, 4, a=1, b=0)
	
	add_edge_with_data!(network, 2, 3, a=0, b=0)  # additional edge
	add_edge_with_data!(network, 3, 2, a=0, b=0)  # additional edge

	return network
end

# ╔═╡ f18720e8-4aba-4d9f-8503-f711500790a6
network8 = create_network_Q8()

# ╔═╡ aecc372f-26ca-4b84-95cd-1b00d344018e
price_of_anarchy(network8)

# ╔═╡ 71ce1df5-d449-4486-b2fe-53c0ed8fa20a
md"""
## Question 9

Create a random constructor for the previous graph, where coefficients $a_e$ and $b_e$ are drawn uniformly in $[0, 1]$ for each edge $e$.
Plot a histogram of the price of anarchy over $100$ samples.
Which result from the course do we observe ?
"""

# ╔═╡ d1632c41-63da-4e1f-a5cc-e2f535c48b0d
function random_network_Q9()
	network = Network()

	add_vertex_with_data!(network, inflow=1)
	add_vertex_with_data!(network, inflow=0)
	add_vertex_with_data!(network, inflow=0)
	add_vertex_with_data!(network, inflow=-1)

	for (s, d) in [(1, 2), (1, 3), (2, 4), (3, 4), (2, 3), (3, 2)]
		add_edge_with_data!(network, s, d; a=rand(), b=rand())
	end

	return network
end

# ╔═╡ 6ba2a7e6-6e8e-4294-9f89-74312305d9e4
begin
	poa_list9 = Float64[]
	@progress for _ = 1:100
		network9 = random_network_Q9()
		poa9 = price_of_anarchy(network9; verbose=false)
		push!(poa_list9, poa9)
	end
end

# ╔═╡ 20f09bb9-24b5-432e-8249-267a435e492c
histogram(
	poa_list9, bins=20,
	xlabel="Price of anarchy", ylabel="Frequency", label=nothing
)

# ╔═╡ 7092e0f0-2276-4f37-96a7-694bfdf94545
md"""
## Question 10

Adapt the code to handle cost functions of the form $\ell_e(x_e) = a_e(1+\tfrac{15}{100}(x_e/b_e)^4)$, which is standard to measure the congestion.
Plot a histogram of the price of anarchy over $100$ random samples of $(a_e, b_e)$. Comment.
"""

# ╔═╡ a60a0f59-bc49-4aa5-9388-287886deb8cc
function solve_optimization_Q10(network::Network; goal)
	G = network.G
	vertex_data = network.vertex_data
	edge_data = network.edge_data

    model = Model(optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0))
    
    @variable(model, x[1:ne(G)] >= 0)
    @variable(model, edge_costs[1:ne(G)])
	
    N = build_incidence_matrix(network)
    inflow = [vertex_data[v].inflow for v in 1:nv(G)]
	@constraint(model, N * x .+ inflow .== 0)

	for (j, e) in enumerate(edges(G))
		s, d = src(e), dst(e)
		a, b = edge_data[(s, d)].a, edge_data[(s, d)].b
		if goal == :social_optimum
        	@NLconstraint(
				model, edge_costs[j] == a * x[j] + (0.15 / b^4) * x[j]^5
			)
		elseif goal == :user_equilibrium
        	@NLconstraint(
				model, edge_costs[j] == a * x[j] + (0.15 / b^4) * x[j]^5 / 5
			)
		end
	end

    @NLobjective(model, Min, sum(edge_costs[j] for j in 1:ne(G)))

    optimize!(model)
    
    x_opt = round.(value.(x), digits=3)
    
    return x_opt
end

# ╔═╡ 52c21596-8653-4eb4-85b4-4ec5ea03d49a
function social_cost_Q10(network::Network, x)
	G = network.G
	edge_data = network.edge_data
    cost = 0.
	for (j, e) in enumerate(edges(G))
		s, d = src(e), dst(e)
		a, b = edge_data[(s, d)].a, edge_data[(s, d)].b
		cost += a * x[j] + (0.15 / b^4) * x[j]^5
    end
    return cost
end

# ╔═╡ c68ddc62-54c2-4ed6-9ebb-f835f768897c
function price_of_anarchy_Q10(network::Network; verbose=true)
	x_so = solve_optimization_Q10(network; goal=:social_optimum)
	x_ue = solve_optimization_Q10(network; goal=:user_equilibrium)
	C_so = social_cost_Q10(network, x_so)
	C_ue = social_cost_Q10(network, x_ue)
	poa = C_ue / C_so
	verbose && @info "User equilibrium: $x_ue"
	return poa
end

# ╔═╡ 23b77db3-0809-4830-93e1-56726d5a6092
begin
	poa_list10 = Float64[]
	@progress for _ = 1:100
		network10 = random_network_Q9()
		poa10 = price_of_anarchy(network10; verbose=false)
		push!(poa_list10, poa10)
	end
end

# ╔═╡ bb4b68a8-2698-4f1b-9511-302bc29f9720
histogram(
	poa_list10, bins=20,
	xlabel="Price of anarchy", ylabel="Frequency", label=nothing
)

# ╔═╡ a2882bc0-0aa2-4cbf-a494-05c7578e3a9d
md"""
# More complex settings
"""

# ╔═╡ 440f0f4a-2810-4260-8fa9-6d11152e0f68
md"""
## Question 11

Construct a graph with $6$ nodes.
Define linear costs on each edge. 
Compute the price of anarchy.
"""

# ╔═╡ e81f1a4c-f429-49e4-93aa-f5ee04308626
md"""
## Question 12 

On your new graph, check whether it is possible to add an edge with null cost that increases the cost of the user equilibrium.
"""

# ╔═╡ 72ef1699-eb6e-4afa-9bc8-ec0a285d3799
md"""
## Question 13

By introducing edge-flow variables $x^1$ and $x^2$, rewrite the problems $(P^{SO})$ and $(P^{UE})$ in the case where $K=2$. 
Construct an example on your previous graph and compute the price of anarchy.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
Ipopt = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ProgressLogging = "33c8b6b6-d38a-422a-b730-caa89a2f386c"

[compat]
Graphs = "~1.6.0"
Ipopt = "~1.0.2"
JuMP = "~0.23.2"
Plots = "~1.27.0"
PlutoUI = "~0.7.37"
ProgressLogging = "~0.1.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.2"
manifest_format = "2.0"

[[deps.ASL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6252039f98492252f9e47c312c8ffda0e3b9e78d"
uuid = "ae81ac8f-d209-56e5-92de-9978fef736f9"
version = "0.1.3+0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "af92965fb30777147966f58acb05da51c5616b5f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "4c10eee4af024676200bc7752e536f858c6b8f93"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.3.1"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9950387274246d08af38f6eef8cb5480862a435f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.14.0"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "12fc73e5e0af68ad3137b886e3f7c1eacfca2640"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.17.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "96b0bc6c52df76506efc8a441c6cf1adcb1babc4"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.42.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.DiffResults]]
deps = ["StaticArrays"]
git-tree-sha1 = "c18e98cba888c6c25d1c3b048e4b3380ca956805"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.0.3"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "dd933c4ef7b4c270aacd4eb88fa64c147492acf0"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.10.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3f3a2501fa7236e9b911e0f7a588c657e822bb6d"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.3+0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ae13fcbc7ab8f16b0856729b050ef0c446aa3492"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.4.4+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "Pkg", "Zlib_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "d8a578692e3077ac998b50c0217dfd67f21d1e5f"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.0+0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "1bd6fc0c344fc0cbee1f42f8d2e7ec8253dda2d2"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.25"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "87eb71354d8ec1a96d4a7636bd57a7347dde3ef9"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.10.4+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pkg", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "51d2dfe8e590fbd74e7a842cf6d13d8a2f45dc01"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.6+0"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "9f836fb62492f4b0f0d3b06f55983f2704ed0883"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.64.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "a6c850d77ad5118ad3be4bd188919ce97fffac47"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.64.0+0"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "83ea630384a13fc4f002b77690bc0afeb4255ac9"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.2"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "a32d672ac2c967f3deb8a81d828afc739c838a06"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.68.3+2"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "57c021de207e234108a6f1454003120a1bf350c4"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.6.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[deps.IniFile]]
git-tree-sha1 = "f550e6e32074c939295eb5ea6de31849ac2c9625"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "91b5dcf362c5add98049e6c29ee756910b03051d"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.3"

[[deps.Ipopt]]
deps = ["Ipopt_jll", "MathOptInterface"]
git-tree-sha1 = "8b7b5fdbc71d8f88171865faa11d1c6669e96e32"
uuid = "b6b21f68-93f8-5de0-b562-5493be1d77c9"
version = "1.0.2"

[[deps.Ipopt_jll]]
deps = ["ASL_jll", "Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "MUMPS_seq_jll", "OpenBLAS32_jll", "Pkg"]
git-tree-sha1 = "e3e202237d93f18856b6ff1016166b0f172a49a8"
uuid = "9cc047cb-c261-5740-88fc-0cf96f7bdcc7"
version = "300.1400.400+0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "fa6287a4469f5e048d763df38279ee729fbd44e5"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.4.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b53380851c6e6664204efb2e62cd24fa5c47e4ba"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.2+0"

[[deps.JuMP]]
deps = ["Calculus", "DataStructures", "ForwardDiff", "LinearAlgebra", "MathOptInterface", "MutableArithmetics", "NaNMath", "OrderedCollections", "Printf", "SparseArrays", "SpecialFunctions"]
git-tree-sha1 = "c48de82c5440b34555cb60f3628ebfb9ab3dc5ef"
uuid = "4076af6c-e467-56ae-b986-b466b2749572"
version = "0.23.2"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "4f00cc36fede3c04b8acf9b2e2763decfdcecfa6"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.13"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "7739f837d6447403596a75d19ed01fd08d6f56bf"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.3.0+3"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "42b62845d70a619f063a7da093d995ec8e15e778"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.16.1+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "c9551dd26e31ab17b86cbd00c2ede019c08758eb"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "58f25e56b706f95125dcb796f39e1fb01d913a71"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.10"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.METIS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "1d31872bb9c5e7ec1f618e8c4a56c8b0d9bddc7e"
uuid = "d00139f3-1899-568f-a2f0-47f597d42d70"
version = "5.1.1+0"

[[deps.MUMPS_seq_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "METIS_jll", "OpenBLAS32_jll", "Pkg"]
git-tree-sha1 = "29de2841fa5aefe615dea179fcde48bb87b58f57"
uuid = "d7ed1dd3-d0ae-5e8e-bfb4-87a502085b8d"
version = "5.4.1+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "Printf", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "09be99195f42c601f55317bd89f3c6bbaec227dc"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "1.1.1"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "ba8c0f8732a24facba709388c74ba99dcbfdda1e"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.0.0"

[[deps.NaNMath]]
git-tree-sha1 = "b086b7ea07f8e38cf122f5016af580881ac914fe"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.7"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS32_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c6c2ed4b7acd2137b878eb96c68e63b76199d0f"
uuid = "656ef2d0-ae68-5445-9ca0-591084a874a2"
version = "0.3.17+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ab05aa4cc89736e95915b01e7279e61b1bfe33b8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.14+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PCRE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b2a7af664e098055a7529ad1a900ded962bca488"
uuid = "2f80f16e-611a-54ab-bc61-aa92de5b98fc"
version = "8.44.0+0"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "85b5da0fa43588c75bb1ff986493443f821c70b7"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.2.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "b4f5d02549a10e20780a24fce72bea96b6329e29"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.40.1+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Requires", "Statistics"]
git-tree-sha1 = "a3a964ce9dc7898193536002a6dd892b1b5a6f1d"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "2.0.1"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "bb16469fd5224100e422f0b027d26c5a25de1200"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.2.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "GeometryBasics", "JSON", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "5f6e1309595e95db24342e56cd4dabd2159e0b79"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.27.3"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "bf0a1121af131d9974241ba53f601211e9303a9e"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.37"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "d3538e7f8a790dc8903519090857ef8e1283eecd"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "c6c0f690d0cc7caddb74cef7aa847b824a16b256"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "6bf3f380ff52ce0832ddd3a2a7b9538ed1bcca7d"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.2.1"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "RecipesBase"]
git-tree-sha1 = "dc1e451e15d90347a7decc4221842a022b011714"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.5.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "cdbd3b1338c72ce29d9584fdbe9e9b70eeb5adca"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "0.1.3"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "0b4b7f1393cff97c33891da2a0bf69c6ed241fda"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.1.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "5ba658aeecaaf96923dce0da9e703bd1fe7666f9"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "4f6ec5d99a28e1a749559ef7dd518663c5eca3d5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c3d8ba7f3fa0625b062b82853a7d5229cb728b6b"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.1"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "8977b17906b0a1cc74ab2e3a05faa16cf08a8291"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.16"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "StaticArrays", "Tables"]
git-tree-sha1 = "57617b34fa34f91d536eb265df67c2d4519b8b98"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.5"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "5ce79ce186cc678bbb5c5681ca3379d1ddae11a1"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.7.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "216b95ea110b5972db65aa90f88d8d89dcb8851c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.6"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "34db80951901073501137bdbc3d5a8e7bbd06670"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.1.2"

[[deps.Wayland_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "3e61f0b86f90dacb0bc0e73a0c5a83f6a8636e23"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.19.0+0"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4528479aa01ee1b3b4cd0e6faef0e04cf16466da"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.25.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "1acf5bdf07aa0907e0a37d3718bb88d4b687b74a"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.9.12+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "5be649d550f3f4b95308bf0183b82e2582876527"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.6.9+4"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4e490d5c960c314f33885790ed410ff3a94ce67e"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.9+4"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fe47bd2247248125c428978740e18a681372dd4"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.3+4"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6783737e45d3c59a4a4c4091f5f88cdcf0908cbb"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.0+3"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "daf17f441228e7a3833846cd048892861cff16d6"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.13.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "926af861744212db0eb001d9e40b5d16292080b2"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.0+4"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "4bcbf660f6c2e714f87e960a171b119d06ee163b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.2+4"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "5c8424f8a67c3f2209646d4425f3d415fee5931d"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.27.0+4"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "79c31e7844f6ecf779705fbc12146eb190b7d845"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.4.0+3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e45044cd873ded54b6a5bac0eb5c971392cf1927"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.2+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "ece2350174195bb31de1a63bea3a41ae1aa593b6"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "0.9.1+5"
"""

# ╔═╡ Cell order:
# ╟─a4d76f51-2801-4cb3-9929-a93dc4b2132b
# ╟─cebaf96f-09f9-4ce5-bec5-fb33fd4fabe2
# ╠═7b1cb992-a5ce-11ec-3eb2-ef7b56d0b3f6
# ╟─077d1eef-fdf9-4327-ac1b-7189129bead5
# ╠═a544ef87-05d9-4bda-9e93-551903b4b993
# ╟─bc43ebee-2ed0-4255-bf2b-3352e53b5d92
# ╟─67a5a5d1-e9b2-4800-a71e-a8f631d480f3
# ╟─c546cd21-700f-4c2a-931b-9647b345b454
# ╟─304456e3-8ddd-496e-80aa-4df3e46d772c
# ╟─de311c92-b521-404a-b42c-e39fdfbfae63
# ╟─2b69da06-fb23-4268-a9d4-38525a7ac20c
# ╟─59128377-7a05-4b40-9a1b-91cea30d7a5b
# ╟─927b5dde-31fd-4b3c-bb38-346f013673e5
# ╟─b4ed607f-af8d-49fe-b19e-a207083245a9
# ╟─a7b1d5df-9c7c-4ee0-90a5-f41d94c32dad
# ╟─1a2e2b25-8103-4c64-863a-bab9377531c1
# ╟─f6db4c0a-a17c-4996-82d8-91bf1decd492
# ╟─9be537f0-9740-4af6-9367-6db7f99ae89e
# ╟─e9072bbe-439c-4939-92fd-9ba80a4856d0
# ╟─600475b1-d054-48ac-b162-b28af92b713b
# ╟─a1dff897-fee1-40fb-8710-4c3ae3abdd4f
# ╟─ef34ce22-b494-4ec6-940a-7a6c1869c518
# ╠═e4002f3b-46cd-406b-8ce1-22c5d40495fe
# ╟─3e7e2935-5d7a-4072-a5b2-305ec7722afa
# ╠═f10a9ab7-14d5-4a34-a208-f64496262b2f
# ╟─1cc81642-b941-42ff-9a7d-36e7ffb9617f
# ╠═d9560fd1-57c7-46b8-91b1-ae3613df000d
# ╟─67d180c5-8e40-4167-8a23-4e4bf49bd72b
# ╟─32780d79-3a9e-4124-9ade-eb2b5756303e
# ╠═a317b192-f6cf-4b9c-b75b-7eaad1c372ec
# ╠═f95e3ddc-c187-4a7c-9604-c656f15967bb
# ╠═59f864ae-01ef-4225-be06-622740766c74
# ╟─315c0821-442d-4a20-ac42-1442e1f37142
# ╠═eaeef73e-846f-4da9-9510-52f38bcf0976
# ╟─4ed50233-cc44-49f1-a82f-d2cf2d7df034
# ╠═4b75fb3b-9cd1-4f1d-96f2-ea3f7fd79daf
# ╟─f78a3ac2-c2ae-4e0f-9538-27ed72105c05
# ╟─e26709a3-3e44-4b8b-9c03-6cdd720144c7
# ╠═29c0a166-0aaa-4cbb-8f5c-69f5f5c03e00
# ╠═38910d08-ab48-46bc-a0ec-814058c6e114
# ╠═427e627e-cc6c-4fb0-9f54-f89d7827791e
# ╟─56d0e067-ab56-4fc2-9d78-2d971b8f1455
# ╠═68411bd0-bbbd-4058-beca-103fc9e79e75
# ╟─bae0d516-6db6-4d41-9e3c-ce89d4e9dfaa
# ╠═50ac4e5d-11cb-4caa-b0dd-dabad26d87d6
# ╠═fd786978-3ea2-482f-80c5-604c72decbe4
# ╠═82dd92aa-d117-4088-abfb-dd7f5dfaa305
# ╠═17c2368a-d2d1-4fda-a1f6-aeb45b84853e
# ╠═30231615-1b22-4a26-a0a8-5900ac98d7e9
# ╠═5d1f806a-b9f6-405e-bc0b-baddeec76613
# ╠═3fa1fbc8-f675-4f9a-ab0b-15a1521a614a
# ╟─0e0a3d5b-fcc1-46f2-be12-3e9333e5c9bc
# ╠═62d58c18-312e-4572-acad-9efb00b43a94
# ╠═d95c1cf4-5cbf-483d-9927-ae523ede6dc5
# ╠═afd6c239-00cc-415c-b225-ec70105fd5bb
# ╟─7d3c6b05-6449-4ca0-bc0c-2acccdce6584
# ╟─4f827a72-49e7-43e0-9745-c63bdbf29c4b
# ╠═3d0d3306-b9c2-48cb-9646-d4d8a6550ccc
# ╠═f18720e8-4aba-4d9f-8503-f711500790a6
# ╠═aecc372f-26ca-4b84-95cd-1b00d344018e
# ╟─71ce1df5-d449-4486-b2fe-53c0ed8fa20a
# ╠═d1632c41-63da-4e1f-a5cc-e2f535c48b0d
# ╠═6ba2a7e6-6e8e-4294-9f89-74312305d9e4
# ╠═20f09bb9-24b5-432e-8249-267a435e492c
# ╟─7092e0f0-2276-4f37-96a7-694bfdf94545
# ╠═a60a0f59-bc49-4aa5-9388-287886deb8cc
# ╠═52c21596-8653-4eb4-85b4-4ec5ea03d49a
# ╠═c68ddc62-54c2-4ed6-9ebb-f835f768897c
# ╠═23b77db3-0809-4830-93e1-56726d5a6092
# ╠═bb4b68a8-2698-4f1b-9511-302bc29f9720
# ╟─a2882bc0-0aa2-4cbf-a494-05c7578e3a9d
# ╟─440f0f4a-2810-4260-8fa9-6d11152e0f68
# ╟─e81f1a4c-f429-49e4-93aa-f5ee04308626
# ╟─72ef1699-eb6e-4afa-9bc8-ec0a285d3799
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
