######################################################################
# Mesh
#######################################################################

abstract type AbstractMesh{D} end

struct Mesh{D,T,V<:AbstractArray{SVector{D,T}}} <: AbstractMesh{D}   # D is dimension of parameter space
    vertices::V                         # Iterable vertex container with SVector{D,T} eltype
    adjmat::SparseMatrixCSC{Bool,Int}   # Undirected graph: both dest > src and dest < src
end

function Base.show(io::IO, mesh::Mesh{D}) where {D}
    i = get(io, :indent, "")
    print(io,
"$(i)Mesh{$D}: mesh of a $D-dimensional manifold
$i  Vertices   : $(nvertices(mesh))
$i  Edges      : $(nedges(mesh))")
end

nvertices(m::Mesh) = length(m.vertices)

nedges(m::Mesh) = div(nnz(m.adjmat), 2)

nsimplices(m::Mesh) = length(simplices(m))

vertices(m::Mesh) = m.vertices

edges(m::Mesh, src) = nzrange(m.adjmat, src)

edgedest(m::Mesh, edge) = rowvals(m.adjmat)[edge]

edgevertices(m::Mesh) =
    ((vsrc, m.vertices[edgedest(m, edge)]) for (i, vsrc) in enumerate(m.vertices) for edge in edges(m, i))

function minmax_edge(m::Mesh{D,T}) where {D,T<:Real}
    minlen2 = typemax(T)
    maxlen2 = zero(T)
    verts = vertices(m)
    minedge = zero(first(verts))
    maxedge = zero(first(verts))
    for src in eachindex(verts), edge in edges(m, src)
        dest = edgedest(m, edge)
        dest > src || continue # Need only directed graph
        vec = verts[dest] - verts[src]
        norm2 = vec' * vec
        norm2 < minlen2 && (minlen2 = norm2; minedge = vec)
        norm2 > maxlen2 && (maxlen2 = norm2; maxedge = vec)
    end
    return minedge, maxedge
end

######################################################################
# Compute N-simplices (N = number of vertices)
######################################################################
function simplices(mesh::Mesh{D}, ::Val{N} = Val(D+1)) where {D,N}
    N > 0 || throw(ArgumentError("Need a positive number of vertices for simplices"))
    N == 1 && return Tuple.(1:nvertices(mesh))
    simps = NTuple{N,Int}[]
    buffer = (NTuple{N,Int}[], NTuple{N,Int}[], Int[])
    for src in eachindex(vertices(mesh))
        append!(simps, _simplices(buffer, mesh, src))
    end
    N > 2 && alignnormals!(simps, vertices(mesh))
    return simps
end

# Add (greater) neighbors to last vertex of partials that are also neighbors of scr, till N
function _simplices(buffer::Tuple{P,P,V}, mesh, src) where {N,P<:AbstractArray{<:NTuple{N}},V}
    partials, partials´, srcneighs = buffer
    resize!(srcneighs, 0)
    resize!(partials, 0)
    for edge in edges(mesh, src)
        srcneigh = edgedest(mesh, edge)
        srcneigh > src || continue # Directed graph, to avoid simplex duplicates
        push!(srcneighs, srcneigh)
        push!(partials, padright((src, srcneigh), 0, Val(N)))
    end
    for pass in 3:N
        resize!(partials´, 0)
        for partial in partials
            nextsrc = partial[pass - 1]
            for edge in edges(mesh, nextsrc)
                dest = edgedest(mesh, edge)
                dest > nextsrc || continue # If not directed, no need to check
                dest in srcneighs && push!(partials´, modifyat(partial, pass, dest))
            end
        end
        partials, partials´ = partials´, partials
    end
    return partials
end

modifyat(s::NTuple{N,T}, ind, el) where {N,T} = ntuple(i -> i === ind ? el : s[i], Val(N))

function alignnormals!(simplices, vertices)
    for (i, s) in enumerate(simplices)
        volume = elementvolume(vertices, s)
        volume < 0 && (simplices[i] = switchlast(s))
    end
    return simplices
end

# Project N-1 edges onto (N-1)-dimensional vectors to have a deterministic volume
elementvolume(verts, s::NTuple{N,Int}) where {N} =
    elementvolume(hcat(ntuple(i -> padright(SVector(verts[s[i+1]] - verts[s[1]]), Val(N-1)), Val(N-1))...))
elementvolume(mat::SMatrix{N,N}) where {N} = det(mat)

switchlast(s::NTuple{N,T}) where {N,T} = ntuple(i -> i < N - 1 ? s[i] : s[2N - i - 1] , Val(N))

######################################################################
# Mesh specifications
######################################################################
"""
    MeshSpec

Parent type of mesh specifications, which are currently `MarchingMeshSpec` (constructed with
`marchingmesh`) and `LinearMeshSpec` (constructed with `linearmesh`).

# See also
    `marchingmesh`, `linearmesh`, `buildmesh`
"""
abstract type MeshSpec{L} end

Base.show(io::IO, spec::MeshSpec{L}) where {L} =
    print(io, "MeshSpec{$L} : specifications for building a $(L)D mesh.")

"""
    buildlift(s::MeshSpec{L}, h::Union{Hamiltonian,ParametricHamiltonian}, postlift = missing)

Build a `lift` function that maps the vertices `v` of the `Mesh` built with `buildmesh(s,
h)` to the Brillouin/parameter space of `h`, `(p₁,..., pᵢ, ϕ₁,..., ϕⱼ) = lift(v...)` (see
`bandstructure` for details).

If `postlift` is non-missing, a `lift´ = postlift(lift(v...)...)` is returned instead.
`postlift` is useful when the user needs to perform a custom mapping to the default given by
a `MeshSpec`, e.g. when `h` is a `ParametricHamiltonian` with non-scalar parameters that are
connected non-trivially to the mesh coordinates.


# See also
    `buildmesh`, `marchingmesh`, `linearmesh`
"""
function buildlift end

"""
    buildmesh(s::MeshSpec, h::Union{Hamiltonian,ParametricHamiltonian})

Build a `Mesh` from the spec `s`, using properties of `h` as needed. The use of `h` depends
on the spec. For a `LinearMeshSpec` with `samelength = false`, the Bravais matrix of `h` is
needed to work out the length of each mesh segment in the Brillouin zone, while for other
specs such as `MarchingMeshSpec`, `h` is not needed and may be omitted (see example).

# Examples

```jldoctest
julia> buildmesh(marchingmesh((-π, π), (0, 2π), points = 10))
Mesh{2}: mesh of a 2-dimensional manifold
  Vertices   : 100
  Edges      : 261
```

# See also
    `buildlift`, `marchingmesh`, `linearmesh`
"""
function buildmesh end

#######################################################################
# MarchingMeshSpec
#######################################################################
struct MarchingMeshSpec{L,R,T<:Number,M<:NTuple{L,Tuple{Number,Number}}} <: MeshSpec{L}
    minmaxaxes::M
    axes::SMatrix{L,L,T}
    points::R
end

"""
    marchingmesh(minmaxaxes::Vararg{Tuple{Number,Number},L}; axes = 1.0 * I, points = 13)

Create a `spec::MeshSpec` for a L-dimensional marching-tetrahedra `Mesh` over a parallelepiped
with axes given by the columns of `axes`. The points along axis `i` are distributed between
`first(minmaxaxes[i])` and `last(minmaxaxes[i])`. The number of points on each axis is given
by `points`, or `points[i]` if several are given.

The mapping `lift = buildlift(spec, h, postlift)` that maps the marching `mesh =
buildmesh(spec, h)` vertices to the parameter/bloch phase space of a `Hamiltonian` or
`ParametricHamiltonian` `h` is the identity, or `postlift` if provided.

# Examples

```jldoctest
julia> buildmesh(marchingmesh((-π, π), (0,2π); points = 25))
Mesh{2}: mesh of a 2-dimensional manifold
  Vertices   : 625
  Edges      : 1776

julia> buildmesh(marchingmesh((-π, π), (0,2π); points = (10,10)))
Mesh{2}: mesh of a 2-dimensional manifold
  Vertices   : 100
  Edges      : 261
```

# See also
    `linearmesh`, `buildmesh`

# External links
- Marching tetrahedra (https://en.wikipedia.org/wiki/Marching_tetrahedra) in Wikipedia
"""
marchingmesh(minmaxaxes::Vararg{Tuple{Number,Number},L}; axes = 1.0 * I, points = 13) where {L} =
    MarchingMeshSpec(minmaxaxes, SMatrix{L,L}(axes), points)

marchingmesh(; kw...) = throw(ArgumentError("Need a finite number of axes to define a marching mesh"))

function buildmesh(s::MarchingMeshSpec{D}, h) where {D}
    ranges = ((b, r) -> range(b...; length = r)).(s.minmaxaxes, s.points)
    npoints = length.(ranges)
    cs = CartesianIndices(ntuple(n -> 1:npoints[n], Val(D)))
    ls = LinearIndices(cs)
    csinner = CartesianIndices(ntuple(n -> 1:npoints[n]-1, Val(D)))

    # edge vectors for marching tetrahedra in D-dimensions (skip zero vector [first])
    uedges = [c for c in CartesianIndices(ntuple(_ -> 0:1, Val(D)))][2:end]
    # tetrahedra built from the D unit-length uvecs added in any permutation
    perms = permutations(
            ntuple(i -> CartesianIndex(ntuple(j -> i == j ? 1 : 0, Val(D))), Val(D)))
    utets = [cumsum(pushfirst!(perm, zero(CartesianIndex{D}))) for perm in perms]

    # We don't use generators because their non-inferreble eltype causes problems elsewhere
    verts = [s.axes * SVector(getindex.(ranges, Tuple(c))) for c in cs]

    sp = SparseMatrixBuilder{Bool}(length(cs), length(cs))
    for c in cs
        for u in uedges
            dest = c + u    # dest > src
            dest in cs && pushtocolumn!(sp, ls[dest], true)
            dest = c - u    # dest < src
            dest in cs && pushtocolumn!(sp, ls[dest], true)
        end
        finalizecolumn!(sp)
    end
    adjmat = sparse(sp)

    return Mesh(verts, adjmat)
end

buildlift(::MarchingMeshSpec, ::Union{Hamiltonian,ParametricHamiltonian}, postlift = missing) =
    postlift

#######################################################################
# LinearMeshSpec
#######################################################################
struct LinearMeshSpec{N,L,T<:Number,R} <: MeshSpec{1}
    vertices::SVector{N,SVector{L,T}}
    samelength::Bool
    closed::Bool
    points::R
end

"""
    linearmesh(nodes...; points = 13, samelength = true, closed = false)

Create a `MeshSpec` for a one-dimensional `Mesh` connecting the `nodes::NTuple{L,Number}`
with straight segments, where `L` is the embedding mesh dimension. The following named
`nodes` can also be used: `:Γ, :X, :Y, :Z, :K, :Kp, :M` (see `Quantica.BZpoints`).

Each segment in the linear mesh contains a number `points` of points (endpoints included).
If a different number of points for each of the `N` segments is required, use
`points::NTuple{N,Int}`. If `samelength == true` each segment is normalized to have equal
length in mesh coordinates. If `closed == true` the last node is connected to the first node
(they must be equal).

# Examples

```jldoctest
julia> buildmesh(linearmesh(:Γ, :K, :M, :Γ; points = (101, 30, 30)), HamiltonianPresets.graphene())
Mesh{1}: mesh of a 1-dimensional manifold
  Vertices   : 159
  Edges      : 158
```

# See also
    `marchingmesh`, `buildmesh`
"""
linearmesh(nodes...; points = 13, samelength::Bool = true, closed::Bool = false) =
    LinearMeshSpec(sanitize_BZpts(nodes, closed), samelength, closed, points)

function sanitize_BZpts(pts, closed)
    pts´ = parse_BZpoint.(pts)
    if closed
        all(isapprox.(first(pts´), last(pts´))) ||
            throw(ArgumentError("Closed linear meshes should have equal first and last nodes."))
    end
    dim = maximum(length.(pts´))
    pts´´ = SVector(padright.(pts´, Val(dim)))
    return pts´´
end

parse_BZpoint(p::Tuple) = SVector(float.(p))

function parse_BZpoint(p::Symbol)
    pt = get(BZpoints, p, missing)
    pt === missing && throw(ArgumentError("Unknown Brillouin zone point $p, use one of $(keys(BZpoints))"))
    return SVector(float.(pt))
end

const BZpoints =
    ( Γ  = (0,)
    , X  = (pi,)
    , Y  = (0, pi)
    , Z  = (0, 0, pi)
    , K  = (2pi/3, -2pi/3)
    , Kp = (4pi/3, 2pi/3)
    , M  = (pi, 0)
    )

linearmesh_nodes(l, h) = cumsum(SVector(0, segment_lengths(l, h)...))

segment_lengths(s::LinearMeshSpec, h::Hamiltonian) = segment_lengths(s, bravais(h))
segment_lengths(s::LinearMeshSpec, ph::ParametricHamiltonian{P}) where {P} =
    segment_lengths(s, _blockdiag(SMatrix{P,P}(I), bravais(ph)))

function segment_lengths(s::LinearMeshSpec{N,LS,TS}, br::SMatrix{E,LB,TB}) where {TS,TB,N,E,LS,LB}
    T = promote_type(TS, TB)
    verts = padright.(s.vertices, Val(LB))
    dϕs = ntuple(i -> verts[i + 1] - verts[i], Val(N-1))
    if s.samelength
        ls = filltuple(T(1/(N-1)), Val(N-1))
    else
        ibr = pinverse(br)'
        ls = (dϕ -> norm(ibr * dϕ)).(dϕs)
        ls = ls ./ sum(ls)
    end
    return ls
end

function idx_to_node(s, h)
    nodes = SVector.(linearmesh_nodes(s, h))
    nmax = length(nodes)
    nodefunc = nvec -> begin
        n = only(nvec)
        node = if n >= nmax
            nodes[nmax]
        else
            nc = max(n, 1)
            i = Int(floor(nc))
            nodes[i] + rem(nc, 1) * (nodes[i+1] - nodes[i])
        end
        return node
    end
    return nodefunc
end

function buildmesh(s::LinearMeshSpec{N}, h) where {N}
    ranges = ((i, r) -> range(i, i+1, length = r)).(ntuple(identity, Val(N-1)), s.points)
    verts = SVector.(first(ranges))
    for r in Base.tail(ranges)
        pop!(verts)
        append!(verts, SVector.(r))
    end
    s.closed && pop!(verts)
    nv = length(verts)
    nodefunc = idx_to_node(s, h)
    verts .= nodefunc.(verts)
    adjmat = sparse(vcat(1:nv-1, 2:nv), vcat(2:nv, 1:nv-1), true, nv, nv)
    s.closed && (adjmat[end, 1] = adjmat[1, end] = true)
    return Mesh(verts, adjmat)
end

function buildlift(s::LinearMeshSpec{N,L}, h, postlift = missing) where {N,L}
    ls = segment_lengths(s, h)
    nodes = linearmesh_nodes(s, h)
    verts = s.vertices
    l = sum(ls)
    lift = x -> begin
        xc = clamp(only(x), 0, l)
        for (i, node) in enumerate(nodes)
            if node > xc
                p = verts[i-1] + (xc - nodes[i-1])/ls[i-1] * (verts[i]-verts[i-1])
                return applylift(postlift, p)
            end
        end
        return applylift(postlift, last(verts))
    end
    return lift
end
