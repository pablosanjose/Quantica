#######################################################################
# Hamiltonian
#######################################################################
struct HamiltonianHarmonic{L,M,A<:Union{AbstractMatrix{M},SparseMatrixBuilder{M}}}
    dn::SVector{L,Int}
    h::A
end

HamiltonianHarmonic{L,M,A}(dn::SVector{L,Int}, n::Int, m::Int) where {L,M,A<:SparseMatrixCSC{M}} =
    HamiltonianHarmonic(dn, sparse(Int[], Int[], M[], n, m))

HamiltonianHarmonic{L,M,A}(dn::SVector{L,Int}, n::Int, m::Int) where {L,M,A<:Matrix{M}} =
    HamiltonianHarmonic(dn, zeros(M, n, m))

struct Hamiltonian{LA<:AbstractLattice,L,M,A<:AbstractMatrix,
                   H<:HamiltonianHarmonic{L,M,A},
                   O<:Tuple{Vararg{Tuple{Vararg{NameType}}}}} # <: AbstractMatrix{M}
    lattice::LA
    harmonics::Vector{H}
    orbitals::O
end

function Hamiltonian(lat, hs::Vector{H}, orbs, n::Int, m::Int) where {L,M,H<:HamiltonianHarmonic{L,M}}
    sort!(hs, by = h -> abs.(h.dn))
    if isempty(hs) || !iszero(first(hs).dn)
        pushfirst!(hs, H(zero(SVector{L,Int}), empty_sparse(M, n, m)))
    end
    return Hamiltonian(lat, hs, orbs)
end

Base.show(io::IO, ham::Hamiltonian) = show(io, MIME("text/plain"), ham)
function Base.show(io::IO, ::MIME"text/plain", ham::Hamiltonian)
    i = get(io, :indent, "")
    print(io, i, summary(ham), "\n",
"$i  Bloch harmonics  : $(length(ham.harmonics)) ($(displaymatrixtype(ham)))
$i  Harmonic size    : $((n -> "$n × $n")(nsites(ham)))
$i  Orbitals         : $(displayorbitals(ham))
$i  Element type     : $(displayelements(ham))
$i  Onsites          : $(nonsites(ham))
$i  Hoppings         : $(nhoppings(ham))
$i  Coordination     : $(nhoppings(ham) / nsites(ham))")
    ioindent = IOContext(io, :indent => string("  "))
    issuperlattice(ham.lattice) && print(ioindent, "\n", ham.lattice.supercell)
end

Base.summary(h::Hamiltonian{LA}) where {E,L,LA<:Lattice{E,L}} =
    "Hamiltonian{<:Lattice} : Hamiltonian on a $(L)D Lattice in $(E)D space"

Base.summary(::Hamiltonian{LA}) where {E,L,T,L´,LA<:Superlattice{E,L,T,L´}} =
    "Hamiltonian{<:Superlattice} : $(L)D Hamiltonian on a $(L´)D Superlattice in $(E)D space"

Base.eltype(::Hamiltonian{<:Any,<:Any,M}) where {M} = M

Base.isequal(h1::HamiltonianHarmonic, h2::HamiltonianHarmonic) =
    h1.dn == h2.dn && h1.h == h2.h

displaymatrixtype(h::Hamiltonian) = displaymatrixtype(matrixtype(h))
displaymatrixtype(::Type{<:SparseMatrixCSC}) = "SparseMatrixCSC, sparse"
displaymatrixtype(::Type{<:Array}) = "Matrix, dense"
displaymatrixtype(A::Type{<:AbstractArray}) = string(A)
displayelements(h::Hamiltonian) = displayelements(blocktype(h))
displayelements(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = "$N × $N blocks ($T)"
displayelements(::Type{T}) where {T} = "scalar ($T)"
displayorbitals(h::Hamiltonian) =
    replace(replace(string(h.orbitals), "Symbol(\"" => ":"), "\")" => "")

SparseArrays.issparse(h::Hamiltonian{LA,L,M,A}) where {LA,L,M,A<:AbstractSparseMatrix} = true
SparseArrays.issparse(h::Hamiltonian{LA,L,M,A}) where {LA,L,M,A} = false

Base.parent(h::Hamiltonian) = h

# Internal API #

latdim(h::Hamiltonian{LA}) where {E,L,LA<:AbstractLattice{E,L}} = L

matrixtype(::Hamiltonian{LA,L,M,A}) where {LA,L,M,A} = A
blockeltype(::Hamiltonian{<:Any,<:Any,M}) where {M} = eltype(M)

# find SMatrix type that can hold all matrix elements between lattice sites
blocktype(orbs, type::Type{Tv}) where {Tv} =
    _blocktype(orbitaltype(orbs, Tv))
_blocktype(::Type{S}) where {N,Tv,S<:SVector{N,Tv}} = SMatrix{N,N,Tv,N*N}
_blocktype(::Type{S}) where {S<:Number} = S

blocktype(h::Hamiltonian{LA,L,M}) where {LA,L,M} = M

promote_blocktype(hs::Hamiltonian...) = promote_blocktype(blocktype.(hs)...)
promote_blocktype(s1::Type, s2::Type, ss::Type...) =
    promote_blocktype(promote_blocktype(s1, s2), ss...)
promote_blocktype(::Type{SMatrix{N1,N1,T1,NN1}}, ::Type{SMatrix{N2,N2,T2,NN2}}) where {N1,NN1,T1,N2,NN2,T2} =
    SMatrix{max(N1, N2), max(N1, N2), promote_type(T1, T2), max(NN1,NN2)}
promote_blocktype(T1::Type{<:Number}, T2::Type{<:Number}) = promote_type(T1, T2)
promote_blocktype(T::Type) = T

blockdim(h::Hamiltonian) = blockdim(blocktype(h))
blockdim(::Type{S}) where {N,S<:SMatrix{N,N}} = N
blockdim(::Type{T}) where {T<:Number} = 1

# find SVector type that can hold all orbital amplitudes in any lattice sites
orbitaltype(orbs, type::Type{Tv} = Complex{T}) where {T,Tv} =
    _orbitaltype(SVector{1,Tv}, orbs...)
_orbitaltype(::Type{S}, ::NTuple{D,NameType}, os...) where {N,Tv,D,S<:SVector{N,Tv}} =
    (M = max(N,D); _orbitaltype(SVector{M,Tv}, os...))
_orbitaltype(t::Type{SVector{N,Tv}}) where {N,Tv} = t
_orbitaltype(t::Type{SVector{1,Tv}}) where {Tv} = Tv

orbitaltype(h::Hamiltonian{LA,L,M}) where {N,T,LA,L,M<:SMatrix{N,N,T}} = SVector{N,T}
orbitaltype(h::Hamiltonian{LA,L,M}) where {LA,L,M<:Number} = M

function nhoppings(ham::Hamiltonian)
    count = 0
    for h in ham.harmonics
        count += iszero(h.dn) ? (_nnz(h.h) - _nnzdiag(h.h)) : _nnz(h.h)
    end
    return count
end

function nonsites(ham::Hamiltonian)
    count = 0
    for h in ham.harmonics
        iszero(h.dn) && (count += _nnzdiag(h.h))
    end
    return count
end

_nnz(h::AbstractSparseMatrix) = count(!iszero, nonzeros(h)) # Does not include stored zeros
_nnz(h::DenseMatrix) = count(!iszero, h)

function _nnzdiag(s::SparseMatrixCSC)
    count = 0
    rowptrs = rowvals(s)
    nz = nonzeros(s)
    for col in 1:size(s,2)
        for ptr in nzrange(s, col)
            rowptrs[ptr] == col && (count += !iszero(nz[ptr]); break)
        end
    end
    return count
end
_nnzdiag(s::Matrix) = count(!iszero, s[i,i] for i in 1:minimum(size(s)))

Base.isequal(h1::Hamiltonian, h2::Hamiltonian) =
    isequal(h1.lattice, h2.lattice) && isequal(h1.harmonics, h2.harmonics) &&
    isequal(h1.orbitals, h2.orbitals)

# Iterators #

function nonzero_indices(h::Hamiltonian, rowrange = 1:size(h, 1), colrange = 1:size(h, 2))
    rowrange´ = rclamp(rowrange, 1:size(h, 1))
    colrange´ = rclamp(colrange, 1:size(h, 2))
    gen = ((har.dn, rowvals(har.h)[ptr], col)
                for har in h.harmonics
                for col in colrange´
                for ptr in nzrange_inrows(har.h, col, rowrange´)
                if !iszero(nonzeros(har.h)[ptr]))
    return gen
end

function nonzero_indices(har::HamiltonianHarmonic, rowrange = 1:size(h, 1), colrange = 1:size(h, 2))
    rowrange´ = rclamp(rowrange, 1:size(har, 1))
    colrange´ = rclamp(colrange, 1:size(har, 2))
    gen = ((rowvals(har.h)[ptr], col)
                for col in colrange´
                for ptr in nzrange_inrows(har.h, col, rowrange´)
                if !iszero(nonzeros(har.h)[ptr]))
    return gen
end

function nzrange_inrows(h, col, rowrange)
    ptrs = nzrange(h, col)
    rows = rowvals(h)
    ptrmin = first(ptrs)
    ptrmax = last(ptrs)

    for p in ptrs
        rows[p] in rowrange && break
        ptrmin = p + 1
    end

    if ptrmin < ptrmax
        for p in ptrmax:-1:ptrmin
            ptrmax = p
            rows[p] in rowrange && break
        end
    end

    return ptrmin:ptrmax
end

# External API #
"""
    hamiltonian(lat, model; orbitals, type)

Create a `Hamiltonian` by applying `model::TighbindingModel` to the lattice `lat` (see
`hopping` and `onsite` for details on building tightbinding models).

The number of orbitals on each sublattice can be specified by the keyword `orbitals`
(otherwise all sublattices have one orbital by default). The following, and obvious
combinations, are possible formats for the `orbitals` keyword:

    orbitals = :a                # all sublattices have 1 orbital named :a
    orbitals = (:a,)             # same as above
    orbitals = (:a, :b, 3)       # all sublattices have 3 orbitals named :a and :b and :3
    orbitals = ((:a, :b), (:c,)) # first sublattice has 2 orbitals, second has one
    orbitals = ((:a, :b), :c)    # same as above
    orbitals = (Val(2), Val(1))  # same as above, with automatic names
    orbitals = (:A => (:a, :b), :D => :c) # sublattice :A has two orbitals, :D and rest have one
    orbitals = :D => Val(4)      # sublattice :D has four orbitals, rest have one

The matrix sizes of tightbinding `model` must match the orbitals specified. Internally, we
define a block size `N = max(num_orbitals)`. If `N = 1` (all sublattices with one orbital)
the the Hamiltonian element type is `type`. Otherwise it is `SMatrix{N,N,type}` blocks,
padded with the necessary zeros as required. Keyword `type` is `Complex{T}` by default,
where `T` is the number type of `lat`.

    lat |> hamiltonian(model; kw...)

Curried form of `hamiltonian` equivalent to `hamiltonian(lat, model[, funcmodel]; kw...)`.

# Indexing

Indexing into a Hamiltonian `h` works as follows. Access the `HamiltonianHarmonic` matrix at
a given `dn::NTuple{L,Int}` with `h[dn]`. Assign `v` into element `(i,j)` of said matrix
with `h[dn][i,j] = v` or `h[dn, i, j] = v`. Broadcasting with vectors of indices `is` and
`js` is supported, `h[dn][is, js] = v_matrix`.

To add an empty harmonic with a given `dn::NTuple{L,Int}`, do `push!(h, dn)`. To delete it,
do `deleteat!(h, dn)`.

# Examples

```jldoctest
julia> h = hamiltonian(LatticePresets.honeycomb(), hopping(@SMatrix[1 2; 3 4], range = 1/√3), orbitals = Val(2))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> push!(h, (3,3)) # Adding a new Hamiltonian harmonic (if not already present)
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 6 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> h[(3,3)][1,1] = @SMatrix[1 2; 2 1]; h[(3,3)] # element assignment
2×2 SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64} with 1 stored entry:
  [1, 1]  =  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]

julia> h[(3,3)][[1,2],[1,2]] .= Ref(@SMatrix[1 2; 2 1])
2×2 view(::SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64}, [1, 2], [1, 2]) with eltype StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4}:
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]
```

# See also:
    `onsite`, `hopping`, `bloch`, `bloch!`
"""
hamiltonian(lat, ts...; orbitals = missing, kw...) =
    _hamiltonian(lat, sanitize_orbs(orbitals, lat.unitcell.names), ts...; kw...)
_hamiltonian(lat::AbstractLattice, orbs; kw...) = _hamiltonian(lat, orbs, TightbindingModel(); kw...)
_hamiltonian(lat::AbstractLattice, orbs, m::TightbindingModel; type::Type = Complex{numbertype(lat)}, kw...) =
    hamiltonian_sparse(blocktype(orbs, type), lat, orbs, m; kw...)

hamiltonian(t::TightbindingModel...; kw...) = lat -> hamiltonian(lat, t...; kw...)

sanitize_orbs(o::Union{Val,NameType,Integer}, names::NTuple{N}) where {N} =
    ntuple(n -> sanitize_orbs(o), Val(N))
sanitize_orbs(o::NTuple{M,Union{NameType,Integer}}, names::NTuple{N}) where {M,N} =
    (ont = nametype.(o); ntuple(n -> ont, Val(N)))
sanitize_orbs(o::Missing, names) = sanitize_orbs((:a,), names)
sanitize_orbs(o::Pair, names) = sanitize_orbs((o,), names)
sanitize_orbs(os::NTuple{M,Pair}, names::NTuple{N}) where {N,M} =
    ntuple(Val(N)) do n
        for m in 1:M
            first(os[m]) == names[n] && return sanitize_orbs(os[m])
        end
        return (:a,)
    end
sanitize_orbs(os::NTuple{M,Any}, names::NTuple{N}) where {N,M} =
    ntuple(n -> n > M ? (:a,) : sanitize_orbs(os[n]), Val(N))

sanitize_orbs(p::Pair) = sanitize_orbs(last(p))
sanitize_orbs(o::Integer) = (nametype(o),)
sanitize_orbs(o::NameType) = (o,)
sanitize_orbs(o::Val{N}) where {N} = ntuple(_ -> :a, Val(N))
sanitize_orbs(o::NTuple{N,Union{Integer,NameType}}) where {N} = nametype.(o)
sanitize_orbs(p) = throw(ArgumentError("Wrong format for orbitals, see `hamiltonian`"))

Base.Matrix(h::Hamiltonian) = Hamiltonian(h.lattice, Matrix.(h.harmonics), h.orbitals)
Base.Matrix(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, Matrix(h.h))

Base.copy(h::Hamiltonian) = Hamiltonian(copy(h.lattice), copy.(h.harmonics), h.orbitals)
Base.copy(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, copy(h.h))

Base.size(h::Hamiltonian, n) = size(first(h.harmonics).h, n)
Base.size(h::Hamiltonian) = size(first(h.harmonics).h)
Base.size(h::HamiltonianHarmonic, n) = size(h.h, n)
Base.size(h::HamiltonianHarmonic) = size(h.h)

flatsize(h::Hamiltonian, n) = first(flatsize(h)) # h is always square

function flatsize(h::Hamiltonian)
    n = sum(sublatsites(h.lattice) .* length.(h.orbitals))
    return (n, n)
end

function LinearAlgebra.ishermitian(h::Hamiltonian)
    for hh in h.harmonics
        isassigned(h, -hh.dn) || return false
        hh.h == h[-hh.dn]' || return false
    end
    return true
end

bravais(h::Hamiltonian) = bravais(h.lattice)

issemibounded(h::Hamiltonian) = issemibounded(h.lattice)

nsites(h::Hamiltonian) = isempty(h.harmonics) ? 0 : nsites(first(h.harmonics))
nsites(h::HamiltonianHarmonic) = size(h.h, 1)

nsublats(h::Hamiltonian) = nsublats(h.lattice)

norbitals(h::Hamiltonian) = length.(h.orbitals)

# External API #

"""
    sitepositions(lat::AbstractLattice; kw...)
    sitepositions(h::Hamiltonian; kw...)

Build a generator of the positions of sites in the lattice unitcell. Only sites specified
by `siteselector(kw...)` are selected, see `siteselector` for details.

"""
sitepositions(lat::AbstractLattice; kw...) = sitepositions(lat, siteselector(;kw...))
sitepositions(h::Hamiltonian; kw...) = sitepositions(h.lattice, siteselector(;kw...))

"""
    siteindices(lat::AbstractLattice; kw...)
    siteindices(lat::Hamiltonian; kw...)

Build a generator of the unique indices of sites in the lattice unitcell. Only sites
specified by `siteselector(kw...)` are selected, see `siteselector` for details.

"""
siteindices(lat::AbstractLattice; kw...) = siteindices(lat, siteselector(;kw...))
siteindices(h::Hamiltonian; kw...) = siteindices(h.lattice, siteselector(;kw...))

"""
    transform!(h::Hamiltonian, f::Function)

Transform the site positions of the Hamiltonian's lattice in place without modifying the
Hamiltonian harmonics.

    transform!(h::Hamiltonian, bravais::Bravais)

Replace the Bravais matrix of `h`'s lattice with `bravais` in place.

"""
function transform!(h::Hamiltonian, f::Function)
    transform!(h.lattice, f)
    return h
end

function transform!(h::Hamiltonian, br::Bravais)
    transform!(h.lattice, br)
    return h´
end

# Indexing #
Base.push!(h::Hamiltonian{<:Any,L}, dn::NTuple{L,Int}) where {L} = push!(h, SVector(dn...))
Base.push!(h::Hamiltonian{<:Any,L}, dn::Vararg{Int,L}) where {L} = push!(h, SVector(dn...))
function Base.push!(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L} 
    get_or_push!(h.harmonics, dn, size(h))
    return h
end

function get_or_push!(harmonics::Vector{H}, dn::SVector{L,Int}, dims) where {L,M,A,H<:HamiltonianHarmonic{L,M,A}}
    for hh in harmonics
        hh.dn == dn && return hh
    end
    hh = HamiltonianHarmonic{L,M,A}(dn, dims...)
    push!(harmonics, hh)
    return hh
end

Base.getindex(h::Hamiltonian, dn::NTuple) = getindex(h, SVector(dn))
@inline function Base.getindex(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L}
    nh = findfirst(hh -> hh.dn == dn, h.harmonics)
    nh === nothing && throw(BoundsError(h, dn))
    return h.harmonics[nh].h
end
Base.getindex(h::Hamiltonian, dn::NTuple, i::Vararg{Int}) = h[dn][i...]
Base.getindex(h::Hamiltonian{LA, L}, i::Vararg{Int}) where {LA,L} = h[zero(SVector{L,Int})][i...]

Base.deleteat!(h::Hamiltonian{<:Any,L}, dn::Vararg{Int,L}) where {L} =
    deleteat!(h, toSVector(dn))
Base.deleteat!(h::Hamiltonian{<:Any,L}, dn::NTuple{L,Int}) where {L} =
    deleteat!(h, toSVector(dn))
function Base.deleteat!(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L}
    nh = findfirst(hh -> hh.dn == SVector(dn...), h.harmonics)
    nh === nothing || deleteat!(h.harmonics, nh)
    return h
end

Base.isassigned(h::Hamiltonian, dn::Vararg{Int}) = isassigned(h, SVector(dn))
Base.isassigned(h::Hamiltonian, dn::NTuple) = isassigned(h, SVector(dn))
Base.isassigned(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L} =
    findfirst(hh -> hh.dn == dn, h.harmonics) != nothing

## Boolean masking
"""
    &(h1::Hamiltonian{<:Superlattice}, h2::Hamiltonian{<:Superlattice})

Construct a new `Hamiltonian{<:Superlattice}` using an `and` boolean mask, i.e. with a
supercell that contains cells that are both in the supercell of `h1` and `h2`

    &(s1::Superlattice, s2::Superlattice}

Equivalent of the above for `Superlattice`s

# See also:
    `|`, `xor`
"""
(Base.:&)(s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice}) =
    boolean_mask_hamiltonian(Base.:&, s1, s2)

"""
    |(h1::Hamiltonian{<:Superlattice}, h2::Hamiltonian{<:Superlattice})

Construct a new `Hamiltonian{<:Superlattice}` using an `or` boolean mask, i.e. with a
supercell that contains cells that are either in the supercell of `h1` or `h2`

    |(s1::Superlattice, s2::Superlattice}

Equivalent of the above for `Superlattice`s

# See also:
    `&`, `xor`
"""
(Base.:|)(s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice}) =
    boolean_mask_hamiltonian(Base.:|, s1, s2)

"""
    xor(h1::Hamiltonian{<:Superlattice}, h2::Hamiltonian{<:Superlattice})

Construct a new `Hamiltonian{<:Superlattice}` using a `xor` boolean mask, i.e. with a
supercell that contains cells that are either in the supercell of `h1` or `h2` but not in
both

    xor(s1::Superlattice, s2::Superlattice}

Equivalent of the above for `Superlattice`s

# See also:
    `&`, `|`
"""
(Base.xor)(s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice}) =
    boolean_mask_hamiltonian(Base.xor, s1, s2)

function boolean_mask_hamiltonian(f, s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice})
    check_compatible_hsuper(s1, s2)
    return Hamiltonian(f(s1.lattice, s2.lattice), s1.harmonics, s1.orbitals)
end

function check_compatible_hsuper(s1, s2)
    compatible = isequal(s1.harmonics, s2.harmonics) && isequal(s1.orbitals, s2.orbitals)
    compatible || throw(ArgumentError("Hamiltonians are incompatible for boolean masking"))
    return nothing
end

#######################################################################
# auxiliary types
#######################################################################
struct IJV{L,M}
    dn::SVector{L,Int}
    i::Vector{Int}
    j::Vector{Int}
    v::Vector{M}
end

struct IJVBuilder{L,M,E,T,O,LA<:AbstractLattice{E,L,T}}
    lat::LA
    orbs::O
    ijvs::Vector{IJV{L,M}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end

IJV{L,M}(dn::SVector{L} = zero(SVector{L,Int})) where {L,M} =
    IJV(dn, Int[], Int[], M[])

function IJVBuilder(lat::AbstractLattice{E,L,T}, orbs, ijvs::Vector{IJV{L,M}}) where {E,L,T,M}
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, orbs, ijvs, kdtrees)
end

IJVBuilder(lat::AbstractLattice{E,L}, orbs, ::Type{M}) where {E,L,M} =
    IJVBuilder(lat, orbs, IJV{L,M}[])

function IJVBuilder(lat::AbstractLattice{E,L}, orbs, hs::Hamiltonian...) where {E,L}
    M = promote_blocktype(hs...)
    ijvs = IJV{L,M}[]
    builder = IJVBuilder(lat, orbs, ijvs)
    offset = 0
    for h in hs
        for har in h.harmonics
            ijv = builder[har.dn]
            push_block!(ijv, har, offset)
        end
        offset += size(h, 1)
    end
    return builder
end

Base.eltype(b::IJVBuilder{L,M}) where {L,M} = M

function Base.getindex(b::IJVBuilder{L,M}, dn::SVector{L2,Int}) where {L,L2,M}
    L == L2 || throw(error("Tried to apply an $L2-dimensional model to an $L-dimensional lattice"))
    for e in b.ijvs
        e.dn == dn && return e
    end
    e = IJV{L,M}(dn)
    push!(b.ijvs, e)
    return e
end

Base.length(h::IJV) = length(h.i)
Base.isempty(h::IJV) = length(h) == 0
Base.copy(h::IJV) = IJV(h.dn, copy(h.i), copy(h.j), copy(h.v))

function Base.resize!(h::IJV, n)
    resize!(h.i, n)
    resize!(h.j, n)
    resize!(h.v, n)
    return h
end

Base.push!(ijv::IJV, (i, j, v)::Tuple) = (push!(ijv.i, i); push!(ijv.j, j); push!(ijv.v, v))

function push_block!(ijv::IJV{L,M}, h::HamiltonianHarmonic, offset) where {L,M}
    I, J, V = findnz(h.h)
    for (i, j, v) in zip(I, J, V)
        push!(ijv, (i + offset, j + offset, padtotype(v, M)))
    end
    return ijv
end

#######################################################################
# hamiltonian_sparse
#######################################################################
function hamiltonian_sparse(Mtype, lat, orbs, model)
    builder = IJVBuilder(lat, orbs, Mtype)
    return hamiltonian_sparse!(builder, lat, orbs, model)
end

function hamiltonian_sparse!(builder::IJVBuilder{L,M}, lat::AbstractLattice{E,L}, orbs, model) where {E,L,M}
    applyterms!(builder, terms(model)...)
    n = nsites(lat)
    HT = HamiltonianHarmonic{L,M,SparseMatrixCSC{M,Int}}
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n)) for e in builder.ijvs if !isempty(e)]
    return Hamiltonian(lat, harmonics, orbs, n, n)
end

applyterms!(builder, terms...) = foreach(term -> applyterm!(builder, term), terms)

applyterm!(builder::IJVBuilder, term::Union{OnsiteTerm, HoppingTerm}) =
    applyterm!(builder, term)

function applyterm!(builder::IJVBuilder{L}, term::OnsiteTerm) where {L}
    lat = builder.lat
    dn0 = zero(SVector{L,Int})
    ijv = builder[dn0]
    allpos = allsitepositions(lat)
    rsel = resolve(term.selector, lat)
    for s in sublats(rsel), i in siteindices(rsel, s)
        r = allpos[i]
        v = toeltype(term(r, r), eltype(builder), builder.orbs[s], builder.orbs[s])
        push!(ijv, (i, i, v))
    end
    return nothing
end

function applyterm!(builder::IJVBuilder{L}, term::HoppingTerm) where {L}
    lat = builder.lat
    rsel = resolve(term.selector, lat)
    L > 0 && checkinfinite(rsel)
    allpos = allsitepositions(lat)
    for (s2, s1) in sublats(rsel)  # Each is a Pair s2 => s1
        dns = dniter(rsel)
        for dn in dns
            keepgoing = false
            ijv = builder[dn]
            for j in source_candidates(rsel, s2)
                sitej = allpos[j]
                rsource = sitej - lat.bravais.matrix * dn
                is = targets(builder, rsel.selector.range, rsource, s1)
                for i in is
                    # Make sure we don't stop searching until we reach minimum range
                    is_below_min_range((i, j), (dn, zero(dn)), rsel) && (keepgoing = true)
                    ((i, j), (dn, zero(dn))) in rsel || continue
                    keepgoing = true
                    rtarget = allsitepositions(lat)[i]
                    r, dr = _rdr(rsource, rtarget)
                    v = toeltype(term(r, dr), eltype(builder), builder.orbs[s1], builder.orbs[s2])
                    push!(ijv, (i, j, v))
                end
            end
            keepgoing && acceptcell!(dns, dn)
        end
    end
    return nothing
end

# For use in Hamiltonian building
toeltype(t::Number, ::Type{T}, t1::NTuple{1}, t2::NTuple{1}) where {T<:Number} = T(t)
toeltype(t::Number, ::Type{S}, t1::NTuple{1}, t2::NTuple{1}) where {S<:SMatrix} =
    padtotype(t, S)
toeltype(t::SMatrix{N1,N2}, ::Type{S}, t1::NTuple{N1}, t2::NTuple{N2}) where {N1,N2,S<:SMatrix} =
    padtotype(t, S)

toeltype(u::UniformScaling, ::Type{T}, t1::NTuple{1}, t2::NTuple{1}) where {T<:Number} = T(u.λ)
toeltype(u::UniformScaling, ::Type{S}, t1::NTuple{N1}, t2::NTuple{N2}) where {N1,N2,S<:SMatrix} =
    padtotype(SMatrix{N1,N2}(u), S)

# For use in ket building
toeltype(t::Number, ::Type{T}, t1::NTuple{1}) where {T<:Number} = T(t)
toeltype(t::Number, ::Type{S}, t1::NTuple{1}) where {S<:SVector} = padtotype(t, S)
toeltype(t::SVector{N}, ::Type{S}, t1::NTuple{N}) where {N,S<:SVector} = padtotype(t, S)

# Fallback to catch mismatched or undesired block types
toeltype(t::Array, x...) = throw(ArgumentError("Array input in model, please use StaticArrays instead (e.g. SA[1 0; 0 1] instead of [1 0; 0 1])"))
toeltype(t, x...) = throw(DimensionMismatch("Dimension mismatch between model and Hamiltonian. Does the `orbitals` kwarg in your `hamiltonian` match your model?"))

# Although range can be (rmin, rmax) we return all targets within rmax.
# Those below rmin get filtered later by `in rsel`
function targets(builder, range, rsource, s1)
    rmax = maximum(range)
    !isfinite(rmax) && return targets(builder, missing, rsource, s1)
    if !isassigned(builder.kdtrees, s1)
        sitepos = sitepositions(builder.lat.unitcell, s1)
        (builder.kdtrees[s1] = KDTree(sitepos))
    end
    targetlist = inrange(builder.kdtrees[s1], rsource, rmax)
    targetlist .+= builder.lat.unitcell.offsets[s1]
    return targetlist
end

targets(builder, range::Missing, rsource, s1) = siterange(builder.lat, s1)

checkinfinite(rs) =
    rs.selector.dns === missing && (rs.selector.range === missing || !isfinite(maximum(rs.selector.range))) &&
    throw(ErrorException("Tried to implement an infinite-range hopping on an unbounded lattice"))

#######################################################################
# Matrix(::KetModel, ::Hamiltonian), and Vector
#######################################################################
"""
  Vector(km::KetModel, h::Hamiltonian)

Construct a `Vector` representation of `km` applied to Hamiltonian `h`.
"""
Base.Vector(km::KetModel, h::Hamiltonian) = vec(Matrix(km, h))

"""
  Matrix(km::KetModel, h::Hamiltonian; orthogonal = false)
  Matrix(kms::NTuple{N,KetModel}, h::Hamiltonian, orthogonal = false)
  Matrix(kms::AbstractMatrix, h::Hamiltonian; orthogonal = false)
  Matrix(kms::StochasticTraceKets, h::Hamiltonian; orthogonal = false)

Construct an `M×N` `Matrix` representation of the `N` kets `kms` applied to `M×M`
Hamiltonian `h`. If `orthogonal = true`, the columns are made orthogonal through a
Gram-Schmidt process. If `kms::StochasticTraceKets` for `n` random kets (constructed with
`randomkets(n)`), a normalization `1/√n` required for stochastic traces is included.
"""
Base.Matrix(km::KetModel, h::Hamiltonian) = Matrix((km,), h)

function Base.Matrix(km::AbstractMatrix, h::Hamiltonian; orthogonal = false)
    eltype(km) == orbitaltype(h) && size(km, 1) == size(h, 2) || throw(ArgumentError("ket vector or matrix is incompatible with Hamiltonian"))
    kmat = Matrix(km)
    orthogonal && make_orthogonal!(kmat, kms)
    return kmat
end

function Base.Matrix(rk::StochasticTraceKets, h::Hamiltonian)
    ketmodels = Base.Iterators.repeated(rk.ketmodel, rk.repetitions)
    kmat = Matrix(ketmodels, h; orthogonal = rk.orthogonal)
    normk = sqrt(1/size(kmat,2))
    kmat .*= normk  # normalized for stochastic traces
    return kmat
end

function Base.Matrix(kms, h::Hamiltonian; orthogonal = false)
    M = orbitaltype(h)
    kmat = zeros(M, size(h, 2), length(kms))
    for (j, km) in enumerate(kms)
        kvec = view(kmat, :, j)
        ket!(kvec, km, h)
    end
    orthogonal && make_orthogonal!(kmat, kms)
    return kmat
end

function ket!(k, km::KetModel, h)
    M = eltype(k)
    fill!(k, zero(M))
    hsites = allsitepositions(h.lattice)
    for term in km.model.terms
        rs = resolve(term.selector, h.lattice)
        ss = sublats(rs)
        for s in ss
            orbs = h.orbitals[s]
            is = siterange(h.lattice, s)
            for i in is
                i in rs || continue
                r = hsites[i]
                k[i] += generate_amplitude(km, term, r, M, orbs)
            end
        end
    end
    km.normalized && normalize!(k)
    return k
end

function make_orthogonal!(kmat::AbstractMatrix{<:Number}, kms)
    q, r = qr!(kmat)
    kmat .= Matrix(q)
    for (j, km) in enumerate(kms)
        if !km.normalized
            kmat[:,j] .*= r[j, j]
        end
    end
    return kmat
end

make_orthogonal!(kmat, kms) = throw(ArgumentError("The orthogonalize option is only available for kets of scalar eltype, not for $(eltype(kmat))."))

#######################################################################
# unitcell/supercell for Hamiltonians
#######################################################################
function supercell(ham::Hamiltonian, args...; kw...)
    slat = supercell(ham.lattice, args...; kw...)
    return Hamiltonian(slat, ham.harmonics, ham.orbitals)
end

function unitcell(ham::Hamiltonian{<:Lattice}, args...; modifiers = (), kw...)
    sham = supercell(ham, args...; kw...)
    return unitcell(sham; modifiers = modifiers)
end

function unitcell(ham::Hamiltonian{LA,L}; modifiers = ()) where {E,L,T,L´,LA<:Superlattice{E,L,T,L´}}
    lat = ham.lattice
    sc = lat.supercell
    modifiers´ = resolve.(ensuretuple(modifiers), Ref(lat))
    mapping = OffsetArray{Int}(undef, sc.sites, sc.cells.indices...) # store supersite indices newi
    mapping .= 0
    foreach_supersite((s, oldi, olddn, newi) -> mapping[oldi, Tuple(olddn)...] = newi, lat)
    dim = nsites(sc)
    B = blocktype(ham)
    S = typeof(SparseMatrixBuilder{B}(dim, dim))
    harmonic_builders = HamiltonianHarmonic{L´,B,S}[]
    pinvint = pinvmultiple(sc.matrix)
    foreach_supersite(lat) do s, source_i, source_dn, newcol
        for oldh in ham.harmonics
            rows = rowvals(oldh.h)
            vals = nonzeros(oldh.h)
            target_dn = source_dn + oldh.dn
            super_dn = new_dn(target_dn, pinvint)
            wrapped_dn = wrap_dn(target_dn, super_dn, sc.matrix)
            newh = get_or_push!(harmonic_builders, super_dn, dim, newcol)
            for p in nzrange(oldh.h, source_i)
                target_i = rows[p]
                # check: wrapped_dn could exit bounding box along non-periodic direction
                checkbounds(Bool, mapping, target_i, Tuple(wrapped_dn)...) || continue
                newrow = mapping[target_i, Tuple(wrapped_dn)...]
                val = applymodifiers(vals[p], lat, (source_i, target_i), (source_dn, target_dn), modifiers´...)
                iszero(newrow) || pushtocolumn!(newh.h, newrow, val)
            end
        end
        foreach(h -> finalizecolumn!(h.h), harmonic_builders)
    end
    harmonics = [HamiltonianHarmonic(h.dn, sparse(h.h)) for h in harmonic_builders]
    unitlat = unitcell(lat)
    orbs = ham.orbitals
    return Hamiltonian(unitlat, harmonics, orbs)
end

function get_or_push!(hs::Vector{<:HamiltonianHarmonic{L,B,<:SparseMatrixBuilder}}, dn, dim, currentcol) where {L,B}
    for h in hs
        h.dn == dn && return h
    end
    newh = HamiltonianHarmonic(dn, SparseMatrixBuilder{B}(dim, dim))
    currentcol > 1 && finalizecolumn!(newh.h, currentcol - 1) # for columns that have been already processed
    push!(hs, newh)
    return newh
end

wrap_dn(olddn::SVector, newdn::SVector, supercell::SMatrix) = olddn - supercell * newdn

applymodifiers(val, lat, inds, dns) = val

function applymodifiers(val, lat, inds, dns, m::UniformModifier, ms...)
    selected = (inds, dns) in m.selector
    val´ = selected ? m.f(val) : val
    return applymodifiers(val´, lat, inds, dns, ms...)
end

function applymodifiers(val, lat, (row, col), (dnrow, dncol), m::OnsiteModifier, ms...)
    selected = ((row, col), (dnrow, dncol)) in m.selector
    if selected
        r = allsitepositions(lat)[col] + bravais(lat) * dncol
        val´ = selected ? m(val, r) : val
    else
        val´ = val
    end
    return applymodifiers(val´, lat, (row, col), (dnrow, dncol), ms...)
end

function applymodifiers(val, lat, (row, col), (dnrow, dncol), m::HoppingModifier, ms...)
    selected = ((row, col), (dnrow, dncol)) in m.selector
    if selected
        br = bravais(lat)
        r, dr = _rdr(allsitepositions(lat)[col] + br * dncol, allsitepositions(lat)[row] + br * dnrow)
        val´ = selected ? m(val, r, dr) : val
    else
        val´ = val
    end
    return applymodifiers(val´, lat, (row, col), (dnrow, dncol), ms...)
end

#######################################################################
# wrap
#######################################################################
"""
    wrap(h::Hamiltonian, axes; phases = missing)

Build a new Hamiltonian from `h` reducing its dimensions from `L` to `L - length(axes)` by
wrapping the specified Bravais `axes` into a loop. `axes` can be an integer ∈ 1:L or a tuple
of such integers. If `phases` are given (with `length(axes) == length(phases)`), the wrapped
hoppings at a cell distance `dn` along `axes` will be multiplied by a factor
`cis(-dot(phases, dn))`. This is useful, for example, to represent a flux Φ through a loop,
using a single `axes = 1` and `phases = 2π * Φ/Φ₀`.

    wrap(h::Hamiltonian; kw...)

Wrap all axes of `h`, yielding a compactified zero-dimensional Hamiltonian.

    h |> wrap(axes; kw...)

Curried form equivalent to `wrap(h, axes; kw...)`.

# Examples

```jldoctest
julia> LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = 1/√3)) |>
       unitcell((1,-1), (10, 10)) |> wrap(2)
Hamiltonian{<:Lattice} : Hamiltonian on a 1D Lattice in 2D space
  Bloch harmonics  : 3 (SparseMatrixCSC, sparse)
  Harmonic size    : 40 × 40
  Orbitals         : ((:a,), (:a,))
  Element type     : scalar (Complex{Float64})
  Onsites          : 0
  Hoppings         : 120
  Coordination     : 3.0
```
"""
wrap(h::Hamiltonian, axis::Int; kw...) = wrap(h, (axis,); kw...)

wrap(h::Hamiltonian{<:Lattice,L}; kw...) where {L} = wrap(h, ntuple(identity, Val(L)); kw...)

function wrap(h::Hamiltonian{<:Lattice,L}, axes::NTuple{N,Int}; phases = missing) where {L,N}
    all(axis -> 1 <= axis <= L, axes) && allunique(axes) || throw(ArgumentError("wrap axes should be unique and between 1 and the lattice dimension $L"))
    lattice´ = _wrap(h.lattice, axes)
    phases´ = (phases === missing) ? filltuple(0, Val(N)) : phases
    harmonics´ = _wrap(h.harmonics, axes, phases´, size(h))
    return Hamiltonian(lattice´, harmonics´, h.orbitals)
end

wrap(axes::Union{Integer,Tuple}; kw...) = h -> wrap(h, axes; kw...)

_wrap(lat::Lattice, axes) = Lattice(_wrap(lat.bravais, axes), lat.unitcell)

function _wrap(br::Bravais{E,L}, axes) where {E,L}
    mask = deletemultiple_nocheck(SVector{L}(1:L), axes)
    return Bravais(br.matrix[:, mask], br.semibounded[mask])
end

function _wrap(harmonics::Vector{HamiltonianHarmonic{L,M,A}}, axes::NTuple{N,Int}, phases::NTuple{N,Number}, sizeh) where {L,M,A,N}
    harmonics´ = HamiltonianHarmonic{L-N,M,A}[]
    for har in harmonics
        dn = har.dn
        dn´ = deletemultiple_nocheck(dn, axes)
        phase = -sum(phases .* dn[SVector(axes)])
        newh = get_or_push!(harmonics´, dn´, sizeh)
        # map!(+, newh, newh, factor * har.h) # TODO: activate after resolving #37375
        newh.h .+= cis(phase) .* har.h
    end
    return harmonics´
end

#######################################################################
# combine
#######################################################################
"""
    combine(hams::Hamiltonian...; coupling = missing)

Build a new Hamiltonian `h` that combines all `hams` as diagonal blocks, and applies
`coupling::Model`, if provided, to build the off-diagonal couplings. Note that the diagonal
blocks are not modified by the coupling model.
"""
combine(hams::Hamiltonian...; coupling = missing) = _combine(coupling, hams...)

_combine(::Missing, hams...) = _combine(TightbindingModel(), hams...)

function _combine(model::TightbindingModel, hams::Hamiltonian...)
    lat = combine((h -> h.lattice).(hams)...)
    orbs = tuplejoin((h -> h.orbitals).(hams)...)
    builder = IJVBuilder(lat, orbs, hams...)
    model´ = offdiagonal(model, lat, nsublats.(hams))
    ham = hamiltonian_sparse!(builder, lat, orbs, model´)
    return ham
end

#######################################################################
# Bloch routines
#######################################################################
"""
    similarmatrix(h::Hamiltonian)

Create an uninitialized matrix of the same type and size of the Hamiltonian's matrix,
calling `optimize!(h)` first to produce an optimal work matrix in the sparse case.

    similarmatrix(h::Hamiltonian, T::Type{<:AbstractMatrix})

Specifies the desired type `T` of the uninitialized matrix.

    similarmatrix(h::Hamiltonian, method::AbstractDiagonalizeMethod)

Adapts the type of the matrix (e.g. dense/sparse) to the specified `method`

    similarmatrix(x::Union{ParametricHamiltonian, GreensFunction}, ...)

Equivalent to the above, but adapted to the more general type of `x`.
"""
function similarmatrix(h, ::Type{A´} = matrixtype(h)) where {A´<:AbstractMatrix}
    optimize!(h)
    return _similarmatrix(parent(h), matrixtype(h), A´)
end

similarmatrix(h, ::Missing) = similarmatrix(h)

# We only provide the type combinastions that make sense
_similarmatrix(h, ::Type{A}, ::Type{A´}) where {A´,A<:A´} =
    similar(h.harmonics[1].h)

_similarmatrix(h, ::Type{A}, ::Type{AbstractMatrix{T´}}) where {T<:Number,A<:AbstractMatrix{T},T´<:Number} =
    similar(h.harmonics[1].h, T´)

_similarmatrix(h, ::Type{A}, ::Type{AbstractMatrix{T´}}) where {N,T<:SMatrix{N,N},A<:AbstractMatrix{T},T´<:SMatrix{N,N}} =
    similar(h.harmonics[1].h, T´)

_similarmatrix(h, ::Type{A}, ::Type{A´}) where {N,T<:SMatrix{N,N},A<:AbstractMatrix{T},T´<:SMatrix{N,N},A´<:DenseMatrix{T´}} =
    similar(A´, size(h)...)

_similarmatrix(h, ::Type{A}, ::Type{A´}) where {T<:Number,A<:AbstractMatrix{T},T´<:Number,A´<:DenseMatrix{T´}} =
    similar(A´, size(h)...)

_similarmatrix(h, ::Type{A}, ::Type{A´}) where {T<:SMatrix,A<:AbstractMatrix{T},T´<:Number,A´<:DenseMatrix{T´}} =
    similar(A´, flatsize(h)...)

_similarmatrix(h, ::Type{A}, ::Type{AbstractMatrix{T´}}) where {T<:SMatrix,A<:AbstractMatrix{T},T´<:Number} =
    _flatten(h.harmonics[1].h, length.(h.orbitals), h.lattice, T´)

_similarmatrix(h, ::Type{A}, ::Type{A´}) where {T<:SMatrix,A<:AbstractSparseMatrix{T},T´<:Number,A´<:AbstractSparseMatrix{T´}} =
    _flatten(h.harmonics[1].h, length.(h.orbitals), h.lattice, T´)

"""
    optimize!(h::Hamiltonian)

Prepare a sparse Hamiltonian `h` to increase the performance of subsequent calls to
`bloch(h, ϕs)` and `bloch!(matrix, h, ϕs)` by minimizing memory reshufflings. It also
adds missing structural zeros to the diagonal to enable shifts by `α*I` (for
shift-and-invert methods).

No optimization will be performed on non-sparse Hamiltonians, or those defined on
`Superlattice`s, for which Bloch Hamiltonians are lazily evaluated.

Note that when calling `similarmatrix(h)` on a sparse `h`, `optimize!` is called first.

# See also:
    `bloch`, `bloch!`
"""
function optimize!(ham::Hamiltonian{<:Lattice,L,M,A}) where {L,M,A<:SparseMatrixCSC}
    h0 = first(ham.harmonics)
    n, m = size(h0.h)
    iszero(h0.dn) || throw(ArgumentError("First Hamiltonian harmonic is not the fundamental"))
    nh = length(ham.harmonics)
    builder = SparseMatrixBuilder{M}(n, m)
    for col in 1:m
        for i in eachindex(ham.harmonics)
            h = ham.harmonics[i].h
            for p in nzrange(h, col)
                v = i == 1 ? nonzeros(h)[p] : zero(M)
                row = rowvals(h)[p]
                pushtocolumn!(builder, row, v, false) # skips repeated rows
            end
        end
        pushtocolumn!(builder, col, zero(M), false) # if not present already, add structural zeros to diagonal
        finalizecolumn!(builder)
    end
    ho = sparse(builder)
    copy!(h0.h, ho) # Inject new structural zeros into zero harmonics
    return ham
end
# IDEA: could sum and subtract all harmonics instead
# Tested, it is slower

function optimize!(ham::Hamiltonian{<:Lattice,L,M,A}) where {L,M,A<:AbstractMatrix}
    # @warn "Hamiltonian is not sparse. Nothing changed."
    return ham
end

function optimize!(ham::Hamiltonian{<:Superlattice})
    # @warn "Hamiltonian is defined on a Superlattice. Nothing changed."
    return ham
end

"""
    bloch(h::Hamiltonian{<:Lattice}, ϕs)

Build the Bloch Hamiltonian matrix of `h`, for Bloch phases `ϕs = (ϕ₁, ϕ₂,...)` (or an
`SVector(ϕs...)`). In terms of Bloch wavevector `k`, `ϕs = k * bravais(h)`, it is defined as
`H(ϕs) = ∑exp(-im * ϕs' * dn) h_dn` where `h_dn` are Bloch harmonics connecting unit cells
at a distance `dR = bravais(h) * dn`.

    bloch(h::Hamiltonian{<:Lattice})

Build the intra-cell Hamiltonian matrix of `h`, without adding any Bloch harmonics.

    bloch(h::Hamiltonian{<:Lattice}, ϕs, axis::Int)

A nonzero `axis` produces the derivative of the Bloch matrix respect to `ϕs[axis]` (i.e. the
velocity operator along this axis), `∂H(ϕs) = ∑ -im * dn[axis] * exp(-im * ϕs' * dn) h_dn`

    bloch(matrix, h::Hamiltonian{<:Lattice}, ϕs::NTuple{L,Real}, dnfunc::Function)

Generalization that applies a prefactor `dnfunc(dn) * exp(im * ϕs' * dn)` to the `dn`
harmonic.

    bloch(ph::ParametricHamiltonian, pϕs, [axis])

Same as above, but with `pϕs = (p₁,...,pᵢ, ϕ₁, ..., ϕⱼ)`, with `p` values for
`parameters(ph)` and `ϕ` Bloch phases.

    h |> bloch(ϕs, ...)

Curried forms of `bloch`, equivalent to `bloch(h, ϕs, ...)`

# Notes

`bloch` allocates a new matrix on each call. For a non-allocating version of `bloch`, see
`bloch!`.

# Examples

```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(onsite(1) + hopping(2)) |> bloch((0, 0))
2×2 SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  13.0+0.0im
  [2, 1]  =  6.0+0.0im
  [1, 2]  =  6.0+0.0im
  [2, 2]  =  13.0+0.0im
```

# See also:
    `bloch!`, `optimize!`, `similarmatrix`
"""
bloch(ϕs, axis = 0) = h -> bloch(h, ϕs, axis)
bloch(h::Hamiltonian, args...) = bloch!(similarmatrix(h), h, args...)

"""
    bloch!(matrix, h::Hamiltonian, ϕs, [axis])

In-place version of `bloch`. Overwrite `matrix` with the Bloch Hamiltonian matrix of `h` for
the specified Bloch phases `ϕs = (ϕ₁,ϕ₂,...)` (see `bloch` for definition and API).  A
conventient way to obtain a `matrix` is to use `similarmatrix(h,...)`, which will return an
`AbstractMatrix` of the same type as the Hamiltonian's. Note, however, that matrix need not
be of the same type (e.g. it can be dense with `Number` eltype for a sparse `h` with
`SMatrix` block eltype).

    bloch!(matrix, ph::ParametricHamiltonian, pϕs, [axis])

Same as above, but with `pϕs = (p₁,...,pᵢ, ϕ₁, ..., ϕⱼ)`, with `p` values for
`parameters(ph)` and `ϕ` Bloch phases.

# Examples

```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(hopping(2I), orbitals = (Val(2), Val(1)));

julia> bloch!(similarmatrix(h), h, (0, 0))
2×2 SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64} with 4 stored entries:
  [1, 1]  =  [12.0+0.0im 0.0+0.0im; 0.0+0.0im 12.0+0.0im]
  [2, 1]  =  [6.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]
  [1, 2]  =  [6.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]
  [2, 2]  =  [12.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]

julia> bloch!(similarmatrix(h, AbstractMatrix{ComplexF64}), h, (0, 0))
3×3 SparseMatrixCSC{Complex{Float64},Int64} with 9 stored entries:
  [1, 1]  =  12.0+0.0im
  [2, 1]  =  0.0+0.0im
  [3, 1]  =  6.0+0.0im
  [1, 2]  =  0.0+0.0im
  [2, 2]  =  12.0+0.0im
  [3, 2]  =  0.0+0.0im
  [1, 3]  =  6.0+0.0im
  [2, 3]  =  0.0+0.0im
  [3, 3]  =  12.0+0.0im

julia> ph = parametric(h, @hopping!((t; α) -> α * t));

julia> bloch!(similarmatrix(ph, AbstractMatrix{ComplexF64}), ph, (2, 0, 0))
3×3 SparseMatrixCSC{Complex{Float64},Int64} with 9 stored entries:
  [1, 1]  =  24.0+0.0im
  [2, 1]  =  0.0+0.0im
  [3, 1]  =  12.0+0.0im
  [1, 2]  =  0.0+0.0im
  [2, 2]  =  24.0+0.0im
  [3, 2]  =  0.0+0.0im
  [1, 3]  =  12.0+0.0im
  [2, 3]  =  0.0+0.0im
  [3, 3]  =  24.0+0.0im
```

# See also:
    `bloch`, `optimize!`, `similarmatrix`
"""
bloch!(matrix, h::Hamiltonian, ϕs = (), axis = 0) = _bloch!(matrix, h, toSVector(ϕs), axis)

function _bloch!(matrix::AbstractMatrix, h::Hamiltonian{<:Lattice,L,M}, ϕs, axis::Number) where {L,M}
    rawmatrix = parent(matrix)
    if iszero(axis)
        _copy!(rawmatrix, first(h.harmonics).h, h) # faster copy!(dense, sparse) specialization
        add_harmonics!(rawmatrix, h, ϕs, dn -> 1)
    else
        fill!(rawmatrix, zero(M)) # There is no guarantee of same structure
        add_harmonics!(rawmatrix, h, ϕs, dn -> -im * dn[axis])
    end
    return matrix
end

function _bloch!(matrix::AbstractMatrix, h::Hamiltonian{<:Lattice,L,M}, ϕs, dnfunc::Function) where {L,M}
    prefactor0 = dnfunc(zero(ϕs))
    rawmatrix = parent(matrix)
    if iszero(prefactor0)
        fill!(rawmatrix, zero(eltype(rawmatrix)))
    else
        _copy!(rawmatrix, first(h.harmonics).h, h)
        rmul!(rawmatrix, prefactor0)
    end
    add_harmonics!(rawmatrix, h, ϕs, dnfunc)
    return matrix
end

add_harmonics!(zerobloch, h::Hamiltonian{<:Lattice}, ϕs::SVector{0}, _) = zerobloch

function add_harmonics!(zerobloch, h::Hamiltonian{<:Lattice,L}, ϕs::SVector{L}, dnfunc) where {L}
    ϕs´ = ϕs'
    for ns in 2:length(h.harmonics)
        hh = h.harmonics[ns]
        hhmatrix = hh.h
        prefactor = dnfunc(hh.dn)
        iszero(prefactor) && continue
        ephi = prefactor * cis(-ϕs´ * hh.dn)
        _add!(zerobloch, hhmatrix, h, ephi)
    end
    return zerobloch
end

############################################################################################
######## _copy! and _add! call specialized methods in tools.jl #############################
############################################################################################

_copy!(dest, src, h) = copy!(dest, src)
_copy!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, h) = _fast_sparse_copy!(dst, src)
_copy!(dst::DenseMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, h) = _fast_sparse_copy!(dst, src)
_copy!(dst::DenseMatrix{<:SMatrix{N,N}}, src::SparseMatrixCSC{<:SMatrix{N,N}}, h) where {N} = _fast_sparse_copy!(dst, src)
_copy!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:SMatrix}, h) = flatten_sparse_copy!(dst, src, h)
_copy!(dst::DenseMatrix{<:Number}, src::DenseMatrix{<:SMatrix}, h) = flatten_dense_copy!(dst, src, h)

_add!(dest, src, h, α) = _plain_muladd!(dest, src, α)
_add!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, h, α = 1) = _fast_sparse_muladd!(dst, src, α)
_add!(dst::AbstractMatrix{<:SMatrix{N,N}}, src::SparseMatrixCSC{<:SMatrix{N,N}}, h, α = I) where {N} = _fast_sparse_muladd!(dst, src, α)
_add!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:SMatrix}, h, α = I) = flatten_sparse_muladd!(dst, src, h, α)
_add!(dst::DenseMatrix{<:Number}, src::DenseMatrix{<:SMatrix}, h, α = I) = flatten_dense_muladd!(dst, src, h, α)

function flatten_sparse_copy!(dst, src, h)
    fill!(dst, zero(eltype(dst)))
    norbs = length.(h.orbitals)
    offsets = h.lattice.unitcell.offsets
    offsets´ = flatoffsets(offsets, norbs)
    coloffset = 0
    for s´ in sublats(h.lattice)
        N´ = norbs[s´]
        for col in siterange(h.lattice, s´)
            for p in nzrange(src, col)
                val = nonzeros(src)[p]
                row = rowvals(src)[p]
                rowoffset, M´ = flatoffsetorbs(row, h.lattice, norbs, offsets´)
                for j in 1:N´, i in 1:M´
                    dst[i + rowoffset, j + coloffset] = val[i, j]
                end
            end
            coloffset += N´
        end
    end
    return dst
end

function flatten_sparse_muladd!(dst, src, h, α = I)
    norbs = length.(h.orbitals)
    offsets = h.lattice.unitcell.offsets
    offsets´ = flatoffsets(offsets, norbs)
    coloffset = 0
    for s´ in sublats(h.lattice)
        N´ = norbs[s´]
        for col in siterange(h.lattice, s´)
            for p in nzrange(src, col)
                val = α * nonzeros(src)[p]
                row = rowvals(src)[p]
                rowoffset, M´ = flatoffsetorbs(row, h.lattice, norbs, offsets´)
                for j in 1:N´, i in 1:M´
                    dst[i + rowoffset, j + coloffset] += val[i, j]
                end
            end
            coloffset += N´
        end
    end
    return dst
end

function flatten_dense_muladd!(dst, src, h, α = I)
    norbs = length.(h.orbitals)
    offsets = h.lattice.unitcell.offsets
    offsets´ = flatoffsets(offsets, norbs)
    coloffset = 0
    for s´ in sublats(h.lattice)
        N´ = norbs[s´]
        for col in siterange(h.lattice, s´)
            rowoffset = 0
            for s in sublats(h.lattice)
                M´ = norbs[s]
                for row in siterange(h.lattice, s)
                    val = α * src[row, col]
                    for j in 1:N´, i in 1:M´
                        dst[i + rowoffset, j + coloffset] += val[i, j]
                    end
                    rowoffset += M´
                end
            end
            coloffset += N´
        end
    end
    return dst
end

function flatten_dense_copy!(dst, src, h)
    fill!(dst, zero(eltype(dst)))
    return flatten_dense_muladd!(dst, src, h, I)
end

# sublat offsets after flattening (without padding zeros)
flatoffsets(offsets, norbs) = _flatoffsets((0,), offsets, norbs...)
_flatoffsets(offsets´::NTuple{N,Any}, offsets, n, ns...) where {N} =
    _flatoffsets((offsets´..., offsets´[end] + n * (offsets[N+1] - offsets[N])), offsets, ns...)
_flatoffsets(offsets´, offsets) = offsets´

# offset of site i after flattening
@inline flatoffset(args...) = first(flatoffsetorbs(args...))

function flatoffsetorbs(i, lat, norbs, offsets´)
    s = sublat(lat, i)
    N = norbs[s]
    offset = lat.unitcell.offsets[s]
    Δi = i - offset
    i´ = offsets´[s] + (Δi - 1) * N
    return i´, N
end

"""
    flatten(h::Hamiltonian)

Flatten a multiorbital Hamiltonian `h` into one with a single orbital per site. The
associated lattice is flattened also, so that there is one site per orbital for each initial
site (all at the same position). Note that in the case of sparse Hamiltonians, zeros in
hopping/onsite matrices are preserved as structural zeros upon flattening.

    h |> flatten()

Curried form equivalent to `flatten(h)` of `h |> flatten` (included for consistency with
the rest of the API).

# Examples

```jldoctest
julia> h = LatticePresets.honeycomb() |>
           hamiltonian(hopping(@SMatrix[1; 2], range = 1/√3, sublats = :A =>:B),
           orbitals = (Val(1), Val(2)))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 3 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a,), (:a, :a))
  Element type     : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 3
  Coordination     : 1.5

julia> flatten(h)
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 3 (SparseMatrixCSC, sparse)
  Harmonic size    : 3 × 3
  Orbitals         : ((:flat,), (:flat,))
  Element type     : scalar (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 2.0
```
"""
flatten() = h -> flatten(h)

function flatten(h::Hamiltonian)
    all(isequal(1), norbitals(h)) && return copy(h)
    harmonics´ = [flatten(har, h.orbitals, h.lattice) for har in h.harmonics]
    lattice´ = flatten(h.lattice, h.orbitals)
    orbitals´ = (_ -> (:flat, )).(h.orbitals)
    return Hamiltonian(lattice´, harmonics´, orbitals´)
end

flatten(h::HamiltonianHarmonic, orbs, lat) =
    HamiltonianHarmonic(h.dn, _flatten(h.h, length.(orbs), lat))

function _flatten(src::SparseMatrixCSC{<:SMatrix{N,N,T}}, norbs::NTuple{S,Any}, lat, ::Type{T´} = T) where {N,T,S,T´}
    offsets´ = flatoffsets(lat.unitcell.offsets, norbs)
    dim´ = last(offsets´)

    builder = SparseMatrixBuilder{T´}(dim´, dim´, nnz(src) * N * N)

    for col in 1:size(src, 2)
        scol = sublat(lat, col)
        for j in 1:norbs[scol]
            for p in nzrange(src, col)
                row = rowvals(src)[p]
                srow = sublat(lat, row)
                rowoffset´ = flatoffset(row, lat, norbs, offsets´)
                val = nonzeros(src)[p]
                for i in 1:norbs[srow]
                    pushtocolumn!(builder, rowoffset´ + i, val[i, j])
                end
            end
            finalizecolumn!(builder, false)
        end
    end
    matrix = sparse(builder)
    return matrix
end

function _flatten(src::DenseMatrix{<:SMatrix{N,N,T}}, norbs::NTuple{S,Any}, lat, ::Type{T´} = T) where {N,T,S,T´}
    offsets´ = flatoffsets(lat.unitcell.offsets, norbs)
    dim´ = last(offsets´)
    matrix = similar(src, T´, dim´, dim´)

    for col in 1:size(src, 2), row in 1:size(src, 1)
        srow, scol = sublat(lat, row), sublat(lat, col)
        nrow, ncol = norbs[srow], norbs[scol]
        val = src[row, col]
        rowoffset´ = flatoffset(row, lat, norbs, offsets´)
        coloffset´ = flatoffset(col, lat, norbs, offsets´)
        for j in 1:ncol, i in 1:nrow
            matrix[rowoffset´ + i, coloffset´ + j] = val[i, j]
        end
    end
    return matrix
end

function flatten(lat::Lattice, orbs)
    length(orbs) == nsublats(lat) || throw(ArgumentError("Msmatch between sublattices and orbitals"))
    unitcell´ = flatten(lat.unitcell, length.(orbs))
    bravais´ = lat.bravais
    lat´ = Lattice(bravais´, unitcell´)
end

function flatten(unitcell::Unitcell, norbs::NTuple{S,Int}) where {S}
    offsets´ = [flatoffsets(unitcell.offsets, norbs)...]
    ns´ = last(offsets´)
    sites´ = similar(unitcell.sites, ns´)
    i = 1
    for sl in 1:S, site in sitepositions(unitcell, sl), rep in 1:norbs[sl]
        sites´[i] = site
        i += 1
    end
    names´ = unitcell.names
    unitcell´ = Unitcell(sites´, names´, offsets´)
    return unitcell´
end