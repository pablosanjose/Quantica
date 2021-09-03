#######################################################################
# OrbitalStructure
#######################################################################
struct OrbitalStructure{T,N,O<:NTuple{N,Tuple{Vararg{NameType}}}}
    orbtype::Type{T}    # T<:Union{Number,SVector} is Hamiltonian's orbitaltype
    orbitals::O
    offsets::Tuple{Int,Vararg{Int,N}}
    flatoffsets::Tuple{Int,Vararg{Int,N}}
end

function OrbitalStructure(::Type{T}, orbs::O, lat::AbstractLattice) where {T,N,O<:NTuple{N,Any}}
    offsets = ntuple(i -> lat.unitcell.offsets[i], Val(N+1))
    offsets´ = flatoffsets(offsets, length.(orbs))
    return OrbitalStructure(T, orbs, offsets, offsets´)
end

OrbitalStructure(::Type{T}, n) where {T} = OrbitalStructure(T, ((:A,),), (0, n), (0, n))

# Equality does not need equal T, or equal orbital names
Base.:(==)(o1::OrbitalStructure, o2::OrbitalStructure) =
    length.(o1.orbitals) == length.(o2.orbitals) && o1.offsets == o2.offsets

function Base.show(io::IO, o::OrbitalStructure{T,N,O}) where {T,N,O}
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => string(i, "  "))
    print(io,
"$(i)OrbitalStructure:
$i  Orbital Type  : $T
$i  Orbitals      : $(orbitals(o))
$i  Sublattices   : $(length(sublats(o)))
$i  Dimensions    : $(dimh(o))")
end

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
                   H<:HamiltonianHarmonic{L,M,A},O<:OrbitalStructure} # <: AbstractMatrix{M}
    lattice::LA
    harmonics::Vector{H}
    orbstruct::O
    # Enforce sorted-dns-starting-from-zero invariant onto harmonics
    function Hamiltonian{LA,L,M,A,H,O}(lattice, harmonics, orbstruct) where
        {LA<:AbstractLattice,L,M,A<:AbstractMatrix, H<:HamiltonianHarmonic{L,M,A},O<:OrbitalStructure}
        dimh = nsites(lattice)
        all(har -> size(har.h) == (dimh, dimh), harmonics) || throw(DimensionMismatch("Harmonics don't match lattice dimensions"))
        length(harmonics) > 0 && iszero(first(harmonics).dn) ||
            pushfirst!(harmonics, H(zero(SVector{L,Int}), dimh, dimh))
        sort!(harmonics, by = h -> abs.(h.dn))
        return new(lattice, harmonics, orbstruct)
    end
end

Hamiltonian(lat::LA, hs::Vector{H}, orb::O) where {LA<:AbstractLattice,L,M,A<:AbstractMatrix, H<:HamiltonianHarmonic{L,M,A},O<:OrbitalStructure} =
    Hamiltonian{LA,L,M,A,H,O}(lat, hs, orb)

Hamiltonian(lat, hs::Vector{H}, orbs::Tuple) where {L,M,H<:HamiltonianHarmonic{L,M}} =
    Hamiltonian(lat, hs, OrbitalStructure(orbitaltype(M), orbs, lat))

orbitals(h::Hamiltonian) = orbitalstructure(h).orbitals

offsets(h::Hamiltonian) = orbitalstructure(h).offsets

flatoffsets(h::Hamiltonian) = orbitalstructure(h).flatoffsets

Base.eltype(::Hamiltonian{<:Any,<:Any,M}) where {M} = M

Base.isequal(h1::HamiltonianHarmonic, h2::HamiltonianHarmonic) =
    h1.dn == h2.dn && h1.h == h2.h
Base.isequal(o1::OrbitalStructure, o2::OrbitalStructure) =
    o1.orbitals == o1.orbitals && o1.offsets == o2.offsets && o1.flatoffsets == o2.flatoffsets

sublat_site(siteidx, o::OrbitalStructure) = sublat_site(siteidx, o.offsets)

dimh(o::OrbitalStructure) = last(o.offsets)

sublats(o::OrbitalStructure) = 1:length(o.orbitals)

orbitals(o::OrbitalStructure) = o.orbitals

siterange(o::OrbitalStructure, sublat) = (1+o.offsets[sublat]):o.offsets[sublat+1]

# sublat offsets after flattening (without padding zeros)
flatoffsets(offsets, norbs) = _flatoffsets((0,), offsets, norbs...)
_flatoffsets(flatoffsets´::NTuple{N,Any}, offsets, n, ns...) where {N} =
    _flatoffsets((flatoffsets´..., flatoffsets´[end] + n * (offsets[N+1] - offsets[N])), offsets, ns...)
_flatoffsets(flatoffsets´, offsets) = flatoffsets´

# offset of site i after flattening
@inline flatoffset_site(i, orbstruct) = first(flatoffsetorbs_site(i, orbstruct))

function flatoffsetorbs_site(i, orbstruct)
    s = sublat_site(i, orbstruct)
    N = length(orbitals(orbstruct)[s])
    offset = orbstruct.offsets[s]
    offset´ = orbstruct.flatoffsets[s]
    Δi = i - offset
    i´ = offset´ + (Δi - 1) * N
    return i´, N
end

function flatindices(i, orbstruct)
    o, N = flatoffsetorbs_site(i, orbstruct)
    return o+1:o+N
end

displaymatrixtype(h::Hamiltonian) = displaymatrixtype(matrixtype(h))
displaymatrixtype(::Type{<:SparseMatrixCSC}) = "SparseMatrixCSC, sparse"
displaymatrixtype(::Type{<:Array}) = "Matrix, dense"
displaymatrixtype(A::Type{<:AbstractArray}) = string(A)
displayelements(h::Hamiltonian) = displayelements(blocktype(h))
displayelements(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = "$N × $N blocks ($T)"
displayelements(::Type{T}) where {T} = "scalar ($T)"
displayorbitals(h::Hamiltonian) = displayorbitals(orbitalstructure(h))
displayorbitals(o::OrbitalStructure) =
    replace(replace(string(orbitals(o)), "Symbol(\"" => ":"), "\")" => "")

function Base.show(io::IO, ham::Hamiltonian)
    i = get(io, :indent, "")
    print(io, i, summary(ham), "\n",
"$i  Bloch harmonics  : $(length(ham.harmonics)) ($(displaymatrixtype(ham)))
$i  Harmonic size    : $((n -> "$n × $n")(nsites(ham)))
$i  Orbitals         : $(displayorbitals(ham))
$i  Element type     : $(displayelements(ham))
$i  Onsites          : $(nonsites(ham))
$i  Hoppings         : $(nhoppings(ham))
$i  Coordination     : $(coordination(ham))")
    ioindent = IOContext(io, :indent => string("  "))
    issuperlattice(ham.lattice) && print(ioindent, "\n", ham.lattice.supercell)
end

Base.summary(h::Hamiltonian{LA}) where {E,L,LA<:Lattice{E,L}} =
    "Hamiltonian{<:Lattice} : Hamiltonian on a $(L)D Lattice in $(E)D space"

Base.summary(::Hamiltonian{LA}) where {E,L,T,L´,LA<:Superlattice{E,L,T,L´}} =
    "Hamiltonian{<:Superlattice} : $(L)D Hamiltonian on a $(L´)D Superlattice in $(E)D space"

#######################################################################
# flatten
#######################################################################
"""
    flatten(h::Hamiltonian)

Flatten a multiorbital Hamiltonian `h` into one with a single orbital per site. The
associated lattice is flattened also, so that there is one site per orbital for each initial
site (all at the same position). Note that in the case of sparse Hamiltonians, zeros in
hopping/onsite matrices are preserved as structural zeros upon flattening.

    flatten(k::Ket)

Flattens a multiorbital `Ket` to have a scalar `eltype`, instead of `SVector`.

    flatten(s::Subspace)

Rebuild `s` by flattening its basis to have a scalar eltype.

    x |> flatten()

Curried form equivalent to `flatten(x)` or `x |> flatten` (included for consistency with
the rest of the API).

    flatten(x, o::OrbitalStructure)

Flatten object x, if applicable, using the orbital structure `o`, as obtained from a
Hamiltonian `h` with `orbitalstructure(h)`. `x` here is typically an `AbstractArray` of
non-scalar `eltype`.

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

# See also
    `unflatten`
"""
flatten() = h -> flatten(h)

function flatten(h::Hamiltonian)
    isflat(h) && return copy(h)
    harmonics´ = [flatten(har, orbitalstructure(h)) for har in h.harmonics]
    lattice´ = flatten(h.lattice, orbitalstructure(h))
    orbs´ = (_ -> (:flat, )).(orbitals(h))
    return Hamiltonian(lattice´, harmonics´, orbs´)
end

# special method: if already flat, don't make a copy
maybeflatten(h, args...) = isflat(h) ? h : flatten(h, args...)

isflat(h::Hamiltonian) = all(isequal(1), norbitals(h))
isflat(m::AbstractArray{<:Number}) = true
isflat(m) = false

flatten(h::HamiltonianHarmonic, orbstruct) =
    HamiltonianHarmonic(h.dn, flatten(h.h, orbstruct))

function flatten(o::OrbitalStructure{T}) where {T}
    T´ = eltype(T)
    orbs = (_ -> (:flat, )).(orbitals(o))
    offsets = o.flatoffsets
    flatoffsets = o.flatoffsets
    return OrbitalStructure(T´, orbs, offsets, flatoffsets)
end

function flatten(lat::Lattice, orbstruct)
    length(orbitals(orbstruct)) == nsublats(lat) || throw(ArgumentError("Mismatch between sublattices and orbitals"))
    unitcell´ = flatten(lat.unitcell, orbstruct)
    bravais´ = lat.bravais
    lat´ = Lattice(bravais´, unitcell´)
end

function flatten(unitcell::Unitcell, orbstruct) # orbs::NTuple{S,Any}) where {S}
    norbs = length.(orbitals(orbstruct))
    nsublats = length(sublats(orbstruct))
    offsets´ = orbstruct.flatoffsets
    ns´ = last(offsets´)
    sites´ = similar(unitcell.sites, ns´)
    i = 1
    for sl in 1:nsublats, site in sitepositions(unitcell, sl), rep in 1:norbs[sl]
        sites´[i] = site
        i += 1
    end
    names´ = unitcell.names
    unitcell´ = Unitcell(sites´, names´, offsets´)
    return unitcell´
end

flatten(src::AbstractArray{<:Number}, orbstruct) = src
flatten(src::AbstractArray{T}, orbstruct, ::Type{T}) where {T<:Number} = src
flatten(src::AbstractArray{T}, orbstruct, ::Type{T´}) where {T<:Number,T´<:Number} = T´.(src)

function flatten(src::SparseMatrixCSC{<:SMatrix{N,N,T}}, orbstruct, ::Type{T´} = T) where {N,T,T´}
    norbs = length.(orbitals(orbstruct))
    offsets´ = orbstruct.flatoffsets
    dim´ = last(offsets´)

    builder = SparseMatrixBuilder{T´}(dim´, dim´, nnz(src) * N * N)

    for col in 1:size(src, 2)
        scol = sublat_site(col, orbstruct)
        for j in 1:norbs[scol]
            for p in nzrange(src, col)
                row = rowvals(src)[p]
                srow = sublat_site(row, orbstruct)
                rowoffset´ = flatoffset_site(row, orbstruct)
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

# Requires src to be a square matrix with structure like a Hamiltonian
function flatten(src::StridedMatrix{<:SMatrix{N,N,T}}, orbstruct, ::Type{T´} = T) where {N,T,T´}
    dim = last(orbstruct.offsets)
    size(src) == (dim, dim) || throw(ArgumentError("Trying to flatten a matrix of size incompatible with orbital structure"))
    dim´ = last(orbstruct.flatoffsets)
    dst = similar(src, T´, dim´, dim´)
    return flatten!(dst, src, orbstruct)
end

# Requires src to be a matrix with first dimension like a Hamiltonian
function flatten(src::StridedMatrix{<:SVector{N,T}}, orbstruct, ::Type{T´} = T) where {N,T,T´}
    dim = last(orbstruct.offsets)
    size(src, 1) == dim || throw(ArgumentError("Trying to flatten a matrix of size incompatible with orbital structure"))
    dim´ = last(orbstruct.flatoffsets)
    dst = similar(src, T´, dim´, size(src, 2))
    return flatten!(dst, src, orbstruct)
end

function flatten!(dst, src::StridedMatrix{<:SMatrix{N,N,T}}, orbstruct, ::Type{T´} = T) where {N,T,T´}
    size(src) == (dimh(orbstruct), dimh(orbstruct)) || throw(ArgumentError("Dimension mismatch between source array and orbital structure"))
    norbs = length.(orbitals(orbstruct))
    for col in 1:size(src, 2), row in 1:size(src, 1)
        srow, scol = sublat_site(row, orbstruct), sublat_site(col, orbstruct)
        nrow, ncol = norbs[srow], norbs[scol]
        val = src[row, col]
        rowoffset´ = flatoffset_site(row, orbstruct)
        coloffset´ = flatoffset_site(col, orbstruct)
        for j in 1:ncol, i in 1:nrow
            dst[rowoffset´ + i, coloffset´ + j] = val[i, j]
        end
    end
    return dst
end

# for Subspace bases
function flatten!(dst, src::StridedMatrix{<:SVector{N,T}}, orbstruct, ::Type{T´} = T) where {N,T,T´}
    norbs = length.(orbitals(orbstruct))
    for col in 1:size(src, 2), row in 1:size(src, 1)
        srow = sublat_site(row, orbstruct)
        nrow = norbs[srow]
        val = src[row, col]
        rowoffset´ = flatoffset_site(row, orbstruct)
        for i in 1:nrow
            dst[rowoffset´ + i, col] = val[i]
        end
    end
    return dst
end

function flatten_sparse_copy!(dst, src, o::OrbitalStructure)
    fill!(dst, zero(eltype(dst)))
    norbs = length.(orbitals(o))
    coloffset = 0
    for s´ in sublats(o)
        N´ = norbs[s´]
        for col in siterange(o, s´)
            for p in nzrange(src, col)
                val = nonzeros(src)[p]
                row = rowvals(src)[p]
                rowoffset, M´ = flatoffsetorbs_site(row, o)
                for j in 1:N´, i in 1:M´
                    dst[i + rowoffset, j + coloffset] = val[i, j]
                end
            end
            coloffset += N´
        end
    end
    return dst
end

# Specialized flattening copy! and muladd!
function flatten_sparse_muladd!(dst, src, o::OrbitalStructure, α = I)
    norbs = length.(orbitals(o))
    coloffset = 0
    for s´ in sublats(o)
        N´ = norbs[s´]
        for col in siterange(o, s´)
            for p in nzrange(src, col)
                val = α * nonzeros(src)[p]
                row = rowvals(src)[p]
                rowoffset, M´ = flatoffsetorbs_site(row, o)
                for j in 1:N´, i in 1:M´
                    dst[i + rowoffset, j + coloffset] += val[i, j]
                end
            end
            coloffset += N´
        end
    end
    return dst
end

function flatten_dense_muladd!(dst, src, o::OrbitalStructure, α = I)
    norbs = length.(orbitals(o))
    coloffset = 0
    for s´ in sublats(o)
        N´ = norbs[s´]
        for col in siterange(o, s´)
            rowoffset = 0
            for s in sublats(o)
                M´ = norbs[s]
                for row in siterange(o, s)
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

function flatten_dense_copy!(dst, src, o::OrbitalStructure)
    fill!(dst, zero(eltype(dst)))
    return flatten_dense_muladd!(dst, src, o, I)
end

## unflatten ##
"""
    unflatten(x, o::OrbitalStructure)

Rebuild object `x` performing the inverse of `flatten(x)` or `flatten(x, o)`. The target `o`
is required.

    x |> unflatten(o::OrbitalStructure)

Curried form equivalent to `unflatten(x, o)` (included for consistency with the
rest of the API).

# Examples
```jldoctest
julia> h = LP.honeycomb() |> hamiltonian(hopping(I), orbitals = (:up,:down)) |> unitcell;

julia> psi = spectrum(h)[around = -1]
Subspace{0}: eigenenergy subspace on a 0D manifold
  Energy       : -0.9999999999999989
  Degeneracy   : 2
  Bloch/params : Float64[]
  Basis eltype : SVector{2, ComplexF64}

julia> psiflat = flatten(psi)
Subspace{0}: eigenenergy subspace on a 0D manifold
  Energy       : -0.9999999999999989
  Degeneracy   : 2
  Bloch/params : Float64[]
  Basis eltype : ComplexF64

julia> unflatten(psiflat, orbitalstructure(psi))
Subspace{0}: eigenenergy subspace on a 0D manifold
  Energy       : -0.9999999999999989
  Degeneracy   : 2
  Bloch/params : Float64[]
  Basis eltype : SVector{2, ComplexF64}

julia> k = ket(ketmodel(1, sublats = :up), h) |> flatten
Ket{ComplexF64}: ket with a 4 × 1 amplitude matrix
  OrbitalStructure:
    Orbital Type  : ComplexF64
    Orbitals      : ((:flat,), (:flat,))
    Sublattices   : 2
    Dimensions    : 4

julia> unflatten(k, orbitalstructure(h))
Ket{SVector{2, ComplexF64}}: ket with a 2 × 1 amplitude matrix
  OrbitalStructure:
    Orbital Type  : SVector{2, ComplexF64}
    Orbitals      : ((:up, :down), (:up, :down))
    Sublattices   : 2
    Dimensions    : 2
```

# See also
    `flatten`, `orbitalstructure`
"""
unflatten(o::OrbitalStructure) = x -> unflatten(x, o)

# Pending implementation for Hamiltonian, Lattice, Unitcell
unflatten_orbitals(v::AbstractVector, o::OrbitalStructure) =
    isunflat_orbitals(v, o) ? copy(v) : unflatten!(similar(v, orbitaltype(o), dimh(o)), v, o)
unflatten_orbitals(m::AbstractMatrix, o::OrbitalStructure) =
    isunflat_orbitals(m, o) ? copy(m) : unflatten!(similar(m, orbitaltype(o), dimh(o), size(m, 2)), m, o)
unflatten_blocks(m::AbstractMatrix, o::OrbitalStructure) =
    isunflat_blocks(m, o) ? copy(m) : unflatten!(similar(m, blocktype(o), dimh(o), dimh(o)), m, o)

maybe_unflatten_orbitals(x, o) = isunflat_orbitals(x, o) ? x : unflatten_orbitals(x, o)
maybe_unflatten_blocks(x, o) = isunflat_blocks(x, o) ? x : unflatten_blocks(x, o)

isunflat_orbitals(m, o) = orbitaltype(o) == eltype(m)
isunflat_blocks(m, o) = blocktype(o) == eltype(m)

function unflatten!(v::AbstractArray, vflat::AbstractArray{<:Number}, o::OrbitalStructure)
    dimflat = last(o.flatoffsets)
    check_unflatten_dst_dims(v, dimh(o))
    check_unflatten_src_dims(vflat, dimflat)
    check_unflatten_eltypes(v, o)
    v = _unflatten!(v, vflat, o)
    return v
end

# unflatten into SMatrix blocks
function _unflatten!(v::AbstractArray{T}, vflat::AbstractArray, o::OrbitalStructure) where {T<:SMatrix}
    norbs = length.(orbitals(o))
    flatoffsets = o.flatoffsets
    col = 0
    for scol in sublats(o)
        M = colstride(norbs, scol, T)
        for j in flatoffsets[scol]+1:M:flatoffsets[scol+1]
            col +=1
            row = 0
            for srow in sublats(o)
                N = rowstride(norbs, srow)
                for i in flatoffsets[srow]+1:N:flatoffsets[srow+1]
                    row += 1
                    val = view(vflat, i:i+N-1, j:j+M-1)
                    v[row, col] = padtotype(val, T)
                end
            end
        end
    end
    return v
end

# unflatten into SVector orbitals
function _unflatten!(v::AbstractArray{T}, vflat::AbstractArray, o::OrbitalStructure) where {T<:SVector}
    norbs = length.(orbitals(o))
    flatoffsets = o.flatoffsets
    col = 0
    for col in 1:size(v, 2)
        row = 0
        for srow in sublats(o)
            N = rowstride(norbs, srow)
            for i in flatoffsets[srow]+1:N:flatoffsets[srow+1]
                row += 1
                val = view(vflat, i:i+N-1, col:col)
                v[row, col] = padtotype(val, T)
            end
        end
    end
    return v
end

rowstride(norbs, s) = norbs[s]
colstride(norbs, s, ::Type{<:SVector}) = 1
colstride(norbs, s, ::Type{<:SMatrix}) = rowstride(norbs, s)

# dest v should be a vector or matrix such that H*v is possible
check_unflatten_dst_dims(v, dimh) =
    size(v, 1) == dimh ||
        throw(ArgumentError("Dimension of destination array is inconsistent with orbital structure"))

# dest v should be a square matrix like the Hamiltonian
check_unflatten_dst_dims(v::AbstractArray{<:SMatrix}, dimh) =
    size(v, 1) == dimh && size(v, 2) == dimh ||
        throw(ArgumentError("Dimension of destination array is inconsistent with orbital structure"))

check_unflatten_src_dims(vflat, dimflat) =
    size(vflat, 1) == dimflat ||
        throw(ArgumentError("Dimension of source array is inconsistent with orbital structure"))

check_unflatten_eltypes(::AbstractArray{T}, o::OrbitalStructure) where {T} =
    T === orbitaltype(o) || T === blocktype(o) ||
        throw(ArgumentError("Eltype of desination array is inconsistent with orbital structure"))

## unflatten_orbitals_or_reinterpret: call unflatten_orbitals but only if we cannot unflatten via reinterpret
unflatten_orbitals_or_reinterpret(vflat, ::Missing) = vflat
# source is already of the correct orbitaltype(h)
function unflatten_orbitals_or_reinterpret(vflat::AbstractArray{T}, o::OrbitalStructure{T}) where {T}
    check_unflatten_dst_dims(vflat, dimh(o))
    return vflat
end

function unflatten_orbitals_or_reinterpret(vflat::AbstractArray{<:Number}, o::OrbitalStructure{<:Number})
    check_unflatten_dst_dims(vflat, dimh(o))
    return vflat
end

# source can be reinterpreted, because the number of orbitals is the same M for all N sublattices
unflatten_orbitals_or_reinterpret(vflat::AbstractArray{T}, o::OrbitalStructure{S,N,<:NTuple{N,NTuple{M}}}) where {T,S,N,M} =
    reinterpret(SVector{M,T}, vflat)
# otherwise call unflatten_orbitals
unflatten_orbitals_or_reinterpret(vflat, o) = unflatten_orbitals(vflat, o)

#######################################################################
# similarmatrix
#######################################################################
"""
    similarmatrix(h::Hamiltonian)

Create an uninitialized matrix of the same type and size of the Hamiltonian's matrix.

    similarmatrix(h::Hamiltonian, T::Type{<:AbstractMatrix})

Make the matrix of type `B<:T`. Can be used to specify a different eltype from `h`'s, (e.g.
`T=SparseMatrixCSC{Float64}` with a multiorbital `h`)

    similarmatrix(h::Hamiltonian, flatten)

Create an unitialized matrix of the same type as a flattened version of `h`'s, i.e. with a
scalar eltype as in the example above.

    similarmatrix(h::Hamiltonian, T::AbstractDiagonalizeMethod)

Adapts the type of the matrix (e.g. dense/sparse) to the specified `method`

    similarmatrix(x::Union{ParametricHamiltonian, GreensFunction}, ...)

Equivalent to the above, but adapted to the more general type of `x`.

# Examples

```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(hopping(I), orbitals = Val(2));

julia> similarmatrix(h) |> summary
"2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64}"

julia> similarmatrix(h, Matrix{Int}) |> summary
"4×4 Matrix{Int64}"

julia> similarmatrix(h, flatten) |> summary
"4×4 SparseMatrixCSC{ComplexF64, Int64}"

julia> similarmatrix(h, LinearAlgebraPackage()) |> summary
"4×4 Matrix{ComplexF64}"
```

# See also
    `bloch!`
"""
similarmatrix(h, dest_type = missing) = _similarmatrix(dest_type, matrixtype(h), h)

_similarmatrix(::Missing, src_type, h) =
    similar_merged(h.harmonics)
_similarmatrix(::Type{A}, ::Type{A´}, h) where {T<:Number,A<:AbstractSparseMatrix{T},T´<:Number,A´<:AbstractSparseMatrix{T´}} =
    similar_merged(h.harmonics, T)
_similarmatrix(::Type{A}, ::Type{A´}, h) where {N,T<:SMatrix{N,N},A<:AbstractSparseMatrix{T},T´<:SMatrix{N,N},A´<:AbstractSparseMatrix{T´}} =
    similar_merged(h.harmonics, T)
_similarmatrix(::Type{A}, ::Type{A´}, h) where {N,T<:Number,A<:AbstractSparseMatrix{T},T´<:SMatrix{N,N},A´<:AbstractSparseMatrix{T´}} =
    flatten(similar_merged(h.harmonics), orbitalstructure(h), T)
_similarmatrix(::Type{A}, ::Type{A´}, h) where {T<:Number,A<:Matrix{T},T´<:Number,A´<:AbstractMatrix{T´}} =
    similar(A, size(h))
_similarmatrix(::Type{A}, ::Type{A´}, h) where {N,T<:SMatrix{N,N},A<:Matrix{T},T´<:SMatrix{N,N},A´<:AbstractMatrix{T´}} =
    similar(A, size(h))
_similarmatrix(::Type{A}, ::Type{A´}, h) where {N,T<:Number,A<:Matrix{T},T´<:SMatrix{N,N},A´<:AbstractMatrix{T´}} =
    similar(A, flatsize(h))

_similarmatrix(::typeof(flatten), ::Type{A´}, h) where {N,T,S<:SMatrix{N,N,T},A´<:AbstractSparseMatrix{S}} =
    _similarmatrix(AbstractSparseMatrix{T}, A´, h)
_similarmatrix(::typeof(flatten), ::Type{A´}, h) where {N,T,S<:SMatrix{N,N,T},A´<:StridedMatrix{S}} =
    _similarmatrix(Matrix{T}, A´, h)
_similarmatrix(::typeof(flatten), ::Type{A´}, h) where {T<:Number,A´<:AbstractArray{T}} =
    _similarmatrix(A´, A´, h)
_similarmatrix(::Type{SparseMatrixCSC}, ::Type{A´}, h) where {T´,A´<:AbstractMatrix{T´}} =
    _similarmatrix(SparseMatrixCSC{T´}, A´, h)
_similarmatrix(::Type{Matrix}, ::Type{A´}, h) where {T´,A´<:AbstractMatrix{T´}} =
    _similarmatrix(Matrix{T´}, A´, h)

_similarmatrix(dest_type, src_type, h) = throw(ArgumentError("Unexpected `blochtype` ($src_type => $dest_type)"))

# ensure we have stored entries for all harmonics in the sparse case
similar_merged(hs::AbstractVector{H}, ::Type{M´} = M) where {M´,M,H<:HamiltonianHarmonic{<:Any,M,<:AbstractSparseMatrix}} =
    _similar_merged_sparse(hs, M´)
similar_merged(hs::AbstractVector{H}, ::Type{M´} = M) where {M´,M,H<:HamiltonianHarmonic{<:Any,M,<:StridedMatrix}} =
    similar(first(hs).h, M´)

function _similar_merged_sparse(hs::Vector{<:HamiltonianHarmonic{L,M}},::Type{M´} = M) where {L,M,M´}
    h0 = first(hs)
    n, m = size(h0.h)
    iszero(h0.dn) || throw(ArgumentError("First Hamiltonian harmonic is not the fundamental"))
    nh = length(hs)
    builder = SparseMatrixBuilder{M´}(n, m)
    for col in 1:m
        for i in eachindex(hs)
            h = hs[i].h
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
    return ho
end
# IDEA: could sum and subtract all harmonics instead: sum(h->h.h, hs)
# Tested, it is slower

#######################################################################
# Hamiltonian API
#######################################################################
"""
    hamiltonian(lat, model; orbitals, orbtype)

Create a `Hamiltonian` by applying `model::TighbindingModel` to the lattice `lat` (see
`hopping` and `onsite` for details on building tightbinding models).

    lat |> hamiltonian(model; kw...)

Curried form of `hamiltonian` equivalent to `hamiltonian(lat, model; kw...)`.

# Keywords

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
the Hamiltonian element type is `orbtype`. Otherwise it is `SMatrix{N,N,orbtype}` blocks,
padded with the necessary zeros as required. Keyword `orbtype` is `Complex{T}` by default,
where `T` is the number type of `lat`.

# Indexing

Indexing into a Hamiltonian `h` works as follows. Access the `HamiltonianHarmonic` matrix at
a given `dn::NTuple{L,Int}` with `h[dn]`. The special `h[]` syntax stands for `h[(0...)]`
for the zero-harmonic. Assign `v` into element `(i,j)` of said matrix with `h[dn][i,j] = v`.
Broadcasting with vectors of indices `is` and `js` is supported, `h[dn][is, js] = v_matrix`.

A slicing syntax `h[rows, cols]` (without specifying `dn`) is also available, that creates a
special `hs::Slice{<:Hamiltonian}` object that represents a slice of the Hamiltonian matrix
restricted to `rows` and `cols`. Here, `rows` and `cols` are collections of site indices, or
alternatively `SiteSelector`s (see `siteselector`s for details). If `rows::Integer` or
`cols::Integer`, they will be converted to a single-element range (to preserve always a
matrix-like slice, unlike for `AbstractArray` indexing). Slices support `bloch` and `bloch!`
to produce the corresponding matrices, and can also be indexed as `hs[dn::Tuple]` that
produces the equivalent to `h[dn][rows, cols]`.

To add an empty harmonic with a given `dn::NTuple{L,Int}`, do `push!(h, dn)`. To delete it,
do `deleteat!(h, dn)`.

# Examples

```jldoctest
julia> h = hamiltonian(LatticePresets.honeycomb(), hopping(@SMatrix[1 2; 3 4], range = 1/√3), orbitals = Val(2))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> push!(h, (3,3)) # Adding a new Hamiltonian harmonic (if not already present)
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 6 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (ComplexF64)
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> h[(3,3)][1,1] = @SMatrix[1 2; 2 1]; h[(3,3)] # element assignment
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 1 stored entry:
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]                      ⋅                     
                     ⋅                                           ⋅                     

julia> h[(3,3)][[1,2],[1,2]] .= Ref(@SMatrix[1 2; 2 1])
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 4 stored entries:
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]
 [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]

 julia> h = unitcell(h); h[]
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 2 stored entries:
                     ⋅                       [1.0+0.0im 2.0+0.0im; 3.0+0.0im 4.0+0.0im]
 [1.0+0.0im 2.0+0.0im; 3.0+0.0im 4.0+0.0im]                      ⋅                     

```

# See also
    `onsite`, `hopping`, `bloch`, `bloch!`
"""
hamiltonian(lat::AbstractLattice, ts...; orbitals = missing, kw...) =
    _hamiltonian(lat, sanitize_orbs(orbitals, lat.unitcell.names), ts...; kw...)
_hamiltonian(lat::AbstractLattice, orbs; kw...) = _hamiltonian(lat, orbs, TightbindingModel(); kw...)
_hamiltonian(lat::AbstractLattice, orbs, m::TightbindingModel; orbtype::Type = Complex{numbertype(lat)}, kw...) =
    hamiltonian_sparse(blocktype(orbs, orbtype), lat, orbs, m; kw...)

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

"""
    orbitalstructure(x::Union{Hamiltonian,ParametricHamiltonian})
    orbitalstructure(x::Subspace)

Return an `OrbitalStructure` containing information about the orbital structure of `x`

# Examples

```jldoctest
julia> sp = spectrum(LP.honeycomb() |> hamiltonian(hopping(I), orbitals = (:up,:down)) |> unitcell);

julia> sp[around = -1] |> orbitalstructure
OrbitalStructure:
  Orbital Type  : SVector{2, ComplexF64}
  Orbitals      : ((:up, :down), (:up, :down))
  Sublattices   : 2
  Dimensions    : 2
```
"""
orbitalstructure(h::Hamiltonian) = h.orbstruct

"""
    dims(lh::Union{Hamiltonian,AbstractLattice}) -> (E, L)

Return a tuple `(E, L)` of the embedding `E` and lattice dimensions `L` of `AbstractLattice`
or `Hamiltonian` `lh`
"""
dims(lat::AbstractLattice{E,L}) where {E,L} = E, L
dims(h::Hamiltonian) = dims(h.lattice)

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
    siteindices(h::Hamiltonian; kw...)
    siteindices(lat_or_h, s::SiteSelector)

Build a generator of the unique indices of sites in the lattice unitcell. Only sites
specified by `siteselector(kw...)` are selected, see `siteselector` for details.

"""
siteindices(lat::AbstractLattice; kw...) = siteindices(lat, siteselector(;kw...))
siteindices(h::Hamiltonian; kw...) = siteindices(h.lattice, siteselector(;kw...))
siteindices(h::Hamiltonian, s::SiteSelector) = siteindices(h.lattice, s)

"""
    transform!(f::Function, h::Hamiltonian)

Transform the site positions of the Hamiltonian's lattice in place without modifying the
Hamiltonian harmonics.
"""
function transform!(f, h::Hamiltonian)
    transform!(f, h.lattice)
    return h
end

latdim(h::Hamiltonian) = last(dims(h.lattice))

matrixtype(::Hamiltonian{LA,L,M,A}) where {LA,L,M,A} = A

blockeltype(::Hamiltonian{<:Any,<:Any,M}) where {M} = eltype(M)

# find SMatrix type that can hold all matrix elements between lattice sites
blocktype(orbs, type::Type{Tv}) where {Tv} =
    _blocktype(orbitaltype(orbs, Tv))
_blocktype(::Type{S}) where {N,Tv,S<:SVector{N,Tv}} = SMatrix{N,N,Tv,N*N}
_blocktype(::Type{S}) where {S<:Number} = S

blocktype(h::Hamiltonian{LA,L,M}) where {LA,L,M} = M
blocktype(o::OrbitalStructure) = _blocktype(orbitaltype(o))

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

orbitaltype(h::Hamiltonian) = orbitaltype(orbitalstructure(h))
orbitaltype(o::OrbitalStructure) = o.orbtype
orbitaltype(::Type{M}) where {M<:Number} = M
orbitaltype(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = SVector{N,T}

# find SVector type that can hold all orbital amplitudes in any lattice sites
orbitaltype(orbs, type::Type{Tv}) where {Tv} =
    _orbitaltype(SVector{1,Tv}, orbs...)
_orbitaltype(::Type{S}, ::NTuple{D,NameType}, os...) where {N,Tv,D,S<:SVector{N,Tv}} =
    (M = max(N,D); _orbitaltype(SVector{M,Tv}, os...))
_orbitaltype(t::Type{SVector{N,Tv}}) where {N,Tv} = t
_orbitaltype(t::Type{SVector{1,Tv}}) where {Tv} = Tv

coordination(ham) = nhoppings(ham) / nsites(ham)

function nhoppings(ham)
    count = 0
    for h in ham.harmonics
        count += iszero(h.dn) ? (_nnz(h.h) - _nnzdiag(h.h)) : _nnz(h.h)
    end
    return count
end

function nonsites(ham)
    count = 0
    for h in ham.harmonics
        iszero(h.dn) && (count += _nnzdiag(h.h))
    end
    return count
end

_nnz(h::AbstractSparseMatrix) = count(!iszero, nonzeros(h)) # Does not include stored zeros
_nnz(h::StridedMatrix) = count(!iszero, h)

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

check_orbital_consistency(::Hamiltonian{<:Any,<:Any,<:Number}) = nothing

function check_orbital_consistency(h::Hamiltonian)
    slats = sublats(h.lattice)
    norbs = norbitals(h)
    M = maximum(norbs)
    all(norbs .== M) && return nothing
    for srow in slats, scol in slats
        nrow, ncol = norbs[srow], norbs[scol]
        nrow == ncol == M && continue
        rows = siterange(h.lattice, srow)
        cols = siterange(h.lattice, scol)
        for hh in h.harmonics
            hrows = rowvals(hh.h)
            nz = nonzeros(hh.h)
            for col in cols, ptr in nzrange(hh.h, col)
                row = hrows[ptr]
                row in rows || continue
                z = nz[ptr]
                is_consistent_orbs(z, nrow, ncol) || _throw_inconsistent_orbs()
            end
        end
    end
    return nothing
end

function is_consistent_orbs(z::SMatrix{N,N}, rows, cols) where {N}
    for j in cols + 1:N, i in 1:N
        iszero(z[i,j]) || return false
    end
    for j in 1:N-cols, i in rows+1:N
        iszero(z[i,j]) || return false
    end
    return true
end

@noinline _throw_inconsistent_orbs() = throw(ArgumentError("Internal error: orbital structure not correctly encoded into Hamiltonian harmonics"))

bravais(h::Hamiltonian) = bravais(h.lattice)

nsites(h::Hamiltonian) = isempty(h.harmonics) ? 0 : nsites(first(h.harmonics))
nsites(h::HamiltonianHarmonic) = size(h.h, 1)

nsublats(h::Hamiltonian) = nsublats(h.lattice)

norbitals(h::Hamiltonian) = length.(orbitals(h))

flatsize(h::Hamiltonian, n) = first(flatsize(h)) # h is always square

function flatsize(h::Hamiltonian)
    n = sum(sublatlengths(h.lattice) .* length.(orbitals(h)))
    return (n, n)
end

expand_supercell_mask(h::Hamiltonian{<:Superlattice}) =
    Hamiltonian(expand_supercell_mask(h.lattice), h.harmonics, orbitals(h))

Base.size(h::Hamiltonian, n) = size(first(h.harmonics).h, n)
Base.size(h::Hamiltonian) = size(first(h.harmonics).h)
Base.size(h::HamiltonianHarmonic, n) = size(h.h, n)
Base.size(h::HamiltonianHarmonic) = size(h.h)

Base.Matrix(h::Hamiltonian) = Hamiltonian(h.lattice, Matrix.(h.harmonics), orbitalstructure(h))
Base.Matrix(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, Matrix(h.h))

Base.copy(h::Hamiltonian) = Hamiltonian(copy(h.lattice), copy.(h.harmonics), orbitalstructure(h))
Base.copy(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, copy(h.h))

function LinearAlgebra.ishermitian(h::Hamiltonian)
    for hh in h.harmonics
        isassigned(h, -hh.dn) || return false
        hh.h ≈ h[-hh.dn]' || return false
    end
    return true
end

Base.isequal(h1::Hamiltonian, h2::Hamiltonian) =
    isequal(h1.lattice, h2.lattice) && isequal(h1.harmonics, h2.harmonics) &&
    isequal(orbitalstructure(h1), orbitalstructure(h2))

SparseArrays.issparse(h::Hamiltonian{LA,L,M,A}) where {LA,L,M,A<:AbstractSparseMatrix} = true
SparseArrays.issparse(h::Hamiltonian{LA,L,M,A}) where {LA,L,M,A} = false

Base.parent(h::Hamiltonian) = h

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

function nonzero_indices(har::HamiltonianHarmonic, rowrange = 1:size(har, 1), colrange = 1:size(har, 2))
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

# Indexing #

Base.getindex(h::Hamiltonian{<:Any,L}) where {L} = h[zero(SVector{L,Int})]
Base.getindex(h::Hamiltonian, dn::NTuple{L}) where {L} = getindex(h, SVector{L,Int}(dn))

@inline function Base.getindex(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L}
    nh = findfirst(hh -> hh.dn == dn, h.harmonics)
    nh === nothing && throw(BoundsError(h, dn))
    return h.harmonics[nh].h
end

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

# See also
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

# See also
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

# See also
    `&`, `|`
"""
(Base.xor)(s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice}) =
    boolean_mask_hamiltonian(Base.xor, s1, s2)

function boolean_mask_hamiltonian(f, s1::Hamiltonian{<:Superlattice}, s2::Hamiltonian{<:Superlattice})
    check_compatible_hsuper(s1, s2)
    return Hamiltonian(f(s1.lattice, s2.lattice), s1.harmonics, orbitals(s1))
end

function check_compatible_hsuper(s1, s2)
    compatible = isequal(s1.harmonics, s2.harmonics) && isequal(s1.orbstruct, s2.orbstruct)
    compatible || throw(ArgumentError("Hamiltonians are incompatible for boolean masking"))
    return nothing
end

## Hamiltonian algebra

function Base.literal_pow(::typeof(^), h::Hamiltonian, p::Val{P}) where {P}
    P > 0 || throw(ArgumentError("Only positive powers of Hamiltonians are supported"))
    P == 1 && return copy(h)
    hhs = similar(h.harmonics, 0)
    nh = length(h.harmonics)
    dims = size(h)
    hhiter = CartesianIndices(ntuple(i -> 1:nh, p))
    for Is in hhiter
        is = Tuple(Is)
        dn = sum(i -> h.harmonics[i].dn, is)
        hh = get_or_push!(hhs, dn, dims)
        hh.h .+= prod((i -> h.harmonics[i].h).(is))
    end
    return Hamiltonian(h.lattice, hhs, orbitalstructure(h))
end

function Base.:*(h1::Hamiltonian, h2::Hamiltonian) where {P}
    check_compatible_hamiltonians(h1, h2)
    hhs = similar(h1.harmonics, 0)
    dims = size(h1)
    for hh1 in h1.harmonics, hh2 in h2.harmonics
        dn = hh1.dn + hh2.dn
        hh = get_or_push!(hhs, dn, dims)
        mul!(hh.h, hh1.h, hh2.h, 1, 1)
    end
    return Hamiltonian(h1.lattice, hhs, orbitalstructure(h1))
end

Base.:*(h::Hamiltonian, p::Number) = p * h

function Base.:*(p::Number, h::Hamiltonian)
    hhs = copy.(h.harmonics)
    for hh in hhs
        hh.h .*= p
    end
    return Hamiltonian(h.lattice, hhs, orbitalstructure(h))
end

Base.:-(h1::Hamiltonian, h2) = h1 + (-h2)
Base.:-(h::Hamiltonian) = (-1) * h

function Base.:+(h1::Hamiltonian, h2::Hamiltonian)
    check_compatible_hamiltonians(h1, h2)
    hhs = copy.(h1.harmonics)
    dims = size(h1)
    for hh2 in h2.harmonics
        hh = get_or_push!(hhs, hh2.dn, dims)
        hh.h .+= hh2.h
    end
    return Hamiltonian(h1.lattice, hhs, orbitalstructure(h1))
end

Base.:+(id::UniformScaling, h::Hamiltonian) = h + id

function Base.:+(h::Hamiltonian, id::UniformScaling)
    hhs = copy.(h.harmonics)
    M = blocktype(h)
    mat = first(hhs).h
    shift_diagonal!(mat, blocktype(h), h, id)
    return Hamiltonian(h.lattice, hhs, orbitalstructure(h))
end

function shift_diagonal!(mat, ::Type{<:Number}, h, id::UniformScaling)
    mat .+= id
    return mat
end

function shift_diagonal!(mat, ::Type{S}, h, id::UniformScaling) where {N,S<:SMatrix{N,N}}
    for s in sublats(h.lattice)
        shift = id.λ * padprojector(S, orbitals(h)[s])
        for i in siterange(h.lattice, s)
            mat[i,i] += shift
        end
    end
    return mat
end

check_compatible_hamiltonians(h1, h2) =
    isequal(h1.lattice, h2.lattice) && isequal(orbitalstructure(h1), orbitalstructure(h2)) && size(h1) == size(h2) ||
        throw(ArgumentError("Cannot combine Hamiltonians with different lattices, dimensions or orbitals"))
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
    return Hamiltonian(lat, harmonics, orbs)
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
        v = to_blocktype(term(r, r), eltype(builder), builder.orbs[s], builder.orbs[s])
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
                rsource = sitej - bravais(lat) * dn
                is = targets(builder, rsel.selector.range, rsource, s1)
                for i in is
                    # Make sure we don't stop searching until we reach minimum range
                    is_below_min_range((i, j), (dn, zero(dn)), rsel) && (keepgoing = true)
                    ((i, j), (dn, zero(dn))) in rsel || continue
                    keepgoing = true
                    rtarget = allsitepositions(lat)[i]
                    r, dr = _rdr(rsource, rtarget)
                    v = to_blocktype(term(r, dr), eltype(builder), builder.orbs[s1], builder.orbs[s2])
                    push!(ijv, (i, j, v))
                end
            end
            keepgoing && acceptcell!(dns, dn)
        end
    end
    return nothing
end

# For use in Hamiltonian building
to_blocktype(t::Number, ::Type{T}, t1::NTuple{1}, t2::NTuple{1}) where {T<:Number} = T(t)
to_blocktype(t::Number, ::Type{S}, t1::NTuple{1}, t2::NTuple{1}) where {S<:SMatrix} =
    padtotype(t, S)
to_blocktype(t::SMatrix{N1,N2}, ::Type{S}, t1::NTuple{N1}, t2::NTuple{N2}) where {N1,N2,S<:SMatrix} =
    padtotype(t, S)

to_blocktype(u::UniformScaling, ::Type{T}, t1::NTuple{1}, t2::NTuple{1}) where {T<:Number} = T(u.λ)
to_blocktype(u::UniformScaling, ::Type{S}, t1::NTuple{N1}, t2::NTuple{N2}) where {N1,N2,S<:SMatrix} =
    padtotype(SMatrix{N1,N2}(u), S)

# Fallback to catch mismatched or undesired block types
to_blocktype(t::Array, x...) = throw(ArgumentError("Array input in model, please use StaticArrays instead (e.g. SA[1 0; 0 1] instead of [1 0; 0 1])"))
to_blocktype(t, x...) = throw(DimensionMismatch("Dimension mismatch between model and Hamiltonian. Does the `orbitals` kwarg in your `hamiltonian` match your model?"))

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
# unitcell/supercell for Hamiltonians
#######################################################################
function supercell(ham::Hamiltonian, args...; kw...)
    slat = supercell(ham.lattice, args...; kw...)
    return Hamiltonian(slat, ham.harmonics, orbitals(ham))
end

function unitcell(ham::Hamiltonian{<:Lattice}, args...; modifiers = (), mincoordination = missing, kw...)
    sham = supercell(ham, args...; kw...)
    return unitcell(sham; modifiers = modifiers, mincoordination = mincoordination)
end

function unitcell(ham::Hamiltonian{LA,L}; modifiers = (), mincoordination = missing) where {E,L,T,L´,LA<:Superlattice{E,L,T,L´}}
    slat = ham.lattice
    sc = slat.supercell
    supercell_dn = r_to_dn(slat, sc.matrix, SVector{L´}(1:L´))
    pos = allsitepositions(slat)
    br = bravais(slat)
    modifiers´ = resolve.(ensuretuple(modifiers), Ref(slat))

    # Build a version of slat that has a filtered supercell mask according to mincoordination
    slat´ = filtered_superlat!(ham, mincoordination, supercell_dn, br, sc.matrix, pos)
    # store supersite indices newi
    mapping = OffsetArray{Int}(undef, sc.sites, sc.cells.indices...)
    mapping .= 0
    foreach_supersite((s, oldi, olddn, newi) -> mapping[oldi, Tuple(olddn)...] = newi, slat´)

    dim = nsites(slat´.supercell)
    B = blocktype(ham)
    S = typeof(SparseMatrixBuilder{B}(dim, dim))
    harmonic_builders = HamiltonianHarmonic{L´,B,S}[]

    foreach_supersite(slat´) do s, source_i, source_dn, newcol
        for oldh in ham.harmonics
            rows = rowvals(oldh.h)
            vals = nonzeros(oldh.h)
            target_dn = source_dn + oldh.dn
            for p in nzrange(oldh.h, source_i)
                target_i = rows[p]
                wrapped_dn, super_dn = wrap_super_dn(target_i, target_dn, supercell_dn, br, sc.matrix, pos)
                # check: wrapped_dn could exit bounding box along non-periodic direction
                checkbounds(Bool, mapping, target_i, Tuple(wrapped_dn)...) || continue
                newh = get_or_push!(harmonic_builders, super_dn, dim, newcol)
                newrow = mapping[target_i, Tuple(wrapped_dn)...]
                if !iszero(newrow)
                    val = applymodifiers(vals[p], slat, (source_i, target_i), (source_dn, target_dn), modifiers´...)
                    pushtocolumn!(newh.h, newrow, val)
                end
            end
        end
        foreach(h -> finalizecolumn!(h.h), harmonic_builders)
    end
    harmonics = [HamiltonianHarmonic(h.dn, sparse(h.h)) for h in harmonic_builders]
    unitlat = unitcell(slat´)
    return Hamiltonian(unitlat, harmonics, orbitals(ham))
end

filtered_superlat!(sham, ::Missing, args...) = sham.lattice
filtered_superlat!(sham, mc::Int, args...) =
    _filtered_superlat!(expand_supercell_mask(sham), mc, args...)

function _filtered_superlat!(sham::Hamiltonian{LA,L}, mincoord, args...) where {LA,L}
    slat = sham.lattice
    sc = slat.supercell
    mask = sc.mask
    if mincoord > 0
        delsites = NTuple{L+1,Int}[]
        while true
            foreach_supersite(sham.lattice) do _, source_i, source_dn, _
                nn = num_neighbors_supercell(sham.harmonics, source_i, source_dn, mask, args...)
                nn < mincoord && push!(delsites, (source_i, Tuple(source_dn)...))
            end
            foreach(p -> mask[p...] = false, delsites)
            isempty(delsites) && break
            resize!(delsites, 0)
        end
    end
    sc = Supercell(sc.matrix, sc.sites, sc.cells, mask)
    return Superlattice(slat.bravais, slat.unitcell, sc)
end

function num_neighbors_supercell(hhs, source_i, source_dn, mask, args...)
    nn = 0
    for hh in hhs
        ptrs = nzrange(hh.h, source_i)
        rows = rowvals(hh.h)
        nzel = nonzeros(hh.h)
        target_dn = source_dn + hh.dn
        for p in nzrange(hh.h, source_i)
            target_i = rows[p]
            wrapped_dn, _ = wrap_super_dn(target_i, target_dn, args...)
            isonsite = rows[p] == source_i && iszero(hh.dn)
            isincell = isinmask(mask, rows[p], Tuple(wrapped_dn)...)
            isnonzero = !iszero(nzel[p])
            nn += !isonsite && isincell && isnonzero
        end
    end
    return nn
end

function wrap_super_dn(target_i, target_dn, supercell_dn, br, smat, pos)
    r = pos[target_i] + br * target_dn
    super_dn = supercell_dn(r)
    wrapped_dn = target_dn - smat * super_dn
    return wrapped_dn, super_dn
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
    return Hamiltonian(lattice´, harmonics´, orbitals(h))
end

wrap(axes::Union{Integer,Tuple}; kw...) = h -> wrap(h, axes; kw...)

_wrap(lat::Lattice, axes) = Lattice(_wrap(lat.bravais, axes), lat.unitcell)

function _wrap(br::Bravais{E,L}, axes) where {E,L}
    mask = deletemultiple_nocheck(SVector{L}(1:L), axes)
    return Bravais(br.matrix[:, mask])
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
    orbs = tuplejoin(orbitals.(hams)...)
    builder = IJVBuilder(lat, orbs, hams...)
    model´ = offdiagonal(model, lat, nsublats.(hams))
    ham = hamiltonian_sparse!(builder, lat, orbs, model´)
    return ham
end

#######################################################################
# Bloch routines
#######################################################################

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

    bloch(h::Hamiltonian{<:Lattice}, ϕs::NTuple{L,Real}, dnfunc::Function)

Generalization that applies a prefactor `dnfunc(dn) * exp(im * ϕs' * dn)` to the `dn`
harmonic.

    bloch(ph::ParametricHamiltonian, [pϕs, [axis]])

Build the Bloch matrix for `ph`. `pϕs = (ϕs, (;kw...))` or `pϕs = (ϕs..., (;kw...))`
specifies both Bloch phases `ϕs` and the parameters `kw` passed to `ph(; kw...)`. If there are
no `ϕs`, the syntax `pϕs = (;kw...)` is also allowed, which is in that case equivalent to
`bloch(ph(; kw...))`. Similarly, `bloch(ph)` is equivalent to `bloch(ph())`.

    h |> bloch(ϕs, ...)
    ph |> bloch(pϕs, ...)

Curried forms of `bloch`, equivalent to `bloch(h, ϕs, ...)` and `bloch(ph, pϕs, ...)`

# Notes

`bloch` allocates a new matrix on each call. For a non-allocating version of `bloch`, see
`bloch!`.

`bloch` also supports slices of `Hamiltonians` and `ParametricHamiltonians`, produced with
`h[rows, cols]`, see the indexing section of `hamiltonian` and  `parametric` for details.

# Examples

```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(onsite(1) + hopping(2)) |> bloch((0, 0))
2×2 SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  1.0+0.0im
  [2, 1]  =  6.0+0.0im
  [1, 2]  =  6.0+0.0im
  [2, 2]  =  1.0+0.0im
```

# See also
    `bloch!`, `similarmatrix`
"""
bloch(ϕs, axis = 0) = h -> bloch(h, ϕs, axis)
bloch(h::Hamiltonian, args...) = bloch!(similarmatrix(h), h, args...)

"""
    bloch!(matrix, h::Hamiltonian, [ϕs, [axis]])

In-place version of `bloch`. Overwrite `matrix` with the Bloch Hamiltonian matrix of `h` for
the specified Bloch phases `ϕs = (ϕ₁,ϕ₂,...)` (see `bloch` for definition and API). A
conventient way to obtain a `matrix` is to use `similarmatrix(h,matrix_type)`, which will
return an `AbstractMatrix` of the same type as the Hamiltonian's. Note, however, that matrix
need not be of the same type (e.g. it can be dense with `Number` eltype for a sparse `h`
with `SMatrix` block eltype).

    bloch!(matrix, ph::ParametricHamiltonian, [pϕs, [axis]])

Same as above but with `pϕs = (ϕs, (;kw...))`, `pϕs = (ϕs..., (;kw...))` or `pϕs = (;kw...)`
(see `bloch` for details).

# Examples

```jldoctest
julia> h = LatticePresets.honeycomb() |> hamiltonian(hopping(2I), orbitals = (Val(2), Val(1)));

julia> bloch!(similarmatrix(h), h, (0, 0))
2×2 SparseMatrixCSC{SMatrix{2, 2, ComplexF64, 4}, Int64} with 2 stored entries:
                     ⋅                       [6.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]
 [6.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im]                      ⋅                     

julia> bloch!(similarmatrix(h, flatten), h, (0, 0))
3×3 SparseMatrixCSC{ComplexF64, Int64} with 9 stored entries:
 0.0+0.0im  0.0+0.0im  6.0+0.0im
 0.0+0.0im  0.0+0.0im  0.0+0.0im
 6.0+0.0im  0.0+0.0im  0.0+0.0im

julia> ph = parametric(h, @hopping!((t; α, β = 0) -> α * t .+ β));

julia> bloch!(similarmatrix(ph, flatten), ph, (0, 0, (; α = 2)))
3×3 SparseMatrixCSC{ComplexF64, Int64} with 9 stored entries:
  0.0+0.0im  0.0+0.0im  12.0+0.0im
  0.0+0.0im  0.0+0.0im   0.0+0.0im
 12.0+0.0im  0.0+0.0im   0.0+0.0im
```

# See also
    `bloch`, `similarmatrix`
"""
bloch!(matrix, h::Hamiltonian, ϕs, axis = 0) = _bloch!(matrix, h, toSVector(ϕs), axis)
bloch!(matrix, h::Hamiltonian, ϕs::Tuple{SVector,NamedTuple}, args...) = bloch!(matrix, h, first(ϕs), args...)

function bloch!(matrix, h::Hamiltonian)
    maybe_flatten_copy!(parent(matrix), first(h.harmonics).h, orbitalstructure(h)) # faster copy!(dense, sparse) specialization
    return matrix
end

function _bloch!(matrix::AbstractMatrix, h::Hamiltonian{<:Lattice,L,M}, ϕs::SVector{L}, axis::Number) where {L,M}
    rawmatrix = parent(matrix)
    if iszero(axis)
        maybe_flatten_copy!(rawmatrix, first(h.harmonics).h, orbitalstructure(h)) # faster copy!(dense, sparse) specialization
        add_harmonics!(rawmatrix, h, ϕs, dn -> 1)
    else
        fill!(rawmatrix, zero(M)) # There is no guarantee of same structure
        add_harmonics!(rawmatrix, h, ϕs, dn -> -im * dn[axis])
    end
    return matrix
end

function _bloch!(matrix::AbstractMatrix, h::Hamiltonian{<:Lattice,L,M}, ϕs::SVector{L}, dnfunc::Function) where {L,M}
    prefactor0 = dnfunc(zero(ϕs))
    rawmatrix = parent(matrix)
    if iszero(prefactor0)
        fill!(rawmatrix, zero(eltype(rawmatrix)))
    else
        maybe_flatten_copy!(rawmatrix, first(h.harmonics).h, orbitalstructure(h))
        rmul!(rawmatrix, prefactor0)
    end
    add_harmonics!(rawmatrix, h, ϕs, dnfunc)
    return matrix
end

_bloch!(matrix, h::Hamiltonian{<:Lattice,L}, ϕs::SVector{L´}, axis) where {L,L´} =
    L == L´ ? throw(ArgumentError("Unexpected `bloch!` signature")) :
              throw(DimensionMismatch("Mismatch between $L-dimensional Hamiltonian and $L´-dimensional Bloch phases"))

function add_harmonics!(zerobloch, h::Hamiltonian{<:Lattice,L}, ϕs::SVector{L}, dnfunc) where {L}
    ϕs´ = ϕs'
    for ns in 2:length(h.harmonics)
        hh = h.harmonics[ns]
        hhmatrix = hh.h
        prefactor = dnfunc(hh.dn)
        iszero(prefactor) && continue
        ephi = prefactor * cis(-ϕs´ * hh.dn)
        maybe_flatten_add!(zerobloch, hhmatrix, orbitalstructure(h), ephi)
    end
    return zerobloch
end

############################################################################################
######## maybe_flatten_copy! and maybe_flatten_add! call specialized methods #########################################
############################################################################################

maybe_flatten_copy!(dest, src, _) = copy!(dest, src)
maybe_flatten_copy!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, o) = fast_sparse_copy!(dst, src)
maybe_flatten_copy!(dst::StridedMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, o) = fast_sparse_copy!(dst, src)
maybe_flatten_copy!(dst::StridedMatrix{<:SMatrix{N,N}}, src::SparseMatrixCSC{<:SMatrix{N,N}}, o) where {N} = fast_sparse_copy!(dst, src)
maybe_flatten_copy!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:SMatrix}, o) = flatten_sparse_copy!(dst, src, o)
maybe_flatten_copy!(dst::StridedMatrix{<:Number}, src::StridedMatrix{<:SMatrix}, o) = flatten_dense_copy!(dst, src, o)

maybe_flatten_add!(dest, src, _, α) = _plain_muladd!(dest, src, α)
maybe_flatten_add!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:Number}, o, α = 1) = fast_sparse_muladd!(dst, src, α)
maybe_flatten_add!(dst::AbstractMatrix{<:SMatrix{N,N}}, src::SparseMatrixCSC{<:SMatrix{N,N}}, o, α = I) where {N} = fast_sparse_muladd!(dst, src, α)
maybe_flatten_add!(dst::AbstractMatrix{<:Number}, src::SparseMatrixCSC{<:SMatrix}, o, α = I) = flatten_sparse_muladd!(dst, src, o, α)
maybe_flatten_add!(dst::StridedMatrix{<:Number}, src::StridedMatrix{<:SMatrix}, o, α = I) = flatten_dense_muladd!(dst, src, o, α)