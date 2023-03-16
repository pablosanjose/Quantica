############################################################################################
# Hamiltonian builders
#region

abstract type AbstractHamiltonianBuilder{T,E,L,B} end

abstract type AbstractBuilderHarmonic{L,B} end

struct IJVHarmonic{L,B} <: AbstractBuilderHarmonic{L,B}
    dn::SVector{L,Int}
    collector::IJV{B}
end

mutable struct CSCHarmonic{L,B} <: AbstractBuilderHarmonic{L,B}
    dn::SVector{L,Int}
    collector::CSC{B}
end

struct IJVBuilder{T,E,L,B} <: AbstractHamiltonianBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    blockstruct::OrbitalBlockStructure{B}
    harmonics::Vector{IJVHarmonic{L,B}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end

struct CSCBuilder{T,E,L,B} <: AbstractHamiltonianBuilder{T,E,L,B}
    lat::Lattice{T,E,L}
    blockstruct::OrbitalBlockStructure{B}
    harmonics::Vector{CSCHarmonic{L,B}}
end

## Constructors ##

function IJVBuilder(lat::Lattice{T,E,L}, blockstruct::OrbitalBlockStructure{B}) where {E,L,T,B}
    harmonics = IJVHarmonic{L,B}[]
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, blockstruct, harmonics, kdtrees)
end

function CSCBuilder(lat::Lattice{<:Any,<:Any,L}, blockstruct::OrbitalBlockStructure{B}) where {L,B}
    harmonics = CSCHarmonic{L,B}[]
    return CSCBuilder(lat, blockstruct, harmonics)
end

empty_harmonic(b::CSCBuilder{<:Any,<:Any,L,B}, dn) where {L,B} =
    CSCHarmonic{L,B}(dn, CSC{B}(nsites(b.lat)))

empty_harmonic(::IJVBuilder{<:Any,<:Any,L,B}, dn) where {L,B} =
    IJVHarmonic{L,B}(dn, IJV{B}())

## API ##

collector(har::AbstractBuilderHarmonic) = har.collector  # for IJVHarmonic and CSCHarmonic

dcell(har::AbstractBuilderHarmonic) = har.dn

kdtrees(b::IJVBuilder) = b.kdtrees

Base.filter!(f::Function, b::IJVBuilder) =
    foreach(bh -> filter!(f, bh.collector), b.harmonics)

finalizecolumn!(b::CSCBuilder, x...) =
    foreach(har -> finalizecolumn!(collector(har), x...), b.harmonics)

Base.isempty(h::IJVHarmonic) = isempty(collector(h))
Base.isempty(s::CSCHarmonic) = isempty(collector(s))

lattice(b::AbstractHamiltonianBuilder) = b.lat

blockstructure(b::AbstractHamiltonianBuilder) = b.blockstruct

harmonics(b::AbstractHamiltonianBuilder) = b.harmonics

function Base.getindex(b::AbstractHamiltonianBuilder{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    hars = b.harmonics
    for har in hars
        dcell(har) == dn && return collector(har)
    end
    har = empty_harmonic(b, dn)
    push!(hars, har)
    return collector(har)
end

function SparseArrays.sparse(builder::AbstractHamiltonianBuilder{T,<:Any,L,B}) where {T,L,B}
    HT = Harmonic{T,L,B}
    b = blockstructure(builder)
    n = nsites(lattice(builder))
    hars = HT[sparse(b, har, n, n) for har in harmonics(builder) if !isempty(har)]
    return hars
end

function SparseArrays.sparse(b::OrbitalBlockStructure{B}, har::AbstractBuilderHarmonic{L,B}, m::Integer, n::Integer) where {L,B}
    s = sparse(collector(har), m, n)
    return Harmonic(dcell(har), HybridSparseBlochMatrix(b, s))
end

#endregion

############################################################################################
# hamiltonian
#region

hamiltonian(args...; kw...) = lat -> hamiltonian(lat, args...; kw...)

hamiltonian(lat::Lattice, m0::TightbindingModel, m::Modifier, ms::Modifier...; kw...) =
    parametric(hamiltonian(lat, m0; kw...), m, ms...)

hamiltonian(lat::Lattice, m0::ParametricModel, ms::Modifier...; kw...) =
    parametric(hamiltonian(lat, basemodel(m0); kw...), modifier.(terms(m0))..., ms...)

hamiltonian(lat::Lattice, m::Modifier, ms::Modifier...; kw...) =
    parametric(hamiltonian(lat; kw...), m, ms...)

hamiltonian(h::AbstractHamiltonian, m::Modifier, ms::Modifier...) = parametric(h, m, ms...)

hamiltonian(lat::Lattice, m::AbstractBlockModel; kw...) =
    hamiltonian(lat, parent(m), block(m); kw...)

# Base.@constprop :aggressive may be needed for type-stable non-Val orbitals?
function hamiltonian(lat::Lattice{T}, m::TightbindingModel = TightbindingModel(), block = missing; orbitals = Val(1)) where {T}
    orbitals´ = sanitize_orbitals(orbitals)
    blockstruct = OrbitalBlockStructure(T, orbitals´, sublatlengths(lat))
    builder = IJVBuilder(lat, blockstruct)
    apmod = apply(m, (lat, blockstruct))
    # using foreach here foils precompilation of applyterm! for some reason
    applyterm!.(Ref(builder), Ref(block), terms(apmod))
    hars = sparse(builder)
    return Hamiltonian(lat, blockstruct, hars)
end

function applyterm!(builder, block, term::AppliedOnsiteTerm)
    sel = selector(term)
    isempty(cells(sel)) || argerror("Cannot constrain cells in an onsite term, cell periodicity is assumed.")
    lat = lattice(builder)
    dn0 = zerocell(lat)
    ijv = builder[dn0]
    bs = blockstructure(builder)
    bsizes = blocksizes(bs)
    foreach_site(sel, dn0) do s, i, r
        isinblock(i, block) || return nothing
        n = bsizes[s]
        v = term(r, n)
        push!(ijv, (i, i, v))
    end
    return nothing
end

function applyterm!(builder, block, term::AppliedHoppingTerm)
    trees = kdtrees(builder)
    sel = selector(term)
    bs = blockstructure(builder)
    bsizes = blocksizes(bs)
    foreach_cell(sel) do dn
        ijv = builder[dn]
        found = foreach_hop(sel, trees, dn) do (si, sj), (i, j), (r, dr)
            isinblock(i, j, block) || return nothing
            ni = bsizes[si]
            nj = bsizes[sj]
            v = term(r, dr, (ni, nj))
            push!(ijv, (i, j, v))
        end
        return found
    end
    return nothing
end

# block is Missing some index collections, see Inter/IntrablockModel
isinblock(i, j, ::Missing) = true
isinblock(i, j, (rng, rng´)::Tuple) =
    (isinblock(i, rng) && isinblock(j, rng´)) || (isinblock(j, rng) && isinblock(i, rng´))
isinblock(i, j, rng) = isinblock(i, rng) && isinblock(j, rng)
isinblock(i, ::Missing) = true
isinblock(i, rng) = i in rng

#endregion

############################################################################################
# parametric
#region

function parametric(hparent::Hamiltonian)
    modifiers = ()
    allparams = Symbol[]
    allptrs = [Int[] for _ in harmonics(hparent)]
    # We must decouple hparent from the result, which will modify h in various ways
    h = minimal_callsafe_copy(hparent)
    return ParametricHamiltonian(hparent, h, modifiers, allptrs, allparams)
end

parametric(h::Hamiltonian, m::AbstractModifier, ms::AbstractModifier...) =
    _parametric!(parametric(h), m, ms...)
parametric(p::ParametricHamiltonian, ms::AbstractModifier...) =
    _parametric!(copy(p), ms...)

# This should not be exported, because it doesn't modify p in place (because of modifiers)
function _parametric!(p::ParametricHamiltonian, ms::Modifier...)
    ams = apply.(ms, Ref(parent(p)))
    return _parametric!(p, ams...)
end

function _parametric!(p::ParametricHamiltonian, ms::AppliedModifier...)
    hparent = parent(p)
    h = hamiltonian(p)
    allmodifiers = (modifiers(p)..., ms...)
    allparams = parameters(p)
    merge_parameters!(allparams, ms...)
    allptrs = pointers(p)
    merge_pointers!(allptrs, ms...)
    return ParametricHamiltonian(hparent, h, allmodifiers, allptrs, allparams)
end

merge_pointers!(p, m, ms...) = merge_pointers!(_merge_pointers!(p, m), ms...)

function merge_pointers!(p)
    for pn in p
        unique!(sort!(pn))
    end
    return p
end

function _merge_pointers!(p, m::AppliedOnsiteModifier)
    p0 = first(p)
    for (ptr, _) in pointers(m)
        push!(p0, ptr)
    end
    return p
end

function _merge_pointers!(p, m::AppliedHoppingModifier)
    for (pn, pm) in zip(p, pointers(m)), (ptr, _) in pm
        push!(pn, ptr)
    end
    return p
end

#endregion

############################################################################################
# Hamiltonian call API
#   call!(::AbstractHamiltonian; params...) returns a Hamiltonian with params applied
#   call!(::AbstractHamiltonian, ϕs; params...) returns a HybridSparseMatrix with Bloch phases
#     ϕs and params applied
#   h(ϕs...; params...) is a copy decoupled from future call!'s
#region

(h::Hamiltonian)(phi...; params...) = copy(call!(h, phi; params...))

call!(h::Hamiltonian; params...) = h  # mimic partial call!(p::ParametricHamiltonian; params...)
call!(h::Hamiltonian, φs; params...) = flat_bloch!(h, sanitize_SVector(φs))
call!(h::Hamiltonian{<:Any,<:Any,0}, ::Tuple{}; params...) = flat(h[])

# returns a flat sparse matrix
function flat_bloch!(h::Hamiltonian{T}, φs::SVector, axis = missing) where {T}
    hbloch = bloch(h)
    needs_initialization(hbloch) && initialize_bloch!(hbloch, harmonics(h))
    fbloch = flat(hbloch)
    fill!(fbloch, zero(Complex{T}))  # This preserves sparsity structure
    addblochs!(fbloch, h, φs, axis)
    return fbloch
end

function addblochs!(dst::SparseMatrixCSC, h::Hamiltonian, φs, axis)
    checkbloch(h, φs)
    hars = harmonics(h)
    isvelocity = axis isa Integer
    for har in hars
        iszero(dcell(har)) && isvelocity && continue
        e⁻ⁱᵠᵈⁿ = cis(-dot(φs, dcell(har)))
        isvelocity && (e⁻ⁱᵠᵈⁿ *= - im * dcell(har)[axis])
        merged_mul!(dst, matrix(har), e⁻ⁱᵠᵈⁿ, 1, 1)  # see tools.jl
    end
    return dst
end

is_bloch_initialized(h) = !needs_full_update(bloch(h))

function initialize_bloch!(bloch, hars)
    fbloch = flat_unsafe(bloch)
    fbloch´ = merge_sparse(flat.(matrix.(hars)))
    copy!(fbloch, fbloch´)
    needs_no_sync!(bloch)
    return bloch
end

@noinline checkbloch(::AbstractHamiltonian{<:Any,<:Any,L}, ::SVector{L´}) where {L,L´} =
    L == L´ || throw(ArgumentError("Need $L Bloch phases, got $(L´)"))

# ouput of a call!(h, ϕs)
call!_output(h::Hamiltonian) = flat(bloch(h))
call!_output(h::Hamiltonian{<:Any,<:Any,0}) = flat(h[])

#endregion

############################################################################################
# ParametricHamiltonian call API
#region

(p::ParametricHamiltonian)(; kw...) = copy(call!(p; kw...))
(p::ParametricHamiltonian)(phi, phis...; kw...) = copy(call!(call!(p; kw...), (phi, phis...)))

call!(p::ParametricHamiltonian, phi; kw...) = call!(call!(p; kw...), phi)
call!(p::ParametricHamiltonian, ft::FrankenTuple) = call!(p, Tuple(ft); NamedTuple(ft)...)

function call!(ph::ParametricHamiltonian; kw...)
    reset_to_parent!(ph)
    h = hamiltonian(ph)
    applymodifiers!(h, modifiers(ph)...; kw...)
    return h
end

function reset_to_parent!(ph)
    h = hamiltonian(ph)
    hparent = parent(ph)
    nnzfraction = 0.3  # threshold to revert to full copyto!
    for (har, har´, ptrs) in zip(harmonics(h), harmonics(hparent), pointers(ph))
        m, m´ = matrix(har), matrix(har´)
        nz = nonzeros(needs_initialization(m) ? unflat(m) : unflat_unsafe(m))
        nz´ = nonzeros(unflat(m´))
        if length(ptrs) < length(nz) * nnzfraction
            @simd for ptr in ptrs
                nz[ptr] = nz´[ptr]
            end
        else
            copyto!(nz, nz´)
        end
        needs_flat_sync!(m)
    end
    return ph
end

applymodifiers!(h, m, m´, ms...; kw...) = applymodifiers!(applymodifiers!(h, m; kw...), m´, ms...; kw...)

applymodifiers!(h, m::Modifier; kw...) = applymodifiers!(h, apply(m, h); kw...)

function applymodifiers!(h, m::AppliedOnsiteModifier; kw...)
    nz = nonzeros(unflat(first(harmonics(h))))
    for (ptr, r, norbs) in pointers(m)
        nz[ptr] = m(nz[ptr], r, norbs; kw...)
    end
    return h
end

function applymodifiers!(h, m::AppliedOnsiteModifier{B}; kw...) where {B<:SMatrixView}
    nz = nonzeros(unflat(first(harmonics(h))))
    for (ptr, r, norbs) in pointers(m)
        val = view(nz[ptr], 1:norbs, 1:norbs)
        nz[ptr] = m(val, r, norbs; kw...) # this allocates, currently unavoidable
    end
    return h
end

function applymodifiers!(h, m::AppliedHoppingModifier; kw...)
    for (har, p) in zip(harmonics(h), pointers(m))
        nz = nonzeros(unflat(har))
        for (ptr, r, dr, orborb) in p
            nz[ptr] = m(nz[ptr], r, dr, orborb; kw...)
        end
    end
    return h
end

function applymodifiers!(h, m::AppliedHoppingModifier{B}; kw...) where {B<:SMatrixView}
    for (har, p) in zip(harmonics(h), pointers(m))
        nz = nonzeros(unflat(har))
        for (ptr, r, dr, (norbs, norbs´)) in p
            val = view(nz[ptr], 1:norbs, 1:norbs´)
            nz[ptr] = m(val, r, dr, (norbs, norbs´); kw...)  # this allocates, unavoidable
        end
    end
    return h
end

# ouput of a *full* call!(p, ϕs; kw...)
call!_output(p::ParametricHamiltonian) = call!_output(hamiltonian(p))

#endregion

############################################################################################
# indexing into AbstractHamiltonian
#region

Base.getindex(h::AbstractHamiltonian{<:Any,<:Any,L}) where {L} = h[zero(SVector{L,Int})]
Base.getindex(h::AbstractHamiltonian, dn::Union{Integer,Tuple}) = getindex(h, SVector(dn))

function Base.getindex(h::AbstractHamiltonian{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    for har in harmonics(h)
        dn == dcell(har) && return matrix(har)
    end
    @boundscheck(boundserror(harmonics(h), dn))
end

Base.isassigned(h::AbstractHamiltonian, dn::Tuple) = isassigned(h, SVector(dn))

function Base.isassigned(h::AbstractHamiltonian{<:Any,<:Any,L}, dn::SVector{L,Int}) where {L}
    for har in harmonics(h)
        dn == dcell(har) && return true
    end
    return false
end

#endregion

############################################################################################
# coordination
#region

function nhoppings(h::AbstractHamiltonian)
    count = 0
    for har in harmonics(h)
        count += iszero(dcell(har)) ? (nnz(matrix(har)) - nnzdiag(matrix(har))) : nnz(matrix(har))
    end
    return count
end

nonsites(h::AbstractHamiltonian) = nnzdiag(h[])

coordination(h::AbstractHamiltonian) = iszero(nhoppings(h)) ? 0.0 : round(nhoppings(h) / nsites(lattice(h)), digits = 5)

#endregion

############################################################################################
# unitcell_hamiltonian
#    Builds the intra-unitcell 0D Hamiltonian. If parent is p::ParametricHamiltonian, the
#    obtained uh::Hamiltonian is aliased with p, so call!(p,...) also updates uh
#region

function unitcell_hamiltonian(h::Hamiltonian)
    lat = lattice(lattice(h), bravais = ())
    bs = blockstructure(h)
    hars = [Harmonic(SVector{0,Int}(), matrix(first(harmonics(h))))]
    return Hamiltonian(lat, bs, hars)
end

unitcell_hamiltonian(ph::ParametricHamiltonian) = unitcell_hamiltonian(hamiltonian(ph))

#endregion


# ############################################################################################
# # foreach_hop(h::AbstractHamiltonian) and foreach_site(h::AbstractHamiltonian)
# #region

# foreach_hop(f, ph::ParametricHamiltonian) = foreach_hop(f, hamiltonian(ph))
# foreach_site(f, ph::AbstractHamiltonian) = foreach_site(f, lattice(hamiltonian(ph)))

# function foreach_hop(f, h::Hamiltonian, sublatsrc = missing)
#     lat = lattice(h)
#     found = false
#     for har in harmonics(h)
#         found = found || foreach_hop(f, har, lat, sublatsrc)
#     end
#     return found
# end

# function foreach_hop(f, har::Harmonic, lat, sublatsrc = missing)
#     found = false
#     colrng = sublatsrc === missing ? axes(h, 2) : siterange(lat, sublatsrc)
#     mat = unflat(har)
#     dn = dcell(har)
#     for col in colrng, ptr in nzrange(mat, col)
#         row = rowvals(mat)[ptr]
#         row == col && continue
#         f(site(lat, col, zero(dn)) => site(lat, row, dn), dn)
#         found = true
#     end
#     return found
# end

# #endregion

# ############################################################################################
# # store_onsites - AbstractHamiltonian with onsites as structural SparseMatrix elements
# #    Note: if hasdiagonal, then no-op
# #region

# # if newptrs::Vector{Int} it will be filled with the list of inserted pointers
# function store_onsites(h::Hamiltonian, newptrs = nothing)
#     hasdiagonal(h[]) && return h
#     hh´ = harmonics(h)[2:end]
#     h0 = first(harmonics(h))
#     h0´ = Harmonic(dcell(h0), store_diagonal(matrix(h0), newptrs)) 
#     pushfirst!(hh´, h0´)
#     return Hamiltonian(lattice(h), orbitalstructure(h), hh´)
# end

# function store_onsites(p::ParametricHamiltonian)
#     newptrs = Int[]
#     hasdiagonal(parent(p)[]) && return p
#     h´ = store_onsites(parent(p), newptrs)
#     mods = modifiers(p)
#     mods´ = sync_ptrs.(mods, Ref(newptrs))
#     return parametric(h´, mods´...)
# end

# sync_ptrs(m::AppliedOnsiteModifier, newptrs) =
#     AppliedOnsiteModifier(blocktype(m), parametric_function(m), sync_ptrs(pointers(m), newptrs))

# function sync_ptrs(m::AppliedHoppingModifier, newptrs)
#     ptrs´ = pointers(m)[2:end]
#     ptrs0´ = sync_ptrs(first(pointers(m)), newptrs)
#     pushfirst!(ptrs´, ptrs0´)
#     return AppliedHoppingModifier(blocktype(m), parametric_function(m), ptrs´)
# end

# # race across ptrs and newptrs, injecting shifts
# function sync_ptrs(ptrs::Vector{<:Tuple}, newptrs)
#     ptrs´ = copy(ptrs)
#     i = i´ = 1
#     while i <= length(ptrs)
#         (ptr, rest...) = ptrs[i]
#         if i´ > length(newptrs) || ptr + i´ - 1 < newptrs[i´]
#             ptrs´[i] = (ptr + i´ - 1, rest...)
#             i += 1
#         else
#             i´ += 1
#         end
#     end
#     return ptrs´
# end

# ## Tools

# # insert structural zeros in sparse square matrix diagonal if not present
# function store_diagonal(m::SparseMatrixCSC{B}, newptrs = nothing) where {B}
#     size(m, 1) == size(m, 2) || throw(ArgumentError("Expect square matrix"))
#     rows = rowvals(m)
#     vals = nonzeros(m)
#     builder = CSC{B}(size(m, 2), nnz(m))
#     for col in axes(m, 2)
#         hasdiagonal = false
#         for ptr in nzrange(m, col)
#             row, val = rows[ptr], vals[ptr]
#             row == col && (hasdiagonal = true)
#             pushtocolumn!(builder, row, val)
#         end
#         hasdiagonal || pushtocolumn!(builder, col, zero(B))
#         finalizecolumn!(builder)
#         hasdiagonal || record!(newptrs, row_col_ptr(builder, col, col))
#     end
#     return sparse(builder, size(m, 1))
# end

# record!(ptrs::Vector{Int}, ptr) = push!(ptrs, ptr)
# record!(_, ptr) = nothing

# function hasdiagonal(m::SparseMatrixCSC)
#     rows = rowvals(m)
#     for col in axes(m, 2)
#         found = false
#         for ptr in nzrange(m, col)
#             rows[ptr] == col && (found = true; break)
#         end
#         found || return false
#     end
#     return true
# end

# # appends pointers to diagonal elements of m in cols range
# function diagonal_pointers!(ptrs, m::SparseMatrixCSC, cols = 1:min(size(m)...))
#     rows = rowvals(m)
#     for col in cols
#         for ptr in nzrange(m, col)
#             if rows[ptr] == col
#                 push!(ptrs, ptr)
#                 break
#             end
#         end
#     end
#     return ptrs
# end

# function add_diagonal!(A::Matrix, x, cols = 1:size(A,2))
#     checkbounds(A, cols, cols)
#     @inbounds for col in cols
#         A[col, col] += x
#     end
#     return A
# end

# function add_diagonal!(A::SparseMatrixCSC, x, cols = 1:size(A,2))
#     rows = rowvals(m)
#     vals = nonzeros(m)
#     for col in cols
#         found = false
#         for ptr in nzrange(m, col)
#             if rows[ptr] == col
#                 found = true
#                 vals[ptr] += x
#                 break
#             end
#         end
#         found || internalerror("add_diagonal!: not enough structural elements")
#     end
# end


# #endregion