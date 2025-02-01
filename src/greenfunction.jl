############################################################################################
# greenfunction
#region

greenfunction(s::AbstractGreenSolver) = oh -> greenfunction(oh, s)

greenfunction() = h -> greenfunction(h)

greenfunction(h::AbstractHamiltonian, args...) = greenfunction(OpenHamiltonian(h), args...)

function greenfunction(oh::OpenHamiltonian, s::AbstractGreenSolver = default_green_solver(hamiltonian(oh)))
    cs = Contacts(oh)
    h = hamiltonian(oh)
    as = apply(s, h, cs)
    return GreenFunction(h, as, cs)
end

default_green_solver(::AbstractHamiltonian0D) = GS.SparseLU()
default_green_solver(::AbstractHamiltonian1D) = GS.Schur()
default_green_solver(::AbstractHamiltonian) = GS.Bands()

#endregion

############################################################################################
# Contacts call! API
#region

function call!(c::Contacts, ω; params...)
    Σblocks = selfenergyblocks(c)
    call!.(c.selfenergies, Ref(ω); params...) # updates matrices in Σblocks
    return Σblocks
end

call!_output(c::Contacts) = selfenergyblocks(c)

#endregion

############################################################################################
# GreenFuntion call! API
#   `symmetrize` and `post` allow to obtain gij and conj(gji) combinations efficiently.
#   Useful when computing e.g. the spectral density A = (Gʳ-Gʳ')/2πi on a non-diagonal slice
#   These kwargs are currently not documented - see specialmatrices.jl
#region

(g::GreenFunction)(ω; params...) = minimal_callsafe_copy(call!(g, ω; params...))
(g::GreenSlice)(ω; params...) = copy(call!(g, ω; params...))

call!(g::G, ω; params...) where {T,G<:Union{GreenFunction{T},GreenSlice{T}}} =
    call!(g, real_or_complex_convert(T, ω); params...)

function call!(g::GreenFunction{T}, ω::T; params...) where {T}
    ω´ = retarded_omega(ω, solver(g))
    return call!(g, ω´; params...)
end

function call!(g::GreenFunction{T}, ω::Complex{T}; params...) where {T}
    gsolver = solver(g)
    contacts´ = contacts(g)
    Σblocks = call!(contacts´, ω; params...)
    corbs = contactorbitals(contacts´)
    slicer = build_slicer(gsolver, g, ω, Σblocks, corbs; params...)
    return GreenSolution(g, slicer, Σblocks, corbs)
end

# real frequency -> maybe complex frequency
call!(g::GreenSlice{T}, ω::T; kw...) where {T} =
    call!(g, retarded_omega(ω, solver(parent(g))); kw...)

call!(gs::GreenSlice{T}, ω::Complex{T}; post = identity, symmetrize = missing, params...) where {T} =
    getindex!(gs, call!(greenfunction(gs), ω; params...); post, symmetrize)

real_or_complex_convert(::Type{T}, ω::Real) where {T<:Real} = convert(T, ω)
real_or_complex_convert(::Type{T}, ω::Complex) where {T<:Real} = convert(Complex{T}, ω)

retarded_omega(ω::T, s::AppliedGreenSolver) where {T<:Real} =
    ω + im * sqrt(eps(float(T))) * needs_omega_shift(s)

# fallback, may be overridden
needs_omega_shift(s::AppliedGreenSolver) = true

#endregion

############################################################################################
# GreenSolution indexing
#   We convert any index down to cellorbs to pass to slicer, except contacts (Int, Colon)
#   If we index with CellIndices, we bypass mortaring and return a bare matrix
#region

## GreenFunction -> GreenSlice

Base.getindex(g::GreenFunction; kw...) = g[siteselector(; kw...)]
Base.getindex(g::GreenFunction, kw::NamedTuple) = g[siteselector(; kw...)]
Base.getindex(g::GreenFunction, i, j = i) = GreenSlice(g, i, j)  # this calls sites_to_orbs

## GreenSolution -> OrbitalSliceMatrix or AbstractMatrix

# view(g, ::Integer, ::Integer) and view(g, :, :) - intra and inter contacts
Base.view(g::GreenSolution, i::CT, j::CT = i) where {CT<:Union{Integer,Colon}} =
    view(slicer(g), i, j)

# general getindex(::GreenSolution,...) entry point. Relies on GreenSlice constructor
# args could be anything: CellSites, CellOrbs, Colon, Integer, DiagIndices...
function Base.getindex(g::GreenSolution, args...; post = identity, symmetrize = missing, kw...)
    gs = getindex(parent(g), args...; kw...)  # get GreenSlice
    output = getindex!(gs, g; post, symmetrize)
    return output
end

Base.getindex(g::GreenSolution, i::AnyCellOrbitals, j::AnyCellOrbitals) =
    slicer(g)[sanitize_cellorbs(i, g), sanitize_cellorbs(j, g)]

# must ensure that orbindices is not a scalar, to consistently obtain a Matrix
# also, it should not be Colon, as GreenSlicers generally assume an explicit range
sanitize_cellorbs(c::CellOrbitals, g) = sites_to_orbs(c, g)  # convert : to range
sanitize_cellorbs(c::CellOrbitals) = c
sanitize_cellorbs(c::CellOrbital, _...) = CellOrbitals(cell(c), orbindex(c):orbindex(c))
sanitize_cellorbs(c::CellOrbitalsGrouped, _...) = CellOrbitals(cell(c), orbindices(c))
sanitize_cellorbs(::CellOrbitals{<:Any,Colon}) =
    internalerror("sanitize_cellorbs: Colon indices leaked!")

## common getindex! shortcut in terms of GreenSlice

# index object g over a slice enconded in gs::GreenSlice, using its preallocated output
getindex!(gs::GreenSlice, g; kw...) = getindex!(call!_output(gs), g, orbinds_or_contactinds(gs)...; kw...)

# ifelse(rows && cols are contacts, (rows, cols), (orbrows, orbcols))
# I.e: if rows, cols are contact indices retrieve them instead of orbslices.
orbinds_or_contactinds(g) = orbinds_or_contactinds(rows(g), cols(g), orbrows(g), orbcols(g))
orbinds_or_contactinds(
    r::Union{Colon,Integer,DiagIndices{Colon},DiagIndices{<:Integer}},
    c::Union{Colon,Integer,DiagIndices{Colon},DiagIndices{<:Integer}}, _, _) = (r, c)
orbinds_or_contactinds(_, _, or, oc) = (or, oc)

## getindex! - core functions

# fastpath for intra and inter-contact, only for g::GreenSolution
getindex!(output, g::GreenSolution, i::Union{Integer,Colon}, j::Union{Integer,Colon}; symmetrize = missing, kw...) =
    maybe_symmetrized_view!(output, g, i, j, symmetrize; kw...)

# indexing over single cells. g can be any type that implements g[::AnyCellOrbitals...]
getindex!(output, g, i::AnyCellOrbitals, j::AnyCellOrbitals; symmetrize = missing, kw...) =
    maybe_symmetrized_getindex!(output, g, i, j, symmetrize; kw...)

# fallback for multicell case
getindex!(output, g, ci::Union{AnyOrbitalSlice,AnyCellOrbitals}, cj::Union{AnyOrbitalSlice,AnyCellOrbitals}; kw...) =
    getindex_cells!(output, g, cellinds_iterable_axis(ci), cellinds_iterable_axis(cj); kw...)

cellinds_iterable_axis(ci::CellIndices) = ((ci,), missing)
cellinds_iterable_axis(ci::AnyOrbitalSlice) = (cellsdict(ci), ci)

# at this point, either cis or cjs is a multi-cell iterator, so an output view is needed
function getindex_cells!(output, g, (cis, iaxis), (cjs, jaxis); kw...)
    for ci in cis, cj in cjs
        rows, cols = cell_orbindices(ci, iaxis), cell_orbindices(cj, jaxis)
        output´ = view(maybe_orbmat_parent(output), rows, cols)
        getindex!(output´, g, ci, cj; kw...)    # will call the single-cell method above
    end
    return output
end

# output may be a Matrix (inhomogenous rows/cols) or an OrbitalSliceMatrix (homogeneous)
maybe_orbmat_parent(output::OrbitalSliceMatrix) = parent(output)
maybe_orbmat_parent(output) = output

# orbital index range in parent(output::OrbitalSliceMatrix) for a given cell
cell_orbindices(ci, axis) = orbrange(axis, cell(ci))
# if output::Matrix, we take all indices, since output was already cropped along this axis
cell_orbindices(_, ::Missing) = Colon()

## GreenSlicer -> Matrix, dispatches to each solver's implementation

# simplify CellOrbitalGroups to CellOrbitals. No CellOrbitals{L,Colon} should get here
Base.getindex(s::GreenSlicer, i::AnyCellOrbitals, j::AnyCellOrbitals = i; kw...) =
    getindex(s, sanitize_cellorbs(i), sanitize_cellorbs(j); kw...)

# fallback, for GreenSlicers that forgot to implement getindex
Base.getindex(s::GreenSlicer, ::CellOrbitals, ::CellOrbitals; kw...) =
    argerror("getindex of $(nameof(typeof(s))) not implemented")

# fallback, for GreenSlicers that don't implement view
Base.view(g::GreenSlicer, args...) =
    argerror("GreenSlicer of type $(nameof(typeof(g))) doesn't implement view for these arguments")

## Diagonal and sparse indexing

# These i,j are already gone through sites_to_orbs, unless they are Colon or Integer (orbinds_or_contactinds)
getindex!(output, g, i::DiagIndices, j::DiagIndices = i; kw...) =
    getindex_diagonal!(output, g, parent(i), kernel(i); kw...)

getindex!(output, g, i::OrbitalPairIndices, j::OrbitalPairIndices = i; kw...) =
    getindex_sparse!(output, g, parent(i), kernel(i); kw...)

#endregion

############################################################################################
# getindex_diagonal! and getindex_sparse! : GreenSolution diagonal and sparse indexing
#   They rely on fill_diagonal! and fill_sparse! to fill the output matrix with mat elements
#   We can swith `g` with another object `o` that supports getindex_diag or getindex_sparse
#   returning `mat::AbstractMatrix`
#   `mat` is not an OrbitalSliceGrouped for performance reasons. Instead the object `o`
#   should also implement a `blockstructure` method so we know how to map sites to indices.
#region

function getindex_diagonal!(output, g::GreenSolution, i::Union{Colon,Integer}, ker; symmetrize = missing, kw...)
    v = maybe_symmetrized_matrix(view(g, i, i), symmetrize)
    return fill_diagonal!(output, v, contact_kernel_ranges(ker, i, g), ker; kw...)
end

function getindex_diagonal!(output, g, o::CellOrbitalsGrouped, ker; symmetrize = missing, kw...)
    rngs = orbital_kernel_ranges(ker, o)
    mat = getindex_diag(g, o, symmetrize)
    fill_diagonal!(output, mat, rngs, ker; kw...)
    return output
end

# g may be a GreenSolution or any other type with a minimal API (e.g. EigenProduct)
function getindex_diagonal!(output, g, i::OrbitalSliceGrouped, ker; symmetrize = missing, kw...)
    offset = 0
    for o in cellsdict(i)
        rngs = orbital_kernel_ranges(ker, o)
        mat = getindex_diag(g, o, symmetrize)
        fill_diagonal!(output, mat, rngs, ker; offset, kw...)
        offset += length(rngs)
    end
    return output
end

function getindex_sparse!(output, g, hmask::Hamiltonian, ker; symmetrize = missing, kw...)
    bs = blockstructure(g)  # used to find blocks of g for nonzeros in hmask
    oj = sites_to_orbs(sites(zerocell(hmask), :), bs)  # full cell
    for har in harmonics(hmask)
        oi = CellIndices(dcell(har), oj)
        mat_cell = getindex_sparse(g, oi, oj, symmetrize)
        fill_sparse!(har, mat_cell, bs, ker; kw...)
    end
    fill_sparse!(parent(output), hmask)
    return output
end

# non-optimized fallback, can be specialized by solvers
getindex_diag(g, oi, symmetrize) = maybe_symmetrized_getindex(g, oi, oi, symmetrize)
getindex_sparse(g, oi, oj, symmetrize) = maybe_symmetrized_getindex(g, oi, oj, symmetrize)

# input index ranges for each output index to fill
contact_kernel_ranges(::Missing, o::Colon, gω) = 1:norbitals(contactorbitals(gω))
contact_kernel_ranges(::Missing, i::Integer, gω) = 1:norbitals(contactorbitals(gω), i)
contact_kernel_ranges(kernel, ::Colon, gω) = orbranges(contactorbitals(gω))
contact_kernel_ranges(kernel, i::Integer, gω) = orbranges(contactorbitals(gω), i)
orbital_kernel_ranges(::Missing, o::CellOrbitalsGrouped) = eachindex(orbindices(o))
orbital_kernel_ranges(kernel, o::CellOrbitalsGrouped) = orbranges(o)

## core drivers
#   write to the diagonal of output, with a given offset, the elements Tr(kernel*mat_cell[i,i])
#   for each orbital or site encoded in range i
function fill_diagonal!(output, mat_cell, blockrngs, kernel; post = identity, offset = 0)
    for (n, rng) in enumerate(blockrngs)
        i = n + offset
        output[i, i] = post(apply_kernel(kernel, mat_cell, rng))
    end
    return output
end

#   overwrite nonzero h elements with Tr(kernel*mat_cell[irng,jrng]) for each site pair i,j
function fill_sparse!(har::Harmonic, mat_cell, bs, kernel; post = identity)
    B = blocktype(har)
    hmat = unflat(har)
    rows = rowvals(hmat)
    nzs = nonzeros(hmat)
    for col in axes(hmat, 2), ptr in nzrange(hmat, col)
        row = rows[ptr]
        orows, ocols = flatrange(bs, row), flatrange(bs, col)
        val = post(apply_kernel(kernel, mat_cell, orows, ocols))
        nzs[ptr] = mask_block(B, val)
    end
    needs_flat_sync!(matrix(har))
    return har
end

#  overwrite sparse output with an vcat of all Harmonics in h
function fill_sparse!(output::SparseMatrixCSC, h::Hamiltonian)
    out_nzs = nonzeros(output)
    for col in axes(output, 2)
        out_ptr_rng = nzrange(output, col)
        iptr = 1
        for har in harmonics(h)
            hmat = flat(har)
            hnzs = nonzeros(hmat)
            for ptr in nzrange(hmat, col)
                out_nzs[out_ptr_rng[iptr]] = hnzs[ptr]
                iptr += 1
            end
        end
    end
    return output
end

apply_kernel(kernel, mat_cell, rows, cols = rows) = apply_kernel(kernel, view_or_scalar(mat_cell, rows, cols))
apply_kernel(kernel::Missing, v) = v
apply_kernel(kernel, v) = trace_prod(kernel, v)

view_or_scalar(mat_cell, rows::UnitRange, cols::UnitRange) = view(mat_cell, rows, cols)
view_or_scalar(mat_cell, orb::Integer, orb´::Integer) = mat_cell[orb, orb´]

#endregion

############################################################################################
# FixedParamGreenSolver
#   support for g(; params...) --> GreenFunction (not a wrapper, completely independent)
#   FixedParamGreenSolver doesn't need to implement the AppliedGreenSolver API, since it
#   forwards to its parent
#region

struct FixedParamGreenSolver{P,G<:GreenFunction} <: AppliedGreenSolver
    gparent::G
    params::P
end

Base.parent(s::FixedParamGreenSolver) = s.gparent

parameters(s::FixedParamGreenSolver) = s.params

function (g::GreenFunction)(; params...)
    h´ = minimal_callsafe_copy(parent(g))
    c´ = minimal_callsafe_copy(contacts(g))
    s´ = minimal_callsafe_copy(solver(g), h´, c´)
    gparent = GreenFunction(h´, s´, c´)
    return GreenFunction(h´, FixedParamGreenSolver(gparent, params), c´)
end

(g::GreenSlice)(; params...) = GreenSlice(parent(g)(; params...), greenindices(g)...)

# params are ignored, solver.params are used instead. T required to disambiguate.
function call!(g::GreenFunction{T,<:Any,<:Any,<:FixedParamGreenSolver}, ω::Complex{T}; params...) where {T}
    s = solver(g)
    return call!(s.gparent, ω; s.params...)
end

function minimal_callsafe_copy(s::FixedParamGreenSolver, parentham, parentcontacts)
    solver´ = minimal_callsafe_copy(solver(s.gparent), parentham, parentcontacts)
    gparent = GreenFunction(parentham, solver´, parentcontacts)
    s´ = FixedParamGreenSolver(gparent, s.params)
    return s´
end

default_hamiltonian(g::GreenFunction{<:Any,<:Any,<:Any,<:FixedParamGreenSolver}) =
    default_hamiltonian(parent(g); parameters(solver(g))...)

#endregion

############################################################################################
# selfenergy(::GreenSolution, contactinds::Int...; onlyΓ = false)
#   if no contactinds are provided, all are returned. Otherwise, a view of Σ over contacts.
# selfenergy!(Σ::AbstractMatrix, ::GreenSolution, cinds...)
#   add blocks from cinds selfenergies to Σ (or all), and return a view over them
# similar_contactΣ(g)
#   Matrix to hold MatrixBlocks of any self-energy
# similar_contactΣ(g, cind)
#   Matrix that can hold a view of one self-energy
#region

selfenergy(g::GreenSolution, cinds::Int...; kw...) =
    selfenergy!(similar_contactΣ(g), g, cinds...; kw...)

# we support also the case for one contact, but it is only used elsewhere
function similar_contactΣ(g::Union{GreenFunction{T},GreenSolution{T}}, cind...) where {T}
    n = norbitals(contactorbitals(g), cind...)
    Σ = zeros(Complex{T}, n, n)
    return Σ
end

function maybe_selfenergy_view(Σ, g, cind, cinds...)
    inds = copy(contactinds(g, cind))
    foreach(i -> append!(inds, contactinds(g, i)), cinds)
    isempty(cinds) || unique!(sort!(inds))
    Σv = view(Σ, inds, inds)
    return Σv
end

maybe_selfenergy_view(Σ, g) = Σ

function selfenergy!(Σ::AbstractMatrix{T}, g::GreenSolution, cinds...; onlyΓ = false) where {T}
    fill!(Σ, zero(T))
    addselfenergy!(Σ, g, cinds...)
    onlyΓ && extractΓ!(Σ)   # faster to do this on Σ than on Σv
    Σv = maybe_selfenergy_view(Σ, g, cinds...)
    return Σv
end

addselfenergy!(Σ, g::GreenSolution, cind::Int, cinds...) =
    addselfenergy!(addselfenergy!(Σ, selfenergies(g)[cind]), g, cinds...)
addselfenergy!(Σ, ::GreenSolution) = Σ

# RegularSelfEnergy case
function addselfenergy!(Σ, b::MatrixBlock)
    v = view(Σ, blockrows(b), blockcols(b))
    v .+= blockmat(b)
    return Σ
end

# ExtendedSelfEnergy case
function addselfenergy!(Σ, (V´, g⁻¹, V)::NTuple{<:Any,MatrixBlock})
    v = view(Σ, blockrows(V´), blockcols(V))
    Vd = denseblockmat(V)
    copy!(Vd, blockmat(V))
    ldiv!(lu(blockmat(g⁻¹)), Vd)
    mul!(v, blockmat(V´), Vd, 1, 1)
    return Σ
end

function extractΓ!(Σ)
    Σ .-= Σ'
    Σ .*= im
    return Σ
end

#endregion

############################################################################################
# selfenergyblocks
#    Build MatrixBlocks from contacts, including extended inds for ExtendedSelfEnergySolvers
#region

function selfenergyblocks(contacts::Contacts)
    Σs = selfenergies(contacts)
    solvers = solver.(Σs)
    extoffset = norbitals(contactorbitals(contacts))
    cinds = contactinds(contacts)
    Σblocks = selfenergyblocks(extoffset, cinds, 1, (), solvers...)
    return Σblocks
end

# extoffset: current offset where extended indices start
# contactinds: orbital indices for all selfenergies in contacts
# ci: auxiliary index for current selfenergy being processed
# blocks: tuple accumulating all MatrixBlocks from all selfenergies
# solvers: selfenergy solvers that will update the MatrixBlocks
selfenergyblocks(extoffset, contactinds, ci, blocks) = blocks

function selfenergyblocks(extoffset, contactinds, ci, blocks, s::RegularSelfEnergySolver, ss...)
    c = contactinds[ci]
    Σblock = MatrixBlock(call!_output(s), c, c)
    return selfenergyblocks(extoffset, contactinds, ci + 1, (blocks..., Σblock), ss...)
end

function selfenergyblocks(extoffset, contactinds, ci, blocks, s::ExtendedSelfEnergySolver, ss...)
    Vᵣₑ, gₑₑ⁻¹, Vₑᵣ = shiftedmatblocks(call!_output(s), contactinds[ci], extoffset)
    extoffset += size(gₑₑ⁻¹, 1)
    # there is no minus sign here!
    return selfenergyblocks(extoffset, contactinds, ci + 1, (blocks..., (Vᵣₑ, gₑₑ⁻¹, Vₑᵣ)), ss...)
end

function shiftedmatblocks((Vᵣₑ, gₑₑ⁻¹, Vₑᵣ)::NTuple{3,AbstractArray}, cinds, shift)
    extsize = size(gₑₑ⁻¹, 1)
    Vᵣₑ´ = MatrixBlock(Vᵣₑ, cinds, shift+1:shift+extsize)
    # adds denseblock for selfenergy ldiv
    Vₑᵣ´ = MatrixBlock(Vₑᵣ, shift+1:shift+extsize, cinds, Matrix(Vₑᵣ))
    gₑₑ⁻¹´ = MatrixBlock(gₑₑ⁻¹, shift+1:shift+extsize, shift+1:shift+extsize)
    return Vᵣₑ´, gₑₑ⁻¹´, Vₑᵣ´
end

#endregion

############################################################################################
# ContactOrbitals constructors
#   Build a ContactOrbitals from a Hamiltonian and a set of latslices
#region

ContactOrbitals(h::AbstractHamiltonian{<:Any,<:Any,L}) where {L} =
    ContactOrbitals{L}()

ContactOrbitals(h::AbstractHamiltonian, os, oss...) =
    ContactOrbitals(blockstructure(h), os, oss...)

# probably unused, since oss comes from SelfEnergy, so it is already OrbitalSliceGrouped
ContactOrbitals(bs::OrbitalBlockStructure, oss...) =
    ContactOrbitals(bs, sites_to_orbs.(oss, Ref(bs))...)

function ContactOrbitals(bs::OrbitalBlockStructure, oss::OrbitalSliceGrouped...)
    codicts = cellsdict.(oss)
    codictall = combine(codicts...)
    contactinds = Vector{Int}[]
    corb_to_ind = dictionary(corb => ind for (ind, corb) in enumerate(cellorbs(codictall)))
    for codict in codicts
        i = [corb_to_ind[co] for co in cellorbs(codict)]
        push!(contactinds, i)
    end
    offsets = lengths_to_offsets(length, codictall)
    corbs = ContactOrbitals(codictall, [codicts...], contactinds, offsets)
    return corbs
end

#endregion

############################################################################################
# TMatrixSlicer <: GreenSlicer
#    Given a slicer that works without any contacts, implement slicing with contacts through
#    a T-Matrix equation g(i, j) = g0(i, j) + g0(i, k)T(k,k')g0(k', j), and T = (1-Σ*g0)⁻¹*Σ
#region

struct TMatrixSlicer{C,L,V<:AbstractArray{C},S} <: GreenSlicer{C}
    g0slicer::S
    tmatrix::V
    gcontacts::V
    contactorbs::ContactOrbitals{L}
end

struct NothingSlicer{C} <: GreenSlicer{C}
end

#region ## Constructors ##

# if there are no Σblocks, return g0slicer. Otherwise build TMatrixSlicer.
maybe_TMatrixSlicer(g0slicer::GreenSlicer, Σblocks::Tuple{}, contactorbs) = g0slicer
maybe_TMatrixSlicer(g0slicer, Σblocks, contactorbs) =
    TMatrixSlicer(g0slicer, Σblocks, contactorbs)

# Uses getindex(g0slicer) to construct g0contacts
function TMatrixSlicer(g0slicer::GreenSlicer{C}, Σblocks, contactorbs) where {C}
    if isempty(Σblocks)
        tmatrix = gcontacts = view(zeros(C, 0, 0), 1:0, 1:0)
    else
        nreg = norbitals(contactorbs)
        g0contacts = zeros(C, nreg, nreg)
        fill_g0contacts!(g0contacts, g0slicer, contactorbs)
        Σblocks´ = tupleflatten(Σblocks...)
        tmatrix, gcontacts = t_g_matrices(g0contacts, contactorbs, Σblocks´...)
    end
    return TMatrixSlicer(g0slicer, tmatrix, gcontacts, contactorbs)
end

function fill_g0contacts!(mat, g0slicer, contactorbs)
    off = offsets(contactorbs)
    for (j, sj) in enumerate(cellsdict(contactorbs)), (i, si) in enumerate(cellsdict(contactorbs))
        irng = off[i]+1:off[i+1]
        jrng = off[j]+1:off[j+1]
        g0view = view(mat, irng, jrng)
        copy!(g0view, g0slicer[si, sj])
    end
    return mat
end

# Takes a precomputed g0contacts (for dummy g0slicer that doesn't implement indexing)
function TMatrixSlicer(g0contacts::AbstractMatrix{C}, Σblocks, contactorbs) where {C}
    tmatrix, gcontacts = t_g_matrices(g0contacts, contactorbs, Σblocks...)
    g0slicer = NothingSlicer{C}()
    return TMatrixSlicer(g0slicer, tmatrix, gcontacts, contactorbs)
end

# empty Σblocks
function t_g_matrices(g0contacts::AbstractMatrix{C}, contactorbs) where {C}
    tmatrix = gcontacts = view(zeros(C, 0, 0), 1:0, 1:0)
    return tmatrix, gcontacts
end

# Check whether Σblocks are all spzeros, and if not, compute G and T
function t_g_matrices(g0contacts::AbstractMatrix{C}, contactorbs, Σblocks::MatrixBlock...) where {C}
    if isspzeros(Σblocks)
        gcontacts = g0contacts
        tmatrix = zero(g0contacts)
    else
        tmatrix, gcontacts = t_g_matrices!(copy(g0contacts), contactorbs, Σblocks...)
    end
    return tmatrix, gcontacts
end

# rewrites g0contacts
function t_g_matrices!(g0contacts::AbstractMatrix{C}, contactorbs, Σblocks::MatrixBlock...) where {C}
    nreg = norbitals(contactorbs)                            # number of regular orbitals
    n = max(nreg, maxrows(Σblocks), maxcols(Σblocks))        # includes extended orbitals
    Σmatext = Matrix{C}(undef, n, n)
    Σbm = BlockMatrix(Σmatext, Σblocks)
    update!(Σbm)                                             # updates Σmat with Σblocks
    Σmatᵣᵣ = view(Σmatext, 1:nreg, 1:nreg)
    Σmatₑᵣ = view(Σmatext, nreg+1:n, 1:nreg)
    Σmatᵣₑ = view(Σmatext, 1:nreg, nreg+1:n)
    Σmatₑₑ = view(Σmatext, nreg+1:n, nreg+1:n)
    Σmat = copy(Σmatᵣᵣ)
    Σmat´ = ldiv!(lu!(Σmatₑₑ), Σmatₑᵣ)
    mul!(Σmat, Σmatᵣₑ, Σmat´, 1, 1)                  # Σmat = Σmatᵣᵣ + ΣmatᵣₑΣmatₑₑ⁻¹ Σmatₑᵣ
    den = Matrix{C}(I, nreg, nreg)
    mul!(den, Σmat, g0contacts, -1, 1)                       # den = 1-Σ*g0
    luden = lu!(den)
    tmatrix = ldiv!(luden, Σmat)                             # tmatrix = (1 - Σ*g0)⁻¹Σ
    gcontacts = rdiv!(g0contacts, luden)                     # gcontacts = g0 * (1 - Σ*g0)⁻¹
    return tmatrix, gcontacts
end

#endregion

#region ## API ##

Base.view(s::TMatrixSlicer, i::Integer, j::Integer) =
    view(s.gcontacts, contactinds(s.contactorbs, i), contactinds(s.contactorbs, j))

Base.view(s::TMatrixSlicer, ::Colon, ::Colon) = view(s.gcontacts, :, :)

function Base.getindex(s::TMatrixSlicer, i::CellOrbitals, j::CellOrbitals)
    g0 = s.g0slicer
    g0ij = ensure_mutable_matrix(g0[i, j])
    tkk´ = s.tmatrix
    isempty(tkk´) && return g0ij
    k = s.contactorbs
    g0ik = mortar((g0[si, sk] for si in (i,), sk in cellsdict(k)))
    g0k´j = mortar((g0[sk´, sj] for sk´ in cellsdict(k), sj in (j,)))
    gij = mul!(g0ij, g0ik, tkk´ * g0k´j, 1, 1)  # = g0ij + g0ik * tkk´ * g0k´j
    return gij
end

ensure_mutable_matrix(m::SMatrix) = Matrix(m)
ensure_mutable_matrix(m::AbstractMatrix) = m

minimal_callsafe_copy(s::TMatrixSlicer, parentham, parentcontacts) = TMatrixSlicer(
    minimal_callsafe_copy(s.g0slicer, parentham, parentcontacts),
    s.tmatrix, s.gcontacts, s.contactorbs)

Base.view(::NothingSlicer, i::Union{Integer,Colon}...) =
    internalerror("view(::NothingSlicer): unreachable reached")

Base.getindex(::NothingSlicer, i::CellOrbitals...) = argerror("Slicer does not support generic indexing")

minimal_callsafe_copy(s::NothingSlicer, parentham, parentcontacts) = s

#endregion
#endregion

############################################################################################
# GreenSolutionCache
#   Cache that memoizes columns of GreenSolution[ci,cj] on columns of single CellSite{L}
#   It does not support more general indices, but upon creation, the cache includes the data
#   already computed for the intra-contacts Green function (with noncontact sites as undefs)
#region

struct GreenSolutionCache{T,L,G<:GreenSolution{T,<:Any,L}}
    gω::G
    cache::Dict{Tuple{SVector{L,Int},SVector{L,Int},Int},Matrix{Complex{T}}}
end

function GreenSolutionCache(gω::GreenSolution{T,<:Any,L}) where {T,L}
    cache = Dict{Tuple{SVector{L,Int},SVector{L,Int},Int},Matrix{Complex{T}}}()
    # if contacts exists, we preallocate columns for each of their sites
    if ncontacts(gω) > 0
        gmat = gω[:]
        g = parent(gω)
        h = hamiltonian(g)
        bs = blockstructure(h)
        co = contactorbitals(g)
        corngs = collect(orbranges(co))
        cls = latslice(g, :)
        nrows = flatsize(h)
        j = 0
        for colsc in cellsdict(cls)
            nj = cell(colsc)
            for j´ in siteindices(colsc)
                j += 1
                jrng = corngs[j]
                i = 0
                for rowsc in cellsdict(cls)
                    ni = cell(rowsc)
                    undefs = Matrix{Complex{T}}(undef, nrows, length(jrng))
                    for i´ in siteindices(rowsc)
                        i += 1
                        irng = corngs[i]
                        irng´ = flatrange(bs, i´)
                        copy!(view(undefs, irng´, :), view(gmat, irng, jrng))
                    end
                    push!(cache, (ni, nj, j´) => undefs)
                end
            end
        end
    end
    return GreenSolutionCache(gω, cache)
end

function Base.getindex(c::GreenSolutionCache{<:Any,L}, ci::CellSite, cj::CellSite) where {L}
    ci´, cj´ = sanitize_cellindices(ci, Val(L)), sanitize_cellindices(cj, Val(L))
    ni, i = cell(ci´), siteindex(ci´)
    nj, j = cell(cj´), siteindex(cj´)
    if haskey(c.cache, (ni, nj, j))
        gs = c.cache[(ni, nj, j)]
    else
        gs = c.gω[sites(ni, :), cj]
        push!(c.cache, (ni, nj, j) => gs)
    end
    h = hamiltonian(c.gω)
    rows = flatrange(h, i)
    return view(gs, rows, :)
end

#endregion
