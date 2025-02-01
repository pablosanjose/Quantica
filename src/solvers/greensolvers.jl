############################################################################################
# Green solvers
#   All new solver::AbstractGreenSolver must live in the GreenSolvers module, and must implement
#     - apply(solver, h::AbstractHamiltonian, c::Contacts) -> AppliedGreenSolver
#   All new s::AppliedGreenSolver must implement (with Σblock a [possibly nested] tuple of MatrixBlock's)
#      - build_slicer(s, g::GreenFunction, ω, Σblocks, ::ContactOrbitals; params...) -> GreenSlicer
#      - minimal_callsafe_copy(s, parentham, parentcontacts)  # injects aliases from parent
#      - optional: needs_omega_shift(s) (has a `true` default fallback)
#   A gs::GreenSlicer's allows to compute G[gi, gi´]::AbstractMatrix for indices gi
#   To do this, it must implement contact slicing (unless it relies on TMatrixSlicer)
#      - view(gs, ::Int, ::Int) -> g(ω; kw...) between specific contacts (has error fallback)
#      - view(gs, ::Colon, ::Colon) -> g(ω; kw...) between all contacts (has error fallback)
#      - Both of the above are of type `SubArray`
#   It must also implement generic slicing, and minimal copying
#      - gs[i::CellOrbitals, j::CellOrbitals] -> must return a `Matrix` for type stability
#      - minimal_callsafe_copy(gs, parentham, parentcontacts)
#   The user-facing indexing API accepts:
#      - i::Integer -> Sites of Contact number i
#      - sites(cell::Tuple, sind::Int)::Subcell -> Single site in a cell
#      - sites(cell::Tuple, sindcollection)::Subcell -> Site collection in a cell
#      - sites(cell::Tuple, slat::Symbol)::Subcell -> Whole sublattice in a cell
#      - sites(cell::Tuple, :) ~ cell::Union{NTuple,SVector} -> All sites in a cell
#      - sel::SiteSelector ~ NamedTuple -> forms a LatticeSlice
#   Optional: to properly plot boundaries, an ::AbstractGreenSolver may also implement
#      - boundaries(s::AbstractGreenSolver) -> collection of (dir => cell)::Pair{Int,Int}
#   Aliasing: Green solvers may only alias fields from the parent Hamiltonian and Contacts
############################################################################################

module GreenSolvers

using Quantica: Quantica, AbstractGreenSolver, I

struct SparseLU <:AbstractGreenSolver end

struct Schur{T<:AbstractFloat} <: AbstractGreenSolver
    shift::T                      # Tunable parameter in algorithm, see Ω in scattering.pdf
    boundary::T                   # Cell index for boundary (float to allow boundary at Inf)
    axis::Int                     # free axis to use for n-dimensional AbstractHamiltonians
end

Schur(; shift = 1.0, boundary = Inf, axis = 1) = Schur(shift, float(boundary), axis)

struct KPM{B<:Union{Missing,NTuple{2}},A} <: AbstractGreenSolver
    order::Int
    bandrange::B
    kernel::A
    padfactor::Float64  # for automatically computed bandrange
end

KPM(; order = 100, bandrange = missing, padfactor = 1.01, kernel = I) =
    KPM(order, bandrange, kernel, padfactor)

struct Spectrum{K} <:AbstractGreenSolver
    spectrumkw::K
end

Spectrum(; spectrumkw...) = Spectrum(NamedTuple(spectrumkw))

struct Bands{B<:Union{Missing,Pair},A,K} <: AbstractGreenSolver
    bandsargs::A    # sorted to make slices easier
    bandskw::K
    boundary::B
end

Bands(bandargs...; boundary = missing, bandskw...) =
    Bands(sort.(bandargs), NamedTuple(bandskw), boundary)

Bands(bandargs::Quantica.Mesh; kw...) =
    argerror("Positional arguments of GS.Bands should be collections of Bloch phases or parameters")

end # module

const GS = GreenSolvers

include("green/sparselu.jl")
include("green/spectrum.jl")
include("green/schur.jl")
include("green/kpm.jl")
include("green/bands.jl")
include("green/internal.jl")    # solvers for internal use only
