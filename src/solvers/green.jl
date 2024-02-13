############################################################################################
# Green solvers
#   All new solver::AbstractGreenSolver must live in the GreenSolvers module, and must implement
#     - apply(solver, h::AbstractHamiltonian, c::Contacts) -> AppliedGreenSolver
#   All new s::AppliedGreenSolver must implement (with Σblock a [possibly nested] tuple of MatrixBlock's)
#      - s(ω, Σblocks, ::ContactOrbitals) -> GreenSlicer
#      - minimal_callsafe_copy(s)
#      - optional: needs_omega_shift(s) (has a `true` default fallback)
#   A gs::GreenSlicer's allows to compute G[gi, gi´]::AbstractMatrix for indices gi
#   To do this, it must implement contact slicing (unless it relies on TMatrixSlicer)
#      - view(gs, ::Int, ::Int) -> g(ω; kw...) between specific contacts (has error fallback)
#      - view(gs, ::Colon, ::Colon) -> g(ω; kw...) between all contacts (has error fallback)
#   It must also implement generic slicing, and minimal copying
#      - gs[i::CellOrbitals, j::CellOrbitals] -> must return a Matrix for type stability
#      - minimal_callsafe_copy(gs)
#   The user-facing indexing API accepts:
#      - i::Integer -> Sites of Contact number i
#      - cellsites(cell::Tuple, sind::Int)::Subcell -> Single site in a cell
#      - cellsites(cell::Tuple, sindcollection)::Subcell -> Site collection in a cell
#      - cellsites(cell::Tuple, slat::Symbol)::Subcell -> Whole sublattice in a cell
#      - cellsites(cell::Tuple, :) ~ cell::Union{NTuple,SVector} -> All sites in a cell
#      - sel::SiteSelector ~ NamedTuple -> forms a LatticeSlice
############################################################################################

module GreenSolvers

using Quantica: Quantica, AbstractGreenSolver, ensureloaded, I

struct SparseLU <:AbstractGreenSolver end

struct Schur{T<:AbstractFloat} <: AbstractGreenSolver
    shift::T                      # Tunable parameter in algorithm, see Ω in scattering.pdf
    boundary::T                   # Cell index for boundary (float to allow boundary at Inf)
end

Schur(; shift = 1.0, boundary = Inf) = Schur(shift, float(boundary))

struct KPM{B<:Union{Missing,NTuple{2}},A} <: AbstractGreenSolver
    order::Int
    bandrange::B
    kernel::A
    padfactor::Float64  # for automatically computed bandrange
end

function KPM(; order = 100, bandrange = missing, padfactor = 1.01, kernel = I)
    bandrange === missing && ensureloaded(:Arpack)
    return KPM(order, bandrange, kernel, padfactor)
end

# Used in kpm.jl
function bandrange_arpack(h::AbstractMatrix{T}) where {T}
    R = real(T)
    ϵL, _ = Quantica.Arpack.eigs(h, nev=1, tol=1e-4, which = :LR);
    ϵR, _ = Quantica.Arpack.eigs(h, nev=1, tol=1e-4, which = :SR);
    ϵmax = R(real(ϵL[1]))
    ϵmin = R(real(ϵR[1]))
    return (ϵmin, ϵmax)
end

## Alternative bandrange, requires ensureloaded(:ArnoldiMethod) in KPM constructor
# function bandrange_arnoldi(h::AbstractMatrix{T}) where {T}
#     # ensureloaded(:ArnoldiMethod)
#     R = real(T)
#     decompl, _ = Quantica.ArnoldiMethod.partialschur(h, nev=1, tol=1e-4, which = Main.ArnoldiMethod.LR());
#     decomps, _ = Quantica.ArnoldiMethod.partialschur(h, nev=1, tol=1e-4, which = Main.ArnoldiMethod.SR());
#     ϵmax = R(real(decompl.eigenvalues[1]))
#     ϵmin = R(real(decomps.eigenvalues[1]))
#     return (ϵmin, ϵmax)
# end

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
