############################################################################################
# AppliedModelGreenSolver - only for internal use
#   Given a function f(ω) that returns a function fω(::CellOrbital, ::CellOrbital),
#   implement an AppliedGreenSolver that returns a s::ModelGreenSlicer that returns fω(i,j)
#   for each single orbital when calling getindex(s, ::CellOrbitals...). view not supported.
#region

struct AppliedModelGreenSolver{F} <: AppliedGreenSolver
    f::F
end

struct ModelGreenSlicer{C<:Complex,L,F} <: GreenSlicer{C}
    ω::C
    fω::F
    contactorbs::ContactOrbitals{L}
    g0contacts::Matrix{C}
end

## API ##

function (g::GreenSlice)(x::UniformScaling)
    s = AppliedModelGreenSolver(Returns((i, j) -> ifelse(i == j, x.λ, zero(x.λ))))
    g´ = swap_solver(g, s)
    return g´(0)
end

## GreenFunction API ##

function build_slicer(s::AppliedModelGreenSolver, g, ω::C, Σblocks, contactorbs; params...) where {C}
    n = norbitals(contactorbs)
    fω = s.f(ω)
    g0contacts = Matrix{C}(undef, n, n)
    slicer = ModelGreenSlicer(ω, fω, contactorbs, g0contacts)
    fill_g0contacts!(g0contacts, slicer, contactorbs)
    return slicer
end

needs_omega_shift(s::AppliedModelGreenSolver) = false

minimal_callsafe_copy(s::Union{ModelGreenSlicer,AppliedModelGreenSolver}, args...) = s

Base.getindex(s::ModelGreenSlicer, is::CellOrbitals, js::CellOrbitals) =
    [s.fω(i, j) for i in cellorbs(is), j in cellorbs(js)]

Base.view(s::ModelGreenSlicer, i::Integer, j::Integer) =
    view(s.g0contacts, contactinds(s.contactorbs, i), contactinds(s.contactorbs, j))

Base.view(s::ModelGreenSlicer, ::Colon, ::Colon) = view(s.g0contacts, :, :)

#endregion
