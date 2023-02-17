############################################################################################
# SparseLU - for 0D AbstractHamiltonians
#region

struct AppliedSparseLU{C} <:AppliedGreenSolver
    invgreen::InverseGreenBlockSparse{C}
end

mutable struct SparseLUSlicer{C} <:GreenSlicer{C}
    fact::SparseSuite.UMFPACK.UmfpackLU{C,Int}  # of full system plus extended orbs
    nonextrng::UnitRange{Int}                    # range of non-extended orbital indices
    unitcinds::Vector{Vector{Int}}               # non-extended fact indices per contact
    unitcindsall::Vector{Int}                    # merged and uniqued unitcinds
    source::Matrix{C}                            # preallocation for ldiv! solve
    unitg::Matrix{C}
    function SparseLUSlicer{C}(fact, nonextrng, unitcinds, unitcindsall, source) where {C}
        s = new()
        s.fact = fact
        s.nonextrng = nonextrng
        s.unitcinds = unitcinds
        s.unitcindsall = unitcindsall
        s.source = source
        return s
    end
end

SparseLUSlicer(fact::Factorization{C}, unitcinds, unitcindsall, source) where {C} =
    SparseLUSlicer(fact, unitcinds, unitcindsall, source)

#region ## API ##

function apply(::GS.SparseLU, h::AbstractHamiltonian0D, cs::Contacts)
    invgreen = inverse_green(h, cs)
    return AppliedSparseLU(invgreen)
end

apply(::GS.SparseLU, ::OpenHamiltonian) =
    argerror("Can only use SparseLU with bounded Hamiltonians")

# Σblocks and contactblockstruct are not used here, because they are already inside invgreen
function (s::AppliedSparseLU{C})(ω, Σblocks, contactblockstruct) where {C}
    invgreen = s.invgreen
    nonextrng = orbrange(invgreen)
    unitcinds = invgreen.unitcinds
    unitcindsall = invgreen.unitcindsall
    source = s.invgreen.source
    # the H0 and Σs inside invgreen have already been updated by the parent call!(g, ω; ...)
    update!(invgreen, ω)
    igmat = matrix(invgreen)
    @show sum(nonzeros(igmat))
    @show sum(igmat.colptr)
    @show sum(igmat.rowval)

    fact = try
        lu(igmat)
    catch
        argerror("Encountered a singular G⁻¹(ω) at ω = $ω, cannot factorize")
    end

    so = SparseLUSlicer{C}(fact, nonextrng, unitcinds, unitcindsall, source)
    return so
end

unitcellinds_contacts(s::SparseLUSlicer) = s.unitcinds
unitcellinds_contacts_merged(s::SparseLUSlicer) = s.unitcindsall

minimal_callsafe_copy(s::SparseLUSlicer) =
    SparseLUSlicer(s.fact, s.nonextrng, s.unitcinds, s.unitcindsall, copy(s.source))

#endregion

############################################################################################
# SparseLUSlicer indexing
#region

function Base.view(s::SparseLUSlicer, i::ContactIndex, j::ContactIndex)
    dstinds = s.unitcinds[Int(i)]
    srcinds = s.unitcinds[Int(j)]
    source = view(s.source, :, 1:length(srcinds))
    return _view(s, dstinds, srcinds, source)
end

Base.view(s::SparseLUSlicer, ::Colon, ::Colon) =
    compute_or_retrieve_green(s, s.unitcindsall, s.unitcindsall, s.source)

function compute_or_retrieve_green(s::SparseLUSlicer{C}, dstinds, srcinds, source) where {C}
    if isdefined(s, :unitg)
        g = view(s.unitg, dstinds, srcinds)
    else
        fact = s.fact
        one!(source, srcinds)
        gext = ldiv!(fact, source)
        dstinds´ = ifelse(dstinds === Colon(), s.nonextrng, dstinds)
        g = view(gext, dstinds´, :)
        if srcinds === Colon()
            s.unitg = copy(view(gext, s.nonextrng, s.nonextrng))
        end
    end
    return g
end

function Base.view(s::SparseLUSlicer, i::CellOrbitals, j::CellOrbitals)
    # cannot use s.source, because it has only ncols = number of orbitals in contacts
    source = similar_source(s, j)
    v = compute_or_retrieve_green(s, orbindices(i), orbindices(j), source)
    return v
end

similar_source(s::SparseLUSlicer, ::CellOrbitals{<:Any,Colon}) =
    similar(s.source, size(s.source, 1), maximum(s.nonextrng))
similar_source(s::SparseLUSlicer, j::CellOrbitals) =
    similar(s.source, size(s.source, 1), norbs(j))

# getindex must return a Matrix
Base.getindex(s::SparseLUSlicer, i::CellOrbitals, j::CellOrbitals) = copy(view(s, i, j))

#endregion

#endregion

