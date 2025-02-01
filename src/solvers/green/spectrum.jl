############################################################################################
# Spectrum - for 0D AbstractHamiltonians
#region

struct AppliedSpectrumGreenSolver{B,S<:Spectrum{<:Any,B}} <: AppliedGreenSolver
    spectrum::S
end

struct SpectrumGreenSlicer{C,B,S<:AppliedSpectrumGreenSolver{B}} <: GreenSlicer{C}
    ω::C
    solver::S
end

#region ## Constructor ##


#end

#region ## API ##

spectrum(s::AppliedSpectrumGreenSolver) = s.spectrum

# Parent ham needs to be non-parametric, so no need to alias
minimal_callsafe_copy(s::AppliedSpectrumGreenSolver, parentham, parentcontacts) =
    AppliedSpectrumGreenSolver(s.spectrum)

minimal_callsafe_copy(s::SpectrumGreenSlicer, parentham, parentcontacts) =
    SpectrumGreenSlicer(s.ω, s.solver)

#endregion

#region ## apply ##

apply(s::GS.Spectrum, h::Hamiltonian{<:Any,<:Any,0}, ::Contacts) =
    AppliedSpectrumGreenSolver(spectrum(h; s.spectrumkw...))

apply(s::GS.Spectrum, h::ParametricHamiltonian, ::Contacts) =
    argerror("Cannot use GS.Spectrum with ParametricHamiltonian. Apply parameters with h(;params...) first.")

apply(::GS.Spectrum, h::AbstractHamiltonian, ::Contacts) =
    argerror("Can only use Spectrum with bounded Hamiltonians")

#endregion

#region ## call ##

function build_slicer(s::AppliedSpectrumGreenSolver, g, ω, Σblocks, corbitals; params...)
    g0slicer = SpectrumGreenSlicer(complex(ω), s)
    gslicer = maybe_TMatrixSlicer(g0slicer, Σblocks, corbitals)
    return gslicer
end

#endregion

#endregion

############################################################################################
# SpectrumGreenSlicer indexing
#   We don't implement view over contacts because that is taken care of by TMatrixSlicer
#region

function Base.getindex(s::SpectrumGreenSlicer{C}, i::CellOrbitals{0}, j::CellOrbitals{0}) where {C}
    oi, oj = orbindices(i), orbindices(j)
    es, psis = spectrum(s.solver)
    vi, vj = view(psis, oi, :), view(psis, oj, :)
    vj´ = vj' ./ (s.ω .- es)
    return vi * vj´
end

#endregion

############################################################################################
# densitymatrix
#   specialized DensityMatrix method for GS.Spectrum
#region

struct DensityMatrixSpectrumSolver{T,G<:GreenSlice{T},A,P}
    gs::G
    orbaxes::A
    es::Vector{Complex{T}}
    fs::Vector{Complex{T}}
    psis::P
    fpsis::P
end

## Constructor

function densitymatrix(s::AppliedSpectrumGreenSolver, gs::GreenSlice)
    # SpectrumGreenSlicer is 0D, so there is a single cellorbs in dict.
    # If rows/cols are contacts, we need their orbrows/orbcols (unlike for gs(ω; params...))
    has_selfenergy(gs) && argerror("The Spectrum densitymatrix solver currently support only `nothing` contacts")
    es, psis = spectrum(s)
    fpsis = copy(psis)
    fs = copy(es)
    orbaxes = orbrows(gs), orbcols(gs)
    solver = DensityMatrixSpectrumSolver(gs, orbaxes, es, fs, psis, fpsis)
    return DensityMatrix(solver, gs)
end

## call

function (s::DensityMatrixSpectrumSolver)(µ, kBT; params...)
    bs = blockstructure(s.gs)
    psis, fpsis, es, fs = s.psis, s.fpsis, s.es, s.fs
    β = inv(kBT)
    (@. fs = fermi(es - µ, β))
    fpsis .= psis .* transpose(fs)
    ρcell = EigenProduct(bs, psis, fpsis)
    result = call!_output(s.gs)
    getindex!(result, ρcell, s.orbaxes...)
    return result
end

#endregion
#endregion
