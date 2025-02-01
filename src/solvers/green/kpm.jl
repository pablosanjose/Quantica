############################################################################################
# KPMBuilder
#   computes μⁱʲₙ = ⟨ψⱼ|kernel T_n(h)|ψᵢ⟩
#region

struct KPMBuilder{O,M,T,K,H}
    h::H
    kernel::O
    ket::K
    ket1::K
    ket2::K
    bandCH::T   # (center, halfwidth)
    mulist::M
end

function KPMBuilder(h, kernel, ket, bandCH, order)
    mulist = empty_mulist(ket, order)
    ket1, ket2 = similar(ket), similar(ket)
    return KPMBuilder(h, kernel, ket, ket1, ket2, bandCH, mulist)
end

empty_mulist(::Vector{C}, order) where {C<:Complex} =
    zeros(C, order + 1)
empty_mulist(ket::Matrix{C}, order) where {C<:Complex} =
    [zeros(C, size(ket, 2), size(ket, 2)) for _ in 1:(order + 1)]

function momentaKPM(h, ket, (center, halfwidth); order = 10, kernel = I)
    pmeter = Progress(order; desc = "Computing moments: ")
    builder = KPMBuilder(h, kernel, ket, (center, halfwidth), order)
    mulist = addmomentaKPM!(builder, pmeter)
    jackson!(mulist)
    return mulist
end

# Single step algorithm (non-identity Kernel)
function addmomentaKPM!(b::KPMBuilder{<:AbstractMatrix}, pmeter)
    seed_singlestep_KPM!(b)
    mulist, h, bandCH, kmat, kmat0, kmat1 = b.mulist, b.h, b.bandCH, b.ket, b.ket1, b.ket2
    order = length(mulist) - 1
    for n in 3:(order+1)
        ProgressMeter.next!(pmeter; showvalues = ())
        iterateKPM!(kmat0, h', kmat1, bandCH)
        mulat!(mulist, n, kmat0', kmat, 1, 1)           # μ[n] += kmat0'*kmat1
        kmat0, kmat1 = kmat1, kmat0
    end
    return mulist
end

# Double step algorithm (identity Kernel)
function addmomentaKPM!(b::KPMBuilder{<:UniformScaling}, pmeter)
    μ0, μ1 = seed_doublestep_KPM!(b)
    mulist, h, bandCH, A, kmat0, kmat1 = b.mulist, b.h, b.bandCH, b.kernel, b.ket1, b.ket2
    order = length(mulist) - 1
    for n in 3:2:(order+1)
        ProgressMeter.next!(pmeter; showvalues = ())
        ProgressMeter.next!(pmeter; showvalues = ())    # twice because of 2-step
        iterateKPM!(kmat0, h', kmat1, bandCH)
        mulat!(mulist, n, kmat1', kmat1, 2, 1)
        minusat!(mulist, n, μ0)                         # μ[n] += 2*kmat1'*kmat1 - μ0
        n + 1 > order + 1 && break
        mulat!(mulist, n + 1, kmat1', kmat0, 2, 1)
        minusat!(mulist, n + 1, μ1)                     # μ[n+1] += 2*kmat1'*kmat0 - μ1
        kmat0, kmat1 = kmat1, kmat0
    end
    A.λ ≈ 1 || (mulist .*= A.λ)
    return mulist
end

function seed_singlestep_KPM!(b::KPMBuilder)
    mul!(b.ket1, b.kernel', b.ket)
    mulscaled!(b.ket2, b.h', b.ket1, b.bandCH)
    mulat!(b.mulist, 1, b.ket1', b.ket, 1, 1)
    mulat!(b.mulist, 2, b.ket2', b.ket, 1, 1)
    return nothing
end

function seed_doublestep_KPM!(b::KPMBuilder{<:UniformScaling,Vector{T}}) where {T<:Matrix}
    copy!(b.ket1, b.ket)
    mulscaled!(b.ket2, b.h', b.ket1, b.bandCH)
    μ0, μ1 = copy(b.mulist[1]), copy(b.mulist[2])
    mulat!(b.mulist, 1, b.ket1', b.ket1, 1, 1)
    mulat!(b.mulist, 2, b.ket2', b.ket1, 1, 1)
    μ0 .= b.mulist[1] .- μ0
    μ1 .= b.mulist[2] .- μ1
    return μ0, μ1
end

function seed_doublestep_KPM!(b::KPMBuilder{<:UniformScaling,Vector{T}}) where {T<:Number}
    copy!(b.ket1, b.ket)
    mulscaled!(b.ket2, b.h', b.ket1, b.bandCH)
    b.mulist[1] += μ0 = b.ket1' * b.ket1
    b.mulist[2] += μ1 = b.ket2' * b.ket1
    return μ0, μ1
end

# y = rescaled(h) * x
function mulscaled!(y, h´, x, (center, halfwidth))
    mul!(y, h´, x)
    invhalfwidth = 1/halfwidth
    @. y = (y - center * x) * invhalfwidth
    return y
end

# kmat0 = 2 * rescaled(h) * kmat1 - kmat1
function iterateKPM!(kmat0, h, kmat1, (center, halfwidth))
    α = 2/halfwidth
    β = 2center/halfwidth
    mul!(kmat0, h, kmat1, α, -1)
    @. kmat0 = kmat0 - β * kmat1
    return kmat0
end

mulat!(C::Vector{<:Matrix}, n, A, B, α, β) = mul!(C[n], A, B, α, β)
mulat!(C::Vector{<:Number}, n, A, B, α, β) = (C[n] = α * A * B + β * C[n])

minusat!(A::Vector{<:Matrix}, n, x) = (A[n] .-= x)
minusat!(A::Vector{<:Number}, n, x) = (A[n] -= x)

function jackson!(μ::AbstractVector)
    order = length(μ) - 1
    @inbounds for n in eachindex(μ)
        μ[n] *= ((order - n + 1) * cos(π * n / (order + 1)) +
                sin(π * n / (order + 1)) * cot(π / (order + 1))) / (order + 1)
    end
    return μ
end

#endregion

############################################################################################
# AppliedKPMGreenSolver
#region

struct AppliedKPMGreenSolver{T<:AbstractFloat,M<:Union{Complex{T},AbstractMatrix{Complex{T}}}} <: AppliedGreenSolver
    momenta::Vector{M}
    bandCH::Tuple{T,T}
end

#region ## API ##

moments(s::AppliedKPMGreenSolver) = s.moments

# Parent ham needs to be non-parametric, so no need to alias
minimal_callsafe_copy(s::AppliedKPMGreenSolver, parentham, parentcontacts) = s

needs_omega_shift(s::AppliedKPMGreenSolver) = false

#endregion

#region ## apply ##

function apply(s::GS.KPM,  h::Hamiltonian{T,<:Any,0}, cs::Contacts) where {T}
    isempty(cs) && argerror("The KPM solver requires at least one contact to be added that defines where the Green function will be computed. A dummy contact can be created with `attach(nothing; sites...)`.")
    hmat = h(())
    bandCH = T.(band_ceter_halfwidth(hmat, s.bandrange, s.padfactor))
    ket = contact_basis(h, cs)
    momenta = momentaKPM(hmat, ket, bandCH; order = s.order, kernel = s.kernel)
    return AppliedKPMGreenSolver(momenta, bandCH)
end

apply(::GS.KPM, h::AbstractHamiltonian, cs::Contacts) =
    argerror("Can only use KPM with bounded non-parametric Hamiltonians")

band_ceter_halfwidth(_, (emin, emax), padfactor) = 0.5 * (emin + emax), 0.5 * (emax - emin)

function band_ceter_halfwidth(h, ::Missing, padfactor)
    @warn "Computing spectrum bounds... Consider using the `bandrange` option for faster performance."
    (emin, emax) = bandrange(h)  # requires an extension
    @warn  "Computed real bandrange = ($emin, $emax)"
    bandCH = 0.5 * (emin + emax), 0.5 * (emax - emin) * padfactor
    return bandCH
end

function bandrange(h::AbstractMatrix{T}) where {T}
    R = real(T)
    # ϵL, _ = get_eigen(ES.Arpack(nev=1, tol=1e-4, which = :LR), h)
    # ϵR, _ = get_eigen(ES.Arpack(nev=1, tol=1e-4, which = :SR), h)
    ϵL, _ = get_eigen(ES.ArnoldiMethod(nev=1, tol=1e-4, which = :LR), h)
    ϵR, _ = get_eigen(ES.ArnoldiMethod(nev=1, tol=1e-4, which = :SR), h)
    ϵmax = R(real(ϵL[1]))
    ϵmin = R(real(ϵR[1]))
    return (ϵmin, ϵmax)
end

function contact_basis(h::AbstractHamiltonian{T}, contacts) where {T}
    n = flatsize(h)
    # Orbital indices in merged contacts, all belonging to a single unit cell
    mergedorbinds = orbindices(only(cellsdict(contacts)))
    basis = zeros(Complex{T}, n, length(mergedorbinds))
    one!(basis, mergedorbinds)
    return basis
end

#endregion

#region ## call ##

function build_slicer(s::AppliedKPMGreenSolver{T}, g, ω, Σblocks, corbitals; params...) where {T}
    g0contacts = KPMgreen(s.momenta, ω, s.bandCH)
    # We rely on TMatrixSlicer to incorporate contact self-energies, and to slice contacts
    gslicer = TMatrixSlicer(g0contacts, Σblocks, corbitals)
    return gslicer
end

#endregion
#endregion

############################################################################################
# KPM Green
#   h = rescaled(H) = (H - ω0)/Δ, where (ω0, Δ) = bandsCH = (center, halfwidth)
#   G_H(ω) = 1/(ω-H) = 1/(ω - Δ*h - ω0*I) = Δ⁻¹/((ω - ω0)/Δ - h) = Δ⁻¹ G_h((ω-ω0)/Δ)
#region

function KPMgreen(momenta::Vector{<:Matrix}, ω, (ω0, Δ) = (0, 1))
    Δ⁻¹ = 1/Δ
    ω´ =  (ω - ω0) * Δ⁻¹
    g0 = zero(first(momenta))
    for (i, μi) in enumerate(momenta)
        g0n = Δ⁻¹ * KPMgreen_coefficient(i - 1, ω´)
        g0 .+= g0n .* μi
    end
    return g0
end

function KPMgreen_coefficient(n, ω)
    σ = ifelse(imag(ω) < 0, -1, 1)
    ωc = complex(ω)
    g0n = -2 * im * σ * cis(-n * σ * acos(ωc)) / (ifelse(iszero(n), 2, 1) * sqrt(1 - ωc^2))
    return g0n
end

#endregion

############################################################################################
# densitymatrix
#   specialized DensityMatrix method for GS.KPM
#region

struct DensityMatrixKPMSolver{T,M,G<:GreenSlice}
    gs::G
    momenta::Vector{M}
    bandCH::Tuple{T,T}
end

## Constructor

function densitymatrix(s::AppliedKPMGreenSolver, gs::GreenSlice{T}) where {T}
    has_selfenergy(gs) && argerror("The KPM densitymatrix solver currently support only `nothing` contacts")
    momenta = slice_momenta(s.momenta, gs)
    solver = DensityMatrixKPMSolver(gs, momenta, s.bandCH)
    return DensityMatrix(solver, gs)
end

function slice_momenta(momenta, gs)
    r = _maybe_contactinds(gs, rows(gs))
    c = _maybe_contactinds(gs, cols(gs))
    momenta´ = [_maybe_diagonal(view(m, r, c), gs) for m in momenta]
    return momenta´
end

_maybe_contactinds(gs, ::Colon) = Colon()
_maybe_contactinds(gs, i::Integer) = contactinds(gs, i)
_maybe_contactinds(gs, i::DiagIndices) = _maybe_contactinds(gs, parent(i))
_maybe_contactinds(gs, _) = throw(argerror("KPM doesn't support generic indexing"))

_maybe_diagonal(blockmat, gs) = _maybe_diagonal(blockmat, gs, rows(gs))
_maybe_diagonal(blockmat, gs, is) = blockmat

function _maybe_diagonal(blockmat::AbstractArray{C}, gs, is::DiagIndices) where {C}
    blockrngs = contact_kernel_ranges(kernel(is), parent(is), gs)  # see greenfunction.jl
    # note: blockrngs can be a lazy Iterator of unknown length
    d = Diagonal(Returns(zero(C)).(blockrngs))
    fill_diagonal!(d, blockmat, blockrngs, kernel(is))  # parent(is) should be Colon or Integer
    return d.diag
end

## call

function (d::DensityMatrixKPMSolver)(mu, kBT; params...)
    ω0, Δ = d.bandCH
    kBT´ = kBT / Δ
    mu´ = (mu - ω0) / Δ
    result = copy(call!_output(d.gs))
    ρ = serialize(result)   # obtain underlying array of result
    if kBT´ ≈ 0
        ϕ = acos(mu´)
        ρ .= first(d.momenta) .* (1-ϕ/π)
        for n in 1:length(d.momenta)-1
            @. ρ += d.momenta[n+1] * 2 * (sin(π*n)-sin(n*ϕ))/(π*n)
        end
    else
        throw(argerror("KPM densitymatrix currently doesn't support finite temperatures. Open an issue if you need this feature."))
    end
    return result
end

#endregion
#endregion
