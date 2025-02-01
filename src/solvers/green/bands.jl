############################################################################################
# Series
#   A generalization of dual numbers to arbitrary powers of differential ε, also negative
#   When inverting (dividing), negative powers may be produced if leading terms are zero
#   Higher terms can be lost throughout operations.
#       Series{N}(f(x), f'(x)/1!, f''(x)/2!,..., f⁽ᴺ⁾(x)/N!) = Series[f(x + ε), {ε, 0, N-1}]
#   If we need derivatives respect to x/α instead of x, we do rescale(::Series, α)
#region

struct Series{N,T}
    x::SVector{N,T}   # term coefficients
    pow::Int          # power of first term
end

Series(x::Tuple, pow = 0) = Series(SVector(x), pow)
Series(x...) = Series(SVector(x), 0)

Series{N}(x...) where {N} = Series{N}(SVector(x))
Series{N}(t::Tuple) where {N} = Series{N}(SVector(t))
Series{N}(d::Series) where {N} = Series{N}(d.x)
Series{N}(x::SVector{<:Any,T}, pow = 0) where {N,T} =
    Series(SVector(padtuple(x, zero(T), Val(N))), pow)

function rescale(d::Series{N}, α::Number) where {N}
    αp = cumprod((α ^ d.pow, ntuple(Returns(α), Val(N-1))...))
    d´ = Series(Tuple(d.x) .* αp, d.pow)
    return d´
end

chopsmall(d::Series) = Series(chopsmall(d.x), d.pow)

function trim(d::Series{N}) where {N}
    nz = leading_zeros(d)
    iszero(nz) && return d
    pow = d.pow + nz
    t = ntuple(i -> d[i + pow - 1], Val(N))
    return Series(t, pow)
end

trim(x) = x

function leading_zeros(d::Series{N}) where {N}
    @inbounds for i in 0:N-1
        iszero(d[d.pow + i]) || return i
    end
    return 0
end

function trim_and_map(func, d::Series{N}, d´::Series{N}) where {N}
    f, f´ = trim(d), trim(d´)
    pow = min(f.pow, f´.pow)
    t = ntuple(i -> func(f[pow + i - 1], f´[pow + i - 1]), Val(N))
    return Series(t, pow)
end

scalar(d::Series) = d[0]
scalar(d) = d

Base.first(d::Series) = first(d.x)

function Base.getindex(d::Series{N,T}, i::Integer) where {N,T}
    i´ = i - d.pow + 1
    checkbounds(Bool, d.x, i´) ? (@inbounds d.x[i´]) : zero(T)
end

Base.eltype(::Series{<:Any,T}) where {T} = T

Base.one(::Type{<:Series{N,T}}) where {N,T} = Series{N}(one(T))
Base.one(d::S) where {S<:Series} = one(S)
Base.zero(::Type{<:Series{N,T}}) where {N,T} = Series(zero(SVector{N,T}), 0)
Base.zero(d::S) where {S<:Series} = zero(S)
Base.iszero(d::Series) = iszero(d.x)
Base.transpose(d::Series) = d  # act as a scalar

Base.:-(d::Series) = Series(-d.x, d.pow)
Base.:+(d::Series, d´::Series) = trim_and_map(+, d, d´)
Base.:-(d::Series, d´::Series) = trim_and_map(-, d, d´)
Base.:+(d::Number, d´::Series{N}) where {N} = Series{N}(d) + d´
Base.:-(d::Number, d´::Series{N}) where {N} = Series{N}(d) - d´
Base.:*(d::Number, d´::Series) = Series(d * d´.x, d´.pow)
Base.:*(d´::Series, d::Number) = Series(d * d´.x, d´.pow)
Base.:/(d::Series{N}, d´::Series{N}) where {N} = d * inv(d´)
Base.:/(d::Series, d´::Number) = Series(d.x / d´, d.pow)
Base.:^(d::Series, n::Integer) = Base.power_by_squaring(d, n)

# necessary for the case n = 0 and n = 1 in power_by_squaring
Base.copy(d::Series) = d

function Base.:*(d::Series{N}, d´::Series{N}) where {N}
    x, x´ = promote(d.x, d´.x)
    dp = Series(x, d.pow)
    dp´ = Series(x´, d´.pow)
    return dp * dp´
end

function Base.:*(d::Series{N,T}, d´::Series{N,T}) where {N,T}
    iszero(d´) && return d´
    f, f´ = trim(d), trim(d´)
    pow = f.pow + f´.pow
    s = product_matrix(f.x) * f´.x
    return Series(s, pow)
end

function Base.inv(d::Series)
    d´ = trim(d) # remove leading zeros
    iszero(d´) && argerror("Divide by zero")
    pow = d´.pow
    # impose d * inv(d) = 1. This is equivalent to Ud * inv(d).x = 1.x, where
    # Ud = hcat(d.x, shift(d.x, 1), ...)
    s = inv(product_matrix(d´.x))[:, 1]  # faster than \
    invd = Series(Tuple(s), -pow)
    return invd
end

# Ud = [x1 0 0 0; x2 x1 0 0; x3 x2 x1 0; x4 x3 x2 x1]

# product of two series d*d´ is Ud * d´.x
function product_matrix(s::SVector{N}) where {N}
    t = ntuple(Val(N)) do i
        shiftpad(s, i - 1)
    end
    return hcat(t...)
end

# shift SVector to the right by i, padding on the left with zeros
shiftpad(s::SVector{N,T}, i) where {N,T} =
    SVector(ntuple(j -> j - i > 0 ? s[j - i] : zero(T), Val(N)))

#endregion

############################################################################################
# BandSimplex: encodes energy and momenta of vertices, and derived quantitities
#region

struct Expansions{D,T,DD}
    cis::Tuple{Complex{T},Vararg{Complex{T},D}}
    J0::NTuple{D,Complex{T}}
    Jmat::SMatrix{D,D,Complex{T},DD}
    log::NTuple{D,T}
end

struct BandSimplex{D,T,S1<:SVector{<:Any,<:Real},S2<:SMatrix{<:Any,D,<:Real},S3<:SMatrix{<:Any,<:Any,<:Real},R<:Ref{<:Expansions}}     # D = manifold dimension
    ei::S1        # eᵢ::SVector{D´,T} = energy of vertex i
    kij::S2       # kᵢ[j]::SMatrix{D´,D,T,DD´} = coordinate j of momentum for vertex i
    eij::S3       # ϵᵢʲ::SMatrix{D´,D´,T,D´D´} = e_j - e_i
    dualphi::S1   # dual::SVector{D´,T}, first hyperdual coefficient
    VD::T         # D!V = |det(kᵢʲ - kᵢ⁰)|
    refex::R      # Ref to Series expansions for necessary functions
end

# Precomputes the Series expansion coefficients for cis, J(z->0) and J(z)
function Expansions(::Val{D}, ::Type{T}) where {D,T}  # here order = D
    C = complex(T)
    # series coefs of cis(t) around t = 0
    cis = ntuple(n -> C(im)^(n-1)/(factorial(n-1)), Val(D+1))
    # series coefs of Ci(t) - im * Si(t) around t = 0, starting at order 1
    J0 = ntuple(n -> C(-im)^n/(n*factorial(n)), Val(D))
    # series coefs of Ci(t) - im * Si(t) around t = t_0 is cis(t0) * Jmat * SA[1/t, 1/t², ...]
    Jmat = ntuple(Val(D*D)) do ij
        j, i = fldmod1(ij, D)
        j > i ? zero(C) : ifelse(isodd(i), 1, -1) * C(im)^(i-j) / (i*factorial(i-j))
    end |> SMatrix{D,D,C}
    # series coefs of ln(x) around x = Δ is ln(Δ) + log .* SA[1/Δ, 1/Δ², ...]
    log = ntuple(n -> -T(-1)^n/n, Val(D))
    return Expansions(cis, J0, Jmat, log)
end

#region ## Constructors ##

BandSimplex(ei, kij, refex...) = BandSimplex(real(ei), real(kij), refex...)

BandSimplex(sb::Subband, simpinds, refex...) =  # for simpinds = (v1::Int, v2::Int,...)
    BandSimplex(energies(sb, simpinds), transpose(base_coordinates(sb, simpinds)), refex...)

function BandSimplex(ei::SVector{D´,T}, kij::SMatrix{D´,D,T}, refex = Ref(Expansions(Val(D), T))) where {D´,D,T}
    D == D´ - 1 ||
        argerror("The dimension $D of Bloch phases in simplex should be one less than the number of vertices $(D´)")
    ei, eij = snap_and_diff(ei)
    k0 = kij[1, :]
    U = kij[SVector{D}(2:D´),:]' .- k0          # edges as columns
    VD = T(abs(det(U)) / (2π)^D)
    dualphi = generate_dual_phi(eij)
    return BandSimplex(ei, kij, eij, dualphi, VD, refex)
end

# make similar ei[i] exactly the same, and compute the pairwise difference
function snap_and_diff(es)
    mes = MVector(es)
    for j in eachindex(es), i in j+1:lastindex(es)
        ei, ej = es[i], es[j]
        if ei ≈ ej
            mes[i] = mes[j]
        end
    end
    ess = SVector(mes)
    return ess, chopsmall(ess' .- ess)
end

# e_j such that e^j_k are all nonzero
generate_dual_e(::Type{SVector{D´,T}}) where {D´,T} = SVector(ntuple(i -> T(i^2), Val(D´)))

# dϕ such that M = tʲ₁tʲ₂tʲ₃...(tʲ₁-tʲ₂)(tʲ₁-tʲ₃)...(tʲ₂-tʲ₃)... is maximal
# This is a (pseudo-)vandermonde determinant, and tʲₖ = dϕʲₖ/eʲₖ
# The empirical solution turns out to be tʲₙ ≈ normalize((-1)ⁿsqrt((1+n^2)/2)) (n != j, e.g. j = 0)
function generate_dual_phi(eij::SMatrix{D´,D´,T}) where {D´,T}
    t⁰ₙ = SVector(ntuple(n -> T((-1)^n*sqrt(0.5*(1 + n^2))), Val(D´-1)))
    t⁰ₙ = SVector(zero(T), normalize(t⁰ₙ)...)
    dϕₙ = multiply_if_nonzero.(t⁰ₙ, eij[:, 1])
    return dϕₙ
end

#endregion

#region ## API ##

g_integrals(s::BandSimplex, ω, dn, val...) = isreal(ω) ?
    _g_integrals(s, real(ω), dn, val...) :  # this may be faster
    _g_integrals(s, ω, dn, val...)

function _g_integrals(s::BandSimplex, ω, dn, val...)
    g₀, gi = iszero(dn) ?
        g_integrals_local(s, ω, val...) :
        g_integrals_nonlocal(s, ω, dn, val...)
    return NaN_to_Inf(g₀), NaN_to_Inf.(gi)
end

# Since complex(1) * (Inf+Inf*im) is NaN, we convert the result to Inf in this case
NaN_to_Inf(x::T) where {T} = ifelse(isnan(x), T(Inf), x)

#endregion

#endregion

############################################################################################
# g_integrals_local: zero-dn g₀(ω) and gⱼ(ω) with normal or hyperdual numbers for φ
#region

function g_integrals_local(s::BandSimplex{<:Any,<:Any,SVector{D´,T}}, ω, ::Val{N} = Val(0)) where {D´,T,N}
    eⱼ = s.ei
    eₖʲ = s.eij
    g₀, gⱼ = begin
        if N > 0 || is_degenerate(eₖʲ)
            eⱼ´ = generate_dual_e(SVector{D´,T})
            order = ifelse(N > 0, N, D´)
            eⱼseries = Series{order}.(eⱼ, eⱼ´)
            g_integrals_local_e(s, ω, eⱼseries)
        else
            g_integrals_local_e(s, ω, eⱼ)
        end
    end
    return g₀, gⱼ
end

# whether any eₖ == eⱼ for j != k
function is_degenerate(eₖʲ::SMatrix{D´}) where {D´}
    for j in 2:D´, k in 1:j-1
        iszero(eₖʲ[k,j]) && return true
    end
    return false
end

function g_integrals_local_e(s::BandSimplex{D,T}, ω::Number, eⱼ) where {D,T}
    Δⱼ  = ω .- eⱼ
    eₖʲ  = map(chopsmall, transpose(eⱼ) .- eⱼ)  # broadcast too hard for inference
    qⱼ = q_vector(eₖʲ)                              # SVector{D´,T}
    lΔⱼ = logim.(Δⱼ, s.refex)
    Eᴰⱼ = (-1)^D .* Δⱼ.^(D-1) .* lΔⱼ ./ factorial(D-1)
    Eᴰ⁺¹ⱼ = (-1)^(D+1) .* Δⱼ.^D .* lΔⱼ ./ factorial(D)
    if iszero(eₖʲ)                                  # special case, full energy degeneracy
        Δ0 = chopsmall(first(Δⱼ))
        if iszero(Δ0)
            g₀ = zero(complex(T))
            gⱼ = SVector(ntuple(Returns(g₀), Val(D)))
        else
            g₀ = complex(scalar(inv(Δ0)))/factorial(D)
            gⱼ = SVector(ntuple(Returns(g₀/(D+1)), Val(D)))
        end
    else
        g₀ = scalar(sum(qⱼ .* Eᴰⱼ))
        gⱼ = ntuple(Val(D)) do j
            j´ = j + 1
            D´ = D + 1
            x = (Eᴰⱼ[j´] - (-Δⱼ[j´])^(D-1)/factorial(D)) * qⱼ[j´]
            for k in 1:D´
                if k != j´
                    x -= (qⱼ[j´] * Eᴰ⁺¹ⱼ[j´] + qⱼ[k] * Eᴰ⁺¹ⱼ[k]) / eₖʲ[k, j´]
                end
            end
            return scalar(x)
        end |> SVector
    end
    return g₀, gⱼ
end

function q_vector(eₖʲ::SMatrix{D´,D´,S}) where {D´,S}
    qⱼ = ntuple(Val(D´)) do j
        x = one(S)
        for k in 1:D´
            j != k && (x *= eₖʲ[k, j])
        end
        return inv(x)
    end
    return qⱼ
end

# imaginary log with branchcut in the lower plane
logim(x::Complex) = iszero(imag(x)) ? logim(real(x)) : log(-im * x)
logim(x::T) where {T<:Real} = log(abs(x)) - im * T(0.5π) * sign(x)
logim(x, ex) = logim(x)

# required for local degenerate case (expansion of logim(Δ::Series, ex))
function logim(s::Series{N}, ex) where {N}
    s₀ = scalar(s)
    l₀ = logim(s₀)
    log_coeff = tupletake(ex.log, Val(N-1))
    invzΔ = cumprod(ntuple(Returns(1/s₀), Val(N-1)))
    lⱼ = log_coeff .* invzΔ
    l = Series(l₀, lⱼ...)
    return rescale(l, s[1])
end

#endregion

############################################################################################
# g_integrals_nonlocal: finite-dn g₀(ω) and gⱼ(ω) with normal or hyperdual numbers for φ
#region

function g_integrals_nonlocal(s::BandSimplex{D,T}, ω, dn, ::Val{N} = Val(0)) where {D,T,N}
    ϕⱼ = s.kij * dn
    ϕₖʲ = chopsmall.(transpose(ϕⱼ) .- ϕⱼ)
    eₖʲ = s.eij
    g₀, gⱼ = begin
        if N > 0 || is_degenerate(ϕₖʲ, eₖʲ)
            order = ifelse(N > 0, N, D+1)
        ## This dynamical estimate of the order is not type-stable. Not worth it
        # order = N == 0 ? simplex_degeneracy(ϕₖʲ, eₖʲ) + 1 : N
        # if order > 1
            ϕⱼ´ = s.dualphi
            ϕⱼseries = Series{order}.(ϕⱼ, ϕⱼ´)
            g_integrals_nonlocal_ϕ(s, ω, ϕⱼseries)
        else
            g_integrals_nonlocal_ϕ(s, ω, ϕⱼ)
        end
    end
    return g₀, gⱼ
end

# If any ϕₖʲ = 0, or if any tₖʲ and tₗʲ are equal
function is_degenerate(ϕₖʲ::SMatrix{D´}, eₖʲ) where {D´}
    for j in 2:D´, k in 1:j-1
        # iszero(ϕₖʲ[k,j]) && iszero(eₖʲ[k,j]) && return true # fails for g₁
        iszero(ϕₖʲ[k,j]) && return true
        if !iszero(eₖʲ[k,j])
            for l in 1:D´
                if l != k && l != j
                    ϕₖʲ[k,j]*eₖʲ[l,j] ≈ eₖʲ[k,j]*ϕₖʲ[l,j] && return true
                end
            end
        end
    end
    return false
end

function g_integrals_nonlocal_ϕ(s::BandSimplex{D,T}, ω::Number, ϕⱼ) where {D,T}
    eⱼ  = s.ei
    eₖʲ = s.eij
    Δⱼ  = ω .- eⱼ
    ϕₖʲ  = map(chopsmall, transpose(ϕⱼ) .- ϕⱼ)  # broadcast too hard for inference
    tₖʲ = divide_if_nonzero.(ϕₖʲ, eₖʲ)
    eϕⱼ  = cis_scalar.(ϕⱼ, s.refex)
    αₖʲγⱼ  = αγ_matrix(ϕₖʲ, tₖʲ, eₖʲ)               # αₖʲγⱼ :: SMatrix{D´,D´}
    if iszero(eₖʲ)                                  # special case, full energy degeneracy
        Δ0 = chopsmall(first(Δⱼ))
        if iszero(Δ0)
            g₀ = zero(complex(T))
            gⱼ = ntuple(Returns(g₀), Val(D))
        else
            Δ0⁻¹ = inv(Δ0)
            γⱼ = αₖʲγⱼ[1,:]                         # if eₖʲ == 0, then αₖʲ == 1
            λⱼ = γⱼ .* eϕⱼ
            λₖʲ = divide_if_nonzero.(transpose(λⱼ), ϕₖʲ)
            q = (-im)^D * Δ0⁻¹
            g₀ = q * sum(scalar.(λⱼ))
            gⱼ = ntuple(Val(D)) do j
                q * scalar(λⱼ[j+1] + im * sum(λₖʲ[:,j+1] - transpose(λₖʲ)[:,j+1]))
            end
        end
    else
        αₖʲγⱼeϕⱼ = αₖʲγⱼ .* transpose(eϕⱼ)          # αₖʲγⱼeϕⱼ :: SMatrix{D´,D´}
        Jₖʲ = J_scalar.(tₖʲ, eₖʲ, transpose(Δⱼ), s.refex) # Jₖʲ :: SMatrix{D´,D´}
        αₖʲγⱼeϕⱼJₖʲ = αₖʲγⱼeϕⱼ .* Jₖʲ
        Λⱼ = sum(αₖʲγⱼeϕⱼJₖʲ, dims = 1)
        Λₖʲ = Λ_matrix(eₖʲ, ϕₖʲ, Λⱼ, Δⱼ, tₖʲ, αₖʲγⱼeϕⱼ, Jₖʲ)
        q´ = (-im)^(D+1)
        g₀ = q´ * sum(scalar.(Λⱼ))
        gⱼ = ntuple(Val(D)) do j
            q´ * scalar(Λⱼ[j+1] + im * sum(Λₖʲ[:,j+1] - transpose(Λₖʲ)[:,j+1]))
        end
    end
    return Complex{T}(g₀), SVector{D,Complex{T}}(gⱼ)
end

# Series of cis(ϕ)
function cis_scalar(s::Series{N}, ex) where {N}
    @assert iszero(s.pow)
    cis_series = Series(tupletake(ex.cis, Val(N)))
    c = cis(s[0]) * cis_series
    # Go from ds differential to dϕ
    return rescale(c, s[1])
end

cis_scalar(s, ex) = cis(s)

divide_if_nonzero(a, b) = iszero(b) ? a : a/b
multiply_if_nonzero(a, b) = iszero(b) ? a : a*b

function αγ_matrix(ϕedges::S, tedges::S, eedges::SMatrix{D´,D´}) where {D´,S<:SMatrix{D´,D´}}
    js = ks = SVector{D´}(1:D´)
    kjs = tuple.(ks, js')
    α⁻¹ = α⁻¹_scalar.(kjs, Ref(tedges), Ref(eedges))
    γ⁻¹ = γ⁻¹_scalar.(js', Ref(ϕedges), Ref(eedges))
    γα = inv.(α⁻¹ .* γ⁻¹)
    return γα
end

function α⁻¹_scalar((k, j), tedges::SMatrix{D´,D´,S}, eedges) where {D´,S}
    x = one(S)
    j != k && !iszero(eedges[k, j]) || return x
    @inbounds for l in 1:D´
        if l != j  && !iszero(eedges[l, j])
            x *= eedges[l, j]
            if l != k # ekj != 0, already constrained above
                x *= chopsmall(tedges[l, j] - tedges[k, j])
            end
        end
    end
    return x
end

function γ⁻¹_scalar(j, ϕedges::SMatrix{D´,D´,S}, eedges) where {D´,S}
    x = one(S)
    @inbounds for l in 1:D´
        if l != j && iszero(eedges[l, j])
            x *= ϕedges[l, j]
        end
    end
    return x
end

function Λ_matrix(eₖʲ::SMatrix{D´}, ϕₖʲ, Λⱼ, Δⱼ, tₖʲ, αₖʲγⱼeϕⱼ, Jₖʲ) where {D´}
    js = ks = SVector{D´}(1:D´)
    ## Inference currently struggles with this
    # kjs = tuple.(ks, js')
    # Λₖʲ = Λ_scalar.(kjs, ϕₖʲ, transpose(Λⱼ), transpose(Δⱼ), Ref(eₖʲ), Ref(tₖʲ), Ref(αₖʲγⱼeϕⱼ), Ref(Jₖʲ))
    kjs = Tuple(tuple.(ks, js'))
    Λₖʲtup = ntuple(Val(D´*D´)) do i
        (k,j) = kjs[i]
        Λ_scalar((k,j), ϕₖʲ[k,j], Λⱼ[j], Δⱼ[j], eₖʲ, tₖʲ, αₖʲγⱼeϕⱼ, Jₖʲ)
    end
    Λₖʲ = SMatrix{D´,D´}(Λₖʲtup)
    return Λₖʲ
end

function Λ_scalar((k, j), ϕₖʲ, Λⱼ, Δⱼ, emat::SMatrix{D´,D´,T}, tmat, αγeϕmat, Jmat) where {D´,T}
    Λₖʲ = zero(typeof(Λⱼ))
    j == k && return Λₖʲ
    eₖʲ = emat[k,j]
    if iszero(eₖʲ)
        Λₖʲ = Λⱼ / ϕₖʲ
    else
        tₖʲ = tmat[k,j]
        Jₖʲ = Jmat[k,j]
        @inbounds for l in 1:D´
            if !iszero(emat[l,j])
                tₗʲ = tmat[l,j]
                Jₗʲ = Jmat[l,j]
                if tₗʲ == tₖʲ
                    Λₖʲ -= (αγeϕmat[l, j] / eₖʲ) * (inv(tₗʲ) + im * Δⱼ * Jₗʲ)
                else
                    Λₖʲ -= (αγeϕmat[l, j] / eₖʲ) * chopsmall(Jₗʲ - Jₖʲ) * inv(chopsmall(tₗʲ - tₖʲ))
                end
            end
        end
    end
    return Λₖʲ
end

function J_scalar(t::T, e, Δ, ex) where {T<:Real}
    iszero(e) && return zero(complex(T))
    tΔ = t * Δ
    J = iszero(tΔ) ? logim(Δ) : cis(tΔ) * J_integral(tΔ, t, Δ)
    return Complex{T}(J)
end

function J_scalar(t::Series{N,T}, e, Δ, ex) where {N,T<:Real}
    iszero(e) && return zero(Series{N,Complex{T}})
    N´ = N - 1
    C = Complex{T}
    iszero(Δ) && return Series{N}(C(Inf))
    t₀ = t[0]
    tΔ = t₀ * Δ
    if iszero(tΔ)
        J₀ = logim(Δ)
        Jᵢ = tupletake(ex.J0, Val(N´)) # ntuple(n -> C(-im)^n/(n*factorial(n)), Val(N´))
        J = Series{N}(J₀, Jᵢ...)
        cis_coefs = tupletake(ex.cis, Val(N))
        E = Series(cis_coefs)   # cis(tΔ) == 1
        EJ = E * J
    else
        cistΔ =  cis(tΔ)
        J₀ = J_integral(tΔ, t₀, Δ)
        if N > 1
            invzΔ = cumprod(ntuple(Returns(1/tΔ), Val(N-1)))
            Jmat = smatrixtake(ex.Jmat, Val(N´))
            Jᵢ = Tuple(inv(cistΔ) * (Jmat * SVector(invzΔ)))
            J = Series(J₀, Jᵢ...)
        else
            J = Series(J₀)
        end
        cis_coefs = tupletake(ex.cis, Val(N))
        Eᵢ = cistΔ .* cis_coefs
        E = Series(Eᵢ)
        EJ = E * J
    end
    return rescale(EJ, t[1] * Δ)
end

function J_integral(tΔ, t::T, Δ) where {T}
    J = iszero(imag(tΔ)) ?
        cosint(abs(tΔ)) - im*sinint(real(tΔ)) - im*T(0.5π)*sign(Δ) :
        -gamma(0, im*tΔ) - im*T(0.5π)*(sign(real(Δ))+sign(real(tΔ)))
    return Complex{T}(J)
end

#endregion

############################################################################################
# AppliedBandsGreenSolver <: AppliedGreenSolver
#region

struct SubbandSimplices{BS<:BandSimplex}
    simps::Vector{BS}                   # collection of k_∥-ordered simplices
    simpslices::Vector{UnitRange{Int}}  # ranges of simplices with "equal" k_∥ to boundary
end

struct BoundaryOrbs{L}
    boundary::Pair{Int,Int}             # dir => pos
    orbsright::OrbitalSlice{L}          # 1D boundary orbslice coupled to right half-"plane"
    orbsleft::OrbitalSlice{L}           # 1D boundary orbslice coupled to left half-"plane"
end

struct AppliedBandsGreenSolver{B<:Union{Missing,BoundaryOrbs},SB<:Subband,SS<:SubbandSimplices} <: AppliedGreenSolver
    subband::SB                         # single (non-split) subband
    subbandsimps::SS                    # BandSimplices in subband
    boundaryorbs::B                     # missing or BoundaryOrbs
end

#region ## Constructors ##

boundaryorbs(::Missing, h::AbstractHamiltonian) = missing

function boundaryorbs((dir, pos), h::AbstractHamiltonian)
    L = latdim(h)
    rcell = (pos+1) * unitvector(dir, SVector{L,Int})
    lcell = (pos-1) * unitvector(dir, SVector{L,Int})
    orbsright = coupled_orbslice(<=(pos), h, rcell, dir)
    orbsleft  = coupled_orbslice(>=(pos), h, lcell, dir)
    return BoundaryOrbs(dir => pos, orbsright, orbsleft)
end

function coupled_orbslice(condition, h, seedcell, dir)
    lat = lattice(h)
    ls = grow(lat[CellSites(seedcell, :)], h)
    (cells, inds) = group_projected_cell_indices(condition, dir, cellsdict(ls))
    cdict = cellinds_to_dict(cells, unsafe_cellsites.(cells, inds)) # no uniqueness check here
    sslice = SiteSlice(lat, cdict)  # here we get an (unnecessary) uniqueness check
    oslice = sites_to_orbs_nogroups(sslice, h)
    return oslice
end

projected_cell(cell::SVector{L,Int}, dir) where {L} =
    cell[dir] * unitvector(dir, SVector{L,Int})

# A merged version of:
# [(projected_cell(cell(c), dir), inds(c))  for c in cellsdict(ls) if condition(cell(c)[dir])]
function group_projected_cell_indices(condition, dir, d::CellSitesDict{L}) where {L}
    keys´ = SVector{L,Int}[]
    indvals´ = Vector{Int}[]
    if !isempty(d)
        # get [cell => sites(cell, inds)...]
        ps = collect(pairs(d))
        # get [projcell => sites(cell, inds)...]
        map!(kv -> projected_cell(first(kv), dir) => last(kv), ps, ps)
        # remove those pcells that do not satisfy condition
        filter!(kv -> condition(first(kv)[dir]), ps)
        # sort by projcell
        sort!(ps, by = first)
        # Do an append!-merge of equal projcells
        key´ = first(first(ps))
        indval´ = Int[]
        for (key, val) in ps
            if key != key´
                push!(keys´, key´)
                push!(indvals´, unique!(sort!(indval´)))
                key´ = key
                indval´ = copy(siteindices(val))
            else
                append!(indval´, siteindices(val))
            end
        end
        push!(keys´, key´)
        push!(indvals´, unique!(sort!(indval´)))
    end
    return keys´, indvals´
end

#endregion

#region ## API ##

# Parent hamiltonian needs to be non-parametric, so no need to alias
minimal_callsafe_copy(s::AppliedBandsGreenSolver, parentham, parentcontacts) = s

needs_omega_shift(s::AppliedBandsGreenSolver) = false

bands(g::GreenFunction{<:Any,<:Any,<:Any,<:AppliedBandsGreenSolver}) = g.solver.subband

boundaries(s::AppliedBandsGreenSolver{Missing}) =  ()
boundaries(s::AppliedBandsGreenSolver) = (s.boundaryorbs.boundary,)

#endregion

#region ## apply ##

function apply(s::GS.Bands,  h::AbstractHamiltonian{T,<:Any,L}, cs::Contacts) where {T,L}
    L == 0 && argerror("Cannot use GreenSolvers.Bands with 0D AbstractHamiltonians")
    ticks = s.bandsargs
    kw = s.bandskw
    b = bands(h, ticks...; kw..., projectors = true, split = false)
    sb = only(subbands(b))
    ex = Expansions(Val(L), T)
    subbandsimps = subband_simplices!(sb, ex, s)
    boundary = boundaryorbs(s.boundary, h)
    return AppliedBandsGreenSolver(sb, subbandsimps, boundary)
end

# reorders simplices(sb) (simp indices) to match simps::Vector{<:BandSimplex}
function subband_simplices!(sb::Subband, ex::Expansions, s::GS.Bands)
    refex = Ref(ex)
    simpinds = simplices(sb)
    simps = [BandSimplex(sb, simp, refex) for simp in simpinds]
    simpslices = simplex_slices!(simps, simpinds, s)
    return SubbandSimplices(simps, simpslices)
end

# order simplices by their k∥ and compute ranges in each kranges bin
function simplex_slices!(simps::Vector{<:BandSimplex{D}}, simpinds, s::GS.Bands) where {D}
    boundary = s.boundary
    ticks = applied_ticks(s.bandsargs, Val(D))
    if boundary !== missing
        dir = first(boundary)
        checkdirection(dir, simps)
        if D > 1
            p = sortperm(simps, by = simp -> parallel_base_coordinate(simp, dir))
            permute!(simps, p)
            permute!(simpinds, p)
            # Discrete values for L-1 dimensional k∥ mesh
            kticks = ntuple(i -> ifelse(i < dir, ticks[i], ticks[i+1]), Val(D-1))
            simpslices = collect(Runs(simps, (s1, s2) -> in_same_interval((s1, s2), kticks, dir)))
        else
            simpslices = [UnitRange(eachindex(simps))]
        end
        return simpslices
    end
    return UnitRange{Int}[]
end

# in case s.bandargs is empty (relying on defauld_band_ticks)
applied_ticks(bandsargs::Tuple{}, val) = default_band_ticks(val)
applied_ticks(bandsargs, val) = bandsargs

checkdirection(dir, ::Vector{<:BandSimplex{D}}) where {D} =
    1 <= dir <= D || argerror("Boundary direction $dir should be 1 <= dir <= $D")

function parallel_base_coordinate(s::BandSimplex{D}, dir) where {D}
    kmean = mean(s.kij, dims = 1)
    notdir = SVector(ntuple(i -> ifelse(i < dir, i, i+1), Val(D-1)))
    kpar = kmean[notdir]
    return kpar
end

# assumes kticks are sorted, see GS.Bands constructor
function in_same_interval((s1, s2), kticks, dir)
    kvec1 = parallel_base_coordinate(s1, dir)
    kvec2 = parallel_base_coordinate(s2, dir)
    for (k1, k2, ks) in zip(kvec1, kvec2, kticks)
        for i in firstindex(ks):lastindex(ks)-1
            in1 = ks[i] < k1 < ks[i+1]
            in2 = ks[i] < k2 < ks[i+1]
            xor(in1, in2) && return false
            in1 && in2 && break
        end
    end
    return true
end

#endregion

#region ## call ##

function build_slicer(s::AppliedBandsGreenSolver, g, ω, Σblocks, corbitals; params...)
    g0slicer = BandsGreenSlicer(complex(ω), s)
    gslicer = maybe_TMatrixSlicer(g0slicer, Σblocks, corbitals)
    return gslicer
end

#endregion

#endregion

############################################################################################
# BandsGreenSlicer <: GreenSlicer
#region

struct BandsGreenSlicer{C,B,S<:AppliedBandsGreenSolver{B}} <: GreenSlicer{C}
    ω::C
    solver::S
end

#region ## API ##

Base.getindex(s::BandsGreenSlicer{<:Any,Missing}, i::CellOrbitals, j::CellOrbitals) =
    inf_band_slice(s, i, j)
Base.getindex(s::BandsGreenSlicer, i::CellOrbitals, j::CellOrbitals) =
    semi_band_slice(s, i, j)

function inf_band_slice(s::BandsGreenSlicer{C}, i::Union{SparseIndices,CellOrbitals}, j::Union{SparseIndices,CellOrbitals}) where {C}
    solver = s.solver
    gmat = zeros(C, norbitals(i), norbitals(j))
    inf_band_slice!(gmat, s.ω, (i, j), solver.subband, solver.subbandsimps)
    return gmat
end

function inf_band_slice!(gmat, ω, (i, j)::Tuple{Union{SparseIndices,CellOrbitals},Union{SparseIndices,CellOrbitals}},
        subband::Subband, subbandsimps::SubbandSimplices,
        simpinds = eachindex(simplices(subband)))
    dist = dist_or_zero(i, j)
    orbs = orbstuple_or_orbranges(i, j)
    return inf_band_slice!(gmat, ω, dist, orbs, subband, subbandsimps, simpinds)
end

## Fast codepath for diagonal slices
dist_or_zero(i, j) = cell(i) - cell(j)
dist_or_zero(i::S, j::S) where {S<:SparseIndices{<:AnyCellOrbitals}} = zero(cell(parent(i)))

# can be all orbitals in i⊗j, or a range of orbitals for each site along diagonal
orbstuple_or_orbranges(i, j) = orbindices(i), orbindices(j)
orbstuple_or_orbranges(i::S, j::S) where {S<:SparseIndices{<:AnyCellOrbitals}} =
    orbranges(parent(i))

# main driver
function inf_band_slice!(gmat, ω, dist, orbs, subband, subbandsimps, simpinds)
    ψPdict = projected_states(subband)
    for simpind in simpinds
        bandsimplex = subbandsimps.simps[simpind]
        g₀, gⱼs = g_integrals(bandsimplex, ω, dist)
        isinf(g₀) && continue
        v₀, vⱼs... = simplices(subband)[simpind]
        ψ = states(vertices(subband, v₀))
        pind = (simpind, v₀)
        muladd_ψPψ⁺!(gmat, bandsimplex.VD * (g₀ - sum(gⱼs)), ψ, ψPdict, pind, orbs)
        for (j, gⱼ) in enumerate(gⱼs)
            vⱼ = vⱼs[j]
            ψ = states(vertices(subband, vⱼ))
            pind = (simpind, vⱼ)
            muladd_ψPψ⁺!(gmat, bandsimplex.VD * gⱼ, ψ, ψPdict, pind, orbs)
        end
    end
    return gmat
end

function inf_band_slice!(gmat, ω, (si, sj)::Tuple, args...)   # si, sj can be orbslice or cellorbs
    if ncells(si) == ncells(sj) == 1
        # boundary slice is one-cell-wide, no need for views
        i, j = get_single_cellorbs(si), get_single_cellorbs(sj)
        inf_band_slice!(gmat, ω, (i, j), args...)
    else
        offsetj = 0
        for j in get_multiple_cellorbs(sj)
            offseti = 0
            nj = norbitals(j)
            for i in get_multiple_cellorbs(si)
                ni = norbitals(i)
                gv = view(gmat, offseti+1:offseti+ni, offsetj+1:offsetj+nj)
                inf_band_slice!(gv, ω, (i, j), args...)
                offseti += ni
            end
            offsetj += nj
        end
    end
    return gmat
end

get_single_cellorbs(c::CellOrbitals) = c
get_single_cellorbs(c::OrbitalSlice) = only(cellsdict(c))

get_multiple_cellorbs(c::CellOrbitals) = (c,)
get_multiple_cellorbs(c::LatticeSlice) = cellsdict(c)

# Gᵢⱼ(k∥) = G⁰ᵢⱼ(k∥) - G⁰ᵢᵦ(k∥)G⁰ᵦᵦ(k∥)⁻¹G⁰ᵦⱼ(k∥), where β are removed sites at boundary
function semi_band_slice(s::BandsGreenSlicer{C}, i::CellOrbitals{L}, j::CellOrbitals{L}) where {C,L}
    borbs = s.solver.boundaryorbs
    ni, nj = length(orbindices(i)), length(orbindices(j))
    gij = zeros(C, ni, nj)
    (dir, pos) = borbs.boundary
    xi, xj = cell(i)[dir] - pos, cell(j)[dir] - pos
    if sign(xi) == sign(xj) != 0
        subband, subbandsimps = s.solver.subband, s.solver.subbandsimps
        b = ifelse(xi > 0, borbs.orbsright, borbs.orbsleft)  # 1D boundary orbital slice
        n0 = norbitals(b)
        g0j = zeros(C, n0, nj)
        gi0 = zeros(C, ni, n0)
        g00 = zeros(C, n0, n0)
        for simpinds in subbandsimps.simpslices
            fill!(g00, zero(C))
            fill!(g0j, zero(C))
            fill!(gi0, zero(C))
            inf_band_slice!(gij, s.ω, (i, j), subband, subbandsimps, simpinds)
            inf_band_slice!(g00, s.ω, (b, b), subband, subbandsimps, simpinds)
            inf_band_slice!(g0j, s.ω, (b, j), subband, subbandsimps, simpinds)
            inf_band_slice!(gi0, s.ω, (i, b), subband, subbandsimps, simpinds)
            gg = ldiv!(lu!(g00), g0j)
            mul!(gij, gi0, gg, -1, 1)
        end
    end
    return gij
end

# does g += α * ψPψ´ = α * ψP * (ψP)´, where ψP = ψPdict[pind] is the vertex projection onto
# the simplex subspace. If pind = (simpind, vind) is not in ψPdict::Dict, no P is necessary
function muladd_ψPψ⁺!(gmat, α, ψ, ψPdict, pind, orbs)
    if haskey(ψPdict, pind)
        muladd_ψPψ⁺!(gmat, α, ψPdict[pind], orbs)
    else
        muladd_ψPψ⁺!(gmat, α, ψ, orbs)
    end
    return gmat
end

function muladd_ψPψ⁺!(gmat, α, ψ, (rows, cols)::Tuple)
    if size(ψ, 1) == length(rows) == length(cols)
        mul!(gmat, ψ, ψ', α, 1)
    else
        ψrows = view_or_copy(ψ, rows, :)
        ψcols = view_or_copy(ψ, cols, :)
        mul!(gmat, ψrows, ψcols', α, 1)
    end
    return gmat
end

# fill in only rngs blocks in diagonal of gmat (used by fast codepath for diagonal indexing)
function muladd_ψPψ⁺!(gmat, α, ψ, rngs)
    for rng in rngs
        if length(rng) == 1
            i = only(rng)
            gmat[i, i] += α * abs2(ψ[i])
        else
            gv = view(gmat, rng, rng)
            ψv = view(ψ, rng, :)
            mul!(gv, ψv, ψv', α, 1)
        end
    end
    return gmat
end

view_or_copy(ψ, rows::Union{Colon,AbstractRange}, cols::Union{Colon,AbstractRange}) =
    view(ψ, rows, cols)
view_or_copy(ψ, rows, cols) = ψ[rows, cols]

minimal_callsafe_copy(s::BandsGreenSlicer, parentham, parentcontacts) = s  # it is read-only

#endregion

############################################################################################
# getindex_diag: optimized cell indexing for BandsGreenSlicer, used by diagonal indexing
#   no need to compute full ψ * ψ' if only diagonal is needed
#region

# triggers fast codepath above
getindex_diag(gω::GreenSolution{T,<:Any,<:Any,G}, o::CellOrbitalsGrouped, symmetrize) where {T,G<:BandsGreenSlicer{<:Any,Missing}} =
    maybe_symmetrized_matrix(
        inf_band_slice(slicer(gω), SparseIndices(o, Missing), SparseIndices(o, Missing)),
        symmetrize)

#endregion
#endregion
