#######################################################################
# Green function
#######################################################################
abstract type AbstractGreensSolver end

struct GreensFunction{S<:AbstractGreensSolver,L,B<:NTuple{L,Union{Int,Missing}},H<:Hamiltonian}
    solver::S
    h::H
    boundaries::B
end

"""
    greens(h::Hamiltonian, solveobject; boundaries::NTuple{L,Integer} = missing)

Construct the Green's function `g::GreensFunction` of `L`-dimensional Hamiltonian `h` using
the provided `solveobject`. Currently valid `solveobject`s are

- `Schur1D()` (single-shot generalized eigenvalue approach for 1D Hamiltonians)

If a `boundaries = (n₁, n₂, ...)` is provided, a reflecting boundary is assumed for each
non-missing `nᵢ` perpendicular to Bravais vector `i` at a cell distance `nᵢ` from the
origin.

    h |> greens(h -> solveobject(h), args...)

Curried form equivalent to the above, giving `greens(h, solveobject(h), args...)`.

    g(ω, cells::Pair; source = I, dest = I)
    g(ω; source = I, dest = I)[cells::Pair]

From a constructed `g::GreensFunction`, obtain the retarded Green's function matrix at
frequency `ω` between unit cells `src` and `dst`, where `src, dst` are `::NTuple{L,Int}` or
`SVector{L,Int}`. If allowed by the used `solveobject`, `g0=g(ω)` builds an solution object
that can efficiently produce the Greens function between different cells at fixed `ω` with
`g0[cells]` without repeating cell-independent parts of the computation.

If `g` has a scalar eltype (single-orbital sites), the keywords `source` and `dest` allows
to specify projection matrices, possibly non-square, that act as source and sink for the
propagator, so that `g(ω, ...; source = S, dest = D) = D'*g(ω,...)*S`. This avoids building
the full-cell intermediate `g(ω,...)` for efficiency. If `source` and/or `dest` are not
`AbstractMatrix`, but an iterable of sites, `S` and `D` will be constructed as projectors
onto said sites. In the multiorbital case, non-identity `source` and `dest` are not yet
supported.

# Examples

```jldoctest
julia> g = LatticePresets.square() |> hamiltonian(hopping(-1)) |> unitcell((1,0), region = r->0<r[2]<3) |> greens(Schur1D())
GreensFunction{Schur1DGreensSolver}: Green's function using the Schur1D method
  Flat matrix size      : 2 × 2
  Flat deflated size    : 2 × 2
  Original element type : scalar (ComplexF64)
  Boundaries            : (missing,)

julia> g(0.2, 3=>2) ≈ g(0.2)[3=>2]
true
```

# See also
    `Schur1D`
"""
greens(h::Hamiltonian{<:Any,L}, solverobject; boundaries = filltuple(missing, Val(L))) where {L} =
    GreensFunction(greensolver(solverobject, h), h, boundaries)
greens(s; kw...) = h -> greens(h, greensolver(s, h); kw...)

# fallback
greensolver(s::AbstractGreensSolver, h) = s
greensolver(s::Function, h) = s(h)

sanitize_cells((cell0, cell1)::Pair{<:Number,<:Number}) =
    SA[Int(cell0)] => SA[Int(cell1)]
sanitize_cells((cell0, cell1)::Pair{<:NTuple{L,Number},<:NTuple{L,Number}}) where {L} =
    SVector(Int.(cell0)) => SVector(Int.(cell1))
sanitize_cells(cells) =
    throw(ArgumentError("Cells should be of the form `cᵢ => cⱼ`, with each `c` an `Integer` or an `NTuple{L,Integer}`, got $cells"))

Base.size(g::GreensFunction, args...) = size(g.h, args...)
Base.eltype(g::GreensFunction) = eltype(g.h)

flatsize(g::GreensFunction, args...) = flatsize(g.h, args...)
blockeltype(g::GreensFunction) = blockeltype(g.h)
blocktype(g::GreensFunction) = blocktype(g.h)
orbitaltype(g::GreensFunction) = orbitaltype(g.h)
orbitalstructure(g::GreensFunction) = orbitalstructure(g.h)

#######################################################################
# Schur1DGreensSolver
#######################################################################
"""
    Schur1D()

Return a Greens function solver using the generalized eigenvalue approach, whereby given the
energy `ω`, the eigenmodes of the infinite 1D Hamiltonian, and the corresponding infinite
and semi-infinite Greens function can be computed by solving the generalized eigenvalue
equation

    A⋅φχ = λ B⋅φχ
    A = [0 I; -h₊ ω-h₀]
    B = [I 0; 0 h₋]

This is the matrix form of the problem `λ(ω-h₀)φ - h₊φ - λ²h₋φ = 0`, where `φχ = [φ; λφ]`,
and `φ` are `ω`-energy eigenmodes, with (possibly complex) momentum `q`, and eigenvalues are
`λ = exp(-iqa₀)`. The algorithm assumes the Hamiltonian has only `dn = (0,)` and `dn = (±1,
)` Bloch harmonics (`h₀`, `h₊` and `h₋`), and will error otherwise instructing the user to
grow the unit cell. Bound states in the spectrum will yield delta functions in the density
of states that can be resolved by adding a broadening in the form of a small positive
imaginary part to `ω`. If `ω::Real`, a small imaginary part will be added automatically.

For performace, the eigenvalue equation may be `deflated' and `stabilized', i.e. singular
solutions `λ=0,∞` will be removed, and an inverse-free algorithm is used to preserve
precision even in the presence of singularities.

# Examples
```jldoctest
julia> using LinearAlgebra

julia> h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), (10,10)) |> wrap(2);

julia> g = greens(h, Schur1D(), boundaries = (0,))
GreensFunction{Schur1DGreensSolver}: Green's function using the Schur1D method
  Flat matrix size      : 40 × 40
  Flat deflated size    : 20 × 20
  Original element type : scalar (ComplexF64)
  Boundaries            : (0,)

julia> tr(g(0.3, 1=>1))
-32.193416071797216 - 3.4400038418349084im
```

# See also
    `greens`
"""
struct Schur1D end

greensolver(s::Schur1D, h) = Schur1DGreensSolver(s, h)

#### Schur1DGreensSolver ###################################################################

struct Schur1DWorkspace{T}
    nn1::Matrix{T}
    nn2::Matrix{T}
    nr1::Matrix{T}
    nr2::Matrix{T}
    rr1::Matrix{T}
    rr2::Matrix{T}
    rr3::Matrix{T}
    rr4::Matrix{T}
    r2r21::Matrix{T}
    r2r22::Matrix{T}
    n2r1::Matrix{T}
    n2r2::Matrix{T}
end

Schur1DWorkspace(R::AbstractMatrix{T}) where {T} = Schur1DWorkspace{T}(size(R)...)

Schur1DWorkspace{T}(n, r) where {T} = Schur1DWorkspace(Matrix{T}.(undef,
    ((n, n), (n, n), (n, r), (n, r), (r, r), (r, r), (r, r), (r, r), (2r, 2r), (2r, 2r), (n+2r, n+2r), (n+2r, n+2r)))...)

struct Deflator{T,S}
    L::Matrix{T}        # h₊ = L*R'
    R::Matrix{T}        # h₋ = R*L'
    iG::Matrix{T}       # Matrix(-h₀ + (LL' + RR') * im)
    ωshifter::S         # metadata to aid in ω-shifting iG
end

struct Schur1DGreensSolver{D<:Deflator,T,M} <: AbstractGreensSolver
    h0::M
    hp::M
    hm::M
    effmat::EffectiveMatrix{T}
    deflator::D
    tmp::Schur1DWorkspace{T}
end

function Schur1DGreensSolver(s::Schur1D, h)
    latdim(h) == 1 || throw(ArgumentError("Cannot use a Schur1D Green function solver with an $(latdim(h))-dimensional Hamiltonian"))
    H = maybeflatten(h)
    maxdn = maximum(har -> abs(first(har.dn)), h.harmonics)
    maxdn > 1 && throw(ArgumentError("The Hamiltonian has next-nearest unitcell hoppings. Please enlarge the unit cell with `unitcell(h, $maxdn)` to reduce to nearest-cell couplings."))
    h₋, h₀, h₊ = H[(-1,)], H[(0,)], H[(1,)]
    deflator = Deflator(h₊, h₀, h₋)
    L, R = deflator.L, deflator.R
    effmat = EffectiveMatrix(h₀, L, R)
    tmp = Schur1DWorkspace(R)
    return Schur1DGreensSolver(h₀, h₊, h₋, effmat, deflator, tmp)
end

function Deflator(h₊, h₀, h₋)
    h₊ ≈ h₋' || throw(ArgumentError("Deflation requires mutually adjoint intercell h₊ = h₋'. If you intended to build a non-Hermitian Hamiltonian, please use the undeflated method `Schur(deflation = nothing)`."))
    Ls, Rs   = svd_sparse(h₊)
    L, R     = Matrix(Ls), Matrix(Rs)
    G⁻¹      = Matrix(-h₀ + (Ls * Ls' + Rs * Rs') * im)
    ωshifter = diag(G⁻¹)
    return Deflator(L, R, G⁻¹, ωshifter)
end

function Schur1DWorkspace(d::Deflator, h₀::AbstractMatrix{T}) where {T}
    n = size(h₀, 1)
    r = size(d.R, 2)
    return Schur1DWorkspace{T}(n, r)
end

function Base.show(io::IO, g::GreensFunction{<:Schur1DGreensSolver})
    print(io, summary(g), "\n",
"  Flat matrix size      : $(size(g.solver.h0, 1)) × $(size(g.solver.h0, 2))
  Flat deflated size    : $(deflated_size_text(g))
  Original element type : $(displayelements(g.h))
  Boundaries            : $(g.boundaries)")
end

undeflated_size_text(g::GreensFunction) = undeflated_size_text(g.solver.deflator.R)
undeflated_size_text(R::AbstractMatrix) = "$(size(R, 1)) × $(size(R, 1))"

deflated_size_text(g::GreensFunction) = deflated_size_text(g.solver.deflator.R)
deflated_size_text(R::AbstractMatrix) = "$(size(R, 2)) × $(size(R, 2))"

Base.summary(g::GreensFunction{<:Schur1DGreensSolver}) =
    "GreensFunction{Schur1DGreensSolver}: Green's function using the Schur1D method"

Base.size(s::Schur1DGreensSolver, args...) = size(s.deflator, args...)

#### Deflation and modes ###################################################################

# Return deflated Adef and Bdef
function deflated_pencil!(tmp::Schur1DWorkspace, d::Deflator{T}) where {T}
    luiG = lu!(copy!(tmp.nn1, d.iG))
    GR = ldiv!(luiG, copy!(tmp.nr1, d.R))
    GL = ldiv!(luiG, copy!(tmp.nr2, d.L))
    R = d.R
    L = d.L
    r = size(R, 2)
    i1, i2 = 1:r, r+1:2r
    Adef, Bdef = tmp.r2r21, tmp.r2r22
    fill!(Adef, zero(T))
    fill!(Bdef, zero(T))
    one!(view(Adef, i1, i1))
    mul!(view(Adef, i1, i1), L', GL, -im, 1)
    mul!(view(Adef, i1, i2), L', GL, -1, 1)
    mul!(view(Adef, i2, i1), R', GL, im, 1)
    mul!(view(Adef, i2, i2), R', GL, 1, 1)
    one!(view(Bdef, i2, i2))
    mul!(view(Bdef, i1, i1), L', GR, 1, 1)
    mul!(view(Bdef, i1, i2), L', GR, im, 1)
    mul!(view(Bdef, i2, i1), R', GR, -1, 1)
    mul!(view(Bdef, i2, i2), R', GR, -im, 1)
    return Adef, Bdef
end

function shiftω!(d::Deflator, ω)
    diagG⁻¹ = d.ωshifter
    for (n, v) in enumerate(diagG⁻¹)
        d.iG[n, n] = ω + v
    end
    return d
end

function mode_subspaces(s::Schur1DGreensSolver{<:Deflator}, ω)
    d = s.deflator
    shiftω!(d, ω)
    A, B = deflated_pencil!(s.tmp, d)
    return mode_subspaces(s, A, B, imag(ω))
end

# returns invariant subspaces of retarded and advanced eigenmodes
function mode_subspaces(s::Schur1DGreensSolver, A::AbstractArray{T}, B::AbstractArray{T}, imω) where {T}
    ZrL, ZrR, ZaL, ZaR = s.tmp.rr1, s.tmp.rr2, s.tmp.rr3, s.tmp.rr4
    r = size(A, 1) ÷ 2
    if !iszero(r)
        sch = schur!(A, B)
        # Retarded modes
        whichmodes = Vector{Bool}(undef, length(sch.α))
        retarded_modes!(whichmodes, sch, imω)
        ordschur!(sch, whichmodes)
        copy!(ZrL, view(sch.Z, 1:r, 1:sum(whichmodes)))
        copy!(ZrR, view(sch.Z, r+1:2r, 1:sum(whichmodes)))
        # Advanced modes
        advanced_modes!(whichmodes, sch, imω)
        ordschur!(sch, whichmodes)
        copy!(ZaL, view(sch.Z, 1:r, 1:sum(whichmodes)))
        copy!(ZaR, view(sch.Z, r+1:2r, 1:sum(whichmodes)))
    end
    return ZrL, ZrR, ZaL, ZaR
end

# need this barrier for type-stability (sch.α and sch.β are finicky)
function retarded_modes!(whichmodes, sch, imω)
    whichmodes .= abs.(sch.α) .< abs.(sch.β)
    sum(whichmodes) == length(whichmodes) ÷ 2 || throw_imω(imω)
    return whichmodes
end

function advanced_modes!(whichmodes, sch, imω)
    whichmodes .= abs.(sch.β) .< abs.(sch.α)
    sum(whichmodes) == length(whichmodes) ÷ 2 || throw_imω(imω)
    return whichmodes
end

throw_imω(imω) =
    throw(ArgumentError("Couldn't separate advanced from retarded modes. Consider adding a larger positive imaginary part to ω, currently $imω"))

#### g(ω) ##################################################################################

(g::GreensFunction{<:Schur1DGreensSolver})(ω; kw...) =
    Schur1DGreensSolution(g, ensurecomplex(ω); kw...)

ensurecomplex(ω::T) where {T<:Real} = ω + im * default_tol(T)
ensurecomplex(ω::Complex) = ω

#### Schur1DGreensSolution #################################################################

struct Schur1DGreensSolutionWorkspace{T}
    rs1::Matrix{T}
    rs2::Matrix{T}
    ns1::Matrix{T}
    ns2::Matrix{T}
    dr::Matrix{T}
    rr::Matrix{T}
end

Schur1DGreensSolutionWorkspace(srcmat, dstmat, R::AbstractMatrix{T}) where {T} =
    Schur1DGreensSolutionWorkspace{T}(size(srcmat, 2), size(dstmat, 2), size(R)...)

Schur1DGreensSolutionWorkspace{T}(s, d, n, r) where {T} =
    Schur1DGreensSolutionWorkspace(Matrix{T}.(undef, ((r, s), (r, s), (n, s), (n, s), (d, r), (r, r)))...)

struct Schur1DGreensSolution{B<:Union{Int,Missing},T,S<:SubArray{T},O<:OrbitalStructure}
    ω::T
    L::Matrix{T}
    R::Matrix{T}
    dstmat::Matrix{T}
    G∞S::S      # G^∞ * source
    GLR::S      # G^L * R
    GRL::S      # G^R * L
    boundary::B
    orbstruct::O
    tmp::Schur1DGreensSolutionWorkspace{T}
end

function Base.show(io::IO, s::Schur1DGreensSolution{B}) where {B}
    print(io, summary(s), "\n",
"  ω             : $(s.ω)
  Matrix size   : $(undeflated_size_text(s.R))
  Deflated size : $(deflated_size_text(s.R))
  Element type  : $(blocktype(s.orbstruct))
  Boundary      : $(s.boundary)")
end

Base.summary(s::Schur1DGreensSolution) = "Schur1DGreensSolution : Schur1D solution of a GreensFunction at fixed ω"

function Schur1DGreensSolution(g, ω; source = I, dest = I)
    s = g.solver
    L, R, em, G⁻¹, G⁻¹backup = s.deflator.L, s.deflator.R, s.effmat, s.tmp.n2r1, s.tmp.n2r2
    modes = mode_subspaces(s, ω)
    effective_matrix!(G⁻¹, em, ω, modes)
    copy!(G⁻¹backup, G⁻¹)
    srcmat = flat_source_dest_matrix(source, g)
    G∞S = padded_ldiv(G⁻¹, srcmat, em, Val(:LR))
    copy!(G⁻¹, G⁻¹backup)
    GRL = padded_ldiv(G⁻¹, L, em, Val(:R))
    copy!(G⁻¹, G⁻¹backup)
    GLR = padded_ldiv(G⁻¹, R, em, Val(:L))
    dstmat = flat_source_dest_matrix(dest, g)
    boundary = only(g.boundaries)
    tmp = Schur1DGreensSolutionWorkspace(srcmat, dstmat, R)
    orbstruct = orbitalstructure(g.h)

    # Σ = R * L' * GRL * R'
    # @show sum(abs.(Σ - s.hm * ((ω*I - s.h0 - Σ) \ Matrix(s.hp))))
    # Σ = L * R' * GLR * L'
    # @show sum(abs.(Σ - s.hp * ((ω*I - s.h0 - Σ) \ Matrix(s.hm))))

    return Schur1DGreensSolution(ω, L, R, dstmat, G∞S, GLR, GRL, boundary, orbstruct, tmp)
end

function flat_source_dest_matrix(a, g)
    isflat(g.h) || throw(ArgumentError("`source` and `dest` different from the identity not yet supported in multiorbital Hamiltonians"))
    return _flat_source_dest_matrix(a, g)
end

function flat_source_dest_matrix(id::UniformScaling, g)
    n = flatsize(g, 1)
    return Matrix(one(blockeltype(g)) * id, n, n)
end

_flat_source_dest_matrix(sourcemat::AbstractMatrix, g) =
    flatten(sourcemat, orbitalstructure(g))

function _flat_source_dest_matrix(sources, g)
    n = length(sources)
    m = size(g, 1)
    T = blocktype(g)
    sourcemat = zeros(T, m, n)
    for (i, s) in enumerate(sources)
        sourcemat[s, i] = one(T)
    end
    return flat_source_dest_matrix(sourcemat, g)
end

flat_source_dest_dim(s::AbstractArray, g) = size(s, 1) * blockdim(g)
flat_source_dest_dim(::UniformScaling, g) = flatsize(g, 1)

## Generic path (indexing g(ω; kw...)[cells])

# Infinite:
# G∞ₙₙ = G∞₀₀ = (ω*I - h0 - ΣR - ΣL)⁻¹
# G∞ₙₘ = (G₁₁h₊)ⁿ⁻ᵐ G∞₀₀ = G₁₁L (R'G₁₁L)ⁿ⁻ᵐ⁻¹ R'G∞₀₀     for n-m > 0
# G∞ₙₘ = (G₋₁₋₁h₋)ᵐ⁻ⁿ G∞₀₀ = G₋₁₋₁R(L'G₋₁₋₁R)ᵐ⁻ⁿ⁻¹L'G∞₀₀ for n-m < 0
function Base.getindex(s::Schur1DGreensSolution{Missing,T}, cells) where {T}
    src, dst = sanitize_cells(cells)
    dist = only(dst) - only(src)
    G = zeros(T, size(s.dstmat, 2), size(s.G∞S, 2))
    add_G∞!(G, s, dist)
    G´ = maybe_unflatten_blocks(G, s.orbstruct)
    return G´
end

# Semiinifinite:
# Gₙₘ = (Ghⁿ⁻ᵐ - GhⁿGh⁻ᵐ)G∞₀₀ = G∞ₙₘ - GhⁿG∞₀ₘ
# Gₙₘ = G∞ₙₘ - G₁₁L(R'G₁₁L)ⁿ⁻¹ R'G∞₀ₘ       for n > 1
# Gₙₘ = G∞ₙₘ - G₋₁₋₁R(L'G₋₁₋₁R)⁻ⁿ⁻¹L'G∞₀ₘ   for n < -1
function Base.getindex(s::Schur1DGreensSolution{Int,T}, cells) where {T}
    cells´ = sanitize_cells(cells)
    G = zeros(T, size(s.dstmat, 2), size(s.G∞S, 2))
    if !is_across_boundary(cells´, s.boundary)
        m, n = dist_to_boundary.(cells´, s.boundary)  # m, n = dist_src, dist_dest
        add_G∞!(G, s, n-m, 0, 1)
        add_G∞!(G, s, n, -m, -1)
    end
    G´ = maybe_unflatten_blocks(G, s.orbstruct)
    return G´
end

# G += α dstmat' * Gh^dist * Gh^dist´ * G∞S
function add_G∞!(G, s::Schur1DGreensSolution, dist, dist´ = 0, α = 1)
    TG∞ = Ghⁿmul!(s.tmp.ns1, s.G∞S, s, dist´)
    TG∞ = Ghⁿmul!(s.tmp.ns2, TG∞, s, dist)
    mul!(G, s.dstmat', TG∞, α, 1)
    return G
end

# Gs´ = Ghⁿ * Gs = GX * (X´GX)ⁿ⁻¹ * X´ * Gs
function Ghⁿmul!(Gs´, Gs, s::Schur1DGreensSolution, n)
    if n == 0
        Gs´ = Gs  # no need to copy!(Gs´, Gs)
    else
        isplus = n > 0
        GX  = ifelse(isplus, s.GRL,  s.GLR)
        X´ = ifelse(isplus, s.R', s.L')
        X´GX = mul!(s.tmp.rr, X´, GX)
        X´Gs = mul!(s.tmp.rs1, X´, Gs)
        mul!(Gs´, GX, mul!(s.tmp.rs2, (X´GX)^(abs(n)-1), X´Gs))
    end
    return Gs´
end

## Fast-paths

function (g::GreensFunction)(ω, cells; kw...)
    cells´ = sanitize_cells(cells)
    ω´ = ensurecomplex(ω)
    b = only(g.boundaries)
    if is_infinite_local(cells´, b)
        G = infinite_local_fastpath(g, ω´; kw...)
    elseif is_across_boundary(cells´, b)
        G = zero_fastpath(g; kw...)
    elseif is_from_surface(cells´, b)
        G = from_surface_fastpath(g, ω´, dist_to_boundary(cells´, b); kw...)
    else # general form
        G = g(ω´; kw...)[cells]
    end
    G´ = maybe_unflatten_blocks(G, orbitalstructure(g.h))  # only if non-scalar eltype
    return G´
end

is_infinite_local((src, dst), b::Missing) = only(src) == only(dst)
is_infinite_local(cells, b) = false

is_from_surface((src, dst), b::Int) = abs(dist_to_boundary(src, b)) == 1
is_from_surface(cells, b) = false

is_across_boundary((src, dst), b::Int) =
    sign(dist_to_boundary(src, b)) != sign(dist_to_boundary(dst, b)) ||
    dist_to_boundary(src, b) == 0 || dist_to_boundary(dst, b) == 0
is_across_boundary(cells, b) = false

dist_to_boundary(cell, b) = only(cell) - b
dist_to_boundary((src, dst)::Pair, b) = dist_to_boundary(src, b), dist_to_boundary(dst, b)

# Infinite:
# G∞ₙₙ = G∞₀₀ = (ω*I - h0 - ΣR - ΣL)⁻¹
function infinite_local_fastpath(g, ω; source = I, dest = I)
    em, G⁻¹ = g.solver.effmat, g.solver.tmp.n2r1
    modes = mode_subspaces(g.solver, ω)
    effective_matrix!(G⁻¹, em, ω, modes)
    srcmat = flat_source_dest_matrix(source, g)
    dstmat = flat_source_dest_matrix(dest, g)
    G = dstmat' * padded_ldiv(G⁻¹, srcmat, em, Val(:LR))
    return G
end

# Surface-bulk semi-infinite:
# G₁₁ = (ω*I - h0 - ΣR)⁻¹, G₋₁₋₁ = (ω*I - h0 - ΣL)⁻¹
# Gₙ₀ = G₀ₙ = 0
# h₊ = LR', h₋ = RL'
# Gₙ₁ = (G₁₁h₊)ⁿ⁻¹G₁₁       = G₁₁L(R'G₁₁L)ⁿ⁻²R'G₁₁         for n ≥ 2
# Gₙ₁ = (G₋₁₋₁h₋)⁻ⁿ⁻¹G₋₁₋₁  = G₋₁₋₁R(L'G₋₁₋₁R)⁻ⁿ⁻²L'G₋₁₋₁  for n ≤ -2
function from_surface_fastpath(g, ω, (dsrc, ddst); source = I, dest = I)
    dist = ddst - dsrc
    em, G⁻¹ = g.solver.effmat, g.solver.tmp.n2r1
    modes = mode_subspaces(g.solver, ω)
    effective_matrix!(G⁻¹, em, ω, modes)
    isplus = dsrc > 0
    side = ifelse(isplus, Val(:R), Val(:L))
    luG⁻¹ = padded_lu!(G⁻¹, em, side)
    srcmat = flat_source_dest_matrix(source, g)
    dstmat = flat_source_dest_matrix(dest, g)

    if dist == 0
        G = dstmat' * padded_ldiv(luG⁻¹, srcmat, em, side)
    else
        X  = ifelse(isplus, em.L, em.R)
        X´ = ifelse(isplus, em.R', em.L')
        X´Gs = X´ * padded_ldiv(luG⁻¹, srcmat, em, side)
        GX = padded_ldiv(luG⁻¹, X, em, side)
        dGX = dstmat' * GX
        if abs(dist) > 1
            X´GX = mul!(g.solver.tmp.rr1, X´, GX)
            G = dGX * (X´GX)^(abs(dist)-1) * X´Gs
        else
            G = dGX*X´Gs
        end
    end
    return G
end

zero_fastpath(g; source = I, dest = I) =
    zeros(blockeltype(g), flat_source_dest_dim.((source, dest), Ref(g))...)

#######################################################################
# BandGreensSolver
#######################################################################
# struct SimplexData{D,E,T,C<:SMatrix,DD,SA<:SubArray}
#     ε0::T
#     εmin::T
#     εmax::T
#     k0::SVector{D,T}
#     Δks::SMatrix{D,D,T,DD}     # k - k0 = Δks * z
#     volume::T
#     zvelocity::SVector{D,T}
#     edgecoeffs::NTuple{E,Tuple{T,C}} # s*det(Λ)/w.w and Λc for each edge
#     dωzs::NTuple{E,NTuple{2,SVector{D,T}}}
#     defaultdη::SVector{D,T}
#     φ0::SA
#     φs::NTuple{D,SA}
# end

# struct BandGreensSolver{P<:SimplexData,E,H<:Hamiltonian} <: AbstractGreensSolver
#     simplexdata::Vector{P}
#     indsedges::NTuple{E,Tuple{Int,Int}} # all distinct pairs of 1:V, where V=D+1=num verts
#     h::H
# end

# function Base.show(io::IO, g::GreensFunction{<:BandGreensSolver})
#     print(io, summary(g), "\n",
# "  Matrix size    : $(size(g.h, 1)) × $(size(g.h, 2))
#   Element type   : $(displayelements(g.h))
#   Band simplices : $(length(g.solver.simplexdata))")
# end

# Base.summary(g::GreensFunction{<:BandGreensSolver}) =
#     "GreensFunction{Bandstructure}: Green's function from a $(latdim(g.h))D bandstructure"

# function greensolver(b::Bandstructure{D}, h) where {D}
#     indsedges = tuplepairs(Val(D))
#     v = [SimplexData(simplex, band, indsedges) for band in bands(b) for simplex in band.simplices]
#     return BandGreensSolver(v,  indsedges, h)
# end

# edges_per_simplex(L) = binomial(L,2)

# function SimplexData(simplex::NTuple{V}, band, indsedges) where {V}
#     D = V - 1
#     vs = ntuple(i -> vertices(band)[simplex[i]], Val(V))
#     ks = ntuple(i -> SVector(Base.front(Tuple(vs[i]))), Val(V))
#     εs = ntuple(i -> last(vs[i]), Val(V))
#     εmin, εmax = extrema(εs)
#     ε0 = first(εs)
#     k0 = first(ks)
#     Δks = hcat(tuple_minus_first(ks)...)
#     zvelocity = SVector(tuple_minus_first(εs))
#     volume = abs(det(Δks))
#     edgecoeffs = edgecoeff.(indsedges, Ref(zvelocity))
#     dωzs = sectionpoint.(indsedges, Ref(zvelocity))
#     defaultdη = dummydη(zvelocity)
#     φ0 = vertexstate(first(simplex), band)
#     φs = vertexstate.(Base.tail(simplex), Ref(band))
#     return SimplexData(ε0, εmin, εmax, k0, Δks, volume, zvelocity, edgecoeffs, dωzs, defaultdη, φ0, φs)
# end

# function edgecoeff(indsedge, zvelocity::SVector{D}) where {D}
#     basis = edgebasis(indsedge, Val(D))
#     othervecs = Base.tail(basis)
#     edgevec = first(basis)
#     cutvecs = (v -> dot(zvelocity, edgevec) * v - dot(zvelocity, v) * edgevec).(othervecs)
#     Λc = hcat(cutvecs...)
#     Λ = hcat(zvelocity, Λc)
#     s = sign(det(hcat(basis...)))
#     coeff = s * (det(Λ)/dot(zvelocity, zvelocity))
#     return coeff, Λc
# end

# function edgebasis(indsedge, ::Val{D}) where {D}
#     inds = ntuple(identity, Val(D+1))
#     swappedinds = tupleswapfront(inds, indsedge) # places the two edge vertindices first
#     zverts = (i->unitvector(SVector{D,Int}, i-1)).(swappedinds)
#     basis = (z -> z - first(zverts)).(Base.tail(zverts)) # first of basis is edge vector
#     return basis
# end

# function sectionpoint((i, j), zvelocity::SVector{D,T}) where {D,T}
#     z0, z1 = unitvector(SVector{D,Int}, i-1), unitvector(SVector{D,Int}, j-1)
#     z10 = z1 - z0
#     # avoid numerical cancellation errors due to zvelocity perpendicular to edge
#     d = chop(dot(zvelocity, z10), maximum(abs.(zvelocity)))
#     dzdω = z10 / d
#     dz0 = z0 - z10 * dot(zvelocity, z0) / d
#     return dzdω, dz0   # The section z is dω * dzdω + dz0
# end

# # A vector, not parallel to zvelocity, and with all nonzero components and none equal
# function dummydη(zvelocity::SVector{D,T}) where {D,T}
#     (D == 1 || iszero(zvelocity)) && return SVector(ntuple(i->T(i), Val(D)))
#     rng = MersenneTwister(0)
#     while true
#         dη = rand(rng, SVector{D,T})
#         isparallel = dot(dη, zvelocity)^2 ≈ dot(zvelocity, zvelocity) * dot(dη, dη)
#         isvalid = allunique(dη) && !isparallel
#         isvalid && return dη
#     end
#     throw(error("Unexpected error finding dummy dη"))
# end

# function vertexstate(ind, band)
#     ϕind = 1 + band.dimstates*(ind - 1)
#     state = view(band.states, ϕind:(ϕind+band.dimstates-1))
#     return state
# end

# ## Call API

# function greens!(matrix, g::GreensFunction{<:BandGreensSolver,L}, ω::Number, (src, dst)::SVectorPair{L}) where {L}
#     fill!(matrix, zero(eltype(matrix)))
#     dn = dst - src
#     for simplexdata in g.solver.simplexdata
#         g0, gjs = green_simplex(ω, dn, simplexdata, g.solver.indsedges)
#         addsimplex!(matrix, g0, gjs, simplexdata)
#     end
#     return matrix
# end

# function green_simplex(ω, dn, data::SimplexData{L}, indsedges) where {L}
#     dη = data.Δks' * dn
#     phase = cis(dot(dn, data.k0))
#     dω = ω - data.ε0
#     gz = simplexterm.(dω, Ref(dη), Ref(data), data.edgecoeffs, data.dωzs, indsedges)
#     g0z, gjz = first.(gz), last.(gz)
#     g0 = im^(L-1) * phase * sum(g0z)
#     gj = -im^L * phase * sum(gjz)
#     return g0, gj
# end

# function simplexterm(dω, dη::SVector{D,T}, data, coeffs, (dzdω, dz0), (i, j)) where {D,T}
#     bailout = Complex(zero(T)), Complex.(zero(dη))
#     z = dω * dzdω + dz0
#     # Edges with divergent sections do not contribute
#     all(isfinite, z) || return bailout
#     z0 = unitvector(SVector{D,T},i-1)
#     z1 = unitvector(SVector{D,T},j-1)
#     coeff, Λc = coeffs
#     # If dη is zero (DOS) use a precomputed (non-problematic) simplex-constant vector
#     dη´ = iszero(dη) ? data.defaultdη : dη
#     d = dot(dη´, z)
#     d0 = dot(dη´, z0)
#     d1 = dot(dη´, z1)
#     # Skip if singularity in formula
#     (d ≈ d0 || d ≈ d1) && return bailout
#     s = sign(dot(dη´, dzdω))
#     coeff0 = coeff / prod(Λc' * dη´)
#     coeffj = isempty(Λc) ? zero(dη) : (Λc ./ ((dη´)' * Λc)) * sumvec(Λc)
#     params = s, d, d0, d1
#     zs = z, z0, z1
#     g0z = iszero(dη) ? g0z_asymptotic(D, coeff0, params) : g0z_general(coeff0, params)
#     gjz = iszero(dη) ? gjz_asymptotic(D, g0z, coeffj, coeff0, zs, params) :
#                        gjz_general(g0z, coeffj, coeff0, zs, params)
#     return g0z, gjz
# end

# sumvec(::SMatrix{N,M,T}) where {N,M,T} = SVector(ntuple(_->one(T),Val(M)))

# g0z_general(coeff0, (s, d, d0, d1)) =
#     coeff0 * cis(d) * ((cosint_c(-s*(d0-d)) + im*sinint(d0-d)) - (cosint_c(-s*(d1-d)) + im*sinint(d1-d)))

# gjz_general(g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1)) =
#     g0z * (im * z - coeffj) + coeff0 * ((z0-z) * cis(d0) / (d0-d) - (z1-z) * cis(d1) / (d1-d))

# g0z_asymptotic(D, coeff0, (s, d, d0, d1)) =
#     coeff0 * (cosint_a(-s*(d0-d)) - cosint_a(-s*(d1-d))) * (im*d)^(D-1)/factorial(D-1)

# function gjz_asymptotic(D, g0z, coeffj, coeff0, (z, z0, z1), (s, d, d0, d1))
#     g0z´ = g0z
#     for n in 1:(D-1)
#         g0z´ += coeff0 * im^n * (im*d)^(D-1-n)/factorial(D-1-n) *
#                 ((d0-d)^n - (d1-d)^n)/(n*factorial(n))
#     end
#     gjz = g0z´ * (im * z - im * coeffj * d / D) +
#         coeff0 * ((z0-z) * (im*d0)^D / (d0-d) - (z1-z) * (im*d1)^D / (d1-d)) / factorial(D)
#     return gjz
# end

# cosint_c(x::Real) = ifelse(iszero(abs(x)), zero(x), cosint(abs(x))) + im*pi*(x<=0)

# cosint_a(x::Real) = ifelse(iszero(abs(x)), zero(x), log(abs(x))) + im*pi*(x<=0)

# function addsimplex!(matrix, g0, gjs, simplexdata)
#     φ0 = simplexdata.φ0
#     φs = simplexdata.φs
#     vol = simplexdata.volume
#     for c in CartesianIndices(matrix)
#         (row, col) = Tuple(c)
#         x = g0 * (φ0[row] * φ0[col]')
#         for (φ, gj) in zip(φs, gjs)
#             x += (φ[row]*φ[col]' - φ0[row]*φ0[col]') * gj
#         end
#         matrix[row, col] += vol * x
#     end
#     return matrix
# end