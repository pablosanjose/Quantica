using LinearAlgebra: diag, norm
using Quantica: Hamiltonian, ParametricHamiltonian

@testset "basic hamiltonians" begin
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
               LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
               LatticePresets.bcc)
    types = (Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
    ts = (1, 2.0I, @SMatrix[1 2; 3 4], 1.0f0*I)
    orbs = (Val(1), Val(1), Val(2), (Val(1), Val(2)))
    for preset in presets, lat in (preset(), unitcell(preset())), type in types
        E, L = dims(lat)
        dn0 = ntuple(_ -> 1, Val(L))
        for (t, o) in zip(ts, orbs)
            @test hamiltonian(lat, onsite(t) + hopping(t; range = 1), orbitals = o, type = type) isa Hamiltonian
            @test hamiltonian(lat, onsite(t) - hopping(t; dn = dn0), orbitals = o, type = type) isa Hamiltonian
        end
    end
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1, range = 1/√3))
    @test bloch(h) == h.harmonics[1].h
    # Inf range
    h = LatticePresets.square() |> unitcell(region = RegionPresets.square(5)) |>
        hamiltonian(hopping(1, range = Inf))
    @test Quantica.nhoppings(h) == 600

    h = LatticePresets.square() |> hamiltonian(hopping(1, dn = (10,0), range = Inf))
    @test Quantica.nhoppings(h) == 1
    @test isassigned(h, (10,0))
end

@testset "similarmatrix" begin
    types = (ComplexF16, ComplexF32, ComplexF64)
    lat = LatticePresets.honeycomb()
    for T in types
        h0 = hamiltonian(lat, onsite(I) + hopping(2I; range = 1), orbitals = (Val(1), Val(2)), type = T)
        hf = flatten(h0)
        hm = Matrix(h0)
        hs = (h0, hf, hm)
        As = (SparseMatrixCSC, SparseMatrixCSC, Matrix)
        Es = (SMatrix{2,2,T,4}, T, SMatrix{2,2,T,4})
        for (h, A, E) in zip(hs, As, Es)
            sh = similarmatrix(h)
            @test sh isa A{E}
            b1 = bloch!(similarmatrix(flatten(h)), flatten(h), (1,1))
            b2 = bloch!(similarmatrix(h, AbstractMatrix{T}), h, (1,1))
            @test isapprox(b1, b2)
            for T´ in types
                E´s = E <: SMatrix ? (SMatrix{2,2,T´,4}, T´) : (T´,)
                for E´ in E´s
                    s1 = similarmatrix(h, Matrix{E´})
                    s2 = similarmatrix(h, AbstractMatrix{E´})
                    @test s1 isa Matrix{E´}
                    @test s2 isa A{E´}
                end
            end
        end
    end
end

@testset "orbitals and sublats" begin
    orbs = (:a, (:a,), (:a, :b, 3), ((:a, :b), :c), ((:a, :b), (:c,)), (Val(2), Val(1)),
            (:A => (:a, :b), :D => :c), :D => Val(4))
    lat = LatticePresets.honeycomb()
    for orb in orbs
        @test hamiltonian(lat, onsite(I), orbitals = orb) isa Hamiltonian
    end
    @test hamiltonian(lat, onsite(I) + hopping(@SMatrix[1 2], sublats = :A =>:B),
                      orbitals = :A => Val(2)) isa Hamiltonian
    h1 = hamiltonian(lat, onsite(I) + hopping(@SMatrix[1 2], sublats = :A =>:B),
                      orbitals = :A => Val(2))
    h2 = hamiltonian(lat, onsite(I) + hopping(@SMatrix[1 2], sublats = (:A =>:B,)),
                      orbitals = :A => Val(2))
    @test bloch(h1, (1, 2)) == bloch(h2, (1, 2))
end

@testset "onsite dimensions" begin
    lat = LatticePresets.honeycomb()
    ts = (@SMatrix[1 2; 3 4], r -> @SMatrix[r[1] 2; 3 4], 2, r -> 2r[1],
          @SMatrix[1 2; 3 4], r -> @SMatrix[r[1] 2; 3 4])
    os = (Val(1), Val(1), Val(2), Val(2), Val(3), Val(3))
    for (t, o) in zip(ts, os)
        model = onsite(t)
        @test_throws DimensionMismatch hamiltonian(lat, model, orbitals = o)
    end

    ts = (@SMatrix[1 2; 3 4], r -> @SMatrix[r[1] 2; 3 4], 2, r -> 2r[1], 3I, r -> I, 3I)
    os = (Val(2), Val(2), Val(1), Val(1), Val(3), Val(3), (Val(1), Val(3)))
    for (t, o) in zip(ts, os)
        model = onsite(t)
        @test hamiltonian(lat, model, orbitals = o) isa Hamiltonian
    end
    @test bloch(hamiltonian(lat, onsite(3I), orbitals = (Val(1), Val(3))))[1,1] ==
        @SMatrix[3 0 0; 0 0 0; 0 0 0]
    @test bloch(hamiltonian(lat, onsite(3I), orbitals = (Val(1), Val(3))))[2,2] ==
        SMatrix{3,3}(3I)
end

@testset "hopping dimensions" begin
    lat = LatticePresets.honeycomb()
    ts = (@SMatrix[1 2; 2 3], (r,dr) -> @SMatrix[r[1] 2; 3 4], 2, (r,dr) -> 2r[1],
          @SMatrix[1 2], @SMatrix[1 ;2], @SMatrix[1 2], @SMatrix[1 2; 2 3])
    os = (Val(1), Val(1), Val(2), Val(2), (Val(2), Val(1)), (Val(2), Val(1)),
         (Val(2), Val(1)), (Val(2), Val(1)))
    ss = (missing, missing, missing, missing, :B => :A, :A => :B, missing, missing)
    for (t, o, s) in zip(ts, os, ss)
        model = hopping(t, sublats = s)
        @test_throws DimensionMismatch hamiltonian(lat, model, orbitals = o)
    end
    ts = (@SMatrix[1 2], @SMatrix[1 ;2])
    os = ((Val(2), Val(1)), (Val(2), Val(1)))
    ss = (:A => :B, :B => :A)
    for (t, o, s) in zip(ts, os, ss)
        model = hopping(t, sublats = s)
        @test hamiltonian(lat, model, orbitals = o) isa Hamiltonian
    end
    @test bloch(hamiltonian(lat, hopping(3I, range = 1/√3), orbitals = (Val(1), Val(2))))[2,1] ==
        @SMatrix[3 0; 0 0]
    @test bloch(hamiltonian(lat, hopping(3I, range = 1/√3), orbitals = (Val(1), Val(2))))[1,2] ==
        @SMatrix[3 0; 0 0]
end

@testset "hermiticity" begin
    lat = LatticePresets.honeycomb()
    @test !ishermitian(hamiltonian(lat, hopping(im, sublats = :A=>:B)))
    @test !ishermitian(hamiltonian(lat, hopping(1, sublats = :A=>:B)))
    @test !ishermitian(hamiltonian(lat, hopping(1, sublats = :A=>:B, dn = (-1,0))))
    @test ishermitian(hamiltonian(lat, hopping(1, sublats = :A=>:B, dn = (1,0))))
    @test !ishermitian(hamiltonian(lat, hopping(im)))
    @test ishermitian(hamiltonian(lat, hopping(1)))

    @test ishermitian(hamiltonian(lat, hopping(im, sublats = :A=>:B, plusadjoint = true)))
    @test ishermitian(hamiltonian(lat, hopping(1, sublats = :A=>:B, plusadjoint = true)))
    @test ishermitian(hamiltonian(lat, hopping(1, sublats = :A=>:B, dn = (1,0), plusadjoint = true)))
    @test ishermitian(hamiltonian(lat, hopping(im, plusadjoint = true)))
    @test ishermitian(hamiltonian(lat, hopping(1, plusadjoint = true)))
end

@testset "unitcell modifiers" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(0)) |> unitcell(2, modifiers = (@onsite!((o, r) -> 1), @hopping!(h -> 1)))
    @test diag(bloch(h)) == ComplexF64[1, 1, 1, 1, 1, 1, 1, 1]
end

@testset "@onsite!" begin
    el = @SMatrix[1 2; 2 1]

    @test @onsite!(o -> 2o)(el) == 2el
    @test @onsite!(o -> 2o)(el, p = 2) == 2el
    @test @onsite!((o;) -> 2o)(el) == 2el
    @test @onsite!((o;) -> 2o)(el, p = 2) == 2el
    @test @onsite!((o; z) -> 2o)(el, z = 2) == 2el
    @test @onsite!((o; z) -> 2o)(el, z = 2, p = 2) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, p = 2) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, z = 1, p = 2) == 2el
    @test @onsite!((o; kw...) -> 2o)(el) == 2el
    @test @onsite!((o; kw...) -> 2o)(el, p = 2) == 2el
    @test @onsite!((o; z, kw...) -> 2o)(el, z = 2) == 2el
    @test @onsite!((o; z, kw...) -> 2o)(el, z = 2, p = 2) == 2el
    @test @onsite!((o; z, y = 2, kw...) -> 2o)(el, z = 2, p = 2) == 2el
    @test @onsite!((o; z, y = 2, kw...) -> 2o)(el, z = 2, y = 3, p = 2) == 2el

    r = SVector(0,0)

    @test @onsite!((o, r;) -> 2o)(el, r) == 2el
    @test @onsite!((o, r;) -> 2o)(el, r, p = 2) == 2el
    @test @onsite!((o, r; z) -> 2o)(el, r, z = 2) == 2el
    @test @onsite!((o, r; z) -> 2o)(el, r, z = 2, p = 2) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, r) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, r, p = 2) == 2el
    @test @onsite!((o; z = 2) -> 2o)(el, r, z = 1, p = 2) == 2el
    @test @onsite!((o, r; kw...) -> 2o)(el, r) == 2el
    @test @onsite!((o, r; kw...) -> 2o)(el, r, p = 2) == 2el
    @test @onsite!((o, r; z, kw...) -> 2o)(el, r, z = 2) == 2el
    @test @onsite!((o, r; z, kw...) -> 2o)(el, r, z = 2, p = 2) == 2el
    @test @onsite!((o, r; z, y = 2, kw...) -> 2o)(el, r, z = 2, p = 2) == 2el
    @test @onsite!((o, r; z, y = 2, kw...) -> 2o)(el, r, z = 2, y = 3, p = 2) == 2el

    @test @onsite!((o; z, y = 2, kw...) -> 2o) isa Quantica.UniformOnsiteModifier
    @test @onsite!((o, r; z, y = 2, kw...) -> 2o) isa Quantica.OnsiteModifier{2}

    @test parameters(@onsite!((o, r; z, y = 2, kw...) -> 2o)) == (:z, :y)
end

@testset "@hopping!" begin
    el = @SMatrix[1 2; 2 1]

    @test @hopping!(t -> 2t)(el) == 2el
    @test @hopping!(t -> 2t)(el, p = 2) == 2el
    @test @hopping!((t;) -> 2t)(el) == 2el
    @test @hopping!((t;) -> 2t)(el, p = 2) == 2el
    @test @hopping!((t; z) -> 2t)(el, z = 2) == 2el
    @test @hopping!((t; z) -> 2t)(el, z = 2, p = 2) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, p = 2) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, z = 1, p = 2) == 2el
    @test @hopping!((t; kw...) -> 2t)(el) == 2el
    @test @hopping!((t; kw...) -> 2t)(el, p = 2) == 2el
    @test @hopping!((t; z, kw...) -> 2t)(el, z = 2) == 2el
    @test @hopping!((t; z, kw...) -> 2t)(el, z = 2, p = 2) == 2el
    @test @hopping!((t; z, y = 2, kw...) -> 2t)(el, z = 2, p = 2) == 2el
    @test @hopping!((t; z, y = 2, kw...) -> 2t)(el, z = 2, y = 3, p = 2) == 2el

    r = dr = SVector(0,0)

    @test @hopping!((t, r, dr;) -> 2t)(el, r, dr) == 2el
    @test @hopping!((t, r, dr;) -> 2t)(el, r, dr, p = 2) == 2el
    @test @hopping!((t, r, dr; z) -> 2t)(el, r, dr, z = 2) == 2el
    @test @hopping!((t, r, dr; z) -> 2t)(el, r, dr, z = 2, p = 2) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, r, dr) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, r, dr, p = 2) == 2el
    @test @hopping!((t; z = 2) -> 2t)(el, r, dr, z = 1, p = 2) == 2el
    @test @hopping!((t, r, dr; kw...) -> 2t)(el, r, dr) == 2el
    @test @hopping!((t, r, dr; kw...) -> 2t)(el, r, dr, p = 2) == 2el
    @test @hopping!((t, r, dr; z, kw...) -> 2t)(el, r, dr, z = 2) == 2el
    @test @hopping!((t, r, dr; z, kw...) -> 2t)(el, r, dr, z = 2, p = 2) == 2el
    @test @hopping!((t, r, dr; z, y = 2, kw...) -> 2t)(el, r, dr, z = 2, p = 2) == 2el
    @test @hopping!((t, r, dr; z, y = 2, kw...) -> 2t)(el, r, dr, z = 2, y = 3, p = 2) == 2el

    @test @hopping!((t; z, y = 2, kw...) -> 2t) isa Quantica.UniformHoppingModifier
    @test @hopping!((t, r, dr; z, y = 2, kw...) -> 2t) isa Quantica.HoppingModifier{3}

    @test parameters(@hopping!((o, r, dr; z, y = 2, kw...) -> 2o)) == (:z, :y)
end

@testset "parametric" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(2)) |> unitcell(10)
    T = typeof(h)
    @test parametric(h, @onsite!(o -> 2o))() isa T
    @test parametric(h, @onsite!((o, r) -> 2o))() isa T
    @test parametric(h, @onsite!((o, r; a = 2) -> a*o))() isa T
    @test parametric(h, @onsite!((o, r; a = 2) -> a*o))(a=1) isa T
    @test parametric(h, @onsite!((o, r; a = 2) -> a*o), @hopping!(t -> 2t))(a=1) isa T
    @test parametric(h, @onsite!((o, r) -> o), @hopping!((t, r, dr) -> r[1]*t))() isa T
    @test parametric(h, @onsite!((o, r) -> o), @hopping!((t, r, dr; a = 2) -> r[1]*t))() isa T
    @test parametric(h, @onsite!((o, r) -> o), @hopping!((t, r, dr; a = 2) -> r[1]*t))(a=1) isa T
    @test parametric(h, @onsite!((o, r; b) -> o), @hopping!((t, r, dr; a = 2) -> r[1]*t))(b=1) isa T
    @test parametric(h, @onsite!((o, r; b) -> o*b), @hopping!((t, r, dr; a = 2) -> r[1]*t))(a=1, b=2) isa T

    # Issue #35
    for orb in (Val(1), Val(2))
        h = LatticePresets.triangular() |> hamiltonian(hopping(I) + onsite(I), orbitals = orb) |> unitcell(10)
        ph = parametric(h, @onsite!((o, r; b) -> o+b*I), @hopping!((t, r, dr; a = 2) -> t+r[1]*I),
                       @onsite!((o, r; b) -> o-b*I), @hopping!((t, r, dr; a = 2) -> t-r[1]*I))
        @test isapprox(bloch(ph(a=1, b=2), (1, 2)), bloch(h, (1, 2)))
    end
    # Issue #37
    for orb in (Val(1), Val(2))
        h = LatticePresets.triangular() |> hamiltonian(hopping(I) + onsite(I), orbitals = orb) |> unitcell(10)
        ph = parametric(h, @onsite!(o -> o*cis(1)))
        @test ph()[1,1] ≈ h[1,1]*cis(1)
    end
end

@testset "boolean masks" begin
    for b in ((), (1,1), 4)
        h1 = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(2)) |>
             supercell(b, region = RegionPresets.circle(10))
        h2 = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(2)) |>
             supercell(b, region = RegionPresets.circle(20))

        @test isequal(h1 & h2, h1)
        @test isequal(h1, h2) || !isequal(h1 & h2, h2)
        @test isequal(h1, h2) || !isequal(h1 | h2, h1)
        @test  isequal(h1 | h2, h2)

        @test isequal(unitcell(h1 & h2), unitcell(h1))
        @test isequal(h1, h2) || !isequal(unitcell(h1 & h2), unitcell(h2))
        @test isequal(h1, h2) || !isequal(unitcell(h1 | h2), unitcell(h1))
        @test isequal(unitcell(h1 | h2), unitcell(h2))

        h1 = h1.lattice
        h2 = h2.lattice

        @test isequal(h1 & h2, h1)
        @test isequal(h1, h2) || !isequal(h1 & h2, h2)
        @test isequal(h1, h2) || !isequal(h1 | h2, h1)
        @test  isequal(h1 | h2, h2)

        @test isequal(unitcell(h1 & h2), unitcell(h1))
        @test isequal(h1, h2) || !isequal(unitcell(h1 & h2), unitcell(h2))
        @test isequal(h1, h2) || !isequal(unitcell(h1 | h2), unitcell(h1))
        @test isequal(unitcell(h1 | h2), unitcell(h2))
    end
end

@testset "unitcell seeds" begin
    p1 = SA[100,0]
    p2 = SA[0,20]
    lat = LatticePresets.honeycomb()
    model = hopping(1, range = 1/√3) + onsite(2)
    h1 = lat |> hamiltonian(model) |> supercell(region = r -> norm(r-p1)<3, seed = p1)
    h2 = lat |> hamiltonian(model) |> supercell(region = r -> norm(r-p2)<3, seed = p2)
    h = unitcell(h1 | h2)
    h3 = lat |> hamiltonian(model) |> unitcell(region = r -> norm(r-p1)<3 || norm(r-p2)<3, seed = p2)

    @test Quantica.nsites(h) == 130
    @test Quantica.nsites(h3) == 64
end