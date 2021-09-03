using LinearAlgebra: tr

@testset "greens schur1D selfenergy accuracy" begin
    h1 = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), region = r->abs(r[2])<3)
    h2 = LP.honeycomb() |> hamiltonian(hopping(1) + onsite(0.02, sublats = :A) - onsite(0.02, sublats = :B)) |> unitcell((4,4), region = r -> -5 < r[1] < 5)
    function res(g, ω)
        s = g.solver
        g0 = g(ω)
        L, R, GRL, GLR= g0.L, g0.R, g0.GRL, g0.GLR
        Σ = R * L' * GRL * R'
        resR = sum(abs, Σ - s.hm * ((ω*I - s.h0 - Σ) \ Matrix(s.hp)))
        Σ = L * R' * GLR * L'
        resL = sum(abs, Σ - s.hp * ((ω*I - s.h0 - Σ) \ Matrix(s.hm)))
        return resR + resL
    end
    g = greens(h1, Schur1D(), boundaries = (0,))
    for h in (h1, h2)
        g = greens(h, Schur1D(), boundaries = (0,))
        residual = maximum(res(g, ω + 1E-8im) for ω in range(-3.2, 3.2, length = 101))
        @test residual < Quantica.default_tol(residual)
    end
end

@testset "greens multiorbital" begin
    h = LP.honeycomb() |> hamiltonian(hopping(SA[0 1; 1 0], range = 4), orbitals = Val(2)) |> unitcell((2,-1), region = r->abs(r[2])<3) |> unitcell(2)
    g = greens(h, Schur1D(), boundaries = (0,))
    g0 = greens(flatten(h), Schur1D(), boundaries = (0,))
    o = orbitalstructure(h)
    # for n in -3:3, m in -3:3
    for n in 1:1, m in 1:1
        @test flatten(g(0.2, n=>m), o) ≈ g0(0.2, n=>m)
    end
end

@testset "greens schur1D positivity" begin
    # Broken until https://github.com/Reference-LAPACK/lapack/issues/477 comes to julia
    if VERSION >= v"1.7.0-DEV"
        h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1), region = r->abs(r[2])<3)
        g = greens(h, Schur1D())
        dos = [-imag(tr(g(w, 1=>1))) for w in range(-3.2, 3.2, length = 101)]
        @test all(x -> Quantica.chop(x) >= 0, dos)

        h = LP.honeycomb() |> hamiltonian(hopping(1, range = 2)) |> unitcell((1,-1), region = r->abs(r[2])<3)
        @test_throws ArgumentError greens(h, Schur1D())

        h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1),(2,2)) |> wrap(2)
        g = greens(h, Schur1D())
        dos = [-imag(tr(g(w, 1=>1))) for w in range(-3.2, 3.2, length = 101)]
        @test all(x -> Quantica.chop(x) >= 0, dos)

        h = LatticePresets.honeycomb() |> hamiltonian(hopping((r, dr) -> 1/(dr'*dr), range = 1)) |> unitcell((1,-1),(2,2)) |> wrap(2)
        g = greens(h, Schur1D())
        dos = [-imag(tr(g(w, 1=>1))) for w in range(-3.2, 3.2, length = 101)]
        @test all(x -> Quantica.chop(x) >= 0, dos)
    end

    h = LP.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((2,-2),(3,3)) |> unitcell(1, 1, indices = not(1)) |> wrap(2)
    g = greens(h, Schur1D())
    g = greens(h, Schur1D())
    @test Quantica.chop(imag(tr(g(0.2, 1=>1)))) <= 0
    dos = [-imag(tr(g(w, 1=>1))) for w in range(-3.2, 3.2, length = 101)]
    @test all(x -> Quantica.chop(x) >= 0, dos)
end

@testset "greens schur1D spatial positivity" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1)) |> unitcell((1,-1),(3,3)) |> unitcell((1,0))
    g = greens(h, Schur1D(), boundaries = (0,))
    g0 = g(0.01)
    ldos = [-imag(tr(g0[n=>n])) for n in 1:200]
    @test all(x -> Quantica.chop(x) >= 0, ldos)
end

@testset "greens fastpath equivalence" begin
    if VERSION >= v"1.7.0-DEV"
        h1 = LP.honeycomb() |> hamiltonian(hopping(SA[0 1; 1 0], range = 1), orbitals = Val(2)) |> unitcell((1,-1), region = r->abs(r[2])<2)
        h2 = LP.honeycomb() |> hamiltonian(hopping(1) + onsite(r->randn())) |> unitcell((2,-3), region = r->abs(r[2])<3)
        for h in (h1, h2)
            g = greens(h, Schur1D(), boundaries = (0,))
            ldos_1 = [g(ω)[n=>m] for ω in range(-0.1, 0.1, length = 5), n in -3:3, m in range(-3, 3, length = 7)]
            ldos_2 = [g(ω, n=>m) for ω in range(-0.1, 0.1, length = 5), n in -3:3, m in range(-3, 3, length = 7)]
            @test isapprox(ldos_1, ldos_2, atol = Quantica.default_tol(Quantica.blockeltype(h)))
        end
    end
end

@testset "greens partial solutions" begin
    g = LatticePresets.honeycomb() |> hamiltonian(hopping(-I)) |> unitcell((3,0), region = r->0<r[2]<5) |> greens(Schur1D())
    @test g(0.2, 2=>3; source = (1,6), dest = (1,6)) ≈ g(0.2, 2=>3)[[1,6],[1,6]]
    g = LatticePresets.honeycomb() |> hamiltonian(hopping(-I), orbitals = Val(2)) |> unitcell((3,0), region = r->0<r[2]<5) |> greens(Schur1D())
    @test_throws ArgumentError g(0.2, 1=>1; source = (1,6), dest = (1,6))
end
