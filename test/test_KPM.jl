using ArnoldiMethod, Random, FFTW

@testset "basic KPM" begin
    h = LatticePresets.honeycomb() |>
        hamiltonian(hopping(1.0I, range = 1/sqrt(3)), orbitals = (Val(1), Val(2))) |>
        unitcell(region = RegionPresets.circle(10))
    brange = bandrangeKPM(h)
    m1 = momentaKPM(h, order = 30, ket = randomkets(1, maporbitals = true))
    m2 = momentaKPM(h, bandrange = brange, order = 30, ket = randomkets(1, maporbitals = true))
    @test abs(m1.mulist[1]) ≈ abs(m2.mulist[1])
    dos1 = dosKPM(h, order = 10, bandrange = (-3,3), ket = randomkets(1, maporbitals = true))
    dos2 = dosKPM(h, order = 10, bandrange = (-3,3), ket = randomkets(1, r -> randn(), maporbitals = true))
    @test issorted(first(dos1))
    @test issorted(first(dos2))
    @test all(>(0), last(dos1))
    @test all(>(0), last(dos2))
    dos = dosKPM(h, order = 10, bandrange = (-3,3), ket = randomkets(2, r -> randn() * SA[1 0; 0 -1], sublats = :B))
    @test all(>(0), last(dos))
    x, energy = densityKPM(h, h, order = 100, resolution = 1, bandrange = (-3,3), ket = ketmodel(1, indices = 1, maporbitals = true))
    @test isapprox(abs(sum(energy)), 0, atol = Quantica.default_tol(eltype(energy)))
    k = ket(ketmodel(1, indices = 1, maporbitals = true), h)
    x´, energy´ = densityKPM(h, h, order = 100, resolution = 1, bandrange = (-3,3), ket = k)
    @test x´ ≈ x
    @test energy´ ≈ energy
    x´, energy´ = densityKPM(h, h[], order = 100, resolution = 1, bandrange = (-3,3), ket = k)
    @test x´ ≈ x
    @test energy´ ≈ energy
end
