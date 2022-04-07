module Quantica

const REPOISSUES = "https://github.com/pablosanjose/Quantica.jl/issues"

using Base.Threads: Iterators

using StaticArrays
using NearestNeighbors
using SparseArrays
using LinearAlgebra
using ProgressMeter
using Random
using SuiteSparse
using FunctionWrappers: FunctionWrapper
using ExprTools
using IntervalTrees
using SparseArrays: getcolptr, AbstractSparseMatrix, AbstractSparseMatrixCSC
using Statistics: mean

export sublat, lattice, supercell, bravais_matrix,
       hopping, onsite, @onsite!, @hopping!, neighbors,
       hamiltonian, parametric, bloch,
       flat, unflat, wrap, transform, transform!, translate, translate!,
       band, mesh, subbands, slice,
       coupler

# export sublat, lattice, dims, supercell, bravais_matrix, siteindices, sitepositions,
#        hopping, onsite, @onsite!, @hopping!, @block!, parameters, neighbors,
#        ket, ketmodel, randomkets, basiskets,
#        hamiltonian, parametric, bloch, bloch!, similarmatrix,
#        flatten, unflatten, orbitalstructure, wrap, transform, translate, combine,
#        spectrum, band, mesh, isometric, subbands,
#        vertices, minima, maxima, gapedge, gap, isinband,
#        energies, states, degeneracy,
#        momentaKPM, dosKPM, averageKPM, densityKPM, bandrangeKPM,
#        greens, greensolver, Schur1D

export LatticePresets, LP, RegionPresets, RP, EigensolverPresets, EP #, HamiltonianPresets, HP
export @SMatrix, @SVector, SMatrix, SVector, SA
export ishermitian, I

include("iterators.jl")
include("builders.jl")
include("sparsetools.jl")
include("types.jl")
include("selector.jl")
include("lattice.jl")
include("model.jl")
include("hamiltonian.jl")
include("supercell.jl")
include("transform.jl")
include("apply.jl")
# include("ket.jl")
include("mesh.jl")
include("band.jl")
include("coupler.jl")
# include("KPM.jl")
# include("effective.jl")
# include("greens.jl")
include("sanitize.jl")
include("show.jl")
include("tools.jl")
include("convert.jl")

include("presets/lattices.jl")
include("presets/regions.jl")
include("presets/eigensolvers.jl")
# include("presets/hamiltonians.jl")

include("precompile.jl")

end