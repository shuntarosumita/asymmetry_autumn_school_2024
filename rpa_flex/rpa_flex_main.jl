using LinearAlgebra
using FFTW
using Optim
using Plots
using LaTeXStrings

const σ0 = ComplexF64[1 0; 0 1]
const σ1 = ComplexF64[0 1; 1 0]
const σ2 = ComplexF64[0 -1im; 1im 0]
const σ3 = ComplexF64[1 0; 0 -1]

include("parameters.jl")
include("mesh.jl")
include("gfunction.jl")

function plot_χ_map(χk::AbstractArray)
    nk1, nk2 = size(χk)[1:2]
    k1s = [(2π*(ik1-1)) / nk1 - π for ik1 in 1:nk1]
    k2s = [(2π*(ik2-1)) / nk2 - π for ik2 in 1:nk2]

    # heatmap関数に渡す場合、データの順番に注意
    χ_map = fftshift(
        [χk[ik1, ik2, 1] for ik2 in 1:nk2, ik1 in 1:nk1], (1, 2)
    )

    # プロット
    plt = heatmap(
        k1s, k2s, χ_map, aspect_ratio = :equal,
        xlims = (-π, π), ylims = (-π, π),
        xticks=([-π, π], [L"-\pi", L"\pi"]), yticks=([-π, π], [L"-\pi", L"\pi"]),
        xlabel=L"q_x", ylabel=L"q_y",
        size=(650, 500), guidefontsize=20, tickfontsize=16
    )

    plt
end

println("##### square lattice #####")
let
    system::String = "square"
    n_fill::Float64 = 1.0
    U::Float64 = 2.0
    T::Float64 = 0.22
    mode::String = "RPA"
    nk1::Int64 = 64
    nk2::Int64 = 64
    nk3::Int64 = 1
    nω::Int64 = 1024
    op = [1/sqrt(2);;]

    p = Parameters(system, n_fill, U, T, mode; nk1, nk2, nk3, nω)
    m = Mesh(p)
    g = Gfunction(m)

    println("##################################################")
    println("Parameter set: n = $n_fill, U = $U, T = $T")
    println("emin = $(m.emin), emax = $(m.emax); μ = $(m.μ)")

    # RPA
    calc_Gfunction!(m, g)
    χs_RPA = calc_multipole_susceptibility(m, g, op)

    # FLEX
    p.mode = "FLEX"
    g = Gfunction(m)

    calc_Gfunction!(m, g)
    χs_FLEX = calc_multipole_susceptibility(m, g, op)

    println("##################################################")

    # 感受率のカラーマップを出力
    isdir("images") || mkdir("images")
    plt = plot_χ_map(real.(χs_RPA[m.iω0_b, :, :, 1]))
    savefig(plt, "images/susceptibility_$(system)_RPA.png")
    plt = plot_χ_map(real.(χs_FLEX[m.iω0_b, :, :, 1]))
    savefig(plt, "images/susceptibility_$(system)_FLEX.png")
end
println()

println("##### modified square lattice #####")
let
    system::String = "modified_square"
    n_fill::Float64 = 1.0
    U::Float64 = 2.0
    T::Float64 = 0.15
    mode::String = "RPA"
    nk1::Int64 = 64
    nk2::Int64 = 64
    nk3::Int64 = 1
    nω::Int64 = 1024
    op = σ3 ./ sqrt(2)

    p = Parameters(system, n_fill, U, T, mode; nk1, nk2, nk3, nω)
    m = Mesh(p)
    g = Gfunction(m)

    println("##################################################")
    println("Parameter set: n = $n_fill, U = $U, T = $T")
    println("emin = $(m.emin), emax = $(m.emax); μ = $(m.μ)")

    # RPA
    calc_Gfunction!(m, g)
    χs_RPA = calc_multipole_susceptibility(m, g, op)

    # FLEX
    p.mode = "FLEX"
    g = Gfunction(m)

    calc_Gfunction!(m, g)
    χs_FLEX = calc_multipole_susceptibility(m, g, op)

    println("##################################################")

    # 感受率のカラーマップを出力
    plt = plot_χ_map(real.(χs_RPA[m.iω0_b, :, :, 1]))
    savefig(plt, "images/susceptibility_$(system)_RPA.png")
    plt = plot_χ_map(real.(χs_FLEX[m.iω0_b, :, :, 1]))
    savefig(plt, "images/susceptibility_$(system)_FLEX.png")
end
