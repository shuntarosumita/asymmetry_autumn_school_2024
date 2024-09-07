mutable struct Mesh
    prmt::Parameters

    hk::Array{ComplexF64, 5}
    ek0::Array{Float64, 4}
    ek::Array{Float64, 4}
    uk::Array{ComplexF64, 5}
    μ::Float64
    emin::Float64
    emax::Float64
    W::Float64
    U_mat::Matrix{Float64}

    ω_f::Vector{Float64}
    ω_b::Vector{Float64}
    iω0_f::Int64
    iω0_b::Int64
end

"ハミルトニアン・松原振動数のメッシュを設定する関数"
function Mesh(p::Parameters)::Mesh
    # ハミルトニアンとその固有値・固有ベクトルを計算
    hk::Array{ComplexF64, 5} = set_hamiltonian_matrix(p)
    ek0 = Array{Float64, 4}(undef, p.nk1, p.nk2, p.nk3, p.nwan)
    uk = zeros(ComplexF64, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        tmpe, tmpu = eigen(@view(hk[ik1, ik2, ik3, :, :]))
        ek0[ik1, ik2, ik3, :] .= real.(tmpe)
        uk[ik1, ik2, ik3, :, :] .= tmpu
    end

    μ::Float64 = set_μ(p, ek0) # 化学ポテンシャル
    ek::Array{Float64, 4} = ek0 .- μ
    emin::Float64 = minimum(ek0)
    emax::Float64 = maximum(ek0)
    W::Float64 = emax - emin

    U_mat::Matrix{Float64} = set_interaction(p) # 相互作用を表す行列

    # 松原振動数(フェルミオンとボソン)
    ω_f::Vector{Float64} = [
        (2iω+1) * π * p.T for iω in (-p.nω÷2):(p.nω÷2-1)
    ]
    ω_b::Vector{Float64} = [
        2iω * π * p.T for iω in (-p.nω÷2):(p.nω÷2-1)
    ]
    iω0_f = iω0_b = p.nω÷2+1

    return Mesh(
        p, hk, ek0, ek, uk, μ, emin, emax, W, U_mat, ω_f, ω_b, iω0_f, iω0_b
    )
end

"一体ハミルトニアンの定義"
function set_hamiltonian_matrix(p::Parameters)
    # 遷移積分
    t1::Float64 = 0.0
    t2::Float64 = 0.0
    if p.system == "square"
        t1 = 1.0
    elseif p.system == "modified_square"
        t1 = 1.0
        t2 = 0.15
    end

    # systemの値に応じてハミルトニアンを定義(スピン軌道相互作用は無視し、遷移積分のみ)
    hk = Array{ComplexF64, 5}(undef, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
        k1::Float64 = (2π*(ik1-1)) / p.nk1
        k2::Float64 = (2π*(ik2-1)) / p.nk2

        if p.system == "square"
            hk[ik1, ik2, ik3, :, :] .= - 2t1 * (cos(k1) + cos(k2))
        elseif p.system == "modified_square"
            ### 畳み込みを計算するためにFFTを利用するので、ブリルアンゾーンの周期性を満たすハミルトニアンを定義する
            f1::ComplexF64 = (1 + cis(k1)) * (1 + cis(-k2))
            hk[ik1, ik2, ik3, :, :] .= (
                - t1 .* (real(f1) .* σ1 .- imag(f1) .* σ2)
                .- 2t2 * cos(k1) * cos(k2) .* σ0
                .- 2t2 * sin(k1) * sin(k2) .* σ3
            )
        end
    end

    hk
end

"化学ポテンシャルの計算"
function set_μ(p::Parameters, ek::Array{Float64, 4})
    # 軌道・副格子あたりの電子数
    n_0::Float64 = p.n_fill

    # 与えられた電子数密度になるようにBrent法を用いて化学ポテンシャルを決定する
    res = optimize(
        μ -> (2/p.nspin * calc_electron_density(p, ek, μ) - n_0)^2,
        3*minimum(ek), 3*maximum(ek), rel_tol=1e-4, Brent()
    )
    Optim.minimizer(res)[1]
end

"電子数密度の計算"
function calc_electron_density(p::Parameters, ek::Array{Float64, 4}, μ::Float64)
    E = fill(one(Float64), size(ek)...)
    sum(E ./ (E .+ exp.(p.β .* (ek .- μ)))) / (p.nk1 * p.nk2 * p.nk3 * p.norb)
end

"相互作用行列の計算"
function set_interaction(p::Parameters)
    # オンサイトハバード相互作用の計算
    ## スピン自由度の有無(スピン軌道結合を入れるか入れないか)によって表式が異なる
    U_mat::Matrix{Float64} = zeros(Float64, p.nwan^2, p.nwan^2)
    if p.nspin == 1
        for ζ in 1:p.norb
            # udud, dudu
            ξ12::Int = p.nwan*(ζ-1) + ζ
            ξ34::Int = p.nwan*(ζ-1) + ζ
            U_mat[ξ12, ξ34] += p.U
        end
    elseif p.nspin == 2
        for ζ in 1:p.norb, s in 1:p.nspin
            sbar::Int = 3 - s

            # udud, dudu
            ξ12::Int = p.nwan*((p.nspin*(ζ-1) + s)-1) + p.nspin*(ζ-1) + sbar
            ξ34::Int = p.nwan*((p.nspin*(ζ-1) + s)-1) + p.nspin*(ζ-1) + sbar
            U_mat[ξ12, ξ34] += p.U

            # uudd, dduu
            ξ12 = p.nwan*((p.nspin*(ζ-1) + s)-1) + p.nspin*(ζ-1) + s
            ξ34 = p.nwan*((p.nspin*(ζ-1) + sbar)-1) + p.nspin*(ζ-1) + sbar
            U_mat[ξ12, ξ34] -= p.U
        end
    end

    U_mat
end

# 高速フーリエ変換
"τからiωへのフーリエ変換"
function τ_to_ωn(p::Parameters, isboson::Bool, obj_τ::Array{ComplexF64, n}) where n
    phase::Vector{ComplexF64} = (
        isboson ?
        [(-1)^(iτ-1) for iτ in 1:p.nω] : # ボソン
        [(-1)^(iτ-1) * cispi((iτ-1)/p.nω) for iτ in 1:p.nω] # フェルミオン
    )

    obj_ωn::Array{ComplexF64, n} = p.β .* ifft(phase .* obj_τ, [1])
    return obj_ωn
end

"iωからτへのフーリエ変換"
function ωn_to_τ(p::Parameters, isboson::Bool, obj_ωn::Array{ComplexF64, n}) where n
    phase::Vector{ComplexF64} = (
        isboson ?
        [(-1)^(iτ-1) for iτ in 1:p.nω] : # ボソン
        [(-1)^(iτ-1) * cispi(-(iτ-1)/p.nω) for iτ in 1:p.nω] # フェルミオン
    )

    obj_τ::Array{ComplexF64, n} = p.T .* phase .* fft(obj_ωn, [1])
    return obj_τ
end

"波数空間から実空間へのフーリエ変換"
function k_to_r(p::Parameters, obj_k::Array{ComplexF64, n}) where n
    obj_r::Array{ComplexF64, n} = ifft(obj_k, [2, 3, 4])
    return obj_r
end

"実空間から波数空間へのフーリエ変換"
function r_to_k(p::Parameters, obj_r::Array{ComplexF64, n}) where n
    obj_k::Array{ComplexF64, n} = fft(obj_r, [2, 3, 4])
    return obj_k
end
