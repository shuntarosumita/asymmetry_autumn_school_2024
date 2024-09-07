mutable struct Gfunction
    U_pval::Float64
    μ::Float64

    g0iωk::Array{ComplexF64, 6}
    giωk::Array{ComplexF64, 6}
    g0iτk::Array{ComplexF64, 6}
    giτr::Array{ComplexF64, 6}
    χ0iβ0::Array{ComplexF64, 2}
    χ0iωk::Array{ComplexF64, 6}
    χiωk::Array{ComplexF64, 6}
    Viτ_dum::Array{ComplexF64, 3}
    Viω_dum::Array{ComplexF64, 3}
    Σiωk::Array{ComplexF64, 6}
end

"グリーン関数の初期化"
function Gfunction(m::Mesh)::Gfunction
    p::Parameters = m.prmt
    μ::Float64 = m.μ # 相互作用なしの場合に求めた化学ポテンシャル

    # 行列の確保
    g0iωk = Array{ComplexF64, 6}(undef, p.nω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    giωk = Array{ComplexF64, 6}(undef, p.nω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    g0iτk = Array{ComplexF64, 6}(undef, p.nω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    giτr = Array{ComplexF64, 6}(undef, p.nω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    χ0iβ0 = Array{ComplexF64, 2}(undef, p.nwan^2, p.nwan^2)
    χ0iωk = Array{ComplexF64, 6}(undef, p.nω, p.nk1, p.nk2, p.nk3, p.nwan^2, p.nwan^2)
    χiωk = Array{ComplexF64, 6}(undef, p.nω, p.nk1, p.nk2, p.nk3, p.nwan^2, p.nwan^2)
    Viτ_dum = Array{ComplexF64, 3}(undef, p.nω, p.nwan^2, p.nwan^2)
    Viω_dum = Array{ComplexF64, 3}(undef, p.nω, p.nwan^2, p.nwan^2)
    Σiωk::Array{ComplexF64, 6} = zeros(ComplexF64, p.nω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)

    g = Gfunction(p.U, μ, g0iωk, giωk, g0iτk, giτr, χ0iβ0, χ0iωk, χiωk, Viτ_dum, Viω_dum, Σiωk)

    return g
end

function calc_Gfunction!(m::Mesh, g::Gfunction)
    # G_0, G, χ_0, χ の計算
    set_g0iτk!(m, g, g.μ)
    set_giωk!(m, g, g.μ)
    set_giτr!(m, g)
    set_χ0iωk!(m, g)
    set_χiωk!(m, g)

    if m.prmt.mode == "FLEX"
        FLEXcheck = solve_FLEX!(m, g)
        FLEXcheck === missing && println("FLEX not converged!")
    end
end

"フェルミ分布関数"
function fermi(E::Real, T::Real)
    # expを使うよりtanhを使う方が安定する
    0.5 * (1 - tanh(E / (2T)))
end

"裸のグリーン関数G_0(τ, k)"
function set_g0iτk!(m::Mesh, g::Gfunction, μ::Float64)
    p::Parameters = m.prmt

    # G_0(τ, k)をハミルトニアンの固有値と固有ベクトルを用いて求める
    g.g0iτk .= zeros(ComplexF64, p.nω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1, iτ in 1:p.nω
        τ::Float64 = p.β * (iτ - 1) / p.nω

        for α in 1:p.nwan
            ek::Float64 = m.ek0[ik1, ik2, ik3, α] - μ
            tmp::Float64 = (
                ek > 0 ?
                - fermi(-ek, p.T) * exp(-ek * τ) :
                - fermi( ek, p.T) * exp( ek * (p.β - τ))
            )

            for ξ2 in 1:p.nwan, ξ1 in 1:p.nwan
                g.g0iτk[iτ, ik1, ik2, ik3, ξ1, ξ2] += (
                    tmp * m.uk[ik1, ik2, ik3, ξ1, α] * conj(m.uk[ik1, ik2, ik3, ξ2, α])
                )
            end
        end
    end

    g.g0iτk
end

"裸のグリーン関数G_0(iϵ, k)と自己エネルギーの入ったグリーン関数G(iϵ, k)"
function set_giωk!(m::Mesh, g::Gfunction, μ::Float64)
    id = Matrix{ComplexF64}(I, m.prmt.nwan, m.prmt.nwan) # 単位行列

    # 裸のグリーン関数
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.prmt.nω
        g.g0iωk[iω, ik1, ik2, ik3, :, :] .= @views(
            (1im*m.ω_f[iω] + μ) .* id .- m.hk[ik1, ik2, ik3, :, :]
        ) \ I
    end

    # 自己エネルギーを考慮したグリーン関数
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.prmt.nω
        g.giωk[iω, ik1, ik2, ik3, :, :] .= @views(
            (1im*m.ω_f[iω] + μ) .* id .- m.hk[ik1, ik2, ik3, :, :]
            .- g.Σiωk[iω, ik1, ik2, ik3, :, :]
        ) \ I
    end

    g.g0iωk, g.giωk
end

"グリーン関数の対称化"
function symmetrize_giωk!(g::Gfunction)
    # G(k, iω) = G(-k, iω)^T
    giωk_revk::Array{ComplexF64, 6} = permutedims(
        reverse(
            circshift(g.giωk, (0, -1, -1, -1, 0, 0)),
            dims=(2, 3, 4)
        ),
        (1, 2, 3, 4, 6, 5)
    )
    g.giωk .= (g.giωk .+ giωk_revk) ./ 2.0
end

"グリーン関数G(τ, r)"
function set_giτr!(m::Mesh, g::Gfunction)
    # FT: G(iϵ, k) -> G(τ, k)
    ## (iϵ)^{-1}の寄与を避けるため、G_0を引いてフーリエ変換して後で戻す
    giτk::Array{ComplexF64, 6} = ωn_to_τ(m.prmt, false, g.giωk .- g.g0iωk) .+ g.g0iτk

    # FT: G(τ, k) -> G(τ, r)
    g.giτr .= k_to_r(m.prmt, giτk)
end

"既約感受率χ_0(iω, q)"
function set_χ0iωk!(m::Mesh, g::Gfunction)
    p::Parameters = m.prmt

    # -G(-τ, -r) = G(β-τ, -r)
    giτr_rev::Array{ComplexF64, 6} = reverse(
        circshift(g.giτr, (-1, -1, -1, -1, 0, 0)),
        dims=(1, 2, 3, 4)
    )

    ## -G(τ=-0, -r)_{1, 2} = -(δ_{r, 0} δ_{1, 2} + G(+0, -r)_{1, 2})
    for iξ in 1:p.nwan
        giτr_rev[1, 1, 1, 1, iξ, iξ] += 1
    end
    giτr_rev[1, :, :, :, :, :] .*= -1

    # 2つのグリーン関数の積
    ## χ_0(τ, r) = - G(τ, r) G(-τ, -r)
    χ0iτr = zeros(ComplexF64, p.nω, p.nk1, p.nk2, p.nk3, p.nwan^2, p.nwan^2)
    for ξ4 in 1:p.nwan, ξ3 in 1:p.nwan, ξ2 in 1:p.nwan, ξ1 in 1:p.nwan
        ξ12::Int64 = p.nwan * (ξ1-1) + ξ2
        ξ34::Int64 = p.nwan * (ξ3-1) + ξ4
        χ0iτr[:, :, :, :, ξ12, ξ34] .+= @views(
            g.giτr[:, :, :, :, ξ1, ξ3] .* giτr_rev[:, :, :, :, ξ4, ξ2]
        )
    end

    ## χ0(τ=β-0, r=0)_{12, 34} = χ0(-0, 0)_{12, 34} = - (δ_{1, 3} + G(+0, 0)_{1, 3}) G(+0, 0)_{4, 2}
    g.χ0iβ0 .= zeros(ComplexF64, p.nwan^2, p.nwan^2)
    for ξ4 in 1:p.nwan, ξ3 in 1:p.nwan, ξ2 in 1:p.nwan, ξ1 in 1:p.nwan
        ξ12::Int64 = p.nwan * (ξ1-1) + ξ2
        ξ34::Int64 = p.nwan * (ξ3-1) + ξ4
        g.χ0iβ0[ξ12, ξ34] += giτr_rev[1, 1, 1, 1, ξ1, ξ3] * g.giτr[1, 1, 1, 1, ξ4, ξ2]
    end

    # V_dum(τ)
    for iτ in 1:p.nω
        g.Viτ_dum[iτ, :, :] .= (1/p.nspin) .* m.U_mat * @view(χ0iτr[iτ, 1, 1, 1, :, :]) * m.U_mat
    end

    # FFTにかける量として、(iω)^{-1}の寄与を避けたものを定義しておく
    χ0iτr_FFT = similar(χ0iτr)
    for iτ in 1:p.nω
        χ0iτr_FFT[iτ, :, :, :, :, :] .= @views(
            2 .* χ0iτr[iτ, :, :, :, :, :]
            .- χ0iτr[mod(iτ+1, 1:p.nω), :, :, :, :, :]
            .- χ0iτr[mod(iτ-1, 1:p.nω), :, :, :, :, :]
        )
    end
    χ0iτr_FFT[1, 1, 1, 1, :, :]    .+= g.χ0iβ0 .- @view(χ0iτr[1, 1, 1, 1, :, :])
    χ0iτr_FFT[p.nω, 1, 1, 1, :, :] .+= @view(χ0iτr[1, 1, 1, 1, :, :]) .- g.χ0iβ0

    # FT: χ_0(τ, r) -> χ_0(iω, r)
    χ0iωr::Array{ComplexF64, 6} = τ_to_ωn(p, true, χ0iτr_FFT)
    for iω in 1:p.nω
        if iω != m.iω0_b
            ### FFTで得られたのは(iω)^{-2}の寄与
            χ0iωr[iω, :, :, :, :, :] .*= (p.nω * p.T / m.ω_b[iω])^2

            ### r = 0のとき、(iω)^{-1}の寄与を加える
            χ0iωr[iω, 1, 1, 1, :, :] .+= (g.χ0iβ0 .- @view(χ0iτr[1, 1, 1, 1, :, :])) ./ (1im*m.ω_b[iω])
        else
            ### ボソンの場合、ゼロ振動数は特別に扱う必要がある
            χ0iωr[iω, :, :, :, :, :] .= zero(ComplexF64)
            for iτ in 1:p.nω
                χ0iωr[iω, :, :, :, :, :] .+= (p.β / p.nω) .* χ0iτr[iτ, :, :, :, :, :]
            end
            χ0iωr[iω, 1, 1, 1, :, :] .+= (p.β / 2p.nω) .* (g.χ0iβ0 .- @view(χ0iτr[1, 1, 1, 1, :, :]))
        end
    end

    # V_dum(iω)
    for iω in 1:p.nω
        g.Viω_dum[iω, :, :] .= (1/p.nspin) .* m.U_mat * @view(χ0iωr[iω, 1, 1, 1, :, :]) * m.U_mat
    end

    # FT: χ_0(iω, r) -> χ_0(iω, q)
    g.χ0iωk .= r_to_k(p, χ0iωr)
end

"一般化感受率χ(iω, q)"
function set_χiωk!(m::Mesh, g::Gfunction)
    # RPA (FLEX近似) での感受率
    ## χ = [1 - χ_0 U]^(-1) χ_0
    id = Matrix{ComplexF64}(I, m.prmt.nwan^2, m.prmt.nwan^2) # 単位行列
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.prmt.nω
        denom::Matrix{ComplexF64} = copy(id)
        @views mul!(denom, g.χ0iωk[iω, ik1, ik2, ik3, :, :], m.U_mat, -1, 1)
        g.χiωk[iω, ik1, ik2, ik3, :, :] .= denom \ @view(g.χ0iωk[iω, ik1, ik2, ik3, :, :])
    end

    g.χiωk
end

"自己エネルギーを計算するための相互作用バーテックスV_n(τ, r)"
function calc_Viτr(m::Mesh, g::Gfunction)
    p::Parameters = m.prmt

    # V_n(iω, q)の計算
    ## スピン自由度の有無(スピン軌道結合を入れるか入れないか)によって表式が異なる
    Viωk = Array{ComplexF64, 6}(undef, p.nω, p.nk1, p.nk2, p.nk3, p.nwan^2, p.nwan^2)
    if p.nspin == 2
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1, iω in 1:p.nω
            Viωk[iω, ik1, ik2, ik3, :, :] .= @views(
                m.U_mat * (g.χiωk[iω, ik1, ik2, ik3, :, :] .- 0.5.*g.χ0iωk[iω, ik1, ik2, ik3, :, :]) * m.U_mat
            )
        end

    elseif p.nspin == 1
        id = Matrix{ComplexF64}(I, p.nwan^2, p.nwan^2)
        denom = copy(id)
        χciωk = Array{ComplexF64, 6}(undef, p.nω, p.nk1, p.nk2, p.nk3, p.nwan^2, p.nwan^2)
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1, iω in 1:p.nω
            denom .= id
            @views mul!(denom, g.χ0iωk[iω, ik1, ik2, ik3, :, :], m.U_mat, 1, 1)
            @views χciωk[iω, ik1, ik2, ik3, :, :] .= denom \ g.χ0iωk[iω, ik1, ik2, ik3, :, :]
        end
        for ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1, iω in 1:p.nω
            @views Viωk[iω, ik1, ik2, ik3, :, :] .= (
                1.5 .* (m.U_mat * (g.χiωk[iω, ik1, ik2, ik3, :, :] .- 0.5.*g.χ0iωk[iω, ik1, ik2, ik3, :, :]) * m.U_mat) .+
                0.5 .* (m.U_mat * (χciωk[iω, ik1, ik2, ik3, :, :] .- 0.5.*g.χ0iωk[iω, ik1, ik2, ik3, :, :]) * m.U_mat)
            )
        end
    end

    # FT: V_n(iω, q) -> V_n(iω, r)
    Viωr::Array{ComplexF64, 6} = k_to_r(p, Viωk)

    ## r = 0のみ特殊な取り扱いが必要
    Viωr[:, 1, 1, 1, :, :] .-= g.Viω_dum

    # FT: V_n(iω, r) -> V_n(τ, r)
    Viτr::Array{ComplexF64, 6} = ωn_to_τ(p, true, Viωr)

    ## V_n(τ=β-0, r=0)
    Viβ0::Array{ComplexF64, 2} = @view(Viτr[1, 1, 1, 1, :, :]) .+ (1/p.nspin) .* (m.U_mat * g.χ0iβ0 * m.U_mat)

    ## r = 0のみ特殊な取り扱いが必要
    Viτr[:, 1, 1, 1, :, :] .+= g.Viτ_dum

    Viτr, Viβ0
end

"自己エネルギーΣ(iϵ, k)"
function set_Σiωk!(m::Mesh, g::Gfunction)
    p::Parameters = m.prmt

    # 相互作用バーテックスV_nを得る
    Viτr::Array{ComplexF64, 6}, Viβ0::Array{ComplexF64, 2} = calc_Viτr(m, g)

    # 2つのグリーン関数の積
    ## Σ(τ, r) = V(τ, r) G(τ, r)
    Σiτr = zeros(ComplexF64, p.nω, p.nk1, p.nk2, p.nk3, p.nwan, p.nwan)
    for ξ4 in 1:p.nwan, ξ3 in 1:p.nwan, ξ2 in 1:p.nwan, ξ1 in 1:p.nwan
        ξ12::Int64 = p.nwan * (ξ1-1) + ξ2
        ξ34::Int64 = p.nwan * (ξ3-1) + ξ4
        Σiτr[:, :, :, :, ξ1, ξ3] .+= @views(
            Viτr[:, :, :, :, ξ12, ξ34] .* g.giτr[:, :, :, :, ξ2, ξ4]
        )
    end

    ## Σ(τ=β-0, r=0)_{1, 3} = - Σ(-0, 0)_{1, 3} = - V_n(β-0, 0)_{12, 34} (δ_{2, 4} + G(+0, 0)_{2, 4})
    Σiβ0 = zeros(ComplexF64, p.nwan, p.nwan)
    for ξ4 in 1:p.nwan, ξ3 in 1:p.nwan, ξ2 in 1:p.nwan, ξ1 in 1:p.nwan
        ξ12::Int64 = p.nwan * (ξ1-1) + ξ2
        ξ34::Int64 = p.nwan * (ξ3-1) + ξ4
        Σiβ0[ξ1, ξ3] += - Viβ0[ξ12, ξ34] * (
            ifelse(ξ2 == ξ4, 1, 0) + g.giτr[1, 1, 1, 1, ξ2, ξ4]
        )
    end

    # FFTにかける量として、(iϵ)^{-1}の寄与を避けたものを定義しておく
    Σiτr_FFT = similar(Σiτr)

    ## τ = 0
    Σiτr_FFT[1, :, :, :, :, :] .= @views(
        2 .* Σiτr[1, :, :, :, :, :]
        .- Σiτr[2, :, :, :, :, :]
        .+ Σiτr[p.nω, :, :, :, :, :] # 符号に注意
    )
    Σiτr_FFT[1, 1, 1, 1, :, :] .+= - Σiβ0 .- @view(Σiτr[1, 1, 1, 1, :, :])

    ## τ ≠ 0, β-Δτ
    for iτ in 2:p.nω-1
        Σiτr_FFT[iτ, :, :, :, :, :] .= @views(
            2 .* Σiτr[iτ, :, :, :, :, :]
            .- Σiτr[iτ+1, :, :, :, :, :]
            .- Σiτr[iτ-1, :, :, :, :, :]
        )
    end

    ## τ = β-Δτ
    Σiτr_FFT[p.nω, :, :, :, :, :] .= @views(
        2 .* Σiτr[p.nω, :, :, :, :, :]
        .+ Σiτr[1, :, :, :, :, :] # 符号に注意
        .- Σiτr[p.nω-1, :, :, :, :, :]
    )
    Σiτr_FFT[p.nω, 1, 1, 1, :, :] .+= - Σiβ0 .- @view(Σiτr[1, 1, 1, 1, :, :])

    # FT: Σ(τ, r) -> Σ(iϵ, r)
    Σiωr::Array{ComplexF64, 6} = τ_to_ωn(p, false, Σiτr_FFT)
    for iω in 1:p.nω
        ### FFTで得られたのは(iϵ)^{-2}の寄与
        Σiωr[iω, :, :, :, :, :] .*= (p.nω * p.T / m.ω_f[iω])^2

        ### r = 0のとき、(iϵ)^{-1}の寄与を加える
        Σiωr[iω, 1, 1, 1, :, :] .+= (- Σiβ0 .- @view(Σiτr[1, 1, 1, 1, :, :])) ./ (1im*m.ω_f[iω])
    end

    # FT: Σ(iϵ, r) -> Σ(iϵ, k)
    g.Σiωk .= r_to_k(p, Σiωr)

    g.Σiωk
end

"グリーン関数を用いた化学ポテンシャルの計算"
function set_μ_from_green_func!(m::Mesh, g::Gfunction)
    # 軌道・副格子あたりの電子数
    n_0::Float64 = m.prmt.n_fill

    # 与えられた電子数密度になるようにBrent法を用いて化学ポテンシャルを決定する
    res = optimize(
        μ -> (calc_electron_density_from_green_func!(m, g, μ) - n_0)^2,
        m.emin - 2*m.W, m.emax + 2*m.W, rel_tol=1e-4, Brent()
    )
    g.μ = Optim.minimizer(res)[1]
end

"グリーン関数を用いた電子数密度の計算"
function calc_electron_density_from_green_func!(m::Mesh, g::Gfunction, μ::Float64)
    p::Parameters = m.prmt

    set_g0iτk!(m, g, μ)
    set_giωk!(m, g, μ)
    symmetrize_giωk!(g)

    # FT: G(iϵ, k) -> G(τ, k)
    giτk::Array{ComplexF64, 6} = ωn_to_τ(p, false, g.giωk .- g.g0iωk) .+ g.g0iτk

    # sum_k tr[G(τ=+0, k)] / Nd
    sumg::Float64 = sum(
        real(giτk[1, ik1, ik2, ik3, iξ, iξ])
        for iξ in 1:p.nwan, ik3 in 1:p.nk3, ik2 in 1:p.nk2, ik1 in 1:p.nk1
    ) / (p.nk1 * p.nk2 * p.nk3 * p.norb)

    2 + 2sumg/p.nspin
end

"χiωkの発散チェックとしてmax{eig(χ0iωk*U)}を計算する"
function max_eigval_χ0U(m::Mesh, g::Gfunction)
    χ0U_eig = Array{ComplexF64}(undef, m.prmt.nω, m.prmt.nk1, m.prmt.nk2, m.prmt.nk3, m.prmt.nwan^2)
    for ik3 in 1:m.prmt.nk3, ik2 in 1:m.prmt.nk2, ik1 in 1:m.prmt.nk1, iω in 1:m.prmt.nω
        χ0U_eig[iω, ik1, ik2, ik3, :] .= eigvals(@view(g.χ0iωk[iω, ik1, ik2, ik3, :, :]) * m.U_mat)
    end

    id = argmax(real.(χ0U_eig))
    if abs(imag(χ0U_eig[id])) > 1e-10
        println("!!!!!! Imaginary part of χ eig_val is very large: $(abs(imag(χ0U_eig[id]))) !!!!!!")
    end

    real(χ0U_eig[id])
end

"max(χ0*U) >= 1の場合に実行する、相互作用Uを繰り込むためのループ"
function U_renormalization!(m::Mesh, g::Gfunction, div_check::Float64, ΔU::Float64)::Tuple{Float64, Float64}
    println("### Check for renormalization max(χ0*U): $div_check, U = $(m.prmt.U)")

    # ループ
    while div_check >= 1.0
        ### max(χ0*U) < 1を満たすまでUを繰り込む
        m.prmt.U -= ΔU
        ΔU /= 2.0
        println("New U set to $(m.prmt.U)")

        m.U_mat = set_interaction(m.prmt)
        div_check = max_eigval_χ0U(m, g)
    end

    println("New U value: $(m.prmt.U) with max(χ0*U) = $div_check")

    div_check, ΔU
end

"FLEX計算で使用するループ"
function loop!(m::Mesh, g::Gfunction)
    giωk_old::Array{ComplexF64, 6} = copy(g.giωk)

    set_giτr!(m, g)
    set_χ0iωk!(m, g)
    set_χiωk!(m, g)

    set_Σiωk!(m, g)

    set_μ_from_green_func!(m, g)
    set_g0iτk!(m, g, g.μ)
    set_giωk!(m, g, g.μ)
    symmetrize_giωk!(g)
    sfc_check::Float64 = maximum(abs.(g.giωk .- giωk_old))

    # 新しいグリーン関数を少し混ぜる
    ## 収束性が悪いときは、mixの値を変えることで改善する場合がある
    g.giωk .= m.prmt.mix .* g.giωk .+ (1-m.prmt.mix) .* giωk_old

    sfc_check
end

"FLEX近似でグリーン関数を求める"
function solve_FLEX!(m::Mesh, g::Gfunction)
    p::Parameters = m.prmt

    # Uを収束させるためのパラメータ
    g.U_pval::Float64 = p.U
    ΔU::Float64 = p.U / 2.0
    U_it::Int64 = 1

    # 収束条件に用いるパラメータ
    div_check::Float64 = max_eigval_χ0U(m, g)
    conv_tol::Float64 = p.g_sfc_tol
    sfc_check::Float64 = 1.0

    # 収束するまでループを実行
    while div_check >= 1.0 || abs(g.U_pval - p.U) > 1e-10 || U_it == 1
        ### 計算に時間がかかりすぎているときのチェック
        if U_it == 100
            println("U iteration reached step 100. Everything okay?")
        end

        ### 1サイクルの後、まだ良くない場合はUをリセットする
        if abs(g.U_pval - p.U) > 1e-10
            p.U += ΔU
            m.U_mat = set_interaction(p)
            div_check = max_eigval_χ0U(m, g)
        end

        ### max(χ_0*U) >= 1 となったときは新しいUを設定する
        div_check, ΔU = U_renormalization!(m, g, div_check, ΔU)

        ### グリーン関数の収束を判定する条件を設定
        #### toleranceとイテレーションの最大回数
        if abs(g.U_pval - p.U) > 1e-10
            conv_tol = 1e-4
            sfc_it_max = 200
        else
            conv_tol = p.g_sfc_tol
            sfc_it_max = 400
        end

        ### 収束させるためのループ
        for it_sfc in 1:sfc_it_max
            sfc_check = loop!(m, g)

            println("$it_sfc: $sfc_check, μ = $(g.μ)")

            sfc_check <= conv_tol && break
        end

        U_it += 1
    end

    # 収束しているかのチェック
    ## Uの収束
    if abs(g.U_pval - p.U) > 1e-10
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("U is not initial input. Stopping gfunction.")

        return missing
    end

    ## giωkの収束
    if sfc_check > conv_tol
        println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        println("giωk not converged (diff = $sfc_check)). Stopping gfunction.")
        return missing
    end

    println("Self consistency loop finished!")
    println("### max(χ0*U) = $(max_eigval_χ0U(m, g))")

    return nothing
end

"一般化感受率を用いて、与えられた多極子演算子に対する感受率を計算する"
function calc_multipole_susceptibility(m::Mesh, g::Gfunction, op::AbstractMatrix)
    p::Parameters = m.prmt

    χ_O = zeros(ComplexF64, p.nω, p.nk1, p.nk2, p.nk3)
    for ζ4 in 1:p.nwan, ζ3 in 1:p.nwan, ζ2 in 1:p.nwan, ζ1 in 1:p.nwan
        ζ12::Int64 = p.nwan * (ζ1-1) + ζ2
        ζ34::Int64 = p.nwan * (ζ3-1) + ζ4
        for iq3 in 1:p.nk3, iq2 in 1:p.nk2, iq1 in 1:p.nk1
            qvec::Vector{Float64} = [
                2(iq1-1)%p.nk1 / p.nk1 - (2(iq1-1)÷p.nk1),
                2(iq2-1)%p.nk2 / p.nk2 - (2(iq2-1)÷p.nk2),
                2(iq3-1)%p.nk3 / p.nk3 - (2(iq3-1)÷p.nk3)
            ]

            # exp(iq.(r3-r1)): 対称性を回復するための依存因子
            ## 異なる内部座標をもつ副格子を取り扱う場合、一般にこの因子が必要
            @views χ_O[:, iq1, iq2, iq3] .+= (
                (cispi(dot(qvec, p.pos[(ζ3-1)÷p.nspin+1] .- p.pos[(ζ1-1)÷p.nspin+1])) * op[ζ1, ζ2] * op[ζ4, ζ3])
                .* g.χiωk[:, iq1, iq2, iq3, ζ12, ζ34]
            )
        end
    end

    if maximum(abs, imag.(χ_O)) > 1e-10
        println("!!!!!! Imaginary part of susceptibility is very large: $(maximum(abs, imag.(χ_O))) !!!!!!")
    end

    χ_O
end
