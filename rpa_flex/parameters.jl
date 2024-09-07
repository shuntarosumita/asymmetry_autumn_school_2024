mutable struct Parameters
    mode::String
    mix::Float64
    g_sfc_tol::Float64

    system::String
    nk1::Int64
    nk2::Int64
    nk3::Int64
    nω::Int64
    T::Float64
    β::Float64
    n_fill::Float64
    nspin::Int64
    norb::Int64
    nwan::Int64
    U::Float64
    pos::Vector{Vector{Float64}}
end

"パラメータを設定する関数"
function Parameters(
        system::String,
        n_fill::Float64,
        U::Float64,
        T::Float64,
        mode::String;
        nk1::Int64,
        nk2::Int64,
        nk3::Int64,
        nω::Int64,
    )::Parameters

    mix::Float64 = 0.2 # グリーン関数のイテレーションで新しいGをどのくらい混ぜるかを決める値

    # 収束条件を決める値(tolerance)
    g_sfc_tol::Float64 = 1e-6

    # 物理系に関するパラメータ
    β::Float64 = 1/T           # 逆温度
    n_fill::Float64 = n_fill   # 電子数密度
    nspin::Int64 = 1           # スピン自由度
    norb::Int64 = (
        if system == "square"
            1
        elseif system == "modified_square"
            2
        end
    )                          # 軌道・副格子自由度
    nwan::Int64 = nspin * norb # 全内部自由度

    # 単位胞内の副格子の内部座標
    pos::Vector{Vector{Float64}} = (
        if system == "square"
            [
                [0.0, 0.0, 0.0]
            ]
        elseif system == "modified_square"
            [
                [0.0, -0.5, 0.0], # 副格子a
                [-0.5, 0.0, 0.0]  # 副格子b
            ]
        end
    )

    return Parameters(
        mode, mix, g_sfc_tol,
        system, nk1, nk2, nk3, nω, T, β, n_fill, nspin, norb, nwan, U, pos
    )
end
