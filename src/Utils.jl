module Utils
export DiskStringSet, mc_q, onehot, clip

function mc_q(r, f, γ=1f0)
    result = Float32.(similar(r))
    q = 0f0
    for i in length(r):-1:1
        if f[1, i]
            q = 0f0
        end
        q += r[1, i]
        result[1, i] = q
        q *= γ
    end
    result
end

function onehot(hot_i, length)
    result = zeros(Float32, length)
    result[hot_i] = 1f0
    result
end

clip(n, ϵ) = clamp(n, 1-ϵ, 1+ϵ)

end # module
