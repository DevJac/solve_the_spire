module Utils
export mc_q, onehot, clip, find, max_file_number

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

function find(needle, haystack)
    matching_i = findall(x -> x == needle, haystack)
    if isempty(matching_i)
        nothing
    else
        only(matching_i)
    end
end

function max_file_number(directory, prefix)
    files = filter(readdir(directory)) do f
        startswith(f, prefix)
    end
    if isempty(files); return 0 end
    file_numbers = map(files) do f
        m = match(r"\d+", f)
        isnothing(m) ? 0 : parse(Int, m.match)
    end
    maximum(file_numbers)
end

end # module
