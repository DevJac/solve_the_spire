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

#################
# DiskStringSet #
#################

struct DiskStringSet
    file :: String
end

function Base.push!(set::DiskStringSet, s::String)
    lines = readlines(open(set.file, create=true))
    if strip(s) in lines; return end
    open(set.file, append=true) do f
        write(f, strip(s) * "\n")
    end
end

function Base.pop!(set::DiskStringSet, s::String)
    lines = readlines(open(set.file, create=true))
    filter!(lines) do line
        strip(line) != strip(s)
    end
    open(set.file, write=true) do f
        for line in lines
            write(f, strip(line) * "\n")
        end
    end
end

function Base.length(set::DiskStringSet)
    lines = readlines(open(set.file, create=true))
    length(lines)
end

function Base.iterate(set::DiskStringSet)
    lines = readlines(open(set.file, create=true))
    if isempty(lines)
        nothing
    else
        (strip(lines[1]), 1)
    end
end

function Base.iterate(set::DiskStringSet, state)
    lines = readlines(open(set.file, create=true))
    if length(lines) == state
        nothing
    else
        (strip(lines[state+1]), state+1)
    end
end

end # module
