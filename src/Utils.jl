module Utils
using Random
using Zygote

export mc_q, onehot, clip, find, max_file_number, valgrad, explore_odds

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

function valgrad(f, x...)
    val, back = pullback(f, x...)
    val, back(1)
end

explore_odds(probs, ϵ=0.01) = sum(p -> maximum(probs) - ϵ > p ? p : 0, probs)

export Smoother, smooth!

mutable struct Smoother
    factor :: Float32
    value  :: Float32
    Smoother(smoothing_factor=0.9; initial_value=0) = new(smoothing_factor, initial_value)
end

function smooth!(smoother::Smoother, value)
    smoother.value = smoother.factor * smoother.value + (1 - smoother.factor) * value
end

export Batcher

struct Batcher{T}
    data :: Vector{T}
    batchsize :: Int
    Batcher(data, batchsize) = new{eltype(data)}(data, batchsize)
end

function Base.iterate(b::Batcher)
    shuffle!(b.data)
    iterate(b, 1)
end

function Base.iterate(b::Batcher, state)
    if b.batchsize > length(b.data)
        return (shuffle!(b.data), 1)
    end
    if state + b.batchsize-1 > length(b.data)
        post_shuffle_size = length(b.data) - state + 1
        result = b.data[state:end]
        shuffle!(b.data)
        (append!(result, b.data[1:post_shuffle_size]), post_shuffle_size+1)
    else
        (b.data[state:state+b.batchsize-1], (state + b.batchsize) % length(b.data))
    end
end

end # module
