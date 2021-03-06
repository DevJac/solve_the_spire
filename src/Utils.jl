module Utils
using Pushover
using Random
using SparseArrays
using Zygote

export mc_q, onehot, clip, find, max_file_number, valgrad, explore_odds, diagcat, nearest, kill_java, pushover

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

function diagcat(args...)
    x = map(args) do arg
        if size(arg) == (); arg = [arg] end
        reshape(arg, size(arg,1), size(arg,2))
    end
    collect(blockdiag(sparse.(x)...))
end

Zygote.@adjoint function diagcat(args...)
    val = diagcat(args...)
    adj = x -> begin
        i = (1, 1)
        tuple_parts = map(args) do arg
            part = x[i[1]:i[1]-1+size(arg,1),i[2]:i[2]-1+size(arg,2)]
            i = i .+ (size(arg,1),size(arg,2))
            part
        end
    end
    val, adj
end

nearest(n, ns) = minimum(x -> (abs(n - x), x), ns)[2]

function kill_java()
    try
        run(`killall -q java`)
    catch e
        if !isa(e, ProcessFailedException); rethrow() end
    end
end

function pushover(message, try_count=2)
    try
        client = PushoverClient(ENV["PUSHOVER_USER_KEY"], ENV["PUSHOVER_STS_API_TOKEN"])
        response = send(client, message)
        @assert response["status"] == 1
    catch e
        @warn "Pushover failed" exception=e
        if try_count > 1
            sleep(30)
            pushover(message, try_count-1)
        end
    end
end

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
    iterate(b, 0)
end

function Base.iterate(b::Batcher, state)
    if b.batchsize > length(b.data)
        return (shuffle!(b.data), 0)
    end
    if state + b.batchsize > length(b.data)
        post_shuffle_size = state + b.batchsize - length(b.data)
        result = b.data[state+1:end]
        shuffle!(b.data)
        (append!(result, b.data[1:post_shuffle_size]), post_shuffle_size)
    else
        (b.data[state+1:state+b.batchsize], (state + b.batchsize) % length(b.data))
    end
end

end # module
