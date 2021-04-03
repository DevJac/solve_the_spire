module Networks
using Flux
using Statistics
using Zygote
export VanillaNetwork, PoolNetwork, NullNetwork

###################
# Vanilla Network #
###################

struct VanillaNetwork{N}
    network :: N
end

Flux.@functor VanillaNetwork

function VanillaNetwork(in, out, hidden, activation=relu, initW=Flux.kaiming_uniform)
    in += 1
    hidden = vcat(in, hidden)
    layers = Any[]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))
    VanillaNetwork(Chain(layers...))
end

function (n::VanillaNetwork)(f, s)
    if isempty(s)
        null(n)
    else
        n(reduce(hcat, map(f, s)))
    end
end

(n::VanillaNetwork)(s) = n.network([1; s])

null(n::VanillaNetwork) = n.network(zeros(size(n.network.layers[1].W, 2)))

Base.length(n::VanillaNetwork) = length(n.network.layers[end].b)

################
# Pool Network #
################

struct PoolNetwork{N}
    network     :: N
    oif_weights :: Array{Float32,2}
end

Flux.@functor PoolNetwork

function PoolNetwork(in, out, hidden, activation=relu, initW=Flux.kaiming_uniform)
    in += 1
    hidden = vcat(in, hidden)
    layers = Any[]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))
    PoolNetwork(Chain(layers...), ones(Float32, out, 4))
end

function (n::PoolNetwork)(f, s)
    if isempty(s)
        null(n)
    else
        n(reduce(hcat, map(f, s)))
    end
end

function (n::PoolNetwork)(s)
    # network out
    no = n.network([1; s])
    # applied order invariant functions
    applied_oif = hcat(sum(no, dims=2), mean(no, dims=2), minimum(no, dims=2), maximum(no, dims=2))
    pooled = sum(applied_oif .* n.oif_weights, dims=2)
    reshape(pooled, length(pooled))
end

null(n::PoolNetwork) = n.network(zeros(size(n.network.layers[1].W, 2)))

Base.length(n::PoolNetwork) = length(n.network.layers[end].b)

################
# Null Network #
################

struct NullNetwork; end

(n::NullNetwork)(s) = Float32[]

Base.length(n::NullNetwork) = 0

end # module
