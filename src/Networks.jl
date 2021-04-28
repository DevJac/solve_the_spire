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
    hidden = vcat(in, hidden)
    layers = Any[]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))
    VanillaNetwork(Chain(layers...))
end

(n::VanillaNetwork)(s) = n.network(s)

Base.length(n::VanillaNetwork) = length(n.network.layers[end].b)

################
# Pool Network #
################

struct PoolNetwork{N}
    network     :: N
end

Flux.@functor PoolNetwork

function PoolNetwork(in, out, hidden, activation=relu, initW=Flux.kaiming_uniform)
    hidden = vcat(in, hidden)
    layers = Any[]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))
    PoolNetwork(Chain(layers...))
end

function (n::PoolNetwork)(s)
    network_out = n.network(s)
    pooled = sum(network_out, dims=2)
    reshape(pooled, length(pooled))
end

Base.length(n::PoolNetwork) = length(n.network.layers[end].b)

################
# Null Network #
################

struct NullNetwork; end

(n::NullNetwork)(s) = Float32[]

Base.length(n::NullNetwork) = 0

end # module
