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
    network        :: N
    laf_parameters :: Array{Float32,2}
end

Flux.@functor PoolNetwork

function PoolNetwork(in, out, hidden, activation=relu, initW=Flux.kaiming_uniform)
    hidden = vcat(in, hidden)
    layers = Any[]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))

    PoolNetwork(Chain(layers...), rand(Float32, out, 12))
end

# See: https://arxiv.org/abs/2012.08482
# Paper: Learning Aggregation Functions
# Authors: Giovanni Pellegrini, Alessandro Tibo, Paolo Frasconi, Andrea Passerini, Manfred Jaeger

function L(a, b, x)
    sum(x.^b)^a
end

function LAF(params_and_x)
    a, b, c, d, e, f, g, h, α, β, γ, δ, x0... = params_and_x
    x = sigmoid.(x0)
    ((α * L(a, b, x)) + (β * L(c, d, x))) / ((γ * L(e, f, x)) + (δ * L(g, h, x)))
end

function (n::PoolNetwork)(s)
    network_out = n.network(Float32.(s))
    with_params = vcat(n.laf_parameters', network_out')
    map(i -> LAF(with_params[:,i]), 1:size(with_params, 2))
end

Base.length(n::PoolNetwork) = length(n.network.layers[end].b)

################
# Null Network #
################

struct NullNetwork; end

(n::NullNetwork)(s) = Float32[]

Base.length(n::NullNetwork) = 0

end # module
