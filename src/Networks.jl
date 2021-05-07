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
    laf_parameters = rand(Float32, out, 12)
    laf_parameters[:,[2,4,6,8]] *= 2
    PoolNetwork(Chain(layers...), laf_parameters)
end

# See: https://arxiv.org/abs/2012.08482
# Paper: Learning Aggregation Functions
# Authors: Giovanni Pellegrini, Alessandro Tibo, Paolo Frasconi, Andrea Passerini, Manfred Jaeger

function L(a, b, x)
    sum(x .^ exp.(b), dims=2) .^ exp.(a)
end

function LAF(params, x0)
    @assert size(params, 2) == 12
    a, b, c, d, e, f, g, h, α, β, γ, δ = eachcol(params)
    x = sigmoid.(x0)
    ((α .* L(a, b, x)) .+ (β .* L(c, d, (1 .- x)))) ./ ((γ .* L(e, f, x)) .+ (δ .* L(g, h, (1 .- x))))
end

function (n::PoolNetwork)(s)
    network_out = n.network(Float32.(s))
    pooled = LAF(n.laf_parameters, network_out)
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
