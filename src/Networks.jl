module Networks
using Flux
using Statistics
export PolicyNetwork, QNetwork, VanillaNetwork, PoolNetwork, value, advantage

#####################
# Utility Functions #
#####################

value(network, s) = mean(network(s), dims=1)

function advantage(network, s)
    q = network(s)
    q .- mean(q, dims=1)
end

##################
# Policy Network #
##################

struct PolicyNetwork{N}
    network :: N
end

Flux.@functor PolicyNetwork

function PolicyNetwork(in, out, hidden, activation=relu, initW=Flux.kaiming_uniform)
    hidden = vcat(in, hidden)
    layers = Any[]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))
    push!(layers, softmax)
    PolicyNetwork(Chain(layers...))
end

(p::PolicyNetwork)(s) = p.network(s)

#############
# Q Network #
#############

struct QNetwork{T, A, V}
    trunk_network  :: T
    action_network :: A
    value_network  :: V
    function QNetwork(in, out, hidden, activation=relu, initW=Flux.kaiming_uniform)
        hidden = vcat(in, hidden)
        layers = Any[]
        for i in 1:length(hidden)-1
            push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
        end
        t = Chain(layers...)
        a = Dense(hidden[end], out, identity)
        v = Dense(hidden[end], 1, identity)
        new{typeof(t), typeof(a), typeof(v)}(t, a, v)
    end
end

Flux.@functor QNetwork

function (q::QNetwork)(s)
    t = q.trunk_network(s)
    a = q.action_network(t)
    v = q.value_network(t)
    v .+ a .- mean(a, dims=1)
end

##################
# Vanilla Network #
##################

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

(n::VanillaNetwork)(f, s) = n(reduce(hcat, map(f, s)))

(n::VanillaNetwork)(s) = n.network(s)

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
    hidden = vcat(in, hidden)
    layers = Any[]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))
    PoolNetwork(Chain(layers...), ones(Float32, out, 4))
end

(n::PoolNetwork)(f, s) = n(reduce(hcat, map(f, s)))

function (n::PoolNetwork)(s)
    # network out
    no = n.network(s)
    # applied order invariant functions
    applied_oif = hcat(sum(no, dims=2), mean(no, dims=2), minimum(no, dims=2), maximum(no, dims=2))
    pooled = sum(applied_oif .* n.oif_weights, dims=2)
    reshape(pooled, length(pooled))
end

Base.length(n::PoolNetwork) = length(n.network.layers[end].b)

end # module
