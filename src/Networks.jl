module Networks
using Flux
using Statistics
using Zygote
export PolicyNetwork, QNetwork, VanillaNetwork, PoolNetwork, PoolEachNetwork, GRUNetwork, value, advantage

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

#####################
# Pool Each Network #
#####################

struct PoolEachNetwork{N,M}
    each_network :: N
    pool_network :: M
end

Flux.@functor PoolEachNetwork

pool(n::PoolEachNetwork) = n.pool_network

function PoolEachNetwork(in, out, hidden, activation=relu, initW=Flux.kaiming_uniform)
    PoolEachNetwork(
        VanillaNetwork(in, out, hidden, activation, initW),
        PoolNetwork(in, out, hidden, activation, initW))
end

function (n::PoolEachNetwork)(f, s)
    each = n.each_network(f, s)
    vcat(
        each,
        repeat(n.pool_network(f, s), 1, size(each, 2)))
end

function (n::PoolEachNetwork)(s)
    each = n.each_network(s)
    vcat(
        each,
        repeat(n.pool_network(s), 1, size(each, 2)))
end

function null(n::PoolEachNetwork)
    each = null(n.each_network)
    vcat(
        each,
        repeat(null(n.pool_network), 1, size(each, 2)))
end

Base.length(n::PoolEachNetwork) = length(n.each_network) * 2

###############
# GRU Network #
###############

struct GRUNetwork{N}
    network :: N
end

Flux.@functor GRUNetwork

function GRUNetwork(in, out, hidden, activation=relu, initW=Flux.kaiming_uniform)
    in += 1
    hidden = vcat(in, hidden)
    layers = Any[]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
        push!(layers, GRU(hidden[i+1], hidden[i+1]))
    end
    push!(layers, Dense(hidden[end], out, identity))
    GRUNetwork(Chain(layers...))
end

function (n::GRUNetwork)(f, s)
    if isempty(s)
        null(n)
    else
        n(reduce(hcat, map(f, s)))
    end
end

(n::GRUNetwork)(s) = n.network([1; s])

null(n::GRUNetwork) = n.network(zeros(size(n.network.layers[1].W, 2)))

Base.length(n::GRUNetwork) = length(n.network.layers[end].b)

end # module
