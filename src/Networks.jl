module Networks
using Flux
using Statistics
export PolicyNetwork, QNetwork, VanillaNetwork, value, advantage

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

function PolicyNetwork(in, out, hidden, activation=mish, initW=Flux.kaiming_uniform)
    pushfirst!(hidden, in)
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
    function QNetwork(in, out, hidden, activation=mish, initW=Flux.kaiming_uniform)
        pushfirst!(hidden, in)
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

function VanillaNetwork(in, out, hidden, activation=mish, initW=Flux.kaiming_uniform)
    pushfirst!(hidden, in)
    layers = Any[]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))
    VanillaNetwork(Chain(layers...))
end

(n::VanillaNetwork)(s) = n.network(s)

end # module
