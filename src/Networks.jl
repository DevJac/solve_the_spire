module Networks
using Flux
using Statistics
export PolicyNetwork, QNetwork, VanillaNetwork, advantage

#####################
# Utility Functions #
#####################

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
    layers = Any[Dense(in, hidden[1], activation, initW=initW)]
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

struct QNetwork{N}
    network :: N
end

Flux.@functor QNetwork

function QNetwork(in, out, hidden, activation=mish, initW=Flux.kaiming_uniform)
    layers = Any[Dense(in, hidden[1], activation, initW=initW)]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))
    QNetwork(Chain(layers...))
end

(q::QNetwork)(s) = q.network(s)

##################
# Vanilla Network #
##################

struct VanillaNetwork{N}
    network :: N
end

Flux.@functor VanillaNetwork

function VanillaNetwork(in, out, hidden, activation=mish, initW=Flux.kaiming_uniform)
    layers = Any[Dense(in, hidden[1], activation, initW=initW)]
    for i in 1:length(hidden)-1
        push!(layers, Dense(hidden[i], hidden[i+1], activation, initW=initW))
    end
    push!(layers, Dense(hidden[end], out, identity))
    VanillaNetwork(Chain(layers...))
end

(n::VanillaNetwork)(s) = n.network(s)

end # module
