using AgentCommands
using BSON
using ChoiceEncoders
using Dates
using Encoders
using Flux
using JSON
using Networks
using NNlib
using OwnTime
using Printf
using Profile
using SARSM
using Sockets
using StatsBase
using STSAgents
using TensorBoardLogger
using Utils

m = BSON.load(ARGS[1])[:model]

all_agents_total = 0
for agent in m.agents
    println()
    ps = []
    if :choice_encoder in fieldnames(typeof(agent))
        append!(ps, params(agent.choice_encoder))
        println(typeof(agent), " ", sum(length, params(agent.choice_encoder)))
    end
    if :policy in fieldnames(typeof(agent))
        append!(ps, params(agent.policy))
        println(typeof(agent), " ", sum(length, params(agent.policy)))
    end
    if :critic in fieldnames(typeof(agent))
        append!(ps, params(agent.critic))
        println(typeof(agent), " ", sum(length, params(agent.critic)))
    end
    if !isempty(ps)
        println("Total: ", sum(length, ps))
        global all_agents_total += sum(length, ps)
    end
end
println("All agents total: ", all_agents_total)
