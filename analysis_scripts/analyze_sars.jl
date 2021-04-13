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

for agent in m.agents
    if :sars in fieldnames(typeof(agent))
        println("\n\n\n")
        println(typeof(agent))
        qs = fill_q(agent.sars)
        for (i, q) in enumerate(qs)
            floor = nothing
            while isnothing(floor)
                try
                    floor = qs[i].state["game_state"]["floor"]
                catch e
                    if isa(e, KeyError); continue end
                    rethrow()
                end
            end
            @printf("%4d (%6.2f, %6.2f) %6.2f %6.2f\n",
                    floor,
                    agent.sars.rewards[i][1],
                    agent.sars.rewards[i][2],
                    q.reward,
                    q.q)
        end
    end
end
