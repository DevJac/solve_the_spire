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
        qs = fill_q(agent.sars, sar -> sar.q - STSAgents.state_value(agent, m, sar.state), 0.98)
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
            sv = STSAgents.state_value(agent, m, qs[i].state)
            @printf("%4d %4d (%6.2f, %6.2f) %6.2f %6.2f (%6.2f - ev: %6.2f) = %6.2f ad_norm: %6.2f",
                    i,
                    floor,
                    agent.sars.rewards[i][1],
                    agent.sars.rewards[i][2],
                    q.reward,
                    q.q,
                    q.q,
                    sv,
                    q.advantage,
                    q.advantage_norm)
            if "combat_state" in keys(qs[i].state["game_state"])
                state = qs[i].state
                as, aps = STSAgents.action_probabilities(agent, m, state)
                @printf("  Action: %20s %4.2f PH: %2d MH: (%s)\n",
                        as[qs[i].action],
                        aps[qs[i].action],
                        state["game_state"]["current_hp"],
                        map(m -> m["current_hp"], state["game_state"]["combat_state"]["monsters"]))
            else
                @printf("\n")
            end
        end
    end
end
