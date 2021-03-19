module STSAgents
using Encoders
using Flux
using Networks
using SARSM
using StatsBase
using TensorBoardLogger
using Utils
using Zygote

export RootAgent, agent_command, action, train!

mutable struct RootAgent
    games          :: Int
    generation     :: Int
    ready_to_train :: Bool
    tb_log
    agents
end

function RootAgent()
    tb_log = TBLogger("tb_logs/agent", tb_append)
    set_step!(tb_log, maximum(TensorBoardLogger.steps(tb_log)))
    agents = [
        CampfireAgent(),
        CombatAgent(),
        DeckAgent(),
        EventAgent(),
        MapAgent(),
        MenuAgent(),
        RewardAgent(),
        ShopAgent(),
        SpecialActionAgent()]
    RootAgent(0, 0, false, tb_log, agents)
end

function agent_command(root_agent::RootAgent, sts_state)
    increment_step!(root_agent.tb_log, 1)
    resulting_command = nothing
    for agent in root_agent.agents
        command = action(agent, root_agent, sts_state)
        if !isnothing(command) && !isnothing(resulting_command)
            @error "Two agents issued commands" resulting_command command
            throw("Two agents issued commands")
        end
        if !isnothing(command)
            resulting_command = command
        end
    end
    resulting_command
end

function train!(root_agent::RootAgent)
    for agent in root_agent.agents
        train!(agent, root_agent)
    end
end

include("agents/CampfireAgent.jl")
include("agents/CombatAgent.jl")
include("agents/DeckAgent.jl")
include("agents/EventAgent.jl")
include("agents/MapAgent.jl")
include("agents/MenuAgent.jl")
include("agents/PotionAgent.jl")
include("agents/RewardAgent.jl")
include("agents/ShopAgent.jl")
include("agents/SpecialActionAgent.jl")

end # module
