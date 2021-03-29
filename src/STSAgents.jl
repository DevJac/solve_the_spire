module STSAgents
using AgentCommands
using Encoders
using Flux
using Memoize
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
    agents = [MenuAgent(), SingleNNAgent()]
    RootAgent(0, 0, false, tb_log, agents)
end

function RootAgent(ra::RootAgent)
    tb_log = TBLogger("tb_logs/agent", tb_append)
    set_step!(tb_log, maximum(TensorBoardLogger.steps(tb_log)))
    RootAgent(ra.games, ra.generation, ra.ready_to_train, tb_log, ra.agents)
end

function agent_command(root_agent::RootAgent, sts_state)
    increment_step!(root_agent.tb_log, 1)
    log_value(root_agent.tb_log, "agent/games", root_agent.games)
    log_value(root_agent.tb_log, "agent/generation", root_agent.generation)
    log_value(root_agent.tb_log, "agent/encoder_cache_length", length(memoize_cache(Encoders.encode)))
    log_value(root_agent.tb_log, "agent/flatten_json_cache_length", length(memoize_cache(flatten_json)))
    overridden_commands = []
    resulting_command = nothing
    for agent in root_agent.agents
        command = action(agent, root_agent, sts_state)
        if !isnothing(command) && !isnothing(resulting_command)
            push!(overridden_commands, resulting_command)
        end
        if !isnothing(command)
            resulting_command = command
        end
    end
    if !isempty(overridden_commands)
        Command(resulting_command, Dict("overridden_commands" => overridden_commands))
    else
        resulting_command
    end
end

function train!(root_agent::RootAgent)
    for agent in root_agent.agents
        train!(agent, root_agent)
    end
    empty!(memoize_cache(Encoders.encode))
    empty!(memoize_cache(flatten_json))
end

include("agents/MenuAgent.jl")
include("agents/SingleNNAgent.jl")

end # module
