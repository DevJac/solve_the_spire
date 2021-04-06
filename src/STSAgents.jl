module STSAgents
using AgentCommands
using ChoiceEncoders
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
    map_agent
    agents
end

function RootAgent()
    tb_log = TBLogger("tb_logs/agent", tb_append)
    set_step!(tb_log, maximum(TensorBoardLogger.steps(tb_log)))
    map_agent = MapAgent()
    agents = [
        CampfireAgent(),
        CombatAgent(),
        DeckAgent(),
        EventAgent(),          # TODO
        map_agent,
        MenuAgent(),
        RewardAgent(),
        ShopAgent(),           # TODO
        SpecialActionAgent(),  # TODO
        PotionAgent()]         # TODO
    RootAgent(0, 0, false, tb_log, map_agent, agents)
end

function RootAgent(ra::RootAgent)
    tb_log = TBLogger("tb_logs/agent", tb_append)
    set_step!(tb_log, maximum(TensorBoardLogger.steps(tb_log)))
    RootAgent(ra.games, ra.generation, ra.ready_to_train, tb_log, ra.map_agent, ra.agents)
end

function agent_command(root_agent::RootAgent, sts_state)
    increment_step!(root_agent.tb_log, 1)
    log_value(root_agent.tb_log, "agent/games", root_agent.games)
    log_value(root_agent.tb_log, "agent/generation", root_agent.generation)
    log_value(root_agent.tb_log, "agent/encoder_cache_length", length(memoize_cache(Encoders.encode)))
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

function all_valid_actions(sts_state)
    gs = sts_state["game_state"]
    actions = all_possible_actions()
    actions = filter(a -> a[1] in sts_state["available_commands"], actions)
    actions = filter(a -> (a[1] != "choose" ||
                           a[2] < length(gs["choice_list"])), actions)
    actions = filter(a -> (a[1] != "potion" || a[2] != "use" ||
                           a[3] < length(gs["potions"]) &&
                           gs["potions"][a[3]+1]["can_use"]), actions)
    actions = filter(a -> (a[1] != "potion" || a[2] != "discard" ||
                           a[3] < length(gs["potions"]) &&
                           gs["potions"][a[3]+1]["can_discard"]), actions)
    actions = filter(a -> (a[1] != "play" || length(a) != 2 ||
                           a[2] <= length(gs["combat_state"]["hand"]) &&
                           gs["combat_state"]["hand"][a[2]]["is_playable"] &&
                           !gs["combat_state"]["hand"][a[2]]["has_target"]), actions)
    actions = filter(a -> (a[1] != "play" || length(a) != 3 ||
                           a[2] <= length(gs["combat_state"]["hand"]) &&
                           a[3] < length(gs["combat_state"]["monsters"]) &&
                           gs["combat_state"]["hand"][a[2]]["is_playable"] &&
                           gs["combat_state"]["hand"][a[2]]["has_target"] &&
                           !gs["combat_state"]["monsters"][a[3]+1]["is_gone"]), actions)
    actions
end

function all_possible_actions()
    actions = []
    for card_i in 1:10
        push!(actions, ("play", card_i))
        for monster_i in 0:4
            push!(actions, ("play", card_i, monster_i))
        end
    end
    for potion_i in 0:4
        push!(actions, ("potion", "use", potion_i))
        push!(actions, ("potion", "discard", potion_i))
    end
    for choice_i in 0:29
        push!(actions, ("choose", choice_i))
    end
    push!(actions, ("end",))
    push!(actions, ("proceed",))
    push!(actions, ("return",))
    push!(actions, ("confirm",))
    push!(actions, ("leave",))
    push!(actions, ("skip",))
    actions
end

function encode_seq(encoder, sequence)
    if !isempty(sequence)
        e = reduce(hcat, map(encoder, sequence))
        [zeros(1, size(e, 2)); e]
    else
        [1; zeros(length(encoder))]
    end
end

function current_map_node(ra::RootAgent)
    ra.map_agent.current_map_node
end

end # module
