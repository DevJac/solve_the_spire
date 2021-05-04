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

const STANDARD_POLICY_LAYERS = [80, 80]
const STANDARD_CRITIC_LAYERS = [80, 80]
const STANDARD_EMBEDDER_LAYERS = [40]
const STANDARD_EMBEDDER_OUT = 20
const STANDARD_TRAINING_EPOCHS = 20
const STANDARD_KL_DIV_EARLY_STOP = 1000 # disabled, no limit
const STANDARD_OPTIMIZER = () -> ADADelta()

mutable struct RootAgent
    errors         :: Int
    games          :: Int
    generation     :: Int
    ready_to_train :: Bool
    tb_log
    map_agent
    combat_agent
    agents
end

function RootAgent()
    tb_log = TBLogger("tb_logs/agent", tb_append)
    set_step!(tb_log, maximum(TensorBoardLogger.steps(tb_log)))
    map_agent = MapAgent()
    combat_agent = CombatAgent()
    agents = [
        MenuAgent(),
        combat_agent,
        CampfireAgent(),
        DeckAgent(),
        EventAgent(),          # TODO
        map_agent,
        RewardAgent(),
        ShopAgent(),
        SpecialActionAgent(),
        PotionAgent()]
    RootAgent(0, 0, 0, false, tb_log, map_agent, combat_agent, agents)
end

function RootAgent(ra::RootAgent)
    tb_log = TBLogger("tb_logs/agent", tb_append)
    set_step!(tb_log, maximum(TensorBoardLogger.steps(tb_log)))
    RootAgent(ra.errors, ra.games, ra.generation, ra.ready_to_train, tb_log, ra.map_agent, ra.combat_agent, ra.agents)
end

function agent_command(root_agent::RootAgent, sts_state)
    if "error" in keys(sts_state)
        root_agent.errors += 1
        log_value(root_agent.tb_log, "agent/errors", root_agent.errors)
        if root_agent.errors % 100 == 0; kill_java() end
        return "state"
    end
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
        println("Training: $(typeof(agent))")
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
    actions = filter(a -> (a[1] != "potion" || a[2] != "use" || length(a) != 3 ||
                           a[3] < length(gs["potions"]) &&
                           gs["potions"][a[3]+1]["can_use"] &&
                           !gs["potions"][a[3]+1]["requires_target"]), actions)
    actions = filter(a -> (a[1] != "potion" || a[2] != "use" || length(a) != 4 ||
                           a[3] < length(gs["potions"]) &&
                           gs["potions"][a[3]+1]["can_use"] &&
                           gs["potions"][a[3]+1]["requires_target"] &&
                           "combat_state" in keys(gs) &&
                           a[4] < length(gs["combat_state"]["monsters"]) &&
                           !gs["combat_state"]["monsters"][a[4]+1]["is_gone"]), actions)
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
        push!(actions, ("potion", "discard", potion_i))
        push!(actions, ("potion", "use", potion_i))
        for monster_i in 0:4
            push!(actions, ("potion", "use", potion_i, monster_i))
        end
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

function initial_hp_stats(ra::RootAgent)
    ra.combat_agent.initial_hp_stats
end

function floor_partial_credit(ra::RootAgent)
    ra.combat_agent.floor_partial_credit
end

function train!(train_log, agent, ra, epochs)
    sars = fill_q(agent.sars)
    if length(sars) < 2; return end
    target_agent = deepcopy(agent)
    local loss
    kl_divs = Float32[]
    actual_value = Float32[]
    estimated_value = Float32[]
    estimated_advantage = Float32[]
    entropys = Float32[]
    explore = Float32[]
    for (epoch, batch) in enumerate(Batcher(sars, 5000))
        prms = params(
            agent.choice_encoder,
            agent.policy)
        loss, grads = valgrad(prms) do
            -mean(batch) do sar
                target_aps = Zygote.@ignore action_probabilities(target_agent, ra, sar.state)[2]
                target_ap = Zygote.@ignore max(1e-8, target_aps[sar.action])
                online_aps = action_probabilities(agent, ra, sar.state)[2]
                online_ap = online_aps[sar.action]
                advantage = Zygote.@ignore sar.q_norm - state_value(target_agent, ra, sar.state)
                Zygote.ignore() do
                    push!(kl_divs, Flux.Losses.kldivergence(online_aps, target_aps))
                    push!(actual_value, online_ap * sar.q_norm)
                    push!(estimated_value, online_ap * state_value(target_agent, ra, sar.state))
                    push!(estimated_advantage, online_ap * advantage)
                    push!(entropys, entropy(online_aps))
                    push!(explore, explore_odds(online_aps))
                end
                sar.weight * min(
                    (online_ap / target_ap) * advantage,
                    clip(online_ap / target_ap, 0.2) * advantage)
            end
        end
        log_value(train_log, "train/policy_loss", loss, step=epoch)
        log_value(train_log, "train/kl_div", mean(kl_divs), step=epoch)
        log_value(train_log, "train/actual_value", mean(actual_value), step=epoch)
        log_value(train_log, "train/estimated_value", mean(estimated_value), step=epoch)
        log_value(train_log, "train/estimated_advantage", mean(estimated_advantage), step=epoch)
        log_value(train_log, "train/entropy", mean(entropys), step=epoch)
        log_value(train_log, "train/explore", mean(explore), step=epoch)
        @assert !any(isnan, (loss, mean(kl_divs), mean(actual_value), mean(estimated_value),
                             mean(estimated_advantage), mean(entropys), mean(explore)))
        Flux.Optimise.update!(agent.policy_opt, prms, grads)
        if epoch >= epochs || mean(kl_divs) > STANDARD_KL_DIV_EARLY_STOP; break end
        empty!(kl_divs); empty!(actual_value); empty!(estimated_value); empty!(estimated_advantage)
        empty!(entropys); empty!(explore)
    end
    log_value(ra.tb_log, "$(typeof(agent))/policy_loss", loss)
    log_value(ra.tb_log, "$(typeof(agent))/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "$(typeof(agent))/actual_value", mean(actual_value))
    log_value(ra.tb_log, "$(typeof(agent))/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "$(typeof(agent))/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "$(typeof(agent))/entropy", mean(entropys))
    log_value(ra.tb_log, "$(typeof(agent))/explore", mean(explore))
    for (epoch, batch) in enumerate(Batcher(sars, 1000))
        if epoch > epochs; break end
        prms = params(agent.critic)
        loss, grads = valgrad(prms) do
            mean(batch) do sar
                predicted_q = state_value(agent, ra, sar.state)
                actual_q = sar.q_norm
                (predicted_q - actual_q)^2
            end
        end
        log_value(train_log, "train/critic_loss", loss, step=epoch)
        @assert !isnan(loss)
        Flux.Optimise.update!(agent.critic_opt, prms, grads)
    end
    log_value(ra.tb_log, "$(typeof(agent))/critic_loss", loss)
    empty!(agent.sars)
end

end # module
