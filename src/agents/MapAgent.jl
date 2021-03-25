export MapAgent, action, train!

mutable struct MapAgent
    player_embedder
    deck_embedder
    relics_embedder
    potions_embedder
    all_map_embedder
    single_map_embedder
    policy
    critic
    policy_opt
    critic_opt
    sars
    last_rewarded_floor
end
function MapAgent()
    player_embedder = VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50])
    deck_embedder = PoolNetwork(length(card_encoder), 20, [50])
    relics_embedder = VanillaNetwork(length(relics_encoder), 20, [50])
    potions_embedder = VanillaNetwork(length(potions_encoder), 20, [50])
    all_map_embedder = PoolNetwork(length(map_encoder), 20, [50])
    single_map_embedder = VanillaNetwork(length(map_encoder), 20, [50])
    policy = VanillaNetwork(
        sum(length, [
            player_embedder,
            deck_embedder,
            relics_embedder,
            potions_embedder,
            all_map_embedder,
            single_map_embedder]),
        1, [200, 50, 50])
    critic = VanillaNetwork(
        sum(length, [
            player_embedder,
            deck_embedder,
            relics_embedder,
            potions_embedder,
            all_map_embedder]),
        1, [200, 50, 50])
    MapAgent(
        player_embedder,
        deck_embedder,
        relics_embedder,
        potions_embedder,
        all_map_embedder,
        single_map_embedder,
        policy,
        critic,
        ADADelta(),
        ADADelta(),
        SARS(),
        0)
end

function action(agent::MapAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            if awaiting(agent.sars) == sar_reward
                r = 0
                last_rewarded_floor = 0
                add_reward(agent.sars, r, 0)
                log_value(ra.tb_log, "MapAgent/reward", r)
                log_value(ra.tb_log, "MapAgent/length_sars", length(agent.sars.rewards))
            end
        elseif gs["screen_type"] == "MAP"
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - last_rewarded_floor
                last_rewarded_floor = gs["floor"]
                add_reward(agent.sars, r)
                log_value(ra.tb_log, "MapAgent/reward", r)
                log_value(ra.tb_log, "MapAgent/length_sars", length(agent.sars.rewards))
            end
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            log_value(ra.tb_log, "MapAgent/state_value", state_value(agent, ra, sts_state))
            log_value(ra.tb_log, "MapAgent/step_explore", explore_odds(probabilities))
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            action = actions[action_i]
            return "choose $action"
        end
    end
end

function train!(agent::MapAgent, ra::RootAgent, epochs=1000)
    train_log = TBLogger("tb_logs/train_MapAgent")
    sars = fill_q(agent.sars)
    log_histogram(ra.tb_log, "MapAgent/rewards", map(sar -> sar.reward, sars))
    log_histogram(ra.tb_log, "MapAgent/q", map(sar -> sar.q, sars))
    target_agent = deepcopy(agent)
    kl_div_smoother = Smoother()
    local loss
    kl_divs = Float32[]
    actual_value = Float32[]
    estimated_value = Float32[]
    estimated_advantage = Float32[]
    entropys = Float32[]
    explore = Float32[]
    for epoch in 1:epochs
        batch = sars
        prms = params(
            agent.player_embedder,
            agent.deck_embedder,
            agent.relics_embedder,
            agent.potions_embedder,
            agent.all_map_embedder,
            agent.policy)
        loss, grads = valgrad(prms) do
            -mean(batch) do sar
                target_aps = action_probabilities(target_agent, ra, sar.state)[2]
                target_ap = target_aps[sar.action]
                online_aps = action_probabilities(agent, ra, sar.state)[2]
                online_ap = online_aps[sar.action]
                advantage = sar.q - state_value(target_agent, ra, sar.state)
                Zygote.ignore() do
                    push!(kl_divs, Flux.Losses.kldivergence(online_aps, target_aps))
                    push!(actual_value, online_ap * sar.q)
                    push!(estimated_value, online_ap * state_value(target_agent, ra, sar.state))
                    push!(estimated_advantage, online_ap * advantage)
                    push!(entropys, entropy(online_aps))
                    push!(explore, explore_odds(online_aps))
                end
                min(
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
        Flux.Optimise.update!(agent.policy_opt, prms, grads)
        if smooth!(kl_div_smoother, mean(kl_divs)) > 0.01; break end
        empty!(kl_divs); empty!(actual_value); empty!(estimated_value); empty!(estimated_advantage)
        empty!(entropys); empty!(explore)
    end
    log_value(ra.tb_log, "MapAgent/policy_loss", loss)
    log_value(ra.tb_log, "MapAgent/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "MapAgent/actual_value", mean(actual_value))
    log_value(ra.tb_log, "MapAgent/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "MapAgent/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "MapAgent/entropy", mean(entropys))
    log_value(ra.tb_log, "MapAgent/explore", mean(explore))
    for (epoch, batch) in enumerate(Batcher(sars, 100))
        if epoch > epochs; break end
        prms = params(agent.critic)
        loss, grads = valgrad(prms) do
            mean(batch) do sar
                predicted_q = state_value(agent, ra, sar.state)
                actual_q = sar.q
                (predicted_q - actual_q)^2
            end
        end
        log_value(train_log, "train/critic_loss", loss, step=epoch)
        Flux.Optimise.update!(agent.critic_opt, prms, grads)
    end
    log_value(ra.tb_log, "MapAgent/critic_loss", loss)
    empty!(agent.sars)
end

function action_probabilities(agent::MapAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    next_nodes = isempty(gs["screen_state"]["next_nodes"]) ? [Dict("x" => 3, "y" => 16)] : gs["screen_state"]["next_nodes"]
    player_e = agent.player_embedder(player_basic_encoder(sts_state))
    deck_e = agent.deck_embedder(card_encoder, gs["deck"])
    relics_e = agent.relics_embedder(relics_encoder(gs["relics"]))
    potions_e = agent.potions_embedder(potions_encoder(gs["potions"]))
    all_map_e = agent.all_map_embedder(node -> map_encoder(sts_state, node["x"], node["y"]), next_nodes)
    single_map_e = agent.single_map_embedder(node -> map_encoder(sts_state, node["x"], node["y"]), next_nodes)
    Zygote.ignore() do
        @assert size(player_e) == (length(player_basic_encoder),)
        @assert size(deck_e) == (20,)
        @assert size(relics_e) == (20,)
        @assert size(potions_e) == (20,)
        @assert size(all_map_e) == (20,)
        @assert size(single_map_e) == (20, length(next_nodes))
    end
    action_weights = agent.policy(vcat(
        repeat(player_e, 1, size(single_map_e, 2)),
        repeat(deck_e, 1, size(single_map_e, 2)),
        repeat(relics_e, 1, size(single_map_e, 2)),
        repeat(potions_e, 1, size(single_map_e, 2)),
        repeat(all_map_e, 1, size(single_map_e, 2)),
        single_map_e))
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    actions = collect(0:length(next_nodes)-1)
    Zygote.@ignore @assert length(actions) == length(probabilities)
    actions, probabilities
end

function state_value(agent::MapAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    next_nodes = isempty(gs["screen_state"]["next_nodes"]) ? [Dict("x" => 3, "y" => 16)] : gs["screen_state"]["next_nodes"]
    player_e = agent.player_embedder(player_basic_encoder(sts_state))
    deck_e = agent.deck_embedder(card_encoder, gs["deck"])
    relics_e = agent.relics_embedder(relics_encoder(gs["relics"]))
    potions_e = agent.potions_embedder(potions_encoder(gs["potions"]))
    all_map_e = agent.all_map_embedder(node -> map_encoder(sts_state, node["x"], node["y"]), next_nodes)
    only(agent.critic(vcat(
        player_e,
        deck_e,
        relics_e,
        potions_e,
        all_map_e)))
end
