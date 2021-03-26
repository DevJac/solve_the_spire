export DeckAgent, action, train!

mutable struct DeckAgent
    relics_embedder
    map_embedder
    deck_embedder
    all_card_embedder
    single_card_embedder
    policy
    critic
    policy_opt
    critic_opt
    sars
    last_rewarded_floor :: Int
end
function DeckAgent()
    relics_embedder = VanillaNetwork(length(relics_encoder), 20, [50])
    map_embedder = VanillaNetwork(length(map_encoder), 20, [50])
    deck_embedder = PoolNetwork(length(card_encoder), 40, [100])
    all_card_embedder = PoolNetwork(length(card_encoder), 20, [50])
    single_card_embedder = VanillaNetwork(length(card_encoder), 20, [50])
    policy = VanillaNetwork(
        sum(length, [
            relics_embedder,
            map_embedder,
            deck_embedder,
            all_card_embedder,
            single_card_embedder
        ]) + 4 + 2,
        1, [200, 50, 50])
    critic = VanillaNetwork(
        sum(length, [
            relics_embedder,
            map_embedder,
            deck_embedder,
            all_card_embedder
        ]) + 4 + 2,
        1, [200, 50, 50])
    DeckAgent(
        relics_embedder,
        map_embedder,
        deck_embedder,
        all_card_embedder,
        single_card_embedder,
        policy,
        critic,
        ADADelta(),
        ADADelta(),
        SARS(),
        0)
end

function action(agent::DeckAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_rewarded_floor
                agent.last_rewarded_floor = 0
                add_reward(agent.sars, r, 0)
                log_value(ra.tb_log, "DeckAgent/reward", r)
                log_value(ra.tb_log, "DeckAgent/length_sars", length(agent.sars.rewards))
            end
        elseif gs["screen_type"] in ("CARD_REWARD", "GRID")
            if !in("choose", sts_state["available_commands"])
                return "confirm"
            end
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_rewarded_floor
                agent.last_rewarded_floor = gs["floor"]
                add_reward(agent.sars, r)
                log_value(ra.tb_log, "DeckAgent/reward", r)
                log_value(ra.tb_log, "DeckAgent/length_sars", length(agent.sars.rewards))
            end
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            log_value(ra.tb_log, "DeckAgent/state_value", state_value(agent, ra, sts_state))
            log_value(ra.tb_log, "DeckAgent/step_explore", explore_odds(probabilities))
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            action = actions[action_i]
            if action == "skip"
                return "skip"
            end
            return "choose $action"
        end
    end
end

function train!(agent::DeckAgent, ra::RootAgent, epochs=1000)
    train_log = TBLogger("tb_logs/train_DeckAgent")
    sars = fill_q(agent.sars)
    log_histogram(ra.tb_log, "DeckAgent/rewards", map(sar -> sar.reward, sars))
    log_histogram(ra.tb_log, "DeckAgent/q", map(sar -> sar.q, sars))
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
            agent.relics_embedder,
            agent.map_embedder,
            agent.deck_embedder,
            agent.all_card_embedder,
            agent.single_card_embedder,
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
    log_value(ra.tb_log, "DeckAgent/policy_loss", loss)
    log_value(ra.tb_log, "DeckAgent/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "DeckAgent/actual_value", mean(actual_value))
    log_value(ra.tb_log, "DeckAgent/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "DeckAgent/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "DeckAgent/entropy", mean(entropys))
    log_value(ra.tb_log, "DeckAgent/explore", mean(explore))
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
    log_value(ra.tb_log, "DeckAgent/critic_loss", loss)
    empty!(agent.sars)
end

function action_probabilities(agent::DeckAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    local choice_e
    local unselected_screen_cards
    local skip_available = false
    local bowl_available = false
    Zygote.ignore() do
        choice_e = zeros(Float32, 4)
        if gs["screen_type"] == "CARD_REWARD"; choice_e[1] = 1 end
        if gs["screen_type"] == "GRID" && gs["screen_state"]["for_upgrade"]; choice_e[2] = 1 end
        if gs["screen_type"] == "GRID" && gs["screen_state"]["for_transform"]; choice_e[3] = 1 end
        if gs["screen_type"] == "GRID" && gs["screen_state"]["for_purge"]; choice_e[4] = 1 end
        @assert sum(choice_e) in (0, 1)
        if gs["screen_type"] == "GRID"
            screen_cards = gs["screen_state"]["cards"]
            selected_screen_cards = gs["screen_state"]["selected_cards"]
            unselected_screen_cards = filter(c -> !in(c, selected_screen_cards), screen_cards)
            @assert !gs["screen_state"]["any_number"]
        else
            unselected_screen_cards = gs["screen_state"]["cards"]
            skip_available = gs["screen_state"]["skip_available"]
            bowl_available = gs["screen_state"]["bowl_available"]
        end

    end
    relics_e = agent.relics_embedder(relics_encoder(gs["relics"]))
    map_e = agent.map_embedder(map_encoder(sts_state, ra.map_agent.map_node[1], ra.map_agent.map_node[2]))
    deck_e = agent.deck_embedder(card_encoder, gs["deck"])
    all_cards_e = agent.all_card_embedder(card_encoder, unselected_screen_cards)
    single_cards_e = hcat(agent.single_card_embedder(card_encoder, unselected_screen_cards),
                          zeros(Float32, length(agent.single_card_embedder)))
    skip_e = [zeros(Float32, 1, size(single_cards_e, 2)-1) 1]
    if !skip_available
        single_cards_e = single_cards_e[:,1:end-1]
        skip_e = skip_e[:,1:end-1]
    end
    action_weights = agent.policy(vcat(
        repeat(choice_e, 1, size(single_cards_e, 2)),
        repeat(relics_e, 1, size(single_cards_e, 2)),
        repeat(map_e, 1, size(single_cards_e, 2)),
        repeat(deck_e, 1, size(single_cards_e, 2)),
        repeat(all_cards_e, 1, size(single_cards_e, 2)),
        single_cards_e,
        skip_e,
        fill(Float32(bowl_available), 1, size(single_cards_e, 2))))
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    actions = Zygote.ignore() do
        actions = collect(Union{Int,String}, 0:size(single_cards_e, 2)-1)
        if skip_available && !bowl_available
            actions[end] = "skip"
        end
        actions
    end
    Zygote.@ignore @assert length(actions) == length(probabilities)
    actions, probabilities
end

function state_value(agent::DeckAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    local choice_e
    local unselected_screen_cards
    local skip_available = false
    local bowl_available = false
    Zygote.ignore() do
        choice_e = zeros(Float32, 4)
        if gs["screen_type"] == "CARD_REWARD"; choice_e[1] = 1 end
        if gs["screen_type"] == "GRID" && gs["screen_state"]["for_upgrade"]; choice_e[2] = 1 end
        if gs["screen_type"] == "GRID" && gs["screen_state"]["for_transform"]; choice_e[3] = 1 end
        if gs["screen_type"] == "GRID" && gs["screen_state"]["for_purge"]; choice_e[4] = 1 end
        @assert sum(choice_e) in (0, 1)
        if gs["screen_type"] == "GRID"
            screen_cards = gs["screen_state"]["cards"]
            selected_screen_cards = gs["screen_state"]["selected_cards"]
            unselected_screen_cards = filter(c -> !in(c, selected_screen_cards), screen_cards)
            @assert !gs["screen_state"]["any_number"]
        else
            unselected_screen_cards = gs["screen_state"]["cards"]
            skip_available = gs["screen_state"]["skip_available"]
            bowl_available = gs["screen_state"]["bowl_available"]
        end

    end
    relics_e = agent.relics_embedder(relics_encoder(gs["relics"]))
    map_e = agent.map_embedder(map_encoder(sts_state, ra.map_agent.map_node[1], ra.map_agent.map_node[2]))
    deck_e = agent.deck_embedder(card_encoder, gs["deck"])
    all_cards_e = agent.all_card_embedder(card_encoder, unselected_screen_cards)
    only(agent.critic(vcat(
        choice_e,
        relics_e,
        map_e,
        deck_e,
        all_cards_e,
        Float32(skip_available),
        Float32(bowl_available))))
end
