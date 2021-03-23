export CombatAgent, action, train!

struct CombatAgent
    potions_embedder
    relics_embedder
    player_embedder
    draw_embedder
    discard_embedder
    all_hand_embedder
    all_monster_embedder
    single_hand_embedder
    single_monster_embedder
    policy
    critic
    policy_opt
    critic_opt
    sars
end

function CombatAgent()
    potions_embedder = VanillaNetwork(length(potions_encoder), 20, [50])
    relics_embedder = VanillaNetwork(length(relics_encoder), 20, [50])
    player_embedder = VanillaNetwork(length(player_combat_encoder), 20, [50])
    draw_embedder = PoolNetwork(length(card_encoder), 20, [50])
    discard_embedder = PoolNetwork(length(card_encoder), 20, [50])
    all_hand_embedder = PoolNetwork(length(card_encoder), 20, [50])
    all_monster_embedder = PoolNetwork(length(monster_encoder), 20, [50])
    single_hand_embedder = VanillaNetwork(length(card_encoder), 20, [50])
    single_monster_embedder = VanillaNetwork(length(monster_encoder), 20, [50])
    policy = VanillaNetwork(
        sum(length, [
            potions_embedder,
            relics_embedder,
            player_embedder,
            draw_embedder,
            discard_embedder,
            all_hand_embedder,
            all_monster_embedder,
            single_hand_embedder,
            single_monster_embedder]),
        1, [200, 50, 50])
    critic = VanillaNetwork(
        sum(length, [
            potions_embedder,
            relics_embedder,
            player_embedder,
            draw_embedder,
            discard_embedder,
            all_hand_embedder,
            all_monster_embedder]),
        1, [200, 50, 50])
    CombatAgent(
        potions_embedder,
        relics_embedder,
        player_embedder,
        draw_embedder,
        discard_embedder,
        all_hand_embedder,
        all_monster_embedder,
        single_hand_embedder,
        single_monster_embedder,
        policy,
        critic,
        ADADelta(),
        ADADelta(),
        SARS())
end

function action(agent::CombatAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] in ("NONE", "COMBAT_REWARD", "MAP", "GAME_OVER") && awaiting(agent.sars) == sar_reward
            win = gs["screen_type"] in ("COMBAT_REWARD", "MAP")
            lose = gs["screen_type"] == "GAME_OVER"
            @assert !(win && lose)
            last_hp = agent.sars.states[end]["game_state"]["current_hp"]
            current_hp = gs["current_hp"]
            r = current_hp - last_hp
            if win; r+= 10 end
            add_reward(agent.sars, r, win || lose ? 0 : 1)
            log_value(ra.tb_log, "CombatAgent/length_sars", length(agent.sars.rewards))
        end
        if gs["screen_type"] == "NONE"
            log_value(ra.tb_log, "CombatAgent/state_value", state_value(agent, ra, sts_state))
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            action = actions[action_i]
            if length(action) == 0
                return "end"
            elseif length(action) == 1
                return "play $(action[1])"
            elseif length(action) == 2
                return "play $(action[1]) $(action[2])"
            else
                @error "Unknown CombatAgent action" action
                throw("Unknown CombatAgent action")
            end
        end
    end
end

function train!(agent::CombatAgent, ra::RootAgent, epochs=1000)
    train_log = TBLogger("tb_logs/train_CombatAgent")
    sars = fill_q(agent.sars)
    log_histogram(ra.tb_log, "CombatAgent/rewards", map(sar -> sar.reward, sars))
    log_histogram(ra.tb_log, "CombatAgent/q", map(sar -> sar.q, sars))
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
            agent.potions_embedder,
            agent.relics_embedder,
            agent.player_embedder,
            agent.draw_embedder,
            agent.discard_embedder,
            agent.all_hand_embedder,
            agent.all_monster_embedder,
            agent.single_hand_embedder,
            agent.single_monster_embedder,
            agent.policy)
        loss, grads = valgrad(prms) do
            -mean(batch) do sar
                target_aps = action_probabilities(target_agent, ra, sar.state)[2]
                target_ap = target_aps[sar.action]
                online_aps = action_probabilities(agent, ra, sar.state)[2]
                online_ap = online_aps[sar.action]
                advantage = sar.q - Zygote.ignore(() -> state_value(target_agent, ra, sar.state))
                Zygote.ignore() do
                    push!(kl_divs, Flux.Losses.kldivergence(online_aps, target_aps))
                    push!(actual_value, online_ap * sar.q)
                    push!(estimated_value, online_ap * state_value(target_agent, ra, sar.state))
                    push!(estimated_advantage, online_ap * advantage)
                    push!(entropys, entropy(online_aps))
                    push!(explore, (maximum(online_aps) - 0.01) > online_ap ? 1 : 0)
                end
                min(
                    (online_ap / target_ap) * advantage,
                    clip(online_ap / target_ap, 0.2) * advantage)
            end
        end
        log_value(train_log, "policy/loss", loss, step=epoch)
        log_value(train_log, "policy/kl_div", mean(kl_divs), step=epoch)
        log_value(train_log, "policy/actual_value", mean(actual_value), step=epoch)
        log_value(train_log, "policy/estimated_value", mean(estimated_value), step=epoch)
        log_value(train_log, "policy/estimated_advantage", mean(estimated_advantage), step=epoch)
        log_value(train_log, "policy/entropy", mean(entropys), step=epoch)
        log_value(train_log, "policy/explore", mean(explore), step=epoch)
        Flux.Optimise.update!(agent.policy_opt, prms, grads)
        if smooth!(kl_div_smoother, mean(kl_divs)) > 0.01; break end
        empty!(kl_divs); empty!(actual_value); empty!(estimated_value); empty!(estimated_advantage)
        empty!(entropys); empty!(explore)
    end
    log_value(ra.tb_log, "CombatAgent/policy_loss", loss)
    log_value(ra.tb_log, "CombatAgent/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "CombatAgent/actual_value", mean(actual_value))
    log_value(ra.tb_log, "CombatAgent/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "CombatAgent/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "CombatAgent/entropy", mean(entropys))
    log_value(ra.tb_log, "CombatAgent/explore", mean(explore))
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
        log_value(train_log, "critic/loss", loss, step=epoch)
        Flux.Optimise.update!(agent.critic_opt, prms, grads)
    end
    log_value(ra.tb_log, "CombatAgent/critic_loss", loss)
    empty!(agent.sars)
end

function action_probabilities(agent::CombatAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    potions_e = agent.potions_embedder(potions_encoder(gs["potions"]))
    relics_e = agent.relics_embedder(relics_encoder(gs["relics"]))
    player_e = agent.player_embedder(player_combat_encoder(sts_state))
    draw_e = agent.draw_embedder(card_encoder, gs["combat_state"]["draw_pile"])
    discard_e = agent.draw_embedder(card_encoder, gs["combat_state"]["discard_pile"])
    all_hand_e = agent.all_hand_embedder(card_encoder, gs["combat_state"]["hand"])
    all_monster_e = agent.all_monster_embedder(monster_encoder, gs["combat_state"]["monsters"])
    local actions
    local action_cards_encoded
    local action_monsters_encoded
    local expected_action_length
    Zygote.ignore() do
        @assert size(potions_e) == (20,)
        @assert size(relics_e) == (20,)
        @assert size(player_e) == (20,)
        @assert size(draw_e) == (20,)
        @assert size(discard_e) == (20,)
        @assert size(all_hand_e) == (20,)
        @assert size(all_monster_e) == (20,)
        hand = collect(enumerate(gs["combat_state"]["hand"]))
        playable_hand = filter(c -> c[2]["is_playable"], hand)
        monsters = collect(zip(0:99, gs["combat_state"]["monsters"]))
        attackable_monsters = filter(m -> !m[2]["is_gone"], monsters)
        actions = Any[()]
        action_cards_encoded = Any[zeros(length(card_encoder))]
        action_monsters_encoded = Any[zeros(length(monster_encoder))]
        for card in playable_hand
            if card[2]["has_target"]
                for monster in attackable_monsters
                    push!(actions, (card[1], monster[1]))
                    push!(action_cards_encoded, card_encoder(card[2]))
                    push!(action_monsters_encoded, monster_encoder(monster[2]))
                end
            else
                push!(actions, (card[1],))
                push!(action_cards_encoded, card_encoder(card[2]))
                push!(action_monsters_encoded, zeros(length(monster_encoder)))
            end
        end
        @assert length(actions) == length(action_cards_encoded) == length(action_monsters_encoded)
        expected_action_length = (
            count(c -> c[2]["has_target"], playable_hand) * (length(attackable_monsters)-1) +
            length(playable_hand) + 1)
        @assert length(actions) == expected_action_length
    end
    action_e = vcat(
        agent.single_hand_embedder(reduce(hcat, action_cards_encoded)),
        agent.single_monster_embedder(reduce(hcat, action_monsters_encoded)))
    Zygote.@ignore @assert size(action_e)[2] == expected_action_length
    action_weights = agent.policy(vcat(
        repeat(potions_e, 1, size(action_e)[2]),
        repeat(relics_e, 1, size(action_e)[2]),
        repeat(player_e, 1, size(action_e)[2]),
        repeat(draw_e, 1, size(action_e)[2]),
        repeat(discard_e, 1, size(action_e)[2]),
        repeat(all_hand_e, 1, size(action_e)[2]),
        repeat(all_monster_e, 1, size(action_e)[2]),
        action_e))
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    Zygote.ignore() do
        @assert length(actions) == length(probabilities) == expected_action_length
        actions, probabilities
    end
end

function state_value(agent::CombatAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    potions_e = agent.potions_embedder(potions_encoder(gs["potions"]))
    relics_e = agent.relics_embedder(relics_encoder(gs["relics"]))
    player_e = agent.player_embedder(player_combat_encoder(sts_state))
    draw_e = agent.draw_embedder(card_encoder, gs["combat_state"]["draw_pile"])
    discard_e = agent.draw_embedder(card_encoder, gs["combat_state"]["discard_pile"])
    all_hand_e = agent.all_hand_embedder(card_encoder, gs["combat_state"]["hand"])
    all_monster_e = agent.all_monster_embedder(monster_encoder, gs["combat_state"]["monsters"])
    only(agent.critic(vcat(
        potions_e,
        relics_e,
        player_e,
        draw_e,
        discard_e,
        all_hand_e,
        all_monster_e)))
end
