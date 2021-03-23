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
    single_hand_embedder = VanillaNetwork(length(card_encoder)+1, 20, [50]) # +1 for end turn
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
        end
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

function train!(agent::CombatAgent, ra::RootAgent)
    # TODO
end

function action_probabilities(agent::CombatAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    potions_e = agent.potions_embedder(potions_encoder(sts_state))
    relics_e = agent.relics_embedder(relics_encoder(sts_state))
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
        action_cards_encoded = Any[[zeros(length(card_encoder));1]]
        action_monsters_encoded = Any[zeros(length(monster_encoder))]
        for card in playable_hand
            if card[2]["has_target"]
                for monster in attackable_monsters
                    push!(actions, (card[1], monster[1]))
                    push!(action_cards_encoded, [card_encoder(card);0])
                    push!(action_monsters_encoded, monster_encoder(monster))
                end
            else
                push!(actions, (card[1],))
                push!(action_cards_encoded, [card_encoder(card);0])
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
    # TODO
end












function action(agent::CombatAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] in ("MAP", "COMBAT_REWARD", "GAME_OVER")
            reward(agent, sts_state)
        end
        if gs["screen_type"] == "NONE"
            cs = gs["combat_state"]
            if !any(c -> c["is_playable"], cs["hand"]); return "end" end
            reward(agent, sts_state)
            add_state(agent.sars, sts_state)
            aps, playable_hand = action_probabilities(agent, sts_state)
            selected_card = sample(collect(enumerate(playable_hand)), Weights(aps))
            add_action(agent.sars, selected_card[1])
            card_to_play = selected_card[2]
            card_to_play_index = card_to_play[1]
            if card_to_play[2]["has_target"]
                monsters = collect(enumerate(cs["monsters"]))
                attackable_monsters = filter(m -> !m[2]["is_gone"], monsters)
                min_hp = minimum(map(m -> m[2]["current_hp"], attackable_monsters))
                min_hp_monsters = filter(m -> m[2]["current_hp"] == min_hp, attackable_monsters)
                monster_to_attack_index = sample(min_hp_monsters)[1]-1
                return "play $card_to_play_index $monster_to_attack_index"
            end
            return "play $card_to_play_index"
        end
    end
end

function action_value(agent::CombatAgent, sts_state)
    hand = Zygote.ignore(() -> collect(enumerate(sts_state["game_state"]["combat_state"]["hand"])))
    player_embedded = agent.player_embedder(agent.player_encoder(sts_state))
    draw_discard_embedded = agent.draw_discard_embedder(agent.draw_discard_encoder(sts_state))
    monsters_embedded = agent.monsters_embedder(agent.monsters_encoder(sts_state))

    all_hand_cards_encoded = reduce(hcat, map(c -> agent.hand_card_encoder(c[2]), hand))
    Zygote.ignore(() -> @assert size(all_hand_cards_encoded) == (length(agent.hand_card_encoder), length(hand)))
    all_hand_cards_embedded = agent.all_hand_cards_embedder(all_hand_cards_encoded)
    Zygote.ignore(() -> @assert size(all_hand_cards_embedded) == (length(agent.all_hand_cards_embedder), length(hand)))
    all_hand_cards_pooled = maximum(all_hand_cards_embedded, dims=2)
    Zygote.ignore(() -> @assert size(all_hand_cards_pooled) == (length(agent.all_hand_cards_embedder), 1))

    only(agent.critic(vcat(
        player_embedded,
        draw_discard_embedded,
        monsters_embedded,
        all_hand_cards_pooled)))
end

function action_probabilities(agent::CombatAgent, sts_state)
    hand = Zygote.ignore(() -> collect(enumerate(sts_state["game_state"]["combat_state"]["hand"])))
    player_embedded = agent.player_embedder(agent.player_encoder(sts_state))
    draw_discard_embedded = agent.draw_discard_embedder(agent.draw_discard_encoder(sts_state))
    monsters_embedded = agent.monsters_embedder(agent.monsters_encoder(sts_state))

    all_hand_cards_encoded = reduce(hcat, map(c -> agent.hand_card_encoder(c[2]), hand))
    Zygote.ignore(() -> @assert size(all_hand_cards_encoded) == (length(agent.hand_card_encoder), length(hand)))
    all_hand_cards_embedded = agent.all_hand_cards_embedder(all_hand_cards_encoded)
    Zygote.ignore(() -> @assert size(all_hand_cards_embedded) == (length(agent.all_hand_cards_embedder), length(hand)))
    all_hand_cards_pooled = maximum(all_hand_cards_embedded, dims=2)
    Zygote.ignore(() -> @assert size(all_hand_cards_pooled) == (length(agent.all_hand_cards_embedder), 1))

    playable_hand = filter(c -> c[2]["is_playable"], hand)
    playable_hand_encoded = reduce(hcat, map(c -> agent.hand_card_encoder(c[2]), playable_hand))
    Zygote.ignore(() -> @assert size(playable_hand_encoded) == (length(agent.hand_card_encoder), length(playable_hand)))
    playable_hand_embedded = agent.single_hand_card_embedder(playable_hand_encoded)
    Zygote.ignore(() -> @assert size(playable_hand_embedded) == (length(agent.single_hand_card_embedder), length(playable_hand)))
    selection_weights = agent.hand_card_selector(vcat(
        repeat(player_embedded, 1, length(playable_hand)),
        repeat(draw_discard_embedded, 1, length(playable_hand)),
        repeat(monsters_embedded, 1, length(playable_hand)),
        repeat(all_hand_cards_pooled, 1, length(playable_hand)),
        playable_hand_embedded))
    Zygote.ignore(() -> @assert size(selection_weights) == (1, length(playable_hand)))
    softmax(reshape(selection_weights, length(playable_hand))), playable_hand
end

function reward(agent::CombatAgent, sts_state, continuity=1.0f0)
    if awaiting(agent.sars) == sar_reward
        win = sts_state["game_state"]["screen_type"] in ("MAP", "COMBAT_REWARD")
        lose = sts_state["game_state"]["screen_type"] == "GAME_OVER"
        @assert !(win && lose)
        last_hp = agent.sars.states[end]["game_state"]["current_hp"]
        current_hp = sts_state["game_state"]["current_hp"]
        r = current_hp - last_hp
        if win; r+= 10 end
        add_reward(agent.sars, r, continuity)
    end
end

const policy_opt = ADADelta()
const critic_opt = ADADelta()

function train!(agent::CombatAgent, ra::RootAgent, epochs=1000)
    tb_log = TBLogger("tb_logs/CombatAgent")
    sars = fill_q(agent.sars)
    target_agent = deepcopy(agent)
    kl_div_smoother = Smoother()
    for epoch in 1:epochs
        batch = sample(sars, 1000, replace=false)
        prms = params(
            agent.player_embedder,
            agent.draw_discard_embedder,
            agent.monsters_embedder,
            agent.single_hand_card_embedder,
            agent.all_hand_cards_embedder,
            agent.hand_card_selector)
        local kl_divs = Float32[]
        local actual_value = Float32[]
        local estimated_value = Float32[]
        local estimated_advantage = Float32[]
        local entropys = Float32[]
        local explore = Float32[]
        loss, grads = valgrad(prms) do
            -mean(batch) do sar
                online_aps = action_probabilities(agent, sar.state)[1]
                online_ap = online_aps[sar.action]
                target_aps = Zygote.ignore(() -> action_probabilities(target_agent, sar.state)[1])
                target_ap = target_aps[sar.action]
                advantage = sar.q - Zygote.ignore(() -> action_value(target_agent, sar.state))
                Zygote.ignore() do
                    push!(kl_divs, Flux.Losses.kldivergence(online_aps, target_aps))
                    push!(actual_value, online_ap * sar.q)
                    push!(estimated_value, online_ap * action_value(target_agent, sar.state))
                    push!(estimated_advantage, online_ap * advantage)
                    push!(entropys, entropy(online_aps))
                    push!(explore, (maximum(online_aps) - 0.01) > online_ap ? 1 : 0)
                end
                min(
                    (online_ap / target_ap) * advantage,
                    clip(online_ap / target_ap, 0.2) * advantage)
            end
        end
        log_value(tb_log, "train/policy_loss", loss, step=epoch)
        log_value(tb_log, "train/kl_div", mean(kl_divs), step=epoch)
        log_value(tb_log, "train/actual_value", mean(actual_value), step=epoch)
        log_value(tb_log, "train/estimated_value", mean(estimated_value), step=epoch)
        log_value(tb_log, "train/estimated_advantage", mean(estimated_advantage), step=epoch)
        log_value(tb_log, "train/entropy", mean(entropys), step=epoch)
        log_value(tb_log, "train/explore", mean(explore), step=epoch)
        Flux.Optimise.update!(policy_opt, prms, grads)
        if smooth!(kl_div_smoother, mean(kl_divs)) > 0.02; break end
    end
    for epoch in 1:epochs
        batch = sample(sars, 100, replace=false)
        prms = params(agent.critic)
        loss, grads = valgrad(prms) do
            mean(batch) do sar
                predicted_q = action_value(agent, sar.state)
                Zygote.ignore(() -> @assert !isnan(predicted_q))
                actual_q = sar.q
                Zygote.ignore(() -> @assert !isnan(actual_q))
                (predicted_q - actual_q)^2
            end
        end
        @assert !isnan(loss)
        log_value(tb_log, "train/value_loss", loss, step=epoch)
        Flux.Optimise.update!(critic_opt, prms, grads)
    end
    empty!(agent.sars)
end
