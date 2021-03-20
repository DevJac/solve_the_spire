export CombatAgent, action, train!

struct CombatAgent
    player_encoder
    player_embedder

    draw_discard_encoder
    draw_discard_embedder

    monsters_encoder
    monsters_embedder

    hand_card_encoder
    single_hand_card_embedder
    all_hand_cards_embedder

    hand_card_selector
    critic
    sars
end

function CombatAgent()
    player_encoder = make_player_encoder(DefaultGameData)
    player_embedder = VanillaNetwork(length(player_encoder), 10, [50])

    draw_discard_encoder = make_draw_discard_encoder(DefaultGameData)
    draw_discard_embedder = VanillaNetwork(length(draw_discard_encoder), 10, [50])

    monsters_encoder = make_monsters_encoder(DefaultGameData)
    monsters_embedder = VanillaNetwork(length(monsters_encoder), 10, [50])

    hand_card_encoder = make_hand_card_encoder(DefaultGameData)
    single_hand_card_embedder = VanillaNetwork(length(hand_card_encoder), 10, [50])
    all_hand_cards_embedder = VanillaNetwork(length(hand_card_encoder), 10, [50])

    hand_card_selector = VanillaNetwork(
        sum(length, [
            player_embedder,
            draw_discard_embedder,
            monsters_embedder,
            single_hand_card_embedder,
            all_hand_cards_embedder]),
        1,
        [50, 50, 50, 50, 50])
    critic = VanillaNetwork(
        sum(length, [
            player_embedder,
            draw_discard_embedder,
            monsters_embedder,
            all_hand_cards_embedder]),
        1,
        [40, 40, 40, 40, 40])
    CombatAgent(
        player_encoder,
        player_embedder,

        draw_discard_encoder,
        draw_discard_embedder,

        monsters_encoder,
        monsters_embedder,

        hand_card_encoder,
        single_hand_card_embedder,
        all_hand_cards_embedder,

        hand_card_selector,
        critic,
        SARS())
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

function valgrad(f, x...)
    val, back = pullback(f, x...)
    val, back(1)
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
