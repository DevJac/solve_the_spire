module STSAgents
using Encoders
using Flux
using Networks
using SARSM
using StatsBase
using TensorBoardLogger
using Utils
using Zygote

export action, reward, train!
export CardPlayingAgent

struct CardPlayingAgent
    player_encoder
    draw_discard_encoder
    hand_card_encoder
    hand_card_embedder
    hand_card_selector
    critic_hand_card_embedder
    critic
    sars
end

function CardPlayingAgent()
    player_encoder = make_player_encoder(DefaultGameData)
    draw_discard_encoder = make_draw_discard_encoder(DefaultGameData)
    hand_card_encoder = make_hand_card_encoder(DefaultGameData)
    hand_card_embedder = VanillaNetwork(length(hand_card_encoder), 100, [200, 200])
    hand_card_selector = VanillaNetwork(
        length(player_encoder) + length(draw_discard_encoder) + 100 + length(hand_card_encoder),
        1,
        [200, 200, 200, 200])
    critic_hand_card_embedder = VanillaNetwork(length(hand_card_encoder), 100, [200, 200])
    critic = VanillaNetwork(
        length(player_encoder) + length(draw_discard_encoder) + 100,
        1,
        [200, 200, 200, 200])
    CardPlayingAgent(
        player_encoder,
        draw_discard_encoder,
        hand_card_encoder,
        hand_card_embedder,
        hand_card_selector,
        critic_hand_card_embedder,
        critic,
        SARS())
end

function action_value(agent::CardPlayingAgent, sts_state)
    hand = Zygote.ignore(() -> collect(enumerate(sts_state["game_state"]["combat_state"]["hand"])))
    player_encoded = Zygote.ignore(() -> agent.player_encoder(sts_state))
    draw_discard_encoded = Zygote.ignore(() -> agent.draw_discard_encoder(sts_state))
    embedded_cards = map(c -> agent.critic_hand_card_embedder(Zygote.ignore(() -> agent.hand_card_encoder(c[2]))), hand)
    pooled_cards = maximum(reduce(hcat, embedded_cards), dims=2)
    critic_input = vcat(player_encoded, draw_discard_encoded, pooled_cards)
    agent.critic(critic_input)
end

function action_probabilities(agent::CardPlayingAgent, sts_state)
    hand = Zygote.ignore(() -> collect(enumerate(sts_state["game_state"]["combat_state"]["hand"])))
    playable_hand = Zygote.ignore(() -> filter(c -> c[2]["is_playable"], hand))
    player_encoded = Zygote.ignore(() -> agent.player_encoder(sts_state))
    draw_discard_encoded = Zygote.ignore(() -> agent.draw_discard_encoder(sts_state))
    embedded_cards = map(c -> agent.hand_card_embedder(Zygote.ignore(() -> agent.hand_card_encoder(c[2]))), hand)
    pooled_cards = maximum(reduce(hcat, embedded_cards), dims=2)
    selector_input_separate = map(playable_hand) do hand_card
        vcat(player_encoded, draw_discard_encoded, pooled_cards, Zygote.ignore(() -> agent.hand_card_encoder(hand_card[2])))
    end
    selector_input = reduce(hcat, selector_input_separate)
    selection_weights = reshape(agent.hand_card_selector(selector_input), length(playable_hand))
    softmax(selection_weights), playable_hand
end

function action(agent::CardPlayingAgent, sts_state)
    add_state(agent.sars, sts_state)
    aps, playable_hand = action_probabilities(agent, sts_state)
    selected_card = sample(collect(enumerate(playable_hand)), Weights(aps))
    add_action(agent.sars, selected_card[1])
    return selected_card[2]
end

function reward(agent::CardPlayingAgent, sts_state, continuity=1.0f0)
    if awaiting(agent.sars) == sar_reward
        win = sts_state["game_state"]["screen_type"] in ("MAP", "COMBAT_REWARD")
        lose = sts_state["game_state"]["screen_type"] == "GAME_OVER"
        @assert !(win && lose)
        last_hp = agent.sars.states[end]["game_state"]["current_hp"]
        current_hp = sts_state["game_state"]["current_hp"]
        r = current_hp - last_hp
        if win; r+= 100 end
        add_reward(agent.sars, r, continuity)
    end
end

function valgrad(f, x...)
    val, back = pullback(f, x...)
    val, back(1)
end

function train!(agent::CardPlayingAgent, epochs=100)
    tb_log = TBLogger("tb_logs/card_playing_agent")
    critic_opt = RMSProp()
    policy_opt = RMSProp()
    sars = fill_q(agent.sars)
    for epoch in 1:epochs
        batch = sample(sars, 100)
        prms = params(agent.critic_hand_card_embedder, agent.critic)
        loss, grads = valgrad(prms) do
            mean(batch) do sar
                predicted_q = only(action_value(agent, sar.state))
                actual_q = sar.q
                (predicted_q - actual_q)^2
            end
        end
        log_value(tb_log, "train/value_loss", loss, step=epoch)
        Flux.Optimise.update!(critic_opt, prms, grads)
    end
    target_agent = deepcopy(agent)
    for epoch in 1:epochs
        batch = sample(sars, 100)
        prms = params(agent.hand_card_embedder, agent.hand_card_selector)
        local kl_divs = Float32[]
        loss, grads = valgrad(prms) do
            -mean(batch) do sar
                online_aps = action_probabilities(agent, sar.state)[1]
                online_ap = online_aps[sar.action]
                target_aps = action_probabilities(target_agent, sar.state)[1]
                target_ap = target_aps[sar.action]
                advantage = sar.q - only(action_value(target_agent, sar.state))
                Zygote.ignore(() -> push!(kl_divs, Flux.Losses.kldivergence(online_aps, target_aps)))
                min(
                    (online_ap / target_ap) * advantage,
                    clip(online_ap / target_ap, 0.2) * advantage)
            end
        end
        log_value(tb_log, "train/policy_loss", loss, step=epoch)
        log_value(tb_log, "train/kl_div", mean(kl_divs), step=epoch)
        Flux.Optimise.update!(policy_opt, prms, grads)
    end
    empty!(agent.sars)
end

end # module
