module STSAgents
using Encoders
using Flux
using Networks
using StatsBase

export CardPlayingAgent, select_card_to_play

struct CardPlayingAgent
    player_encoder
    draw_discard_encoder
    hand_card_encoder
    hand_card_embedder
    hand_card_selector
end

function CardPlayingAgent()
    player_encoder = make_player_encoder(DefaultGameData)
    draw_discard_encoder = make_draw_discard_encoder(DefaultGameData)
    hand_card_encoder = make_hand_card_encoder(DefaultGameData)
    hand_card_embedder = VanillaNetwork(length(hand_card_encoder), 100, [500])
    hand_card_selector = VanillaNetwork(
        length(draw_discard_encoder) + length(player_encoder) + length(hand_card_encoder) + 100,
        1,
        [500, 500, 500])
    CardPlayingAgent(player_encoder, draw_discard_encoder, hand_card_encoder, hand_card_embedder, hand_card_selector)
end

function select_card_to_play(agent::CardPlayingAgent, sts_state)
    hand = collect(enumerate(sts_state["game_state"]["combat_state"]["hand"]))
    embedded_cards = map(c -> agent.hand_card_embedder(agent.hand_card_encoder(c[2])), hand)
    pooled_cards = maximum(reduce(hcat, embedded_cards), dims=2)
    player_encoded = agent.player_encoder(sts_state)
    draw_discard_encoded = agent.draw_discard_encoder(sts_state)
    playable_hand = filter(c -> c[2]["is_playable"], hand)
    selector_input_separate = map(playable_hand) do hand_card
        vcat(player_encoded, draw_discard_encoded, pooled_cards, agent.hand_card_encoder(hand_card[2]))
    end
    selector_input = reduce(hcat, selector_input_separate)
    selection_weights = reshape(agent.hand_card_selector(selector_input), length(playable_hand))
    selection_probabilities = softmax(selection_weights)
    selected_card = sample(playable_hand, Weights(selection_probabilities))
    return selected_card
end

end # module
