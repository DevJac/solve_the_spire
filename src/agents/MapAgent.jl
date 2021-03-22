export MapAgent, action, train!

struct MapAgent
    player_embedder
    deck_embedder
    relics_embedder
    potions_embedder
    map_embedder
    policy
    critic
    sars
end
function MapAgent()
    player_embedder = VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50])
    deck_embedder = PoolNetwork(length(card_encoder), 20, [50])
    relics_embedder = VanillaNetwork(length(relics_encoder), 20, [50])
    potions_embedder = VanillaNetwork(length(potions_encoder), 20, [50])
    map_embedder = VanillaNetwork(length(map_embedder), 20, [50])
    policy = VanillaNetwork(
        sum(
            length(player_embedder),
            length(deck_embedder),
            length(relics_embedder),
            length(potions_embedder),
            length(map_embedder)),
        1, [200, 50, 50])
    critic = VanillaNetwork(
        sum(
            length(player_embedder),
            length(deck_embedder),
            length(relics_embedder),
            length(potions_embedder)),
        1, [200, 50, 50])
    MapAgent(
        player_embedder,
        deck_embedder,
        relics_embedder,
        potions_embedder,
        map_embedder,
        policy,
        critic,
        SARS())
end

function action(agent::MapAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            if awaiting(agent.sars) == sar_reward
                add_reward(agent.sars, gs["floor"], 0)
            end
        elseif gs["screen_type"] == "MAP"
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            action = actions[action_i]
            return "choose $action"
        else
            if awaiting(agent.sars) == sar_reward
                add_reward(agent.sars, 0, 0)
            end
        end
    end
end

function train!(agent::MapAgent, ra::RootAgent)
end

function action_probabilities(agent::MapAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    player_e = agent.player_embedder(player_basic_encoder(sts_state))
    deck_e = agent.deck_embedder(card_encoder, gs["deck"])
    relics_e = agent.relics_embedder(relics_encoder(sts_state))
    potions_e = agent.potions_embedder(potions_encoder(sts_state))
    map_e = agent.map_embedder(node -> map_encoder(sts_state, node["x"], node["y"]), gs["screen_state"]["next_nodes"])
    Zygote.ignore() do
        @assert size(player_e) == (length(player_basic_encoder),)
        @assert size(deck_e) == (20,)
        @assert size(relics_e) == (20,)
        @assert size(potions_e) == (20,)
        @assert size(map_e) == (20, length(gs["screen_state"]["next_nodes"]))
    end
    action_weights = agent.policy(vcat(
        repeat(player_e, 1, size(map_e)[2]),
        repeat(deck_e, 1, size(map_e)[2]),
        repeat(relics_e, 1, size(map_e)[2]),
        repeat(potions_e, 1, size(map_e)[2]),
        map_e))
    Zygote.ignore() do
        actions = collect(0:length(gs["screen_state"]["next_nodes"])-1)
        probabilities = softmax(reshape(action_weights, length(action_weights)))
        @assert length(actions) == length(probabilities)
        actions, probabilities
    end
end

function state_value(agent::MapAgent, ra::RootAgent, sts_state)
end
