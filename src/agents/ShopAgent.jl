export ShopAgent, action, train!

mutable struct ShopAgent
    choice_encoder
    policy
    critic
    policy_opt
    critic_opt
    sars
    last_floor_rewarded
    last_visited_shop_floor
end

function ShopAgent()
    choice_encoder = ChoiceEncoder(
        Dict(
            :potions        => VanillaNetwork(length(potions_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :relics         => VanillaNetwork(length(relics_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :player_basic   => VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50]),
            :deck           => PoolNetwork(length(card_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
        ),
        Dict(
            :buy_card       => VanillaNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :buy_relic      => VanillaNetwork(length(relics_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :buy_potion     => VanillaNetwork(length(potions_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :discard_potion => VanillaNetwork(length(potions_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :purge_card     => VanillaNetwork(1, 1, [50]),
            :leave          => NullNetwork()
        ),
        STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_CRITIC_LAYERS)
    ShopAgent(
        choice_encoder,
        policy,
        critic,
        STANDARD_OPTIMIZER(),
        STANDARD_OPTIMIZER(),
        SARS(),
        0, 0)
end

function action(agent::ShopAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            agent.last_visited_shop_floor = 0
            @assert awaiting(agent.sars) == sar_reward || !any(s -> s["game_state"]["seed"] == gs["seed"], agent.sars.states)
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded + floor_partial_credit(ra)
                @assert r >= 0
                agent.last_floor_rewarded = 0
                add_reward(agent.sars, r, 0)
                log_value(ra.tb_log, "ShopAgent/reward", r)
                log_value(ra.tb_log, "ShopAgent/length_sars", length(agent.sars.rewards))
            end
        elseif gs["screen_type"] in ("SHOP_ROOM", "SHOP_SCREEN")
            if gs["screen_type"] == "SHOP_ROOM"
                if gs["floor"] == agent.last_visited_shop_floor
                    return "proceed"
                end
                agent.last_visited_shop_floor = gs["floor"]
                return "choose 0"
            end
            @assert gs["screen_type"] == "SHOP_SCREEN"
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded
                @assert r >= 0
                agent.last_floor_rewarded = gs["floor"]
                add_reward(agent.sars, r)
                log_value(ra.tb_log, "ShopAgent/reward", r)
                log_value(ra.tb_log, "ShopAgent/length_sars", length(agent.sars.rewards))
            end
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            log_value(ra.tb_log, "ShopAgent/state_value", state_value(agent, ra, sts_state))
            log_value(ra.tb_log, "ShopAgent/step_explore", explore_odds(probabilities))
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            action = join(actions[action_i], " ")
            @assert isa(action, String)
            action
        end
    end
end

function setup_choice_encoder(agent::ShopAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    ChoiceEncoders.reset!(agent.choice_encoder)
    add_encoded_state(agent.choice_encoder, :potions, potions_encoder(gs["potions"]))
    add_encoded_state(agent.choice_encoder, :relics, relics_encoder(gs["relics"]))
    add_encoded_state(agent.choice_encoder, :player_basic, player_basic_encoder(sts_state))
    add_encoded_state(agent.choice_encoder, :deck, reduce(hcat, map(card_encoder, gs["deck"])))
    for action in all_valid_actions(sts_state)
        if action[1] == "potion"
            continue
        elseif action[1] == "leave"
            add_encoded_choice(agent.choice_encoder, :leave, nothing, action)
            continue
        elseif action[1] == "choose"
            choice_i = action[2]+1
            choice_name = gs["choice_list"][choice_i]
            if choice_name == "purge"
                add_encoded_choice(agent.choice_encoder, :purge_card, [gs["screen_state"]["purge_cost"]], action)
                continue
            end
            sort_price(a) = sort(a, by=b -> b["price"])
            matching_cards = sort_price(filter(c -> lowercase(c["name"]) == choice_name, gs["screen_state"]["cards"]))
            matching_relics = sort_price(filter(r -> lowercase(r["name"]) == choice_name, gs["screen_state"]["relics"]))
            matching_potions = sort_price(filter(p -> lowercase(p["name"]) == choice_name, gs["screen_state"]["potions"]))
            @assert length(matching_potions) <= 1 || matching_potions[1]["price"] <= matching_potions[2]["price"]
            if !isempty(matching_cards)
                matching_card = matching_cards[1]
                add_encoded_choice(
                    agent.choice_encoder,
                    :buy_card,
                    [matching_card["price"]; card_encoder(matching_card)],
                    action)
                continue
            end
            if !isempty(matching_relics)
                matching_relic = matching_relics[1]
                if matching_relic["id"] == "PrismaticShard"; continue end
                add_encoded_choice(
                    agent.choice_encoder,
                    :buy_relic,
                    [matching_relic["price"]; relics_encoder([matching_relic])],
                    action)
                continue
            end
            if !isempty(matching_potions)
                if !any(p -> p["id"] == "Potion Slot", gs["potions"])
                    for (potion_i, potion) in enumerate(gs["potions"])
                        use_discard = potion["can_use"] ? "use" : "discard"
                        add_encoded_choice(
                            agent.choice_encoder,
                            :discard_potion,
                            potions_encoder([potion]),
                            ("potion", use_discard, potion_i-1))
                    end
                    continue
                end
                matching_potion = matching_potions[1]
                add_encoded_choice(
                    agent.choice_encoder,
                    :buy_potion,
                    [matching_potion["price"]; potions_encoder([matching_potion])],
                    action)
                continue
            end
        end
        @error "Unhandled action" action
        throw("Unhandled action")
    end
end

function action_probabilities(agent::ShopAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    choices_encoded, actions = encode_choices(agent.choice_encoder)
    action_weights = agent.policy(choices_encoded)
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    Zygote.@ignore @assert length(actions) == length(probabilities)
    actions, probabilities
end

function state_value(agent::ShopAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    state_encoded = encode_state(agent.choice_encoder)
    only(agent.critic(state_encoded))
end

function train!(agent::ShopAgent, ra::RootAgent)
    train_log = TBLogger("tb_logs/train_ShopAgent")
    train!(train_log, agent, ra)
end
