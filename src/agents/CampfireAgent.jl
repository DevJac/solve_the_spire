export CampfireAgent, action, train!

mutable struct CampfireAgent
    choice_encoder
    policy
    critic
    sars
    last_floor_rewarded
end

function CampfireAgent()
    choice_encoder = ChoiceEncoder(
        Dict(
            :potions      => VanillaNetwork(length(potions_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :relics       => VanillaNetwork(length(relics_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :player       => VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50]),
            :deck         => PoolNetwork(length(card_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :map          => VanillaNetwork(length(map_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
        ),
        Dict(
            :rest         => NullNetwork(),
            :smith        => NullNetwork(),
            :recall       => NullNetwork(),
            :lift         => NullNetwork(),
            :toke         => NullNetwork(),
            :dig          => NullNetwork()
        ),
        STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_CRITIC_LAYERS)
    CampfireAgent(
        choice_encoder,
        policy,
        critic,
        SARS(),
        0)
end

function action(agent::CampfireAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            @assert awaiting(agent.sars) == sar_reward || !any(s -> s["game_state"]["seed"] == gs["seed"], agent.sars.states)
            if awaiting(agent.sars) == sar_reward
                r = floor_adjusted(gs["floor"] + floor_partial_credit(ra)) - agent.last_floor_rewarded
                @assert r >= 0
                agent.last_floor_rewarded = 0
                add_reward(agent.sars, r, 0)
                log_value(ra.tb_log, "CampfireAgent/reward", r)
                log_value(ra.tb_log, "CampfireAgent/length_sars", length(agent.sars.rewards))
            end
        elseif gs["screen_type"] == "REST"
            if "proceed" in sts_state["available_commands"]
                return "proceed"
            end
            if awaiting(agent.sars) == sar_reward
                r = floor_adjusted(gs["floor"]) - agent.last_floor_rewarded
                @assert r >= 0
                agent.last_floor_rewarded = floor_adjusted(gs["floor"])
                add_reward(agent.sars, r)
                log_value(ra.tb_log, "CampfireAgent/reward", r)
                log_value(ra.tb_log, "CampfireAgent/length_sars", length(agent.sars.rewards))
            end
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            log_value(ra.tb_log, "CampfireAgent/state_value", state_value(agent, ra, sts_state))
            log_value(ra.tb_log, "CampfireAgent/step_explore", explore_odds(probabilities))
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            action = join(actions[action_i], " ")
            @assert isa(action, String)
            action
        end
    end
end

function setup_choice_encoder(agent::CampfireAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    ChoiceEncoders.reset!(agent.choice_encoder)
    add_encoded_state(agent.choice_encoder, :potions, potions_encoder(gs["potions"]))
    add_encoded_state(agent.choice_encoder, :relics, relics_encoder(gs["relics"]))
    add_encoded_state(agent.choice_encoder, :player, player_basic_encoder(sts_state))
    add_encoded_state(agent.choice_encoder, :deck, reduce(hcat, map(card_encoder, gs["deck"])))
    add_encoded_state(agent.choice_encoder, :map, map_encoder(sts_state, current_map_node(ra)...))
    for action in all_valid_actions(sts_state)
        if action[1] == "potion"
            continue
        end
        if action[1] == "choose"
            choice_i = action[2]+1
            choice = gs["choice_list"][choice_i]
            if choice == "rest"
                add_encoded_choice(agent.choice_encoder, :rest, nothing, action)
                continue
            end
            if choice == "smith"
                add_encoded_choice(agent.choice_encoder, :smith, nothing, action)
                continue
            end
            if choice == "recall"
                add_encoded_choice(agent.choice_encoder, :recall, nothing, action)
                continue
            end
            if choice == "lift"
                add_encoded_choice(agent.choice_encoder, :lift, nothing, action)
                continue
            end
            if choice == "toke"
                add_encoded_choice(agent.choice_encoder, :toke, nothing, action)
                continue
            end
            if choice == "dig"
                add_encoded_choice(agent.choice_encoder, :dig, nothing, action)
                continue
            end
        end
        @error "Unhandled action" action
        throw("Unhandled action")
    end
end

function action_probabilities(agent::CampfireAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    choices_encoded, actions = encode_choices(agent.choice_encoder)
    action_weights = agent.policy(choices_encoded)
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    Zygote.@ignore @assert length(actions) == length(probabilities)
    actions, probabilities
end

function state_value(agent::CampfireAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    state_encoded = encode_state(agent.choice_encoder)
    only(agent.critic(state_encoded))
end

function train!(agent::CampfireAgent, ra::RootAgent)
    train_log = TBLogger("tb_logs/train_CampfireAgent")
    train!(train_log, agent, ra)
end
