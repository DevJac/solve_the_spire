export DeckAgent, action, train!

mutable struct DeckAgent
    choice_encoder
    policy
    critic
    policy_opt
    critic_opt
    sars
    last_floor_rewarded
end

function DeckAgent()
    choice_encoder = ChoiceEncoder(
        Dict(
            :relics       => VanillaNetwork(length(relics_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :player       => VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50]),
            :deck         => PoolNetwork(length(card_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :map          => VanillaNetwork(length(map_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
        ),
        Dict(
            :card         => VanillaNetwork(length(card_encoder)+4, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :skip         => NullNetwork(),
            :bowl         => NullNetwork()
        ),
        STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_CRITIC_LAYERS)
    DeckAgent(
        choice_encoder,
        policy,
        critic,
        STANDARD_OPTIMIZER(),
        STANDARD_OPTIMIZER(),
        SARS(),
        0)
end

function action(agent::DeckAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            @assert awaiting(agent.sars) == sar_reward || !any(s -> s["game_state"]["seed"] == gs["seed"], agent.sars.states)
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded + floor_partial_credit(ra)
                @assert r >= 0
                agent.last_floor_rewarded = 0
                add_reward(agent.sars, r, 0)
                log_value(ra.tb_log, "DeckAgent/reward", r)
                log_value(ra.tb_log, "DeckAgent/length_sars", length(agent.sars.rewards))
            end
        elseif gs["screen_type"] in ("CARD_REWARD", "GRID")
            if "confirm" in sts_state["available_commands"]
                return "confirm"
            end
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded
                @assert r >= 0
                agent.last_floor_rewarded = gs["floor"]
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
            action = join(actions[action_i], " ")
            @assert isa(action, String)
            action
        end
    end
end

function setup_choice_encoder(agent::DeckAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    ChoiceEncoders.reset!(agent.choice_encoder)
    add_encoded_state(agent.choice_encoder, :relics, relics_encoder(gs["relics"]))
    add_encoded_state(agent.choice_encoder, :player, player_basic_encoder(sts_state))
    add_encoded_state(agent.choice_encoder, :deck, reduce(hcat, map(card_encoder, gs["deck"])))
    add_encoded_state(agent.choice_encoder, :map, map_encoder(sts_state, current_map_node(ra)...))
    draft_upgrade_etc = zeros(Float32, 4)  # Draft, upgrade, transform, purge
    for action in all_valid_actions(sts_state)
        if action[1] == "potion"
            continue
        end
        if action[1] == "choose" && gs["screen_type"] == "CARD_REWARD"
            choice_i = action[2]+1
            if gs["choice_list"][choice_i] == "bowl"
                add_encoded_choice(agent.choice_encoder, :bowl, nothing, action)
            else
                draft_upgrade_etc[1] = 1
                add_encoded_choice(
                    agent.choice_encoder,
                    :card,
                    [draft_upgrade_etc; card_encoder(gs["screen_state"]["cards"][choice_i])],
                    action)
            end
            continue
        end
        if action[1] == "choose" && gs["screen_type"] == "GRID"
            @assert !gs["screen_state"]["any_number"]
            # Not encoding: any_number, selected_card, num_cards
            if gs["screen_state"]["for_upgrade"]; draft_upgrade_etc[2] = 1 end
            if gs["screen_state"]["for_transform"]; draft_upgrade_etc[3] = 1 end
            if gs["screen_state"]["for_purge"]; draft_upgrade_etc[4] = 1 end
            choice_i = action[2]+1
            add_encoded_choice(
                agent.choice_encoder,
                :card,
                [draft_upgrade_etc; card_encoder(gs["screen_state"]["cards"][choice_i])],
                action)
            continue
        end
        if action[1] == "skip"
            add_encoded_choice(agent.choice_encoder, :skip, nothing, action)
            continue
        end
        @error "Unhandled action" action
        throw("Unhandled action")
    end
end

function action_probabilities(agent::DeckAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    choices_encoded, actions = encode_choices(agent.choice_encoder)
    action_weights = agent.policy(choices_encoded)
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    Zygote.@ignore @assert length(actions) == length(probabilities)
    actions, probabilities
end

function state_value(agent::DeckAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    state_encoded = encode_state(agent.choice_encoder)
    only(agent.critic(state_encoded))
end

function train!(agent::DeckAgent, ra::RootAgent)
    train_log = TBLogger("tb_logs/train_DeckAgent")
    train!(train_log, agent, ra)
end
