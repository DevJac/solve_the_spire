export RewardAgent, action, train!

mutable struct RewardAgent
    choice_encoder
    policy
    critic
    sars
    last_floor_rewarded
    last_card_reward  # (floor, last card reward chosen, last card reward count)
end

function RewardAgent()
    choice_encoder = ChoiceEncoder(
        Dict(
            :potions      => VanillaNetwork(length(potions_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :relics       => VanillaNetwork(length(relics_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :player       => VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50]),
            :deck         => PoolNetwork(length(card_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :map          => VanillaNetwork(length(map_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
        ),
        Dict(
            :give_up_potion      => VanillaNetwork(length(potions_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :choose_relic        => VanillaNetwork(length(relics_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :choose_sapphire_key => NullNetwork()
        ),
        STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_CRITIC_LAYERS)
    RewardAgent(
        choice_encoder,
        policy,
        critic,
        SARS(),
        0, (0, 0, 0))
end

function action(agent::RewardAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            @assert awaiting(agent.sars) == sar_reward || !any(s -> s["game_state"]["seed"] == gs["seed"], agent.sars.states)
            agent.last_card_reward = (0, 0, 0)
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded + floor_partial_credit(ra)
                @assert r >= 0
                agent.last_floor_rewarded = 0
                add_reward(agent.sars, r, 0)
                log_value(ra.tb_log, "RewardAgent/reward", r)
                log_value(ra.tb_log, "RewardAgent/length_sars", length(agent.sars.rewards))
            end
        elseif gs["screen_type"] in ("COMBAT_REWARD", "BOSS_REWARD", "CHEST")
            if gs["screen_type"] == "CHEST"
                # TODO: Give agent a choice if player has a cursed key
                return "choose 0"
            end
            if gs["screen_type"] == "COMBAT_REWARD"
                # The conditions below will choose all no-brainer rewards,
                # and at least look at each card reward; what's left will be
                # decided by the neural networks.
                card_reward_count = count(r -> r["reward_type"] == "CARD", gs["screen_state"]["rewards"])
                card_i = 0
                for (i, reward) in enumerate(gs["screen_state"]["rewards"])
                    if reward["reward_type"] in ("GOLD", "STOLEN_GOLD", "EMERALD_KEY")
                        return "choose $(i-1)"
                    end
                    if reward["reward_type"] == "CARD"
                        card_i += 1
                        if agent.last_card_reward[1] != gs["floor"]
                            agent.last_card_reward = (gs["floor"], 0, card_reward_count)
                        end
                        if card_i > agent.last_card_reward[2] || card_reward_count < agent.last_card_reward[3]
                            agent.last_card_reward = (gs["floor"], card_i, card_reward_count)
                            return "choose $(i-1)"
                        end
                    end
                    if reward["reward_type"] == "POTION" && any(p -> p["id"] == "Potion Slot", gs["potions"])
                        return "choose $(i-1)"
                    end
                    if (reward["reward_type"] == "RELIC" &&
                        !any(r -> r["reward_type"] == "SAPPHIRE_KEY", gs["screen_state"]["rewards"]))
                        return "choose $(i-1)"
                    end
                end
                if !in("choose", sts_state["available_commands"]) || all(c -> c == "card", gs["choice_list"])
                    return "proceed"
                end
            end
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded
                @assert r >= 0
                agent.last_floor_rewarded = gs["floor"]
                add_reward(agent.sars, r)
                log_value(ra.tb_log, "RewardAgent/reward", r)
                log_value(ra.tb_log, "RewardAgent/length_sars", length(agent.sars.rewards))
            end
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            log_value(ra.tb_log, "RewardAgent/state_value", state_value(agent, ra, sts_state))
            log_value(ra.tb_log, "RewardAgent/step_explore", explore_odds(probabilities))
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            action = join(actions[action_i], " ")
            @assert isa(action, String)
            action
        end
    end
end

function setup_choice_encoder(agent::RewardAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    ChoiceEncoders.reset!(agent.choice_encoder)
    add_encoded_state(agent.choice_encoder, :potions, potions_encoder(gs["potions"]))
    add_encoded_state(agent.choice_encoder, :relics, relics_encoder(gs["relics"]))
    add_encoded_state(agent.choice_encoder, :player, player_basic_encoder(sts_state))
    add_encoded_state(agent.choice_encoder, :deck, reduce(hcat, map(card_encoder, gs["deck"])))
    add_encoded_state(agent.choice_encoder, :map, map_encoder(sts_state, current_map_node(ra)...))
    sapphire_key_offered = (gs["screen_type"] == "COMBAT_REWARD" &&
                            any(r -> r["reward_type"] == "SAPPHIRE_KEY", gs["screen_state"]["rewards"]))
    if sapphire_key_offered
        sk_choice_i = find("sapphire_key", gs["choice_list"])-1
        @assert sk_choice_i >= 1
        add_encoded_choice(agent.choice_encoder, :choose_sapphire_key, nothing, ("choose", sk_choice_i))
        linked_relic = gs["screen_state"]["rewards"][sk_choice_i]["relic"]
        add_encoded_choice(agent.choice_encoder, :choose_relic, relics_encoder([linked_relic]), ("choose", sk_choice_i-1))
        return
    end
    if gs["screen_type"] == "BOSS_REWARD"
        for (i, relic) in enumerate(gs["screen_state"]["relics"])
            add_encoded_choice(agent.choice_encoder, :choose_relic, relics_encoder([relic]), ("choose", i-1))
        end
        return
    end
    @assert !any(p -> p["id"] == "Potion Slot", gs["potions"])
    @assert gs["screen_state"]["rewards"][1]["reward_type"] == "POTION"
    @assert gs["screen_type"] == "COMBAT_REWARD"
    @assert "proceed" in sts_state["available_commands"]
    for (i, potion) in enumerate(gs["potions"])
        potion_action = potion["can_use"] ? "use" : "discard"
        add_encoded_choice(agent.choice_encoder, :give_up_potion, potions_encoder([potion]), ("potion", potion_action, i-1))
    end
    @assert gs["screen_state"]["rewards"][1]["reward_type"] == "POTION"
    potion_rewarded = gs["screen_state"]["rewards"][1]["potion"]
    add_encoded_choice(agent.choice_encoder, :give_up_potion, potions_encoder([potion_rewarded]), ("proceed",))
end

function action_probabilities(agent::RewardAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    choices_encoded, actions = encode_choices(agent.choice_encoder)
    action_weights = agent.policy(choices_encoded)
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    Zygote.@ignore @assert length(actions) == length(probabilities)
    actions, probabilities
end

function state_value(agent::RewardAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    state_encoded = encode_state(agent.choice_encoder)
    only(agent.critic(state_encoded))
end

function train!(agent::RewardAgent, ra::RootAgent)
    train_log = TBLogger("tb_logs/train_RewardAgent")
    train!(train_log, agent, ra)
end
