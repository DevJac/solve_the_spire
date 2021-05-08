export PotionAgent, action, train!

mutable struct PotionAgent
    choice_encoder
    policy
    critic
    policy_opt
    critic_opt
    sars
    last_floor_rewarded
end

function PotionAgent()
    choice_encoder = ChoiceEncoder(
        Dict(
            :potions        => VanillaNetwork(length(potions_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :relics         => VanillaNetwork(length(relics_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :player_combat  => VanillaNetwork(length(player_combat_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :player_basic   => VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50]),
            :deck           => PoolNetwork(length(card_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :hand           => PoolNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :draw           => PoolNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :discard        => PoolNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :monsters       => PoolNetwork(length(monster_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
        ),
        Dict(
            :no_potion      => NullNetwork(),
            :potion         => VanillaNetwork(length(potions_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
        ),
        STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_CRITIC_LAYERS)
    PotionAgent(
        choice_encoder,
        policy,
        critic,
        STANDARD_OPTIMIZER(),
        STANDARD_OPTIMIZER(),
        SARS(),
        0)
end

function action(agent::PotionAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            @assert awaiting(agent.sars) == sar_reward || !any(s -> s["game_state"]["seed"] == gs["seed"], agent.sars.states)
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded + floor_partial_credit(ra)
                @assert r >= 0
                agent.last_floor_rewarded = 0
                add_reward(agent.sars, r, 0)
                log_value(ra.tb_log, "PotionAgent/reward", r)
                log_value(ra.tb_log, "PotionAgent/length_sars", length(agent.sars.rewards))
            end
        elseif any(a -> a[1] == "potion" && a[2] == "use", all_valid_actions(sts_state))
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded
                @assert r >= 0
                agent.last_floor_rewarded = gs["floor"]
                add_reward(agent.sars, r)
                log_value(ra.tb_log, "PotionAgent/reward", r)
                log_value(ra.tb_log, "PotionAgent/length_sars", length(agent.sars.rewards))
            end
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            log_value(ra.tb_log, "PotionAgent/state_value", state_value(agent, ra, sts_state))
            log_value(ra.tb_log, "PotionAgent/step_explore", explore_odds(probabilities))
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            if isempty(actions[action_i]); return nothing end
            action = join(actions[action_i], " ")
            @assert isa(action, String)
            action
        end
    end
end

function setup_choice_encoder(agent::PotionAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    ChoiceEncoders.reset!(agent.choice_encoder)
    add_encoded_state(agent.choice_encoder, :potions, potions_encoder(gs["potions"]))
    add_encoded_state(agent.choice_encoder, :relics, relics_encoder(gs["relics"]))
    add_encoded_state(
        agent.choice_encoder,
        :player_combat,
        encode_seq(player_combat_encoder, "combat_state" in keys(gs) ? [sts_state] : []))
    add_encoded_state(agent.choice_encoder, :player_basic, player_basic_encoder(sts_state))
    add_encoded_state(agent.choice_encoder, :deck, reduce(hcat, map(card_encoder, gs["deck"])))
    add_encoded_state(
        agent.choice_encoder,
        :hand,
        encode_seq(card_encoder, "combat_state" in keys(gs) ? gs["combat_state"]["hand"] : []))
    add_encoded_state(
        agent.choice_encoder,
        :draw,
        encode_seq(card_encoder, "combat_state" in keys(gs) ? gs["combat_state"]["draw_pile"] : []))
    add_encoded_state(
        agent.choice_encoder,
        :discard,
        encode_seq(card_encoder, "combat_state" in keys(gs) ? gs["combat_state"]["discard_pile"] : []))
    add_encoded_state(
        agent.choice_encoder,
        :monsters,
        encode_seq(monster_encoder, "combat_state" in keys(gs) ? gs["combat_state"]["monsters"] : []))
    add_encoded_choice(agent.choice_encoder, :no_potion, nothing, ())
    for action in all_valid_actions(sts_state)
        if action[1] != "potion"
            continue
        elseif action[1] == "potion" && action[2] == "use"
            choice_i = action[3]+1
            add_encoded_choice(agent.choice_encoder, :potion, potions_encoder([gs["potions"][choice_i]]), action)
            continue
        elseif action[1] == "potion" && action[2] == "discard"
            continue
        end
        @error "Unhandled action" action
        throw("Unhandled action")
    end
end

function action_probabilities(agent::PotionAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    choices_encoded, actions = encode_choices(agent.choice_encoder)
    action_weights = agent.policy(choices_encoded)
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    Zygote.@ignore @assert length(actions) == length(probabilities)
    actions, probabilities
end

function state_value(agent::PotionAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    state_encoded = encode_state(agent.choice_encoder)
    only(agent.critic(state_encoded))
end

function train!(agent::PotionAgent, ra::RootAgent)
    train_log = TBLogger("tb_logs/train_PotionAgent")
    train!(train_log, agent, ra)
end
