export SpecialActionAgent, action, train!

mutable struct SpecialActionAgent
    choice_encoder
    policy
    critic
    sars
    last_rewarded_target :: Float32
    hand_select_actions  :: Vector{String}
end

function SpecialActionAgent()
    hand_select_actions = readlines("game_data/hand_select_actions.txt")
    choice_encoder = ChoiceEncoder(
        Dict(
            :potions        => VanillaNetwork(length(potions_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :relics         => VanillaNetwork(length(relics_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :player         => VanillaNetwork(length(player_combat_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :hand           => PoolNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :draw           => PoolNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :discard        => PoolNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :monsters       => PoolNetwork(length(monster_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
        ),
        Dict(
            :choose_card    => VanillaNetwork(length(card_encoder) + length(hand_select_actions), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :confirm        => NullNetwork()
        ),
        STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_CRITIC_LAYERS)
    SpecialActionAgent(
        choice_encoder,
        policy,
        critic,
        SARS(),
        0,
        hand_select_actions)
end

function action(agent::SpecialActionAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] in ("HAND_SELECT", "COMBAT_REWARD", "MAP", "GAME_OVER") && awaiting(agent.sars) == sar_reward
            win = gs["screen_type"] in ("COMBAT_REWARD", "MAP")
            lose = gs["screen_type"] == "GAME_OVER"
            @assert !(win && lose)
            monster_hp_max = win ? 0 : maximum(m -> m["current_hp"], gs["combat_state"]["monsters"])
            monster_hp_sum = win ? 0 : sum(m -> m["current_hp"], gs["combat_state"]["monsters"])
            monster_hp_loss_ratio = win ? 1 : Float32(mean([
                1 - (monster_hp_max / initial_hp_stats(ra).monster_hp_max),
                1 - (monster_hp_sum / initial_hp_stats(ra).monster_hp_sum)]))
            player_hp_loss_ratio = lose ? 1 : 1 - (gs["current_hp"] / initial_hp_stats(ra).player_hp)
            target_reward = clamp(monster_hp_loss_ratio, 0, 1) - clamp(player_hp_loss_ratio, -1, 1) * 2
            if win; target_reward += 0.1 end
            if lose; target_reward -= 0.1 end
            r = target_reward - agent.last_rewarded_target
            agent.last_rewarded_target = target_reward
            add_reward(agent.sars, r, win || lose ? 0 : 1)
            log_value(ra.tb_log, "SpecialActionAgent/reward", r)
            log_value(ra.tb_log, "SpecialActionAgent/length_sars", length(agent.sars.rewards))
            if win || lose; agent.last_rewarded_target = 0 end
        end
        if gs["screen_type"] == "HAND_SELECT"
            if !in("choose", sts_state["available_commands"])
                return "proceed"
            end
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            log_value(ra.tb_log, "SpecialActionAgent/state_value", state_value(agent, ra, sts_state))
            log_value(ra.tb_log, "SpecialActionAgent/step_explore", explore_odds(probabilities))
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            action = join(actions[action_i], " ")
            @assert isa(action, String)
            action
        end
    end
end

function setup_choice_encoder(agent::SpecialActionAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    ChoiceEncoders.reset!(agent.choice_encoder)
    add_encoded_state(agent.choice_encoder, :potions, potions_encoder(gs["potions"]))
    add_encoded_state(agent.choice_encoder, :relics, relics_encoder(gs["relics"]))
    add_encoded_state(agent.choice_encoder, :player, player_combat_encoder(sts_state))
    add_encoded_state(agent.choice_encoder, :hand, encode_seq(card_encoder, gs["combat_state"]["hand"]))
    add_encoded_state(agent.choice_encoder, :draw, encode_seq(card_encoder, gs["combat_state"]["draw_pile"]))
    add_encoded_state(agent.choice_encoder, :discard, encode_seq(card_encoder, gs["combat_state"]["discard_pile"]))
    add_encoded_state(agent.choice_encoder, :monsters, reduce(hcat, map(monster_encoder, gs["combat_state"]["monsters"])))
    current_action_encoded = zeros(Float32, length(agent.hand_select_actions))
    current_action_i = find(get(gs, "current_action", ""), agent.hand_select_actions)
    if !isnothing(current_action_i); current_action_encoded[current_action_i] = 1 end
    @assert sum(current_action_encoded) in (0, 1)
    for action in all_valid_actions(sts_state)
        if action[1] == "potion"
            continue
        end
        if action[1] == "confirm"
            add_encoded_choice(agent.choice_encoder, :confirm, nothing, action)
            continue
        end
        if action[1] == "choose"
            choice_i = action[2]+1
            add_encoded_choice(
                agent.choice_encoder,
                :choose_card,
                [current_action_encoded; card_encoder(gs["screen_state"]["hand"][choice_i])],
                action)
            continue
        end
        @error "Unhandled action" action
        throw("Unhandled action")
    end
end

function action_probabilities(agent::SpecialActionAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    choices_encoded, actions = encode_choices(agent.choice_encoder)
    action_weights = agent.policy(choices_encoded)
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    Zygote.@ignore @assert length(actions) == length(probabilities)
    actions, probabilities
end

function state_value(agent::SpecialActionAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    state_encoded = encode_state(agent.choice_encoder)
    only(agent.critic(state_encoded))
end

function train!(agent::SpecialActionAgent, ra::RootAgent)
    train_log = TBLogger("tb_logs/train_SpecialActionAgent")
    train!(train_log, agent, ra)
end
