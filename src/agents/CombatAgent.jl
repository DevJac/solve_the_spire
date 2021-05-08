export CombatAgent, action, train!

struct InitialHPStats
    floor          :: Int
    monster_hp_max :: Float32
    monster_hp_sum :: Float32
    player_hp      :: Float32
end

mutable struct CombatAgent
    choice_encoder
    policy
    critic
    policy_opt
    critic_opt
    sars
    initial_hp_stats     :: InitialHPStats
    floor_partial_credit :: Float32
    last_rewarded_target :: Float32
end

function CombatAgent()
    choice_encoder = ChoiceEncoder(
        Dict(
            :potions      => VanillaNetwork(length(potions_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :relics       => VanillaNetwork(length(relics_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :player       => VanillaNetwork(length(player_combat_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :hand         => PoolNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :draw         => PoolNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :discard      => PoolNetwork(length(card_encoder)+1, STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :monsters     => PoolNetwork(length(monster_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
        ),
        Dict(
            :card         => VanillaNetwork(length(card_encoder), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :card_monster => VanillaNetwork(sum(length, [card_encoder, monster_encoder]), STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS),
            :end          => NullNetwork()
        ),
        STANDARD_EMBEDDER_OUT, STANDARD_EMBEDDER_LAYERS)
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_CRITIC_LAYERS)
    CombatAgent(
        choice_encoder,
        policy,
        critic,
        STANDARD_OPTIMIZER(),
        STANDARD_OPTIMIZER(),
        SARS(),
        InitialHPStats(0, 0, 0, 0), 0, 0)
end

function action(agent::CombatAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["floor"] != agent.initial_hp_stats.floor
            agent.initial_hp_stats = InitialHPStats(0, 0, 0, 0)
            agent.floor_partial_credit = 0
            agent.last_rewarded_target = 0
        end
        if (gs["screen_type"] in ("NONE", "COMBAT_REWARD", "MAP", "GAME_OVER") &&
            (gs["screen_type"] != "GAME_OVER" || "combat_state" in keys(gs)))
            win = gs["screen_type"] in ("COMBAT_REWARD", "MAP")
            lose = gs["screen_type"] == "GAME_OVER"
            @assert !(win && lose)
            if gs["floor"] != agent.initial_hp_stats.floor
                monster_hp_max = win ? 0 : maximum(m -> m["current_hp"], gs["combat_state"]["monsters"])
                monster_hp_sum = win ? 0 : sum(m -> m["current_hp"], gs["combat_state"]["monsters"])
                agent.initial_hp_stats = InitialHPStats(gs["floor"], monster_hp_max, monster_hp_sum, gs["current_hp"])
            end
            monster_hp_max = win ? 0 : maximum(m -> m["current_hp"], gs["combat_state"]["monsters"])
            monster_hp_sum = win ? 0 : sum(m -> m["current_hp"], gs["combat_state"]["monsters"])
            monster_hp_loss_ratio = win ? 1 : Float32(mean([
                1 - (monster_hp_max / agent.initial_hp_stats.monster_hp_max),
                1 - (monster_hp_sum / agent.initial_hp_stats.monster_hp_sum)]))
            agent.floor_partial_credit = monster_hp_loss_ratio
            if awaiting(agent.sars) == sar_reward
                player_hp_loss_ratio = lose ? 1 : 1 - (gs["current_hp"] / agent.initial_hp_stats.player_hp)
                target_reward = clamp(monster_hp_loss_ratio, 0, 1) - clamp(player_hp_loss_ratio, -1, 1) * 2
                if win; target_reward += 0.1 end
                if lose; target_reward -= 0.1 end
                r = target_reward - agent.last_rewarded_target
                agent.last_rewarded_target = target_reward
                add_reward(agent.sars, r, win || lose ? 0 : 1)
                log_value(ra.tb_log, "CombatAgent/reward", r)
                log_value(ra.tb_log, "CombatAgent/length_sars", length(agent.sars.rewards))
            end
        end
        if gs["screen_type"] == "NONE"
            actions, probabilities = action_probabilities(agent, ra, sts_state)
            log_value(ra.tb_log, "CombatAgent/state_value", state_value(agent, ra, sts_state))
            log_value(ra.tb_log, "CombatAgent/step_explore", explore_odds(probabilities))
            action_i = sample(1:length(actions), Weights(probabilities))
            add_state(agent.sars, sts_state)
            add_action(agent.sars, action_i)
            action = join(actions[action_i], " ")
            @assert isa(action, String)
            action
        end
    end
end

function setup_choice_encoder(agent::CombatAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    ChoiceEncoders.reset!(agent.choice_encoder)
    add_encoded_state(agent.choice_encoder, :potions, potions_encoder(gs["potions"]))
    add_encoded_state(agent.choice_encoder, :relics, relics_encoder(gs["relics"]))
    add_encoded_state(agent.choice_encoder, :player, player_combat_encoder(sts_state))
    add_encoded_state(agent.choice_encoder, :hand, encode_seq(card_encoder, gs["combat_state"]["hand"]))
    add_encoded_state(agent.choice_encoder, :draw, encode_seq(card_encoder, gs["combat_state"]["draw_pile"]))
    add_encoded_state(agent.choice_encoder, :discard, encode_seq(card_encoder, gs["combat_state"]["discard_pile"]))
    add_encoded_state(agent.choice_encoder, :monsters, reduce(hcat, map(monster_encoder, gs["combat_state"]["monsters"])))
    for action in all_valid_actions(sts_state)
        if action[1] == "potion"
            continue
        end
        if action == ("end",)
            add_encoded_choice(agent.choice_encoder, :end, nothing, action)
            continue
        end
        if action[1] == "play" && length(action) == 2
            card_i = action[2]
            add_encoded_choice(agent.choice_encoder, :card, card_encoder(gs["combat_state"]["hand"][card_i]), action)
            continue
        end
        if action[1] == "play" && length(action) == 3
            card_i = action[2]
            monster_i = action[3]+1
            add_encoded_choice(
                agent.choice_encoder,
                :card_monster,
                [card_encoder(gs["combat_state"]["hand"][card_i]); monster_encoder(gs["combat_state"]["monsters"][monster_i])],
                action)
            continue
        end
        @error "Unhandled action" action
        throw("Unhandled action")
    end
end

function action_probabilities(agent::CombatAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    choices_encoded, actions = encode_choices(agent.choice_encoder)
    action_weights = agent.policy(choices_encoded)
    probabilities = softmax(reshape(action_weights, length(action_weights)))
    Zygote.@ignore @assert length(actions) == length(probabilities)
    actions, probabilities
end

function state_value(agent::CombatAgent, ra::RootAgent, sts_state)
    Zygote.@ignore setup_choice_encoder(agent, ra, sts_state)
    state_encoded = encode_state(agent.choice_encoder)
    only(agent.critic(state_encoded))
end

function train!(agent::CombatAgent, ra::RootAgent)
    train_log = TBLogger("tb_logs/train_CombatAgent")
    train!(train_log, agent, ra)
end
