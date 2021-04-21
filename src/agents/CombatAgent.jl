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
            :potions      => VanillaNetwork(length(potions_encoder), 20, [50]),
            :relics       => VanillaNetwork(length(relics_encoder), 20, [50]),
            :player       => VanillaNetwork(length(player_combat_encoder), 20, [50]),
            :hand         => PoolNetwork(length(card_encoder)+1, 20, [50]),
            :draw         => PoolNetwork(length(card_encoder)+1, 20, [50]),
            :discard      => PoolNetwork(length(card_encoder)+1, 20, [50]),
            :monsters     => PoolNetwork(length(monster_encoder), 20, [50])
        ),
        Dict(
            :card         => VanillaNetwork(length(card_encoder), 20, [50]),
            :card_monster => VanillaNetwork(sum(length, [card_encoder, monster_encoder]), 20, [50]),
            :end          => NullNetwork()
        ),
        20, [50])
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_CRITIC_LAYERS)
    CombatAgent(
        choice_encoder,
        policy,
        critic,
        ADADelta(),
        ADADelta(),
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
        if gs["screen_type"] in ("NONE", "COMBAT_REWARD", "MAP", "GAME_OVER")
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
                target_reward = monster_hp_loss_ratio - player_hp_loss_ratio
                if win; target_reward += 1 end
                if lose; target_reward -= 1 end
                r = target_reward - agent.last_rewarded_target
                agent.last_rewarded_target = target_reward
                add_reward(agent.sars, clamp(r, -1, 1), win || lose ? 0 : 1)
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

function train!(agent::CombatAgent, ra::RootAgent, epochs=STANDARD_TRAINING_EPOCHS)
    train_log = TBLogger("tb_logs/train_CombatAgent")
    sars = fill_q(agent.sars)
    if isempty(sars); return end
    for (epoch, batch) in enumerate(Batcher(sars, 100))
        if epoch > epochs; break end
        prms = params(agent.critic)
        loss, grads = valgrad(prms) do
            mean(batch) do sar
                predicted_q = state_value(agent, ra, sar.state)
                actual_q = sar.q
                (predicted_q - actual_q)^2
            end
        end
        log_value(train_log, "train/critic_loss", loss, step=epoch)
        Flux.Optimise.update!(agent.critic_opt, prms, grads)
    end
    log_value(ra.tb_log, "CombatAgent/critic_loss", loss)
    target_agent = deepcopy(agent)
    kl_div_smoother = Smoother()
    local loss
    kl_divs = Float32[]
    actual_value = Float32[]
    estimated_value = Float32[]
    estimated_advantage = Float32[]
    entropys = Float32[]
    explore = Float32[]
    for (epoch, batch) in enumerate(Batcher(sars, 10_000))
        prms = params(
            agent.choice_encoder,
            agent.policy)
        loss, grads = valgrad(prms) do
            -mean(batch) do sar
                target_aps = Zygote.@ignore action_probabilities(target_agent, ra, sar.state)[2]
                target_ap = Zygote.@ignore max(1e-8, target_aps[sar.action])
                online_aps = action_probabilities(agent, ra, sar.state)[2]
                online_ap = online_aps[sar.action]
                advantage = Zygote.@ignore sar.q - state_value(target_agent, ra, sar.state)
                Zygote.ignore() do
                    push!(kl_divs, Flux.Losses.kldivergence(online_aps, target_aps))
                    push!(actual_value, online_ap * sar.q)
                    push!(estimated_value, online_ap * state_value(target_agent, ra, sar.state))
                    push!(estimated_advantage, online_ap * advantage)
                    push!(entropys, entropy(online_aps))
                    push!(explore, explore_odds(online_aps))
                end
                min(
                    (online_ap / target_ap) * advantage,
                    clip(online_ap / target_ap, 0.2) * advantage)
            end
        end
        log_value(train_log, "train/policy_loss", loss, step=epoch)
        log_value(train_log, "train/kl_div", mean(kl_divs), step=epoch)
        log_value(train_log, "train/actual_value", mean(actual_value), step=epoch)
        log_value(train_log, "train/estimated_value", mean(estimated_value), step=epoch)
        log_value(train_log, "train/estimated_advantage", mean(estimated_advantage), step=epoch)
        log_value(train_log, "train/entropy", mean(entropys), step=epoch)
        log_value(train_log, "train/explore", mean(explore), step=epoch)
        Flux.Optimise.update!(agent.policy_opt, prms, grads)
        if epoch >= epochs || smooth!(kl_div_smoother, mean(kl_divs)) > STANDARD_KL_DIV_EARLY_STOP; break end
        empty!(kl_divs); empty!(actual_value); empty!(estimated_value); empty!(estimated_advantage)
        empty!(entropys); empty!(explore)
    end
    log_value(ra.tb_log, "CombatAgent/policy_loss", loss)
    log_value(ra.tb_log, "CombatAgent/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "CombatAgent/actual_value", mean(actual_value))
    log_value(ra.tb_log, "CombatAgent/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "CombatAgent/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "CombatAgent/entropy", mean(entropys))
    log_value(ra.tb_log, "CombatAgent/explore", mean(explore))
    empty!(agent.sars)
end
