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
            :potions        => VanillaNetwork(length(potions_encoder), 20, [50]),
            :relics         => VanillaNetwork(length(relics_encoder), 20, [50]),
            :player_combat  => VanillaNetwork(length(player_combat_encoder)+1, 20, [50]),
            :player_basic   => VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50]),
            :deck           => PoolNetwork(length(card_encoder), 20, [50]),
            :hand           => PoolNetwork(length(card_encoder)+1, 20, [50]),
            :draw           => PoolNetwork(length(card_encoder)+1, 20, [50]),
            :discard        => PoolNetwork(length(card_encoder)+1, 20, [50]),
            :monsters       => PoolNetwork(length(monster_encoder)+1, 20, [50])
        ),
        Dict(
            :no_potion      => NullNetwork(),
            :potion         => VanillaNetwork(length(potions_encoder), 20, [50])
        ),
        20, [50])
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    PotionAgent(
        choice_encoder,
        policy,
        critic,
        ADADelta(),
        ADADelta(),
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
                agent.last_floor_rewarded = 0
                add_reward(agent.sars, r, 0)
                log_value(ra.tb_log, "PotionAgent/reward", r)
                log_value(ra.tb_log, "PotionAgent/length_sars", length(agent.sars.rewards))
            end
        elseif any(a -> a[1] == "potion" && a[2] == "use", all_valid_actions(sts_state))
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded
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

function train!(agent::PotionAgent, ra::RootAgent, epochs=STANDARD_TRAINING_EPOCHS)
    train_log = TBLogger("tb_logs/train_PotionAgent")
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
    log_value(ra.tb_log, "PotionAgent/critic_loss", loss)
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
        if epoch > epochs; break end
        prms = params(
            agent.choice_encoder,
            agent.policy)
        loss, grads = valgrad(prms) do
            -mean(batch) do sar
                target_aps = Zygote.@ignore action_probabilities(target_agent, ra, sar.state)[2]
                target_ap = Zygote.@ignore target_aps[sar.action]
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
        if smooth!(kl_div_smoother, mean(kl_divs)) > STANDARD_KL_DIV_EARLY_STOP; break end
        empty!(kl_divs); empty!(actual_value); empty!(estimated_value); empty!(estimated_advantage)
        empty!(entropys); empty!(explore)
    end
    log_value(ra.tb_log, "PotionAgent/policy_loss", loss)
    log_value(ra.tb_log, "PotionAgent/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "PotionAgent/actual_value", mean(actual_value))
    log_value(ra.tb_log, "PotionAgent/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "PotionAgent/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "PotionAgent/entropy", mean(entropys))
    log_value(ra.tb_log, "PotionAgent/explore", mean(explore))
    empty!(agent.sars)
end
