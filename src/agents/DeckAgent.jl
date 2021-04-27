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
        RMSProp(0.000_1),
        RMSProp(0.000_1),
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

function train!(agent::DeckAgent, ra::RootAgent, epochs=STANDARD_TRAINING_EPOCHS)
    train_log = TBLogger("tb_logs/train_DeckAgent")
    sars = fill_q(agent.sars)
    if length(sars) < 2; return end
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
                advantage = Zygote.@ignore sar.q_norm - state_value(target_agent, ra, sar.state)
                Zygote.ignore() do
                    push!(kl_divs, Flux.Losses.kldivergence(online_aps, target_aps))
                    push!(actual_value, online_ap * sar.q_norm)
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
    log_value(ra.tb_log, "DeckAgent/policy_loss", loss)
    log_value(ra.tb_log, "DeckAgent/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "DeckAgent/actual_value", mean(actual_value))
    log_value(ra.tb_log, "DeckAgent/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "DeckAgent/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "DeckAgent/entropy", mean(entropys))
    log_value(ra.tb_log, "DeckAgent/explore", mean(explore))
    for (epoch, batch) in enumerate(Batcher(sars, 100))
        if epoch > epochs; break end
        prms = params(agent.critic)
        loss, grads = valgrad(prms) do
            mean(batch) do sar
                predicted_q = state_value(agent, ra, sar.state)
                actual_q = sar.q_norm
                (predicted_q - actual_q)^2
            end
        end
        log_value(train_log, "train/critic_loss", loss, step=epoch)
        Flux.Optimise.update!(agent.critic_opt, prms, grads)
    end
    log_value(ra.tb_log, "DeckAgent/critic_loss", loss)
    empty!(agent.sars)
end
