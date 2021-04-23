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
            :potions        => VanillaNetwork(length(potions_encoder), 20, [50]),
            :relics         => VanillaNetwork(length(relics_encoder), 20, [50]),
            :player_basic   => VanillaNetwork(length(player_basic_encoder), length(player_basic_encoder), [50]),
            :deck           => PoolNetwork(length(card_encoder), 20, [50])
        ),
        Dict(
            :buy_card       => VanillaNetwork(length(card_encoder)+1, 20, [50]),
            :buy_relic      => VanillaNetwork(length(relics_encoder)+1, 20, [50]),
            :buy_potion     => VanillaNetwork(length(potions_encoder)+1, 20, [50]),
            :discard_potion => VanillaNetwork(length(potions_encoder), 20, [50]),
            :purge_card     => VanillaNetwork(1, 1, [50]),
            :leave          => NullNetwork()
        ),
        20, [50])
    policy = VanillaNetwork(length(choice_encoder), 1, STANDARD_POLICY_LAYERS)
    critic = VanillaNetwork(state_length(choice_encoder), 1, STANDARD_CRITIC_LAYERS)
    ShopAgent(
        choice_encoder,
        policy,
        critic,
        ADADelta(),
        ADADelta(),
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

function train!(agent::ShopAgent, ra::RootAgent, epochs=STANDARD_TRAINING_EPOCHS)
    train_log = TBLogger("tb_logs/train_ShopAgent")
    sars = fill_q(agent.sars)
    if isempty(sars); return end
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
    log_value(ra.tb_log, "ShopAgent/policy_loss", loss)
    log_value(ra.tb_log, "ShopAgent/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "ShopAgent/actual_value", mean(actual_value))
    log_value(ra.tb_log, "ShopAgent/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "ShopAgent/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "ShopAgent/entropy", mean(entropys))
    log_value(ra.tb_log, "ShopAgent/explore", mean(explore))
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
    log_value(ra.tb_log, "ShopAgent/critic_loss", loss)
    empty!(agent.sars)
end
