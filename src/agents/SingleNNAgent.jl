export SingleNNAgent, action, train!

mutable struct SingleNNAgent
    json_words
    actions
    path_embedder
    value_embedder
    pooler
    policy
    critic
    policy_opt
    critic_opt
    sars
    last_floor_rewarded
end
function SingleNNAgent()
    json_words = readlines("game_data/json_path_words.txt")
    @assert length(json_words) == 108
    actions = make_actions()
    path_embedder = GRUNetwork(108+1, 300, [300])
    value_embedder = GRUNetwork(95+2, 300, [300])
    pooler = PoolNetwork(600, 600, [600])
    policy = VanillaNetwork(600, length(actions), [600])
    critic = VanillaNetwork(600, 1, [600])
    SingleNNAgent(
        json_words,
        actions,
        path_embedder,
        value_embedder,
        pooler,
        policy,
        critic,
        ADADelta(),
        ADADelta(),
        SARS(),
        0)
end

function action(agent::SingleNNAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            if awaiting(agent.sars) == sar_reward
                r = gs["floor"] - agent.last_floor_rewarded
                agent.last_floor_rewarded = 0
                add_reward(agent.sars, r, 0)
                log_value(ra.tb_log, "DeckAgent/reward", r)
                log_value(ra.tb_log, "DeckAgent/length_sars", length(agent.sars.rewards))
            end
        else
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

function action_probabilities(agent::SingleNNAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    flat_gs = Zygote.@ignore flatten_json(gs)
    hyperpoints = Vector{Float32}[]
    for (path, value) in flat_gs
        local p_e
        local v_e
        Zygote.ignore() do
            p_e = [encode_path_word(agent, word) for word in path]
            v_e = encode_path_value(value)
        end
        local hyperpoint_top
        local hyperpoint_bot
        Flux.reset!(agent.path_embedder)
        for e in p_e
            hyperpoint_top = agent.path_embedder(e)
        end
        Flux.reset!(agent.value_embedder)
        for e in v_e
            hyperpoint_bot = agent.value_embedder(e)
        end
        push!(hyperpoints, [hyperpoint_top; hyperpoint_bot])
    end
    pool = agent.pooler(reduce(hcat, hyperpoints))
    action_weights = agent.policy(pool)
    action_mask = Zygote.ignore() do
        actions = collect(enumerate(agent.actions))
        action_mask = zeros(length(actions))
        actions = filter(a -> a[2][1] in sts_state["available_commands"], actions)
        actions = filter(a -> (a[2][1] != "choose" ||
                               a[2][2] < length(gs["choice_list"])), actions)
        actions = filter(a -> (a[2][1] != "potion" || a[2][2] != "use" ||
                               a[2][3] < length(gs["potions"]) &&
                               gs["potions"][a[2][3]+1]["can_use"]), actions)
        actions = filter(a -> (a[2][1] != "potion" || a[2][2] != "discard" ||
                               a[2][3] < length(gs["potions"]) &&
                               gs["potions"][a[2][3]+1]["can_discard"]), actions)
        actions = filter(a -> (a[2][1] != "play" || length(a[2]) != 2 ||
                               a[2][2] <= length(gs["combat_state"]["hand"]) &&
                               !gs["combat_state"]["hand"][a[2][2]]["has_target"]), actions)
        actions = filter(a -> (a[2][1] != "play" || length(a[2]) != 3 ||
                               a[2][2] <= length(gs["combat_state"]["hand"]) &&
                               a[2][3] < length(gs["combat_state"]["monsters"]) &&
                               gs["combat_state"]["hand"][a[2][2]]["has_target"] &&
                               !gs["combat_state"]["monsters"][a[2][3]+1]["is_gone"]), actions)
        for a in actions
            action_mask[a[1]] = 1
        end
        action_mask
    end
    probabilities = softmax(action_weights .* action_mask)
    Zygote.@ignore @assert length(agent.actions) == length(probabilities)
    agent.actions, probabilities
end

function state_value(agent::SingleNNAgent, ra::RootAgent, sts_state)
    gs = sts_state["game_state"]
    flat_gs = Zygote.@ignore flatten_json(gs)
    hyperpoints = []
    for (path, value) in flat_gs
        local p_e
        local v_e
        Zygote.ignore() do
            p_e = [encode_path_word(agent, word) for word in path]
            v_e = encode_path_value(value)
        end
        local hyperpoint_top
        local hyperpoint_bot
        Flux.reset!(agent.path_embedder)
        for e in p_e
            hyperpoint_top = agent.path_embedder(e)
        end
        Flux.reset!(agent.value_embedder)
        for e in v_e
            hyperpoint_bot = agent.value_embedder(e)
        end
        push!(hyperpoints, [hyperpoint_top; hyperpoint_bot])
    end
    pool = agent.pooler(reduce(hcat, hyperpoints))
    only(agent.critic(pool))
end

function train!(agent::SingleNNAgent, ra::RootAgent, epochs=1000)
    train_log = TBLogger("tb_logs/train_SingleNNAgent")
    sars = fill_q(agent.sars, 0.5^(1/1000))
    log_histogram(ra.tb_log, "SingleNNAgent/rewards", map(sar -> sar.reward, sars))
    log_histogram(ra.tb_log, "SingleNNAgent/q", map(sar -> sar.q, sars))
    target_agent = deepcopy(agent)
    kl_div_smoother = Smoother()
    local loss
    kl_divs = Float32[]
    actual_value = Float32[]
    estimated_value = Float32[]
    estimated_advantage = Float32[]
    entropys = Float32[]
    explore = Float32[]
    for epoch in 1:epochs
        batch = sars
        prms = params(
            agent.path_embedder,
            agent.value_embedder,
            agent.pooler,
            agent.policy)
        loss, grads = valgrad(prms) do
            -mean(batch) do sar
                target_aps = action_probabilities(target_agent, ra, sar.state)[2]
                target_ap = target_aps[sar.action]
                online_aps = action_probabilities(agent, ra, sar.state)[2]
                online_ap = online_aps[sar.action]
                advantage = sar.q - state_value(target_agent, ra, sar.state)
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
        if smooth!(kl_div_smoother, mean(kl_divs)) > 0.01; break end
        empty!(kl_divs); empty!(actual_value); empty!(estimated_value); empty!(estimated_advantage)
        empty!(entropys); empty!(explore)
    end
    log_value(ra.tb_log, "SingleNNAgent/policy_loss", loss)
    log_value(ra.tb_log, "SingleNNAgent/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "SingleNNAgent/actual_value", mean(actual_value))
    log_value(ra.tb_log, "SingleNNAgent/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "SingleNNAgent/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "SingleNNAgent/entropy", mean(entropys))
    log_value(ra.tb_log, "SingleNNAgent/explore", mean(explore))
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
    log_value(ra.tb_log, "SingleNNAgent/critic_loss", loss)
    empty!(agent.sars)
end

@memoize function flatten_json(json)
    path_values = Dict()
    function recurse(path, json)
        if isa(json, Dict)
            for (k, v) in json
                recurse([path; k], v)
            end
        elseif isa(json, Array)
            for (i, v) in enumerate(json)
                recurse([path; i], v)
            end
        else
            path_values[path] = json
        end
    end
    recurse([], json)
    path_values
end

function make_actions()
    actions = []
    for card_i in 1:10
        push!(actions, ("play", card_i))
        for monster_i in 0:4
            push!(actions, ("play", card_i, monster_i))
        end
    end
    for potion_i in 0:4
        push!(actions, ("potion", "use", potion_i))
        push!(actions, ("potion", "discard", potion_i))
    end
    for choice_i in 0:29
        push!(actions, ("choose", choice_i))
    end
    push!(actions, ("end",))
    push!(actions, ("proceed",))
    push!(actions, ("return",))
end

function encode_path_word(agent::SingleNNAgent, word)
    r = zeros(length(agent.json_words)+1)
    if isa(word, Integer)
        r[end] = word+1
        @assert sum(r) == word+1
    else
        r[find(word, agent.json_words)] = 1
        @assert sum(r) == 1
    end
    r
end

function encode_path_value(value)
    if isa(value, String) && length(value) > 0
        [encode_char(c) for c in value]
    elseif isa(value, String) && length(value) == 0
        r = zeros(95+2)
        r[end-1] = 1
        [r]
    else
        r = zeros(95+2)
        r[end] = (Float32(value) - 0.5) * 2
        [r]
    end
end

function encode_char(c)
    r = zeros(95+2)
    r[Int(c) - 31] = 1
    @assert sum(r) == 1
    r
end
