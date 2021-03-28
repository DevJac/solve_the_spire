export MapAgent, action, train!

mutable struct MapAgent
end
function MapAgent()
end

function action(agent::MapAgent, ra::RootAgent, sts_state)
end

function train!(agent::MapAgent, ra::RootAgent, epochs=1000)
    train_log = TBLogger("tb_logs/train_MapAgent")
    sars = fill_q(agent.sars)
    log_histogram(ra.tb_log, "MapAgent/rewards", map(sar -> sar.reward, sars))
    log_histogram(ra.tb_log, "MapAgent/q", map(sar -> sar.q, sars))
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
            agent.player_embedder,
            agent.deck_embedder,
            agent.relics_embedder,
            agent.potions_embedder,
            agent.all_map_embedder,
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
    log_value(ra.tb_log, "MapAgent/policy_loss", loss)
    log_value(ra.tb_log, "MapAgent/kl_div", mean(kl_divs))
    log_value(ra.tb_log, "MapAgent/actual_value", mean(actual_value))
    log_value(ra.tb_log, "MapAgent/estimated_value", mean(estimated_value))
    log_value(ra.tb_log, "MapAgent/estimated_advantage", mean(estimated_advantage))
    log_value(ra.tb_log, "MapAgent/entropy", mean(entropys))
    log_value(ra.tb_log, "MapAgent/explore", mean(explore))
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
    log_value(ra.tb_log, "MapAgent/critic_loss", loss)
    empty!(agent.sars)
end

function action_probabilities(agent::MapAgent, ra::RootAgent, sts_state)
end

function state_value(agent::MapAgent, ra::RootAgent, sts_state)
end
