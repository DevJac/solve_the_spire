export StartGameOverAgent, action, train!

struct StartGameOverAgent
    gen_floor_reached :: Vector{Float32}
    gen_score         :: Vector{Float32}
    gen_victory       :: Vector{Float32}
end
StartGameOverAgent() = StartGameOverAgent([], [], [])

function action(agent::StartGameOverAgent, ra::RootAgent, sts_state, handled)
    if "in_game" in keys(sts_state) && !sts_state["in_game"]
        return (true, "start silent")
    end
    if "game_state" in keys(sts_state) && sts_state["screen_type"] == "GAME_OVER"
        gs = sts_state["game_state"]
        push!(agent.gen_floor_reached, gs["floor"])
        push!(agent.gen_score, gs["screen_state"]["score"])
        push!(agent.gen_victory, gs["screen_state"]["victory"])
        log_value(ra.tb_log, "StartGameOverAgent/floor_reached", gs["floor"])
        log_value(ra.tb_log, "StartGameOverAgent/score", gs["screen_state"]["score"])
        log_value(ra.tb_log, "StartGameOverAgent/victory", Float32(gs["screen_state"]["victory"]))
        return (true, "proceed")
    end
    return (false, nothing)
end

function train!(agent::StartGameOverAgent, ra::RootAgent)
    log_value(ra.tb_log, "StartGameOverAgent/gen/mean_floor_reached", mean(agent.gen_floor_reached))
    log_value(ra.tb_log, "StartGameOverAgent/gen/mean_score", mean(agent.gen_score))
    log_value(ra.tb_log, "StartGameOverAgent/gen/mean_victory", mean(agent.gen_victory))
    empty!(agent.gen_floor_reached)
    empty!(agent.gen_score)
    empty!(agent.gen_victory)
end
