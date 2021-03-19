export MenuAgent, action, train!

struct MenuAgent
    gen_floor_reached :: Vector{Float32}
    gen_score         :: Vector{Float32}
    gen_victory       :: Vector{Float32}
end
MenuAgent() = MenuAgent([], [], [])

function action(agent::MenuAgent, ra::RootAgent, sts_state, handled)
    if "in_game" in keys(sts_state) && !sts_state["in_game"]
        return "start silent"
    end
    if "game_state" in keys(sts_state) && sts_state["screen_type"] == "GAME_OVER"
        gs = sts_state["game_state"]
        push!(agent.gen_floor_reached, gs["floor"])
        push!(agent.gen_score, gs["screen_state"]["score"])
        push!(agent.gen_victory, gs["screen_state"]["victory"])
        log_value(ra.tb_log, "MenuAgent/floor_reached", gs["floor"])
        log_value(ra.tb_log, "MenuAgent/score", gs["screen_state"]["score"])
        log_value(ra.tb_log, "MenuAgent/victory", Float32(gs["screen_state"]["victory"]))
        return "proceed"
    end
    nothing
end

function train!(agent::MenuAgent, ra::RootAgent)
    log_value(ra.tb_log, "MenuAgent/gen/mean_floor_reached", mean(agent.gen_floor_reached))
    log_value(ra.tb_log, "MenuAgent/gen/mean_score", mean(agent.gen_score))
    log_value(ra.tb_log, "MenuAgent/gen/mean_victory", mean(agent.gen_victory))
    empty!(agent.gen_floor_reached)
    empty!(agent.gen_score)
    empty!(agent.gen_victory)
end
