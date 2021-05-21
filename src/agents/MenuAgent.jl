export MenuAgent, action, train!

mutable struct MenuAgent
    gen_floor_reached  :: Vector{Float32}
    gen_score          :: Vector{Float32}
    gen_victory        :: Vector{Float32}
    mean_floor_reached :: Vector{Float32}
end
MenuAgent() = MenuAgent([], [], [], [0])

function action(agent::MenuAgent, ra::RootAgent, sts_state)
    if "in_game" in keys(sts_state) && !sts_state["in_game"]
        return "start silent"
    end
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "GAME_OVER"
            floor_reached = gs["floor"]
            push!(agent.gen_floor_reached, floor_reached)
            push!(agent.gen_score, gs["screen_state"]["score"])
            push!(agent.gen_victory, gs["screen_state"]["victory"])
            log_value(ra.tb_log, "MenuAgent/floor_reached", floor_reached)
            log_value(ra.tb_log, "MenuAgent/score", gs["screen_state"]["score"])
            log_value(ra.tb_log, "MenuAgent/victory", Float32(gs["screen_state"]["victory"]))
            println("Floor reached: $floor_reached ($(round(mean(agent.gen_floor_reached), digits=1)))")
            return "proceed"
        end
    end
end

function train!(agent::MenuAgent, ra::RootAgent)
    mfr = mean(agent.gen_floor_reached)
    log_value(ra.tb_log, "MenuAgent/gen/mean_floor_reached", mfr)
    log_value(ra.tb_log, "MenuAgent/gen/mean_score", mean(agent.gen_score))
    log_value(ra.tb_log, "MenuAgent/gen/mean_victory", mean(agent.gen_victory))
    below_max = round(mfr - maximum(agent.mean_floor_reached), digits=1)
    push!(agent.mean_floor_reached, mfr)
    since_max = length(agent.mean_floor_reached) - argmax(agent.mean_floor_reached)
    pushover(strip("Mean floor reached: $mfr ($below_max, $since_max)"))
    empty!(agent.gen_floor_reached)
    empty!(agent.gen_score)
    empty!(agent.gen_victory)
end
