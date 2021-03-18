export CampfireAgent, action, train!

struct CampfireAgent
end

function action(agent::CampfireAgent, ra::RootAgent, sts_state, handled)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "REST"
            if !in("choose", state["available_commands"])
                return (true, "proceed")
            end
            random_choice = sample(0:length(gs["screen_state"]["rest_options"])-1)
            return (true, "choose $random_choice")
        end
    end
    (false, nothing)
end

function train!(agent::CampfireAgent, ra::RootAgent)
end
