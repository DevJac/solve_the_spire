export EventAgent, action, train!

struct EventAgent
end

function action(agent::EventAgent, ra::RootAgent, sts_state)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "EVENT"
            choices = filter(c -> "choice_index" in keys(c), gs["screen_state"]["options"])
            choice_index = sample(map(c -> c["choice_index"], choices))
            return "choose $choice_index"
        end
    end
end

function train!(agent::EventAgent, ra::RootAgent)
end
