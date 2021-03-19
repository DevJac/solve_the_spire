export EventAgent, action, train!

struct EventAgent
end

function action(agent::EventAgent, ra::RootAgent, sts_state, handled)
    if "game_state" in keys(sts_state)
        choices = filter(c -> "choice_index" in keys(c), gs["screen_state"]["options"])
        choice_index = sample(map(c -> c["choice_index"], choices))
        return "choose $choice_index"
    end
end

function train!(agent::EventAgent, ra::RootAgent)
end
