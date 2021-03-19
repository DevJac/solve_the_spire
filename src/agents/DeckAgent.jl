export DeckAgent, action, train!

struct DeckAgent
end

function action(agent::DeckAgent, ra::RootAgent, sts_state, handled)
    if "game_state" in keys(sts_state)
        gs = sts_state["game_state"]
        if gs["screen_type"] == "CARD_REWARD"
            if !in("choose", state["available_commands"])
                return "proceed"
            end
            return "choose 0"
        end
        if gs["screen_type"] == "GRID"
            if !in("choose", state["available_commands"])
                return "confirm"
            end
            random_choice = sample(0:length(gs["choice_list"])-1)
            return "choose $random_choice"
        end
    end
    nothing
end

function train!(agent::DeckAgent, ra::RootAgent)
end
